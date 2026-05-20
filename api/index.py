import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app"))

from pure_ml import PureStandardScaler, PureGradientBoostingClassifier
from data_preprocessing import CONTINUOUS_FEATURES
from explainability import compute_single_patient_shap

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cardiovascular Decision Support System API",
    description="Cost-sensitive staged cardiac diagnosis & Explainable AI backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Patient Input Schema ─────────────────────────────────────────────────────
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# ── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR        = os.path.join(ROOT, "models")
SCALER_PATH       = os.path.join(MODELS_DIR, "scaler.joblib")
EXPLAIN_META_PATH = os.path.join(MODELS_DIR, "explainability_metadata.joblib")
COST_SUMMARY_PATH = os.path.join(MODELS_DIR, "cost_analysis_summary.joblib")
ESCALATION_PATH   = os.path.join(MODELS_DIR, "escalation_models_metadata.joblib")
METRICS_PATH      = os.path.join(MODELS_DIR, "model_comparison_metrics.csv")

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']

CLINICAL_COSTS = {
    'age':0.0,'sex':0.0,'cp':5.0,'exang':10.0,'fbs':15.0,
    'trestbps':10.0,'chol':25.0,'restecg':50.0,'thalach':75.0,
    'oldpeak':100.0,'slope':100.0,'thal':250.0,'ca':350.0
}

# ── Load models at import time ────────────────────────────────────────────────
scaler = explainability_metadata = cost_summary = escalation_models_metadata = None
try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(EXPLAIN_META_PATH):
        explainability_metadata = joblib.load(EXPLAIN_META_PATH)
    if os.path.exists(COST_SUMMARY_PATH):
        cost_summary = joblib.load(COST_SUMMARY_PATH)
    if os.path.exists(ESCALATION_PATH):
        escalation_models_metadata = joblib.load(ESCALATION_PATH)
    print("[API] All models loaded.")
except Exception as e:
    print(f"[API] ERROR: {e}")

# ── Routes ──────────
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "pipeline_active": True}


@app.get("/api/metrics")
def get_model_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    df = pd.read_csv(METRICS_PATH)
    return df.to_dict(orient="records")


@app.get("/api/cost-summary")
def get_cost_summary():
    if cost_summary is not None:
        return cost_summary
    if os.path.exists(COST_SUMMARY_PATH):
        return joblib.load(COST_SUMMARY_PATH)
    raise HTTPException(status_code=404, detail="Cost summary not found.")


@app.post("/api/predict")
def predict_cardiac_risk(patient: PatientData):
    if scaler is None or escalation_models_metadata is None or explainability_metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    raw = patient.dict()
    cont_vals = np.array([raw[f] for f in CONTINUOUS_FEATURES]).reshape(1, -1)
    scaled_cont = scaler.transform(cont_vals)[0]

    scaled_arr = np.zeros(len(FEATURES))
    for i, f in enumerate(FEATURES):
        if f in CONTINUOUS_FEATURES:
            scaled_arr[i] = scaled_cont[CONTINUOUS_FEATURES.index(f)]
        else:
            scaled_arr[i] = raw[f]

    lower, upper = 0.15, 0.85
    escalation_path = []
    current_stage = 'stage_1'
    final_prob = 0.5
    final_cost = 0.0

    for stage_name in ['stage_1','stage_2','stage_3','stage_4']:
        current_stage = stage_name
        meta = escalation_models_metadata[stage_name]
        sub = scaled_arr[meta['indices']]
        prob = meta['model'].predict_proba(sub.reshape(1, -1))[0, 1]
        risk_prob = 1.0 - prob
        final_prob = risk_prob
        final_cost = meta['cumulative_cost']
        escalation_path.append({
            "stage": stage_name.upper().replace("_", " "),
            "probability": float(risk_prob),
            "cost": float(final_cost),
            "features_used": meta['features']
        })
        if risk_prob < lower or risk_prob > upper:
            break

    meta_f = escalation_models_metadata[current_stage]
    bg = np.zeros(len(scaled_arr))
    shap_vals = compute_single_patient_shap(
        model=meta_f['model'],
        patient_scaled=scaled_arr,
        active_indices=meta_f['indices'],
        background_means=bg,
        n_samples=500
    )

    explanations = []
    for f, idx in zip(meta_f['features'], meta_f['indices']):
        explanations.append({
            "feature": f.upper(),
            "raw_value": float(raw[f]),
            "scaled_value": float(scaled_arr[idx]),
            "risk_impact": float(-shap_vals[idx]),
            "cost": CLINICAL_COSTS[f]
        })
    explanations.sort(key=lambda x: abs(x["risk_impact"]), reverse=True)

    if final_prob < 0.35:
        cat = "LOW RISK"
        adv = "Patient displays high probability of stable cardiac health. Standard outpatient follow-up recommended."
    elif final_prob < 0.70:
        cat = "INTERMEDIATE RISK"
        adv = "Elevated risk index. Recommend regular monitoring, outpatient diagnostic reviews, and preventive therapies."
    else:
        cat = "HIGH RISK / URGENT"
        adv = "Severe clinical risk factors detected. Immediate cardiology consultation and advanced diagnostic triage highly advised."

    return {
        "risk_probability": float(final_prob),
        "risk_category": cat,
        "clinical_advice": adv,
        "total_diagnostic_cost": float(final_cost),
        "savings_vs_full": float((1.0 - final_cost / 990.0) * 100.0),
        "diagnostic_stage_reached": current_stage.upper().replace("_", " "),
        "escalation_path": escalation_path,
        "explanations": explanations,
        "baseline_probability": 1.0 - float(explainability_metadata["baseline"])
    }
