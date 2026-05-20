import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure 'src/' is in python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from pure_ml import PureStandardScaler, PureRandomForestClassifier, PureGradientBoostingClassifier
from data_preprocessing import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from explainability import compute_single_patient_shap

app = FastAPI(
    title="Cardiovascular Decision Support System API",
    description="Cost-sensitive staged cardiac diagnosis & Explainable AI backend",
    version="1.0.0"
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Patient Input Schema
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

# Paths to models (made absolute relative to app root to guarantee Vercel compatibility)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(APP_DIR), "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
EXPLAIN_META_PATH = os.path.join(MODELS_DIR, "explainability_metadata.joblib")
COST_SUMMARY_PATH = os.path.join(MODELS_DIR, "cost_analysis_summary.joblib")
ESCALATION_MODELS_PATH = os.path.join(MODELS_DIR, "escalation_models_metadata.joblib")

# Global variables to hold loaded models and metadata
scaler = None
explainability_metadata = None
cost_summary = None
escalation_models_metadata = None

# Feature names in the exact training pipeline order
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Cost mapping for individual features
CLINICAL_COSTS = {
    'age': 0.0, 'sex': 0.0, 'cp': 5.0, 'exang': 10.0, 'fbs': 15.0,
    'trestbps': 10.0, 'chol': 25.0, 'restecg': 50.0, 'thalach': 75.0,
    'oldpeak': 100.0, 'slope': 100.0, 'thal': 250.0, 'ca': 350.0
}

STAGE_FEATURES = {
    'stage_1': ['age', 'sex', 'cp', 'exang', 'fbs'],
    'stage_2': ['age', 'sex', 'cp', 'exang', 'fbs', 'trestbps', 'chol', 'restecg'],
    'stage_3': ['age', 'sex', 'cp', 'exang', 'fbs', 'trestbps', 'chol', 'restecg', 'thalach', 'oldpeak', 'slope'],
    'stage_4': ['age', 'sex', 'cp', 'exang', 'fbs', 'trestbps', 'chol', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal']
}


# Loads all custom serialized machine learning components on module import
# (guarantees readiness in serverless environments like Vercel)
print("[BACKEND] Loading clinical intelligence models on import...")
try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("  - Fitted Scaler loaded.")
    else:
        print("  - WARNING: Scaler file not found.")
        
    if os.path.exists(EXPLAIN_META_PATH):
        explainability_metadata = joblib.load(EXPLAIN_META_PATH)
        print("  - Explainability metadata loaded.")
        
    if os.path.exists(COST_SUMMARY_PATH):
        cost_summary = joblib.load(COST_SUMMARY_PATH)
        print("  - Cost analysis summary loaded.")
        
    if os.path.exists(ESCALATION_MODELS_PATH):
        escalation_models_metadata = joblib.load(ESCALATION_MODELS_PATH)
        print("  - Staged escalation models metadata loaded.")
        
except Exception as e:
    print(f"[BACKEND] ERROR loading serialized components: {str(e)}")


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "pipeline_active": True}


@app.get("/api/metrics")
def get_model_metrics():
    """Serves classification performance metrics comparing models."""
    metrics_path = os.path.join(MODELS_DIR, "model_comparison_metrics.csv")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics not found. Run training pipeline first.")
    
    df = pd.read_csv(metrics_path)
    return df.to_dict(orient="records")


@app.get("/api/cost-summary")
def get_cost_summary():
    """Serves clinical staged escalation simulator results and Pareto frontier metadata."""
    if cost_summary is None:
        if os.path.exists(COST_SUMMARY_PATH):
            return joblib.load(COST_SUMMARY_PATH)
        raise HTTPException(status_code=404, detail="Cost analysis summary not found. Run pipeline first.")
    return cost_summary


@app.post("/api/predict")
def predict_cardiac_risk(patient: PatientData):
    """
    Executes a real-time Staged Diagnostic Escalation prediction for an incoming patient.
    """
    if scaler is None or escalation_models_metadata is None or explainability_metadata is None:
        raise HTTPException(status_code=500, detail="Clinical models not fully loaded on server. Run training pipeline.")
    
    # 1. Standard scale the user's raw continuous parameters
    raw_patient_dict = patient.dict()
    
    # Extract continuous feature values in correct scaling order
    continuous_vals = np.array([raw_patient_dict[f] for f in CONTINUOUS_FEATURES]).reshape(1, -1)
    scaled_continuous = scaler.transform(continuous_vals)[0]
    
    # Map scaled continuous values back, keeping categorical values as-is
    scaled_patient_array = np.zeros(len(FEATURES))
    for i, f in enumerate(FEATURES):
        if f in CONTINUOUS_FEATURES:
            # Map from scaled_continuous
            c_idx = CONTINUOUS_FEATURES.index(f)
            scaled_patient_array[i] = scaled_continuous[c_idx]
        else:
            # Keep raw categorical
            scaled_patient_array[i] = raw_patient_dict[f]
            
    # 2. Dynamic Escalation Loop
    lower_threshold = 0.15
    upper_threshold = 0.85
    
    escalation_path = []
    current_stage = 'stage_1'
    final_prob = 0.5
    final_cost = 0.0
    
    for stage_name in ['stage_1', 'stage_2', 'stage_3', 'stage_4']:
        current_stage = stage_name
        meta = escalation_models_metadata[stage_name]
        
        # Extract sub-features for the current diagnostic stage
        sub_feats = scaled_patient_array[meta['indices']]
        prob = meta['model'].predict_proba(sub_feats.reshape(1, -1))[0, 1]
        
        # In this dataset, target=1 corresponds to "Healthy" and target=0 to "Heart Disease".
        # Therefore, predict_proba[:, 1] is the probability of being healthy.
        # We invert it to represent cardiac risk (probability of heart disease).
        risk_prob = 1.0 - prob
        final_prob = risk_prob
        final_cost = meta['cumulative_cost']
        
        escalation_path.append({
            "stage": stage_name.upper().replace("_", " "),
            "probability": float(risk_prob),
            "cost": float(final_cost),
            "features_used": meta['features']
        })
        
        # Confidence achieved (stop escalation)
        if risk_prob < lower_threshold or risk_prob > upper_threshold:
            break
            
    # 3. Compute local explainability (SHAP values) for the features utilized at the final stage
    meta_final = escalation_models_metadata[current_stage]
    final_features = meta_final['features']
    final_indices = meta_final['indices']
    
    # Get scaled values of utilized features
    utilized_values_scaled = scaled_patient_array[final_indices]
    
    # Get background means (which are all 0.0 in the scaled feature space)
    background_means_scaled = np.zeros(len(scaled_patient_array))
    
    # Compute patient-specific true Shapley values for the active features
    patient_shap = compute_single_patient_shap(
        model=meta_final['model'],
        patient_scaled=scaled_patient_array,
        active_indices=final_indices,
        background_means=background_means_scaled,
        n_samples=500
    )
    
    # Map to structured responses with raw values for clinician display.
    # We negate the Shapley values to represent contribution to CARDIAC RISK (class 0, heart disease).
    explanations = []
    for f, idx in zip(final_features, final_indices):
        raw_val = raw_patient_dict[f]
        scaled_val = scaled_patient_array[idx]
        
        # Negate the healthy-probability Shapley value to get risk-probability contribution
        risk_contrib = -patient_shap[idx]
        
        explanations.append({
            "feature": f.upper(),
            "raw_value": float(raw_val),
            "scaled_value": float(scaled_val),
            "risk_impact": float(risk_contrib),
            "cost": CLINICAL_COSTS[f]
        })
        
    # Sort features by absolute impact on risk decision
    explanations = sorted(explanations, key=lambda x: abs(x["risk_impact"]), reverse=True)
    
    # Categorize clinical risk
    if final_prob < 0.35:
        risk_category = "LOW RISK"
        clinical_advice = "Patient displays high probability of stable cardiac health. Standard outpatient follow-up recommended."
    elif final_prob < 0.70:
        risk_category = "INTERMEDIATE RISK"
        clinical_advice = "Elevated risk index. Recommend regular monitoring, outpatient diagnostic reviews, and preventive therapies."
    else:
        risk_category = "HIGH RISK / URGENT"
        clinical_advice = "Severe clinical risk factors detected. Immediate cardiology consultation and advanced diagnostic triage highly advised."
        
    return {
        "risk_probability": float(final_prob),
        "risk_category": risk_category,
        "clinical_advice": clinical_advice,
        "total_diagnostic_cost": float(final_cost),
        "savings_vs_full": float((1.0 - final_cost / 990.0) * 100.0),
        "diagnostic_stage_reached": current_stage.upper().replace("_", " "),
        "escalation_path": escalation_path,
        "explanations": explanations,
        "baseline_probability": 1.0 - float(explainability_metadata["baseline"])
    }

# Bind static directory if it exists
static_path = "app/static"
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
