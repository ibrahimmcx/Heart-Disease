import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import pure custom ML elements
from pure_ml import (
    PureStandardScaler,
    PureLogisticRegression,
    PureRandomForestClassifier,
    PureGradientBoostingClassifier,
    pure_roc_auc_score
)
from data_preprocessing import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

# Define clinical cost mapping in USD
CLINICAL_COSTS = {
    'age': 0.0,        # Demographics / Intake
    'sex': 0.0,        # Demographics / Intake
    'cp': 5.0,         # Outpatient physical exam question
    'exang': 10.0,     # Basic cardiac history question
    'fbs': 15.0,       # Fasting blood glucose (fingerprick lab test)
    'trestbps': 10.0,  # Resting blood pressure (Sphygmomanometer check)
    'chol': 25.0,      # Lipid panel (Lipid profile blood test)
    'restecg': 50.0,   # Resting electrocardiogram (ECG machine)
    'thalach': 75.0,   # Stress test treadmill target heart rate
    'oldpeak': 100.0,  # Stress test ECG depression review
    'slope': 100.0,    # Stress test ECG slope review
    'thal': 250.0,     # Nuclear perfusion stress test (SPECT scintigraphy)
    'ca': 350.0        # Invasive coronary angiography subset (Fluoroscopy)
}

# Define test categories for staged escalation
STAGE_FEATURES = {
    'stage_1': ['age', 'sex', 'cp', 'exang', 'fbs'],                      # Patient Intake ($30)
    'stage_2': ['age', 'sex', 'cp', 'exang', 'fbs', 'trestbps', 'chol', 'restecg'], # + Basic Lab & ECG ($115)
    'stage_3': ['age', 'sex', 'cp', 'exang', 'fbs', 'trestbps', 'chol', 'restecg', 
                'thalach', 'oldpeak', 'slope'],                                # + Non-invasive Stress ($390)
    'stage_4': ['age', 'sex', 'cp', 'exang', 'fbs', 'trestbps', 'chol', 'restecg', 
                'thalach', 'oldpeak', 'slope', 'ca', 'thal']                   # + Angio & SPECT Nuclear ($990)
}


def calculate_feature_efficiency(model_importances, features):
    """
    Computes diagnostic utility per dollar: Efficiency = Importance / Cost
    """
    efficiency_records = []
    for feat, imp in zip(features, model_importances):
        cost = CLINICAL_COSTS.get(feat, 1.0)
        # Avoid division by zero for free demographic features by setting min cost to $1.0
        eff = imp / max(1.0, cost)
        efficiency_records.append({
            "Feature": feat,
            "Importance": imp,
            "Cost_USD": cost,
            "Efficiency": eff
        })
    return pd.DataFrame(efficiency_records).sort_values(by="Efficiency", ascending=False)


def run_cost_aware_simulation():
    print("\n[COST-ANALYSIS] Running Cost-Aware Decision Support Simulator...")
    
    # Load dataset and prepare test split
    from data_preprocessing import prepare_pipeline, load_data, analyze_and_clean_outliers
    df = load_data()
    df_cleaned = analyze_and_clean_outliers(df)
    X_train, X_test, y_train, y_test, features, scaler = prepare_pipeline(df_cleaned)
    
    # Convert dataframes to numpy array representation
    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)
    X_test_np = np.asarray(X_test)
    y_test_np = np.asarray(y_test)
    
    # Load explainability metadata to get feature importances
    meta_path = "models/explainability_metadata.joblib"
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Run explainability.py before cost_analysis.py!")
    metadata = joblib.load(meta_path)
    importances = np.array(metadata["importances"])
    
    # 1. Compute Feature Efficiency
    eff_df = calculate_feature_efficiency(importances, features)
    print("\n[COST-ANALYSIS] Feature Diagnostic Efficiency Rankings:")
    print(eff_df.to_string(index=False))
    eff_df.to_csv("models/feature_efficiency.csv", index=False)
    
    # 2. Select top 5 optimized features for Variant D (High Efficiency subset)
    opt_features = eff_df["Feature"].head(5).tolist()
    opt_feature_indices = [features.index(f) for f in opt_features]
    print(f"\n[COST-ANALYSIS] Optimized Feature Subset (Variant D): {opt_features}")
    
    # 3. Train Model Variants
    variants = {}
    
    # Helper to train sub-model
    def train_variant(feat_subset):
        sub_indices = [features.index(f) for f in feat_subset]
        X_tr = X_train_np[:, sub_indices]
        X_te = X_test_np[:, sub_indices]
        
        # Fit a pure Gradient Boosting for each variant
        model = PureGradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
        model.fit(X_tr, y_train_np)
        
        probs = model.predict_proba(X_te)[:, 1]
        auc = pure_roc_auc_score(y_test_np, probs)
        
        cost = sum([CLINICAL_COSTS[f] for f in feat_subset])
        return model, auc, cost
        
    print("\n[COST-ANALYSIS] Training Variant Models...")
    # Variant A: Full-Feature
    var_a_model, var_a_auc, var_a_cost = train_variant(features)
    print(f"  - Variant A (Full-Feature): AUC = {var_a_auc:.4f}, Cost = ${var_a_cost:.2f}")
    
    # Variant B: Low-Cost-Only
    low_cost_features = STAGE_FEATURES['stage_1']
    var_b_model, var_b_auc, var_b_cost = train_variant(low_cost_features)
    print(f"  - Variant B (Low-Cost-Only): AUC = {var_b_auc:.4f}, Cost = ${var_b_cost:.2f}")
    
    # Variant C: Immediate-Test-Only (Non-invasive)
    immediate_features = STAGE_FEATURES['stage_3']
    var_c_model, var_c_auc, var_c_cost = train_variant(immediate_features)
    print(f"  - Variant C (Immediate-Test-Only): AUC = {var_c_auc:.4f}, Cost = ${var_c_cost:.2f}")
    
    # Variant D: Optimized Subset
    var_d_model, var_d_auc, var_d_cost = train_variant(opt_features)
    print(f"  - Variant D (Optimized Subset): AUC = {var_d_auc:.4f}, Cost = ${var_d_cost:.2f}")
    
    # Save variant models to disk
    variants_dir = "models/variants"
    os.makedirs(variants_dir, exist_ok=True)
    joblib.dump(var_a_model, os.path.join(variants_dir, "variant_a_model.joblib"))
    joblib.dump(var_b_model, os.path.join(variants_dir, "variant_b_model.joblib"))
    joblib.dump(var_c_model, os.path.join(variants_dir, "variant_c_model.joblib"))
    joblib.dump(var_d_model, os.path.join(variants_dir, "variant_d_model.joblib"))
    
    # 4. Run Staged Diagnostic Escalation Simulation across the holdout test set
    # Fit individual models for the 4 diagnostic escalation stages
    stage_models = {}
    print("\n[COST-ANALYSIS] Training Staged Diagnostic Escalation Stage Models...")
    for stage_name, stage_feats in STAGE_FEATURES.items():
        sub_indices = [features.index(f) for f in stage_feats]
        X_tr = X_train_np[:, sub_indices]
        model = PureGradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
        model.fit(X_tr, y_train_np)
        stage_models[stage_name] = {
            'model': model,
            'features': stage_feats,
            'indices': sub_indices,
            'cumulative_cost': sum([CLINICAL_COSTS[f] for f in stage_feats])
        }
        
    # Simulate escalation logic: stop if prediction probability is highly confident (< 0.15 or > 0.85)
    lower_threshold = 0.15
    upper_threshold = 0.85
    
    simulation_costs = []
    simulation_predictions = []
    simulation_stages = []
    
    for patient_idx in range(len(X_test_np)):
        patient = X_test_np[patient_idx]
        current_prob = 0.5
        escalation_cost = 0.0
        final_stage = 'stage_1'
        
        # Sequentially escalate stages
        for stage_name in ['stage_1', 'stage_2', 'stage_3', 'stage_4']:
            final_stage = stage_name
            meta = stage_models[stage_name]
            
            # Extract sub-features for the patient and predict
            sub_feats = patient[meta['indices']]
            prob = meta['model'].predict_proba(sub_feats.reshape(1, -1))[0, 1]
            current_prob = prob
            escalation_cost = meta['cumulative_cost']
            
            # Check stopping criteria (clinical certainty achieved)
            if prob < lower_threshold or prob > upper_threshold:
                break
                
        simulation_costs.append(escalation_cost)
        simulation_predictions.append(current_prob)
        simulation_stages.append(final_stage)
        
    simulation_costs = np.array(simulation_costs)
    simulation_predictions = np.array(simulation_predictions)
    simulation_stages = np.array(simulation_stages)
    
    # Calculate performance of staged escalation model
    simulation_binary_preds = (simulation_predictions >= 0.5).astype(int)
    sim_accuracy = np.mean(y_test_np == simulation_binary_preds)
    sim_auc = pure_roc_auc_score(y_test_np, simulation_predictions)
    avg_sim_cost = np.mean(simulation_costs)
    
    print("\n[COST-ANALYSIS] Staged Diagnostic Escalation Results:")
    print(f"  - Average Diagnostic Cost per Patient: ${avg_sim_cost:.2f} (Savings of {(1.0 - avg_sim_cost / var_a_cost)*100:.1f}%!)")
    print(f"  - Evaluation Accuracy: {sim_accuracy * 100:.2f}% (Full-Feature: {np.mean(y_test_np == var_a_model.predict(X_test_np))*100:.2f}%)")
    print(f"  - Evaluation ROC-AUC: {sim_auc:.4f}")
    
    # Analyze final stages distribution
    unique, counts = np.unique(simulation_stages, return_counts=True)
    stage_dist = dict(zip(unique, counts))
    print("  - Stage escalation distribution across test patients:")
    for stage, count in stage_dist.items():
        print(f"    * {stage.upper()}: {count} patients ({count / len(X_test_np) * 100:.1f}%)")
        
    # 5. Plot Pareto Frontier Comparison Chart
    plt.figure(figsize=(9, 7))
    
    # Plot standard variants
    plt.scatter([var_b_cost], [var_b_auc], color='#e76f51', s=120, label='Variant B: Low-Cost ($65)', zorder=5)
    plt.scatter([var_c_cost], [var_c_auc], color='#f4a261', s=120, label='Variant C: Immediate ($390)', zorder=5)
    plt.scatter([var_d_cost], [var_d_auc], color='#2a9d8f', s=120, label='Variant D: Optimized ($85)', zorder=5)
    plt.scatter([var_a_cost], [var_a_auc], color='#264653', s=120, label='Variant A: Full-Feature ($990)', zorder=5)
    
    # Plot Staged Escalation
    plt.scatter([avg_sim_cost], [sim_auc], color='#e9c46a', marker='*', s=250, label=f'Staged Escalation (Avg: ${avg_sim_cost:.1f})', zorder=6)
    
    # Draw a line connecting B -> D -> C -> A (representing the diagnostic progression)
    plt.plot([var_b_cost, var_d_cost, var_c_cost, var_a_cost], 
             [var_b_auc, var_d_auc, var_c_auc, var_a_auc], 
             color='gray', linestyle=':', lw=1.5, alpha=0.7)
             
    plt.xlim([-50, 1100])
    plt.ylim([0.75, 1.02])
    plt.xlabel('Total Diagnostic Cost (USD)', fontsize=11, labelpad=10)
    plt.ylabel('Diagnostic Model ROC-AUC Score', fontsize=11, labelpad=10)
    plt.title('Clinical Pareto Frontier: Performance vs. Diagnostic Cost', fontsize=13, weight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    pareto_path = "reports/figures/pareto_frontier.png"
    plt.savefig(pareto_path)
    plt.close()
    print(f"\n[COST-ANALYSIS] Saved clinical Pareto Frontier plot to {pareto_path}")
    
    # Export simulation metadata for backend use
    joblib.dump(stage_models, "models/escalation_models_metadata.joblib")
    
    cost_summary = {
        "variants": {
            "A": {"cost": var_a_cost, "auc": var_a_auc},
            "B": {"cost": var_b_cost, "auc": var_b_auc},
            "C": {"cost": var_c_cost, "auc": var_c_auc},
            "D": {"cost": var_d_cost, "auc": var_d_auc}
        },
        "staged_simulation": {
            "avg_cost": avg_sim_cost,
            "accuracy": sim_accuracy,
            "auc": sim_auc,
            "distribution": {k: int(v) for k, v in stage_dist.items()}
        },
        "optimized_features": opt_features
    }
    joblib.dump(cost_summary, "models/cost_analysis_summary.joblib")
    print("[COST-ANALYSIS] Successfully saved all cost metrics to models/cost_analysis_summary.joblib")


if __name__ == "__main__":
    run_cost_aware_simulation()
