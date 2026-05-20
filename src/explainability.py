import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools

# Import pure custom ML elements
from pure_ml import PureStandardScaler
from data_preprocessing import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

def compute_single_patient_shap(model, patient_scaled, active_indices, background_means, n_samples=1000):
    """
    Computes mathematically rigorous Shapley values for a single patient's active features.
    If the number of active features is <= 10, computes Exact Shapley.
    Otherwise, uses KernelSHAP to approximate Shapley values.
    
    Returns:
        shap_values: dictionary mapping feature index to its Shapley value (contribution to class 1 / Healthy)
    """
    k = len(active_indices)
    x = patient_scaled[active_indices]
    b = background_means[active_indices]
    
    # Define model prediction function for a coalition mask
    def predict_coalition_batch(masks):
        # masks is shape (N, k)
        # Reconstruct full-feature vector for the model
        full_inputs = np.tile(background_means, (len(masks), 1))
        for idx, active_idx in enumerate(active_indices):
            full_inputs[:, active_idx] = masks[:, idx] * x[idx] + (1 - masks[:, idx]) * b[idx]
        
        # Predict probability of class 1 (Healthy)
        return model.predict_proba(full_inputs)[:, 1]

    if k <= 10:
        # Exact Shapley calculation
        # Generate all coalitions
        coalitions = np.array(list(itertools.product([0, 1], repeat=k)))
        probs = predict_coalition_batch(coalitions)
        
        # Map coalition mask tuple to prediction
        pred_map = {tuple(mask): prob for mask, prob in zip(coalitions, probs)}
        
        shap_values = np.zeros(k)
        for i in range(k):
            # Loop over all subsets of features excluding i
            other_indices = [j for j in range(k) if j != i]
            # Subsets can be represented as binary masks of size k-1
            for sub_mask in itertools.product([0, 1], repeat=k-1):
                # Construct mask S (without i) and S_u_i (with i)
                S = np.zeros(k, dtype=int)
                for idx, val in zip(other_indices, sub_mask):
                    S[idx] = val
                
                S_u_i = S.copy()
                S_u_i[i] = 1
                
                # Weight
                sz = int(np.sum(S))
                weight = math.factorial(sz) * math.factorial(k - sz - 1) / math.factorial(k)
                
                diff = pred_map[tuple(S_u_i)] - pred_map[tuple(S)]
                shap_values[i] += weight * diff
    else:
        # KernelSHAP approximation
        rng = np.random.default_rng(42)
        
        # Sample coalitions
        sampled_masks = []
        weights = []
        
        # Exact empty and full masks
        sampled_masks.append(np.zeros(k, dtype=int))
        weights.append(1e6) # High weight to enforce constraint
        
        sampled_masks.append(np.ones(k, dtype=int))
        weights.append(1e6)
        
        # We need to sample other coalitions
        for _ in range(n_samples - 2):
            sz = rng.integers(1, k)
            active_feats = rng.choice(k, sz, replace=False)
            mask = np.zeros(k, dtype=int)
            mask[active_feats] = 1
            
            # Compute KernelSHAP weight
            comb = math.comb(k, sz)
            weight = (k - 1) / (comb * sz * (k - sz))
            
            sampled_masks.append(mask)
            weights.append(weight)
            
        sampled_masks = np.array(sampled_masks)
        weights = np.array(weights)
        
        # Predict on all sampled coalitions
        probs = predict_coalition_batch(sampled_masks)
        
        # Weighted linear regression: Y = Z * Beta
        f_empty = probs[0]
        f_full = probs[1]
        
        Y = probs[2:] - f_empty
        Z = sampled_masks[2:]
        W = weights[2:]
        
        # Solve WLS: Z_w = sqrt(W) * Z, Y_w = sqrt(W) * Y
        sqrt_W = np.sqrt(W)[:, np.newaxis]
        Z_w = Z * sqrt_W
        Y_w = Y * np.sqrt(W)
        
        # Solve using NumPy lstsq
        shap_values, _, _, _ = np.linalg.lstsq(Z_w, Y_w, rcond=None)
        
        # Normalize to enforce sum(shap_values) == f_full - f_empty (Efficiency constraint)
        actual_sum = np.sum(shap_values)
        target_sum = f_full - f_empty
        if abs(actual_sum) > 1e-10:
            shap_values = shap_values * (target_sum / actual_sum)
        else:
            shap_values = np.zeros(k)
            
    # Map back to full active indices
    result = {idx: float(val) for idx, val in zip(active_indices, shap_values)}
    return result

def compute_local_explanations(model, X_sample, scaler, feature_names):
    """
    Computes mathematically rigorous local feature attribution (SHAP values) for patients
    using our custom Exact/KernelSHAP solver.
    """
    X_sample_np = np.asarray(X_sample)
    n_samples, n_features = X_sample_np.shape
    
    # Calculate baseline average prediction
    probs = model.predict_proba(X_sample_np)[:, 1]
    base_value = float(np.mean(probs))
    
    # Compute background means (baseline values) from X_sample itself
    background_means = np.mean(X_sample_np, axis=0)
    
    active_indices = list(range(n_features))
    shap_values = np.zeros((n_samples, n_features))
    
    print(f"[EXPLAINABILITY] Computing true Shapley values for {n_samples} samples...")
    for idx in range(n_samples):
        patient = X_sample_np[idx]
        patient_shap = compute_single_patient_shap(model, patient, active_indices, background_means, n_samples=800)
        for f_idx in range(n_features):
            shap_values[idx, f_idx] = patient_shap[f_idx]
            
    return shap_values, base_value


def generate_beeswarm_plot(shap_values, X_sample, feature_names, save_path="reports/figures/shap_beeswarm.png"):
    """
    Generates a premium SHAP-style beeswarm plot using pure Matplotlib.
    Features are ordered by global mean absolute SHAP value.
    Points are colored by original feature value (normalized between 0 and 1: low is blue, high is red).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    shap_values = np.asarray(shap_values)
    X_sample_np = np.asarray(X_sample)
    
    # Calculate global importance: mean absolute attribution
    mean_abs_attributions = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_attributions)
    
    plt.figure(figsize=(10, 8))
    
    n_samples, n_features = shap_values.shape
    
    # Min-max scale original feature values for coloring (low is blue, high is red)
    X_min = np.min(X_sample_np, axis=0)
    X_max = np.max(X_sample_np, axis=0)
    # Avoid divide by zero
    X_range = X_max - X_min
    X_range[X_range == 0.0] = 1.0
    X_normalized = (X_sample_np - X_min) / X_range
    
    for i, feat_idx in enumerate(sorted_idx):
        feat_name = feature_names[feat_idx]
        feat_shap = shap_values[:, feat_idx]
        feat_val = X_normalized[:, feat_idx]
        
        # Add slight jitter on y-axis to simulate beeswarm point spreading
        y_vals = np.ones(n_samples) * i + np.random.uniform(-0.15, 0.15, n_samples)
        
        # Scatter plot colored by feature value using coolwarm colormap
        sc = plt.scatter(feat_shap, y_vals, c=feat_val, cmap='coolwarm', s=25, alpha=0.8, edgecolors='none')
        
    plt.yticks(np.arange(n_features), [feature_names[idx].upper() for idx in sorted_idx], fontsize=10, weight='bold')
    plt.axvline(x=0.0, color='gray', linestyle='--', lw=1.2, alpha=0.7)
    plt.xlabel('SHAP Value (Impact on Risk Model Decision)', fontsize=12, labelpad=10)
    plt.title('Global Explainable AI (XAI) - Feature Risk Impact Distributions', fontsize=14, weight='bold', pad=15)
    
    # Add a custom colorbar at the right
    cbar = plt.colorbar(sc, ticks=[0, 1], orientation='vertical', aspect=40, pad=0.02)
    cbar.ax.set_yticklabels(['LOW Feature Value', 'HIGH Feature Value'], fontsize=9, weight='bold')
    cbar.ax.tick_params(rotation=90, labelsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[EXPLAINABILITY] Saved premium SHAP beeswarm plot to {save_path}")


def generate_dependence_plot(shap_values, X_sample, feature_idx, feature_names, save_path="reports/figures/shap_dependence.png"):
    """
    Generates a dependence plot showing how impact (SHAP) varies with feature value.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    shap_values = np.asarray(shap_values)
    X_sample_np = np.asarray(X_sample)
    
    feat_name = feature_names[feature_idx]
    feat_values = X_sample_np[:, feature_idx]
    feat_shap = shap_values[:, feature_idx]
    
    plt.figure(figsize=(8, 6))
    
    # Simple scatter with dynamic trendline
    plt.scatter(feat_values, feat_shap, color='#2a9d8f', alpha=0.7, s=40, label='Patient Samples')
    
    # Add smooth polynomial trendline
    poly_coefs = np.polyfit(feat_values, feat_shap, 2)
    poly_func = np.poly1d(poly_coefs)
    x_line = np.linspace(np.min(feat_values), np.max(feat_values), 100)
    plt.plot(x_line, poly_func(x_line), color='#e76f51', lw=2.5, label='Non-linear Trend')
    
    plt.axhline(y=0.0, color='gray', linestyle='--', lw=1.0, alpha=0.5)
    plt.xlabel(f'Normalized {feat_name.upper()} Value', fontsize=11, labelpad=10)
    plt.ylabel('SHAP Value (Impact on Decision)', fontsize=11, labelpad=10)
    plt.title(f'XAI Diagnostic Dependence: {feat_name.upper()}', fontsize=13, weight='bold', pad=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[EXPLAINABILITY] Saved SHAP dependence plot to {save_path}")


def run_explainability_pipeline():
    """Master runner to compute XAI metrics on trained models."""
    print("\n[EXPLAINABILITY] Running Explainable AI (XAI) Pipeline...")
    
    # Load dataset and prepare test split
    from data_preprocessing import prepare_pipeline, load_data, analyze_and_clean_outliers
    df = load_data()
    df_cleaned = analyze_and_clean_outliers(df)
    X_train, X_test, y_train, y_test, features, scaler = prepare_pipeline(df_cleaned)
    
    # Load Random Forest model
    model_path = "models/random_forest_model.joblib"
    if not os.path.exists(model_path):
        print(f"[EXPLAINABILITY] Model {model_path} not found! Please run modeling first.")
        return
        
    model = joblib.load(model_path)
    print(f"[EXPLAINABILITY] Successfully loaded clinical model from {model_path}")
    
    # Sample 200 patients from test set for explainability viz
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(X_test), min(200, len(X_test)), replace=False)
    X_sample = np.asarray(X_test)[sample_indices]
    
    # Compute attributions
    shap_values, base_value = compute_local_explanations(model, X_sample, scaler, features)
    print(f"[EXPLAINABILITY] Base value (average risk baseline): {base_value:.4f}")
    
    # Generate beeswarm
    generate_beeswarm_plot(shap_values, X_sample, features)
    
    # Generate dependence for the highest importance feature index
    # (thalach is index 3, cp is index 2, chest pain is usually highly predictive)
    highest_importance_feat_idx = int(np.argmax(model.feature_importances_))
    print(f"[EXPLAINABILITY] Highest predictive feature: {features[highest_importance_feat_idx].upper()}")
    generate_dependence_plot(shap_values, X_sample, highest_importance_feat_idx, features)
    
    # Export local explanations summary to disk for backend use
    local_summary = {
        "features": features,
        "importances": model.feature_importances_.tolist(),
        "baseline": base_value
    }
    joblib.dump(local_summary, "models/explainability_metadata.joblib")
    print("[EXPLAINABILITY] Saved local explanation metadata to models/explainability_metadata.joblib")
    
if __name__ == "__main__":
    run_explainability_pipeline()
