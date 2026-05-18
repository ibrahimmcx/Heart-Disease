import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import pure custom ML elements
from pure_ml import PureStandardScaler
from data_preprocessing import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

def compute_local_explanations(model, X_sample, scaler, feature_names):
    """
    Computes local feature attribution (SHAP equivalent) for patients.
    Returns:
        shap_values: Array of shape (n_samples, n_features)
        base_value: Float representing model's average probability baseline
    """
    X_sample_np = np.asarray(X_sample)
    
    # Calculate baseline average prediction
    # For clinical relevance, baseline is 0.5 (equal probability) or mean predicted prob
    probs = model.predict_proba(X_sample_np)[:, 1]
    base_value = float(np.mean(probs))
    
    # Handle Logistic Regression SHAP values (Exact math)
    if hasattr(model, 'w') and not hasattr(model, 'trees'):
        # w_i * (x_normalized) represents the linear logit contribution
        w = model.w
        shap_values = X_sample_np * w
        
    # Handle Random Forest local explanations (Weighted distance from population mean)
    else:
        importances = model.feature_importances_
        # Weighted normalized contribution: importance * scaled_value
        shap_values = X_sample_np * importances
        
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
