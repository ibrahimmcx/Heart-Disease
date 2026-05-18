import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our pure NumPy clinical-grade engine components
from pure_ml import (
    PureLogisticRegression,
    PureRandomForestClassifier,
    pure_stratified_kfold,
    pure_grid_search_cv,
    pure_roc_auc_score,
    pure_roc_curve,
    pure_confusion_matrix
)

from data_preprocessing import load_data, analyze_and_clean_outliers, prepare_pipeline

# Configure high-quality visualization style in pure Matplotlib
plt.rcParams.update({'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

# Pure NumPy implementations of core classification metrics
def pure_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def pure_precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def pure_recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def pure_f1_score(y_true, y_pred):
    prec = pure_precision_score(y_true, y_pred)
    rec = pure_recall_score(y_true, y_pred)
    return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def train_and_tune_models(X_train, X_test, y_train, y_test, random_state=42, models_dir="models", figures_dir="reports/figures"):
    """
    Trains and tunes Logistic Regression and Random Forest using GridSearchCV (Pure NumPy).
    Saves models and generates metric comparison plots using pure Matplotlib.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Cast splits to standardized NumPy arrays for speed and consistency
    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)
    X_test_np = np.asarray(X_test)
    y_test_np = np.asarray(y_test)
    
    # Generate stratified K-Folds splits
    cv_splits = pure_stratified_kfold(y_train_np, n_splits=5, random_state=random_state)
    
    # 1. Model Definitions & Parameter Grids using Pure Classes
    models = {
        "Logistic_Regression": {
            "class": PureLogisticRegression,
            "grid": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"]
            }
        },
        "Random_Forest": {
            "class": PureRandomForestClassifier,
            "grid": {
                "n_estimators": [30, 50],
                "max_depth": [5, 8],
                "min_samples_split": [2, 5]
            }
        }
    }
    
    best_models = {}
    performance_records = []
    roc_curves_data = {}
    confusion_matrices = {}
    
    # 2. GridSearch Cross-Validation Loop
    for name, config in models.items():
        print(f"\n[MODELING] Tuning and optimizing {name}...")
        
        # Run sequential pure K-Fold cross validation
        best_params, best_score = pure_grid_search_cv(
            model_class=config["class"],
            param_grid=config["grid"],
            X=X_train_np,
            y=y_train_np,
            cv_splits=cv_splits
        )
        
        # Fit optimal parameters on FULL training set
        from pure_ml import PureStandardScaler
        scaler = PureStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled = scaler.transform(X_test_np)
        
        best_model = config["class"](**best_params)
        best_model.fit(X_train_scaled, y_train_np)
        best_models[name] = best_model
        
        # Save serialized model
        model_path = os.path.join(models_dir, f"{name.lower()}_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"[MODELING] Best parameters for {name}: {best_params}")
        print(f"[MODELING] Best cross-validation ROC-AUC: {best_score:.4f}")
        print(f"[MODELING] Saved optimized {name} to {model_path}")
        
        # 3. Model Evaluation on Holdout Test Set
        y_pred = best_model.predict(X_test_scaled)
        y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics calculation
        acc = pure_accuracy_score(y_test_np, y_pred)
        prec = pure_precision_score(y_test_np, y_pred)
        rec = pure_recall_score(y_test_np, y_pred)
        f1 = pure_f1_score(y_test_np, y_pred)
        auc = pure_roc_auc_score(y_test_np, y_proba)
        
        performance_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC-AUC": auc
        })
        
        # Store for visualization
        fpr, tpr, _ = pure_roc_curve(y_test_np, y_proba)
        roc_curves_data[name] = (fpr, tpr, auc)
        confusion_matrices[name] = pure_confusion_matrix(y_test_np, y_pred)
        
    # 4. Generate Metric Report
    metrics_df = pd.DataFrame(performance_records)
    print("\n[MODELING] Test Set Evaluation Metrics:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics CSV to disk
    metrics_df.to_csv(os.path.join(models_dir, "model_comparison_metrics.csv"), index=False)
    
    # 5. Plot Combined ROC Curves
    plt.figure(figsize=(8, 7))
    colors = {"Logistic_Regression": "#2a9d8f", "Random_Forest": "#f4a261"}
    
    for name, (fpr, tpr, auc) in roc_curves_data.items():
        plt.plot(fpr, tpr, color=colors[name], lw=2, label=f'{name.replace("_", " ")} (AUC = {auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"[MODELING] Generated and saved comparative ROC curves: {roc_path}")
    
    # 6. Plot Confusion Matrices Side-by-Side in Pure Matplotlib
    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        # Draw heatmap using imshow
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        
        # Style and ticks
        classes = ['Healthy', 'Heart Disease']
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticklabels(classes, fontsize=11)
        
        ax.set_title(f'{name.replace("_", " ")}', fontsize=14, pad=10, weight='bold')
        ax.set_xlabel('Predicted Label', labelpad=10, fontsize=12)
        ax.set_ylabel('True Label', labelpad=10, fontsize=12)
        
        # Annotate confusion matrix values on cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color=color, fontsize=16, weight="bold")
        
        # Remove grid lines inside the confusion matrix cells
        ax.grid(False)
    
    plt.suptitle("Confusion Matrices Comparison on Holdout Test Set", y=1.02, fontsize=16, weight='bold')
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[MODELING] Generated and saved comparative confusion matrices: {cm_path}")
    
    return best_models, metrics_df

if __name__ == "__main__":
    df = load_data()
    df_cleaned = analyze_and_clean_outliers(df)
    X_train, X_test, y_train, y_test, features, scaler = prepare_pipeline(df_cleaned)
    best_models, metrics = train_and_tune_models(X_train, X_test, y_train, y_test)
