import os
import sys

# Ensure the 'src/' directory is in python search path for robust imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=========================================================================")
print("  CARDIOVASCULAR DECISION SUPPORT SYSTEM - MASTER RUNNER")
print("=========================================================================\n")

def main():
    # Step 1: Preprocessing & Data Loading
    print("--- STEP 1: PREPROCESSING & PIPELINE SPLITS ---")
    from data_preprocessing import load_data, analyze_and_clean_outliers, prepare_pipeline
    df = load_data()
    df_cleaned = analyze_and_clean_outliers(df)
    X_train, X_test, y_train, y_test, features, scaler = prepare_pipeline(df_cleaned)
    print("[PIPELINE] Data preprocessing completed successfully.\n")
    
    # Step 2: Exploratory Data Analysis (EDA)
    print("--- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---")
    from eda import generate_eda_reports
    generate_eda_reports(df_cleaned)
    print("[PIPELINE] Clinical EDA visualizations saved to reports/figures/.\n")
    
    # Step 3: Hyperparameter Tuning and Model Training
    print("--- STEP 3: HIGH-PERFORMANCE CLINICAL MODELING ---")
    from modeling import train_and_tune_models
    best_models, metrics = train_and_tune_models(X_train, X_test, y_train, y_test)
    print("[PIPELINE] High-performance pure clinical classifiers successfully optimized.\n")
    
    # Step 4: Explainable AI (XAI) calculations
    print("--- STEP 4: EXPLAINABLE AI (XAI) METRICS ---")
    from explainability import run_explainability_pipeline
    run_explainability_pipeline()
    print("[PIPELINE] SHAP-equivalent attributions and distributions saved.\n")
    
    # Step 5: Cost-Aware Clinical Decision Simulator
    print("--- STEP 5: COST-AWARE DIAGNOSTIC SIMULATOR & PARETO FRONTIER ---")
    from cost_analysis import run_cost_aware_simulation
    run_cost_aware_simulation()
    print("[PIPELINE] Cost analysis and Pareto frontier simulation successfully finalized.\n")
    
    print("=========================================================================")
    print("  PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print("  Models, metadata, and scientific charts are ready in:")
    print("    - Models: models/")
    print("    - Visualizations: reports/figures/")
    print("=========================================================================")

if __name__ == "__main__":
    main()
