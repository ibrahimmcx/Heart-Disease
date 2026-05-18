import os
import joblib
import pandas as pd
import numpy as np
from pure_ml import PureStandardScaler, pure_stratified_train_test_split

# Define feature classifications for reference
CONTINUOUS_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

def load_data(filepath="data/heart.csv"):
    """Loads the heart disease dataset from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def analyze_and_clean_outliers(df, threshold=3.5):
    """
    Identifies and handles outliers in continuous medical parameters.
    
    CLINICAL REALISM RATIONALE:
    In biomedical data, extremely high values (e.g., Cholesterol > 400 mg/dl or Resting BP > 180 mmHg)
    are clinically valid and represent very high-risk patients. Dropping them entirely leads to 
    unrealistic models. Therefore, we use a robust threshold (Z-score > 3.5) and clip values rather
    than dropping rows, so that the clinical variance remains intact while preventing gradient
    destabilization in models like Logistic Regression.
    """
    df_cleaned = df.copy()
    
    for col in CONTINUOUS_FEATURES:
        mean = df[col].mean()
        std = df[col].std()
        
        # Calculate Z-scores
        z_scores = (df[col] - mean) / std
        outliers_idx = df_cleaned[z_scores.abs() > threshold].index
        
        if len(outliers_idx) > 0:
            # Clip outliers to the threshold boundary
            upper_bound = mean + threshold * std
            lower_bound = max(0.0, mean - threshold * std) # medical metrics cannot be negative
            
            df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
            df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
            
            print(f"[PREPROCESSING] Clipped {len(outliers_idx)} outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
            
    return df_cleaned

def prepare_pipeline(df, random_state=42, models_dir="models"):
    """
    Main preprocessing pipeline executing:
    1. Stratified train/test split.
    2. Scaling continuous features (fitting ONLY on training set to prevent data leakage).
    3. Serializing scaling objects for production use.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 1. Stratified split to preserve class distributions
    X_train, X_test, y_train, y_test = pure_stratified_train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=random_state
    )
    
    print(f"[PREPROCESSING] Split statistics:")
    print(f"  - Training Set: {X_train.shape[0]} samples (Positives: {y_train.sum() / len(y_train) * 100:.2f}%)")
    print(f"  - Test Set: {X_test.shape[0]} samples (Positives: {y_test.sum() / len(y_test) * 100:.2f}%)")
    
    # 2. Scale continuous features (fitting ONLY on train to avoid Data Leakage)
    scaler = PureStandardScaler()
    
    # Fit scaler on train continuous features
    scaler.fit(X_train[CONTINUOUS_FEATURES])
    
    # Save the scaler for production dashboard use
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[PREPROCESSING] Scaler fitted and successfully saved to {scaler_path}")
    
    # Transform both splits
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_train[CONTINUOUS_FEATURES])
    X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist(), scaler

if __name__ == "__main__":
    df = load_data()
    df_cleaned = analyze_and_clean_outliers(df)
    X_train, X_test, y_train, y_test, features, scaler = prepare_pipeline(df_cleaned)
    print("[PREPROCESSING] Preprocessing pipeline verified successfully!")
