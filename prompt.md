You are an expert machine learning researcher, biomedical AI engineer, and Kaggle Grandmaster.

I am building a serious hackathon/research project for cardiovascular disease prediction using the Heart Disease Dataset (UCI/Cleveland style dataset).

This is NOT a beginner tutorial project.
Do NOT generate toy code.
Do NOT simplify the implementation.
Do NOT skip engineering details.

I want a production-quality, research-oriented implementation focused on:

1. Explainable AI
2. Cost-aware clinical prediction
3. Reproducibility
4. Proper evaluation
5. Clean architecture
6. Clinical interpretability

==================================================
PROJECT GOAL
============

Build a complete machine learning pipeline that predicts heart disease risk while also analyzing diagnostic test costs and clinical usefulness.

The project should simulate a realistic clinical decision-support system.

Dataset columns:

age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal, target

Target:

* target = 1 -> heart disease
* target = 0 -> no heart disease

==================================================
IMPORTANT REQUIREMENTS
======================

The implementation MUST include:

---

1. DATA PREPROCESSING

---

* robust missing value handling
* categorical encoding where necessary
* outlier analysis
* train/validation/test split
* stratified splitting
* feature scaling where appropriate
* reproducibility using random seeds

---

2. EXPLORATORY DATA ANALYSIS

---

Generate high-quality visualizations and analysis for:

* class distribution
* correlation matrix
* feature distributions
* feature-target relationships
* pairplots where useful
* clinical interpretation of variables

Explain findings in comments.

---

3. MODELING

---

Implement and compare:

* Logistic Regression
* Random Forest
* XGBoost

Use:

* cross-validation
* hyperparameter tuning
* proper evaluation

Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Generate:

* confusion matrices
* ROC curves
* comparison tables

---

4. EXPLAINABLE AI

---

This section is CRITICAL.

Use SHAP for:

* global feature importance
* local explanations
* beeswarm plots
* waterfall plots
* dependence plots

Explain clinically WHY certain features matter.

Discuss:

* chest pain type
* oldpeak
* thal
* ca
* resting ECG

==================================================
5. COST-AWARE ANALYSIS
======================

This is the UNIQUE contribution of the project.

Implement a cost-sensitive diagnostic analysis.

Assign simulated or configurable medical test costs to features.

Example:

age -> very low cost
chol -> low cost
restecg -> medium cost
thal -> high cost
ca -> high cost

Create:

A. Full-feature model
B. Low-cost-only model
C. Immediate-test-only model
D. Optimized feature subset model

Compare:

* predictive performance
* total diagnostic cost
* cost/performance tradeoff

Compute:

Feature Efficiency Score:

Efficiency = Feature_Importance / Test_Cost

Analyze:

* which tests provide highest predictive value per dollar
* which expensive tests add limited benefit

Generate:

* cost-performance comparison charts
* Pareto-style analysis

==================================================
6. CLINICAL DECISION SIMULATION
===============================

Implement a staged diagnostic workflow.

Example:

Stage 1:
cheap/basic tests only

Stage 2:
ECG-based tests

Stage 3:
advanced/high-cost tests

The system should simulate escalating diagnostic procedures.

==================================================
7. STREAMLIT APPLICATION
========================

Build a polished Streamlit app with:

* clean UI
* sidebar patient inputs
* prediction probability
* risk category
* SHAP explanation
* estimated diagnostic cost
* feature contribution explanation

The app should feel professional and realistic.

==================================================
8. CODE QUALITY
===============

Requirements:

* modular code
* functions/classes where appropriate
* comments explaining reasoning
* clean structure
* reproducible pipeline
* no messy notebook-only code

Structure example:

/data
/models
/notebooks
/src
/app
/reports

==================================================
9. RESEARCH REPORT STYLE OUTPUT
===============================

Generate text sections suitable for a research paper:

* Introduction
* Methodology
* Results
* Discussion
* Limitations
* Future Work
* Conclusion

The writing should sound academic and professional.

==================================================
10. IMPORTANT
=============

Avoid:

* fake metrics
* data leakage
* unrealistic accuracy claims
* beginner-level explanations
* oversimplified code

Focus on:

* trustworthy ML
* reproducibility
* clinical realism
* explainability
* research quality

The final output should resemble:

* a strong Kaggle competition submission
* a biomedical AI research prototype
* a real hackathon-winning project

Generate COMPLETE code and detailed explanations.
