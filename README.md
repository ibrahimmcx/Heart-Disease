# 🏥 CARDIO-SHIELD: Advanced Clinical Decision Support System (CDSS)

CARDIO-SHIELD is a premium, cost-sensitive staged diagnostic escalation platform for cardiovascular risk prediction. Integrating machine learning classification models with clinical cost constraints, it achieves high decision certainty while saving patient diagnostic budgets.

---

## 🌟 Key Features

1. **Cost-Sensitive Staged Triage (Escalation)**
   * **Stage-by-Stage Diagnostics:** The clinical engine dynamically escalates patients through four diagnostic tiers only if clinical certainty cannot be achieved at the lower, cheaper stages.
   * **Budget-Aware:** Avoids expensive tests (like Fluoroscopy or Scintigraphy) for low-risk patients, saving up to **72.8%** of the standard diagnostic budget.
   * **Confidence Thresholds:** Stops escalation early if predicted risk is below **15%** (Low Risk) or above **85%** (High Risk).

2. **Physiologically Aligned Explainable AI (XAI)**
   * **Local SHAP Explanations:** Displays real-time feature impact indices for the specific features utilized at the patient's final diagnostic stage.
   * **Clinical Direction Correction:** attributions are mapped dynamically according to Pearson correlations with target variables.
     * 🟥 **Red Bars:** Clinical features increasing cardiac risk (e.g., high ST-depression `oldpeak`, age, active angina `exang`).
     * 🟦 **Blue Bars:** Clinical features decreasing cardiac risk or representing cardiovascular fitness (e.g., higher `thalach` max heart rate, chest pain classification `cp`).

3. **Premium Clinician Dashboard**
   * High-contrast glassmorphic dark-mode interface built with Vanilla CSS.
   * Real-time range-slider syncing, dynamic SVG circular risk gauge, interactive triage stepper, and responsive Chart.js attribution charts.

---

## 🛠️ Tech Stack

* **Core Logic & ML:** Python (Pandas, NumPy, Joblib)
* **Custom Models:** Pure Random Forest and Decision Tree Classifiers (custom pure-python implementations)
* **Backend:** FastAPI (Uvicorn), Pydantic
* **Frontend:** HTML5, Vanilla JavaScript, Vanilla CSS, Chart.js, FontAwesome

---

## 🚀 Setup & Execution

### 1. Requirements

Ensure you have Python 3.10+ installed. Install the backend dependencies:
```bash
pip install fastapi uvicorn pydantic joblib pandas numpy
```

### 2. Run Backend Server

The FastAPI backend server is configured to run on **port 8001** to prevent conflicts with default Windows services. Run it using:
```bash
python -m uvicorn app.backend:app --host 127.0.0.1 --port 8001
```

### 3. Open Clinical Dashboard

You can access the frontend in two ways:
* **Option A (Hosted):** Open your browser and navigate directly to:
  ```
  http://127.0.0.1:8001/
  ```
* **Option B (Local File):** Open the local HTML file in your browser by double-clicking it:
  ```
  app/static/index.html
  ```
  *(Note: The JavaScript contains a robust fallback API connector that will automatically route `file://` page origins directly to the `http://127.0.0.1:8001` backend api).*

---

## 🔬 Clinical Risk Scale & Decision Mapping

| Risk Range | Category | Clinical Action |
| :--- | :--- | :--- |
| **< 35%** | `LOW RISK` | Stable cardiac health. Outpatient follow-up. |
| **35% - 70%** | `INTERMEDIATE RISK` | Elevated index. Regular monitoring and preventive therapies. |
| **> 70%** | `HIGH RISK / URGENT` | Severe clinical risk. Immediate cardiology consultation advised. |

---

## 🛡️ Git Workflows & Contribution Protocols

Development follows the guidelines specified in `GIT_WORKFLOW.md`:
* Feature additions branch prefix: `feature/`
* Bug fixes branch prefix: `bugfix/`
* Commit messages style: `feat:`, `fix:`, `docs:`, `refactor:`
* Always merge the latest `develop` changes into your feature branch and resolve any conflicts locally before opening Pull Requests (PRs).
