# Hospital Management & Disease Prediction Web Application

## Project Title
**Hospital Management & Disease Prediction using Bayesian Network, Z-Test, T-Test and P-Value**

---

## 🏥 Live Application Access

The application is running live on localhost:

- **Local URL:** [http://127.0.0.1:5000](http://127.0.0.1:5000) or [http://localhost:5000](http://localhost:5000)

### 🔑 Demo Accounts (One-Click Auto-Fill on Login Page)

| Role | Email | Password | Access Capabilities |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin@hospital.ai` | `Admin@123` | Global hospital dashboard, manage doctors, view all patients & predictions, model metrics |
| **Doctor** | `dr.sarah@hospital.ai` | `Doctor@123` | Register patients, enter clinical vitals, run Bayesian prediction, download PDF reports, view statistical lab |
| **Patient** | `john.doe@patient.ai` | `Patient@123` | Patient self-service portal, view personal profile, medical history, prediction reports, test results |

---

## 🎯 Main Objectives & System Architecture

This web application combines clinical hospital workflow management with advanced probabilistic AI and inferential statistics:

1. **Hospital Management System (HMS):** Role-based access control (Admin, Doctor, Patient), patient registration, electronic medical records, clinical notes, and downloadable official PDF medical reports.
2. **Bayesian Network Disease Risk Inference:** Utilizes `pgmpy` with `DiscreteBayesianNetwork` and exact `VariableElimination` to calculate patient-specific posterior disease risk probability:
   $$P(\text{Disease} \mid \text{Patient Evidence})$$
3. **Rigorous Hypothesis Testing & P-Value Laboratory:**
   - **Two-Sample Z-Test:** For large-sample continuous comparison ($n_1=526, n_2=499 \ge 30$, Central Limit Theorem).
   - **Two-Sample Welch's T-Test:** Robust against heteroscedasticity (checked via Levene's test), plus non-parametric Mann-Whitney U test.
   - **Chi-Square Test of Independence:** For categorical clinical features (especially Chest Pain Type, CP).
   - **Strict Distinction:** Educational and UI distinction between frequentist p-values (statistical significance against $H_0$) and Bayesian disease probabilities (posterior patient risk).

---

## 🔬 Dataset Audit & Medical Data Integrity

During initial inspection of the underlying UCI Heart Disease cohort ($n = 1,025$ observations, 14 features):

- **Chest Pain Type (`cp`):** Confirmed present with 4 distinct categorical encodings:
  - `0`: Typical Angina
  - `1`: Atypical Angina
  - `2`: Non-anginal Pain
  - `3`: Asymptomatic
  - *Methodological Rule:* CP is strictly categorical. In accordance with clinical statistical standards, CP is tested using a **Chi-Square Test of Independence** ($\chi^2 = 280.98, p < 0.0001, \text{Cramér's } V = 0.5235$). T-tests are *not* applied to categorical CP codes.
- **Creatine Kinase (`ck`):** Audited and confirmed **absent** from the raw dataset. In strict compliance with the prompt's instruction ("Do NOT assume CP or CK exists. Do not fabricate medical variables"), synthetic CK columns were not fabricated. The system utilizes real physiological and hemodynamic predictors (`trestbps`, `chol`, `thalach`, `oldpeak`, `age`) for continuous hypothesis tests.

---

## 📊 Bayesian Network Model Performance

Evaluated on an 80/20 stratified hold-out test set ($n_{test} = 205$ patients):

| Metric | Hold-out Performance |
| :--- | :--- |
| **Accuracy** | **91.71%** |
| **ROC - AUC** | **0.9885** |
| **Precision** | **90.74%** |
| **Recall (Sensitivity)** | **93.33%** |
| **F1-Score** | **92.02%** |
| **Brier Calibration Score** | **0.0534** (Superbly calibrated probabilities) |
| **Confusion Matrix** | True Negatives: 90, False Positives: 10, False Negatives: 7, True Positives: 98 |

---

## ⚖️ Important Statistical Rule (Section 19)

| Frequentist P-Value | Bayesian Posterior Probability |
| :--- | :--- |
| **Example:** $p = 0.0001$ ($\alpha = 0.05$) | **Example:** $P(\text{Disease} \mid \mathbf{e}) = 78.0\%$ |
| Measures probability of observing cohort data at least as extreme under the assumption that $H_0$ is true. | Measures the conditional degree of belief that this specific individual has heart disease given their observed symptoms and vitals. |
| **Does NOT** mean the probability that $H_0$ is true. | Directly updates prior belief using Bayes' theorem across the network DAG. |

---

## 🗂️ Project Directory Structure

```text
c:\bayesian project\
├── app.py                      # Main Flask application and routes
├── config.py                   # Configuration and environment settings
├── requirements.txt            # Python dependencies
│
├── data/
│   └── dataset.csv             # Audited cohort dataset (1,025 rows, 14 features)
│
├── models/
│   ├── __init__.py
│   └── bayesian_model.py       # pgmpy DiscreteBayesianNetwork & VariableElimination
│
├── statistics/
│   ├── __init__.py
│   ├── z_test.py               # Two-sample, one-sample, and proportion Z-tests
│   ├── t_test.py               # Student's t-test, Welch's t-test, Mann-Whitney U
│   └── chi_square.py           # Chi-Square test of independence & Cramér's V
│
├── database/
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM models (User, Patient, MedicalRecord, Prediction)
│   └── hospital_ai.db          # SQLite database with initial seed records
│
├── reports/
│   ├── __init__.py
│   ├── generator.py            # PDF report generator using ReportLab
│   └── visualizations.py       # Matplotlib visualization suite (11 clinical figures)
│
├── templates/
│   ├── base.html               # Main layout, navigation, and disclaimer
│   ├── login.html              # Authentication & one-click demo credentials
│   ├── register.html           # Patient and doctor registration
│   ├── dashboard.html          # Role-based hospital dashboard
│   ├── patients.html           # Patient directory & registration modal
│   ├── patient_detail.html     # Patient profile, vitals history & prediction list
│   ├── prediction.html         # Clinical observation entry & quick-fill presets
│   ├── result.html             # Bayesian prediction output, risk tier & distinction
│   ├── statistics.html         # Full statistical laboratory & visualization gallery
│   ├── bayesian_network.html   # DAG topology, CPD explorer, and validation metrics
│   └── doctors.html            # Admin physician management
│
├── static/
│   ├── css/custom.css          # Healthcare UI styles
│   └── images/charts/          # Generated static chart images (1 to 11)
│
└── tests/
    └── test_application.py     # 9 comprehensive pytest test cases
```

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:

```powershell
python -m pytest tests/test_application.py -v
```

All 9 unit and integration tests pass:
- Dataset integrity and column auditing
- Z-test calculations and CLT assumption checks
- Welch's T-test and Levene's variance testing
- Chi-Square test of independence on Chest Pain
- Bayesian network model inference and calibration
- Authentication and role-based clearance
- Web routes (`/statistics`, `/predict`, `/bayesian-network`)
- End-to-end PDF medical report generation
- Programmatic JSON API (`/api/predict`)

---

## ⚕️ Official Medical Disclaimer

> *This application provides an AI-based statistical risk estimate for educational/research purposes and is not a medical diagnosis. Clinical decisions must be made by qualified healthcare professionals.*
🌐 Live Deployment

🚀 Live Website:
https://hospital-management-disease-predict.vercel.app/

The application is deployed on Vercel and can be accessed directly from a web browser.
