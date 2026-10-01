import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

class BayesianDiseaseModel:
    """
    Bayesian Network for Disease Prediction and Risk Inference using pgmpy.
    Provides probabilistic graphical modeling, parameter estimation, variable elimination inference,
    and rigorous cross-validation metrics.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.model = None
        self.inference_engine = None
        self.evaluation_metrics = {}
        self.feature_names = [
            'age_cat', 'sex', 'cp', 'bp_cat', 'chol_cat',
            'thalach_cat', 'exang', 'oldpeak_cat'
        ]
        self.target_name = 'target'
        self.is_trained = False
        self.raw_df = None
        self.disc_df = None

        if self.data_path and os.path.exists(self.data_path):
            self.train_and_evaluate()

    @staticmethod
    def discretize_continuous(age, trestbps, chol, thalach, oldpeak):
        """Discretize continuous clinical variables into clinical diagnostic bins."""
        # Age: <45 (0: Young), 45-59 (1: Middle-aged), >=60 (2: Senior)
        if age < 45:
            age_cat = 0
        elif age <= 59:
            age_cat = 1
        else:
            age_cat = 2

        # Resting Blood Pressure: <120 (0: Normal), 120-139 (1: Prehypertension), >=140 (2: Hypertension)
        if trestbps < 120:
            bp_cat = 0
        elif trestbps <= 139:
            bp_cat = 1
        else:
            bp_cat = 2

        # Cholesterol: <200 (0: Desirable), 200-239 (1: Borderline High), >=240 (2: High)
        if chol < 200:
            chol_cat = 0
        elif chol <= 239:
            chol_cat = 1
        else:
            chol_cat = 2

        # Max Heart Rate (thalach): <130 (0: Low), 130-159 (1: Moderate), >=160 (2: High)
        if thalach < 130:
            thalach_cat = 0
        elif thalach <= 159:
            thalach_cat = 1
        else:
            thalach_cat = 2

        # ST Depression (oldpeak): <1.0 (0: Minimal), 1.0-2.0 (1: Mild), >2.0 (2: Severe)
        if oldpeak < 1.0:
            oldpeak_cat = 0
        elif oldpeak <= 2.0:
            oldpeak_cat = 1
        else:
            oldpeak_cat = 2

        return age_cat, bp_cat, chol_cat, thalach_cat, oldpeak_cat

    def prepare_dataframe(self, df):
        """Prepare dataframe with discretized features."""
        d = df.copy()
        d['age_cat'] = pd.cut(d['age'], bins=[-np.inf, 44, 59, np.inf], labels=[0, 1, 2]).astype(int)
        d['bp_cat'] = pd.cut(d['trestbps'], bins=[-np.inf, 119, 139, np.inf], labels=[0, 1, 2]).astype(int)
        d['chol_cat'] = pd.cut(d['chol'], bins=[-np.inf, 199, 239, np.inf], labels=[0, 1, 2]).astype(int)
        d['thalach_cat'] = pd.cut(d['thalach'], bins=[-np.inf, 129, 159, np.inf], labels=[0, 1, 2]).astype(int)
        d['oldpeak_cat'] = pd.cut(d['oldpeak'], bins=[-np.inf, 0.99, 2.0, np.inf], labels=[0, 1, 2]).astype(int)
        return d

    def build_network_structure(self):
        """
        Define Bayesian Network DAG structure reflecting clinical etiology:
        - Demographic factors (Age, Sex) influence intermediate hemodynamics & symptoms
        - Chest Pain Type (CP), Blood Pressure, Cholesterol, Heart Rate, Angina, ST Depression influence Target
        """
        edges = [
            ('age_cat', 'bp_cat'),
            ('age_cat', 'chol_cat'),
            ('age_cat', 'cp'),
            ('sex', 'target'),
            ('cp', 'target'),
            ('bp_cat', 'target'),
            ('chol_cat', 'target'),
            ('thalach_cat', 'target'),
            ('exang', 'target'),
            ('oldpeak_cat', 'target'),
        ]
        return DiscreteBayesianNetwork(edges)

    def train_and_evaluate(self):
        """Train Bayesian Network and evaluate on hold-out test set."""
        self.raw_df = pd.read_csv(self.data_path)
        self.disc_df = self.prepare_dataframe(self.raw_df)

        cols = self.feature_names + [self.target_name]
        data = self.disc_df[cols]

        train_data, test_data = train_test_split(
            data, test_size=0.20, random_state=42, stratify=data[self.target_name]
        )

        self.model = self.build_network_structure()
        self.model.fit(train_data)
        self.inference_engine = VariableElimination(self.model)

        # Evaluate on test set
        y_true = test_data[self.target_name].values
        y_probs = []
        y_preds = []

        for _, row in test_data.iterrows():
            evidence = {k: int(row[k]) for k in self.feature_names}
            q = self.inference_engine.query(
                variables=[self.target_name],
                evidence=evidence,
                show_progress=False
            )
            # P(Disease = 1)
            p_disease = float(q.values[1])
            y_probs.append(p_disease)
            y_preds.append(1 if p_disease >= 0.5 else 0)

        acc = float(accuracy_score(y_true, y_preds))
        prec = float(precision_score(y_true, y_preds))
        rec = float(recall_score(y_true, y_preds))
        f1 = float(f1_score(y_true, y_preds))
        auc = float(roc_auc_score(y_true, y_probs))
        brier = float(brier_score_loss(y_true, y_probs))
        cm = confusion_matrix(y_true, y_preds).tolist()

        self.evaluation_metrics = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": cm,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "total_samples": len(data),
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1]
        }
        self.is_trained = True
        return self.evaluation_metrics

    def predict_patient_risk(self, patient_data):
        """
        Perform Bayesian inference for a single patient: P(Disease | Patient Evidence).
        Expected patient_data dict keys:
        - age: int
        - sex: int (1=Male, 0=Female)
        - cp: int (0-3)
        - trestbps: float
        - chol: float
        - thalach: float
        - exang: int (0 or 1)
        - oldpeak: float
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train_and_evaluate() first.")

        age_cat, bp_cat, chol_cat, thalach_cat, oldpeak_cat = self.discretize_continuous(
            float(patient_data['age']),
            float(patient_data['trestbps']),
            float(patient_data['chol']),
            float(patient_data['thalach']),
            float(patient_data.get('oldpeak', 0.0))
        )

        evidence = {
            'age_cat': int(age_cat),
            'sex': int(patient_data['sex']),
            'cp': int(patient_data['cp']),
            'bp_cat': int(bp_cat),
            'chol_cat': int(chol_cat),
            'thalach_cat': int(thalach_cat),
            'exang': int(patient_data.get('exang', 0)),
            'oldpeak_cat': int(oldpeak_cat)
        }

        query_result = self.inference_engine.query(
            variables=[self.target_name],
            evidence=evidence,
            show_progress=False
        )

        p_no_disease = float(query_result.values[0])
        p_disease = float(query_result.values[1])

        disease_pct = round(p_disease * 100, 1)
        no_disease_pct = round(p_no_disease * 100, 1)

        # Risk Stratification Tier
        if disease_pct >= 80.0:
            risk_level = "Critical"
            risk_badge = "danger"
            clinical_advice = "Urgent clinical cardiology evaluation, stress echocardiogram, and coronary angiogram review recommended."
        elif disease_pct >= 60.0:
            risk_level = "High"
            risk_badge = "warning"
            clinical_advice = "Comprehensive cardiovascular workup, risk factor modification, and cardiology referral recommended."
        elif disease_pct >= 35.0:
            risk_level = "Moderate"
            risk_badge = "info"
            clinical_advice = "Lifestyle interventions, blood pressure/lipid monitoring, and periodic cardiac follow-up recommended."
        else:
            risk_level = "Low"
            risk_badge = "success"
            clinical_advice = "Standard routine preventive cardiovascular health surveillance recommended."

        # Feature influence breakdown
        cp_labels = {
            0: "Typical Angina (Low baseline disease odds in this encoding)",
            1: "Atypical Angina (Elevated disease association)",
            2: "Non-anginal Pain (Elevated disease association)",
            3: "Asymptomatic (Moderate risk)"
        }
        bp_labels = {0: "Normal (<120 mmHg)", 1: "Prehypertension (120-139 mmHg)", 2: "Hypertension (≥140 mmHg)"}
        chol_labels = {0: "Desirable (<200 mg/dl)", 1: "Borderline High (200-239 mg/dl)", 2: "High (≥240 mg/dl)"}
        thalach_labels = {0: "Low (<130 bpm)", 1: "Moderate (130-159 bpm)", 2: "High (≥160 bpm)"}
        oldpeak_labels = {0: "Minimal (<1.0 mm)", 1: "Mild ST Depression (1.0-2.0 mm)", 2: "Severe ST Depression (>2.0 mm)"}

        evidence_breakdown = [
            {"feature": "Chest Pain Type", "value": cp_labels.get(evidence['cp'], str(evidence['cp'])), "status": "Key Risk Factor"},
            {"feature": "Blood Pressure", "value": f"{patient_data['trestbps']} mmHg ({bp_labels.get(bp_cat)})", "status": "Hemodynamic Marker"},
            {"feature": "Serum Cholesterol", "value": f"{patient_data['chol']} mg/dl ({chol_labels.get(chol_cat)})", "status": "Lipid Profile"},
            {"feature": "Maximum Heart Rate", "value": f"{patient_data['thalach']} bpm ({thalach_labels.get(thalach_cat)})", "status": "Cardiorespiratory Capacity"},
            {"feature": "ST Depression (Oldpeak)", "value": f"{patient_data.get('oldpeak', 0.0)} mm ({oldpeak_labels.get(oldpeak_cat)})", "status": "Ischemic Indicator"},
            {"feature": "Exercise Induced Angina", "value": "Present" if evidence['exang'] == 1 else "Absent", "status": "Symptomatic Marker"},
            {"feature": "Biological Sex", "value": "Male" if evidence['sex'] == 1 else "Female", "status": "Demographic Profile"},
            {"feature": "Age Bracket", "value": f"{patient_data['age']} yrs ({'Young <45' if age_cat==0 else 'Middle 45-59' if age_cat==1 else 'Senior ≥60'})", "status": "Demographic Profile"}
        ]

        return {
            "disease_probability": disease_pct,
            "no_disease_probability": no_disease_pct,
            "risk_estimate": disease_pct,
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "clinical_advice": clinical_advice,
            "evidence_used": evidence,
            "evidence_breakdown": evidence_breakdown,
            "model_version": "v1.2.0-DiscreteBayesian",
            "disclaimer": (
                "This application provides an AI-based statistical risk estimate for educational/research "
                "purposes and is not a medical diagnosis. Clinical decisions must be made by qualified healthcare professionals."
            )
        }

    def get_network_graph_data(self):
        """Export nodes and directed edges for UI network visualization (e.g. Vis.js)."""
        nodes = [
            {"id": "age_cat", "label": "Patient Age\n(<45, 45-59, ≥60)", "group": "demographic"},
            {"id": "sex", "label": "Biological Sex\n(Female, Male)", "group": "demographic"},
            {"id": "cp", "label": "Chest Pain Type\n(4 Categories)", "group": "symptom"},
            {"id": "bp_cat", "label": "Resting Blood Pressure\n(Normal, Prehypertensive, High)", "group": "vital"},
            {"id": "chol_cat", "label": "Serum Cholesterol\n(Desirable, Borderline, High)", "group": "biomarker"},
            {"id": "thalach_cat", "label": "Max Heart Rate\n(Low, Moderate, High)", "group": "vital"},
            {"id": "exang", "label": "Exercise Angina\n(No, Yes)", "group": "symptom"},
            {"id": "oldpeak_cat", "label": "ST Depression\n(Minimal, Mild, Severe)", "group": "ecg"},
            {"id": "target", "label": "Disease Status\n[TARGET]\n(No Disease / Disease)", "group": "target"}
        ]
        edges = [
            {"from": "age_cat", "to": "bp_cat", "label": "hemodynamics"},
            {"from": "age_cat", "to": "chol_cat", "label": "lipid metabolism"},
            {"from": "age_cat", "to": "cp", "label": "symptom onset"},
            {"from": "sex", "to": "target", "label": "gender risk"},
            {"from": "cp", "to": "target", "label": "clinical presentation"},
            {"from": "bp_cat", "to": "target", "label": "hypertension risk"},
            {"from": "chol_cat", "to": "target", "label": "atherosclerosis"},
            {"from": "thalach_cat", "to": "target", "label": "chronotropic capacity"},
            {"from": "exang", "to": "target", "label": "exercise ischemia"},
            {"from": "oldpeak_cat", "to": "target", "label": "myocardial stress"}
        ]
        return {"nodes": nodes, "edges": edges}

    def get_cpd_summary(self):
        """Extract summaries of learned Conditional Probability Distributions (CPDs)."""
        cpds = []
        if not self.model:
            return cpds
        for cpd in self.model.get_cpds():
            cpds.append({
                "variable": cpd.variable,
                "cardinality": cpd.variable_card,
                "evidence": cpd.variables[1:] if len(cpd.variables) > 1 else [],
                "values": cpd.values.tolist()
            })
        return cpds
