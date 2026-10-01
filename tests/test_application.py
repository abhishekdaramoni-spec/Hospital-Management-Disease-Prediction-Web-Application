import pytest
import os
import json
import pandas as pd
import numpy as np

from app import app, db, get_bayesian_model
from database.models import User, Patient, MedicalRecord, Prediction
from statistics.z_test import run_two_sample_z_test, run_one_sample_z_test
from statistics.t_test import run_t_test_analysis
from statistics.chi_square import run_chi_square_test
from reports.generator import generate_patient_pdf_report

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            yield client

def test_dataset_integrity():
    """Verify dataset columns, shape, CP presence and CK absence per instructions."""
    df = pd.read_csv('data/dataset.csv')
    assert df.shape == (1025, 14), f"Unexpected shape {df.shape}"
    assert 'cp' in df.columns, "Chest Pain (cp) must be present"
    assert set(df['cp'].unique()) == {0, 1, 2, 3}, "CP must have categories 0, 1, 2, 3"
    assert 'ck' not in df.columns, "CK must not be present in raw dataset per audit"
    assert 'target' in df.columns, "Target must be present"

def test_statistical_z_test():
    """Verify Two-sample Z-test calculations."""
    df = pd.read_csv('data/dataset.csv')
    res = run_two_sample_z_test(df, feature='thalach', alpha=0.05)
    assert res['z_statistic'] > 10.0, f"Expected large Z for thalach, got {res['z_statistic']}"
    assert res['p_value'] < 0.0001, "Expected p < 0.0001"
    assert res['is_significant'] is True
    assert "Statistically significant" in res['conclusion']
    assert res['assumptions']['assumptions_met'] is True

def test_statistical_t_test():
    """Verify Welch's two-sample T-test calculations and CK note."""
    df = pd.read_csv('data/dataset.csv')
    res = run_t_test_analysis(df, feature='trestbps', alpha=0.05)
    assert abs(res['t_statistic']) > 3.0
    assert res['p_value'] < 0.01
    assert res['is_significant'] is True
    assert "CK" in res['ck_note']
    assert res['assumptions']['equal_variance'] is False # Levene rejects homoscedasticity

def test_statistical_chi_square():
    """Verify Chi-Square Test of Independence for CP vs Target."""
    df = pd.read_csv('data/dataset.csv')
    res = run_chi_square_test(df, feature='cp', alpha=0.05)
    assert res['chi2_statistic'] > 250.0, f"Expected large chi2 for CP, got {res['chi2_statistic']}"
    assert res['degrees_of_freedom'] == 3
    assert res['p_value'] < 0.0001
    assert res['cramers_v'] > 0.4
    assert len(res['observed_table']) == 4
    assert "Statistical Rule" in res['method_note']

def test_bayesian_model_inference():
    """Verify Bayesian model training, inference, and hold-out metrics."""
    model = get_bayesian_model()
    assert model.is_trained is True
    assert model.evaluation_metrics['accuracy'] > 85.0
    assert model.evaluation_metrics['roc_auc'] > 0.90

    # High-risk profile
    high_risk_patient = {
        'age': 65,
        'sex': 1,
        'cp': 2,
        'trestbps': 150,
        'chol': 280,
        'thalach': 165,
        'exang': 0,
        'oldpeak': 0.8
    }
    res_high = model.predict_patient_risk(high_risk_patient)
    assert 'disease_probability' in res_high
    assert 'no_disease_probability' in res_high
    assert round(res_high['disease_probability'] + res_high['no_disease_probability'], 1) == 100.0
    assert res_high['risk_level'] in ['Moderate', 'High', 'Critical']
    assert "Disclaimer" in res_high['disclaimer'] or "disclaimer" in res_high

def test_login_and_auth_flow(client):
    """Test login functionality and session management."""
    # GET login
    resp = client.get('/login')
    assert resp.status_code == 200
    assert b"Hospital AI Portal" in resp.data

    # Invalid login
    bad_login = client.post('/login', data={'email': 'wrong@test.com', 'password': 'wrong'}, follow_redirects=True)
    assert b"Invalid email or password" in bad_login.data

    # Valid doctor login
    good_login = client.post('/login', data={'email': 'dr.sarah@hospital.ai', 'password': 'Doctor@123'}, follow_redirects=True)
    assert b"Welcome back, Dr. Sarah Jenkins" in good_login.data
    assert b"HOSPITAL AI DASHBOARD" in good_login.data

def test_statistics_route(client):
    """Test /statistics route renders Z-test, T-test, and Chi-Square."""
    # Login first
    client.post('/login', data={'email': 'dr.sarah@hospital.ai', 'password': 'Doctor@123'})
    resp = client.get('/statistics')
    assert resp.status_code == 200
    assert b"Statistical Analysis Laboratory" in resp.data
    assert b"Z-Test for Difference in Means" in resp.data
    assert b"Welch's Robust T-Test" in resp.data
    assert b"Chest Pain Type" in resp.data
    assert b"Critical Distinction" in resp.data

def test_prediction_route_and_pdf_download(client):
    """Test running a full disease prediction and downloading PDF report."""
    client.post('/login', data={'email': 'dr.sarah@hospital.ai', 'password': 'Doctor@123'})

    # Submit patient prediction form
    pred_data = {
        'name': 'Test Integration Patient',
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 138,
        'chol': 240,
        'thalach': 158,
        'exang': 0,
        'oldpeak': 1.0,
        'fbs': 0,
        'restecg': 0,
        'slope': 1,
        'ck': 120,
        'clinical_notes': 'Testing full end-to-end integration'
    }
    post_resp = client.post('/predict', data=pred_data, follow_redirects=True)
    assert post_resp.status_code == 200
    assert b"DISEASE RISK PREDICTION RESULT" in post_resp.data
    assert b"Bayesian Disease Risk Probability" in post_resp.data

    # Check that prediction was stored in database
    pt = Patient.query.filter_by(name='Test Integration Patient').first()
    assert pt is not None
    assert len(pt.predictions) > 0
    pred = pt.predictions[0]
    assert pred.disease_probability > 0

    # Test PDF report download
    pdf_resp = client.get(f'/prediction/{pred.id}/pdf')
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers['Content-Type'] == 'application/pdf'
    assert len(pdf_resp.data) > 1000, "PDF size must be substantial"

def test_api_predict_endpoint(client):
    """Test JSON API endpoint for external diagnostic queries."""
    payload = {
        'age': 52,
        'sex': 0,
        'cp': 1,
        'trestbps': 130,
        'chol': 225,
        'thalach': 160,
        'exang': 0,
        'oldpeak': 0.6
    }
    resp = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'disease_probability' in data
    assert 'risk_level' in data
    assert data['disease_probability'] >= 0.0
