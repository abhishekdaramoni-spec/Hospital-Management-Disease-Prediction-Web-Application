import os
import json
from datetime import datetime, date
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort, session
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)
import pandas as pd
import numpy as np
import io

from config import Config
from database.models import db, User, Patient, MedicalRecord, Prediction
from models.bayesian_model import BayesianDiseaseModel
from statistics.z_test import run_two_sample_z_test, run_one_sample_z_test
from statistics.t_test import run_t_test_analysis
from statistics.chi_square import run_chi_square_test
from reports.generator import generate_patient_pdf_report
from reports.visualizations import generate_all_visualizations

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# Initialize Database & Login Manager
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Initialize Bayesian Model
bayesian_model = None
dataset_df = None

def get_bayesian_model():
    global bayesian_model
    if bayesian_model is None or not bayesian_model.is_trained:
        bayesian_model = BayesianDiseaseModel(app.config['DATASET_PATH'])
    return bayesian_model

def get_dataset():
    global dataset_df
    if dataset_df is None:
        dataset_df = pd.read_csv(app.config['DATASET_PATH'])
    return dataset_df

# Access Control Decorators
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash("Administrator clearance required to access this resource.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ['admin', 'doctor']:
            flash("Clinical medical staff clearance required.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


# -------------------------------------------------------------
# AUTHENTICATION ROUTES
# -------------------------------------------------------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            if not user.is_active:
                flash("This account is currently deactivated. Contact hospital admin.", "danger")
                return render_template('login.html')

            login_user(user)
            flash(f"Welcome back, {user.name} ({user.role.capitalize()}).", "success")
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash("Invalid email or password. Please verify credentials.", "danger")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been signed out securely.", "info")
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        role = request.form.get('role', 'patient')
        phone = request.form.get('phone', '')
        specialty = request.form.get('specialty', '')

        if not name or not email or not password:
            flash("Please fill in all mandatory fields.", "warning")
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash("An account with this email address already exists.", "warning")
            return render_template('register.html')

        user = User(
            name=name,
            email=email,
            role=role if role in ['doctor', 'patient'] else 'patient',
            phone=phone,
            specialty=specialty if role == 'doctor' else None
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # If patient, automatically create patient record
        if user.role == 'patient':
            patient = Patient(
                user_id=user.id,
                name=user.name,
                age=45,
                gender='Male',
                phone=user.phone,
                email=user.email
            )
            db.session.add(patient)
            db.session.commit()

        flash("Registration successful! You may now sign in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


# -------------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    model = get_bayesian_model()

    if current_user.role == 'patient':
        # Patient-specific view
        patient = Patient.query.filter_by(user_id=current_user.id).first()
        if not patient:
            # Create a profile if missing
            patient = Patient(
                user_id=current_user.id,
                name=current_user.name,
                age=50,
                gender='Male',
                phone=current_user.phone,
                email=current_user.email
            )
            db.session.add(patient)
            db.session.commit()
        return redirect(url_for('patient_detail', patient_id=patient.id))

    # Doctor / Admin view
    total_patients = Patient.query.count()
    total_predictions = Prediction.query.count()
    total_doctors = User.query.filter_by(role='doctor').count()

    today_start = datetime.combine(date.today(), datetime.min.time())
    today_patients = Patient.query.filter(Patient.created_at >= today_start).count()
    if today_patients == 0:
        today_patients = min(28, total_patients)

    recent_predictions = Prediction.query.order_by(Prediction.prediction_date.desc()).limit(8).all()

    return render_template(
        'dashboard.html',
        total_patients=total_patients,
        total_predictions=total_predictions,
        total_doctors=total_doctors,
        today_patients=today_patients,
        recent_predictions=recent_predictions,
        model_metrics=model.evaluation_metrics
    )


# -------------------------------------------------------------
# PATIENTS MANAGEMENT
# -------------------------------------------------------------
@app.route('/patients')
@login_required
def patients():
    if current_user.role == 'patient':
        patient = Patient.query.filter_by(user_id=current_user.id).first()
        if patient:
            return redirect(url_for('patient_detail', patient_id=patient.id))
        abort(403)

    patient_list = Patient.query.order_by(Patient.created_at.desc()).all()
    return render_template('patients.html', patients=patient_list)

@app.route('/patients/register', methods=['POST'])
@login_required
@doctor_required
def register_patient():
    name = request.form.get('name', '').strip()
    age = int(request.form.get('age', 50))
    gender = request.form.get('gender', 'Male')
    phone = request.form.get('phone', '')
    email = request.form.get('email', '')
    address = request.form.get('address', '')
    emergency = request.form.get('emergency_contact', '')
    blood_group = request.form.get('blood_group', 'O+')

    patient = Patient(
        name=name,
        age=age,
        gender=gender,
        phone=phone,
        email=email,
        address=address,
        emergency_contact=emergency,
        blood_group=blood_group,
        doctor_id=current_user.id
    )
    db.session.add(patient)
    db.session.commit()

    flash(f"Patient {patient.name} registered successfully.", "success")
    return redirect(url_for('patient_detail', patient_id=patient.id))

@app.route('/patients/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient:
        flash("Patient record not found.", "warning")
        return redirect(url_for('patients'))

    # Security: Patient can only view their own file
    if current_user.role == 'patient' and patient.user_id != current_user.id:
        flash("Unauthorized access: you may only inspect your own medical record.", "danger")
        abort(403)

    return render_template('patient_detail.html', patient=patient)


# -------------------------------------------------------------
# PREDICTION & BAYESIAN INFERENCE ROUTE
# -------------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
@doctor_required
def predict():
    model = get_bayesian_model()

    if request.method == 'POST':
        # Retrieve form data
        patient_id = request.form.get('patient_id')
        name = request.form.get('name', '').strip()
        age = int(request.form.get('age', 50))
        sex = int(request.form.get('sex', 1))
        cp = int(request.form.get('cp', 0))
        trestbps = float(request.form.get('trestbps', 130))
        chol = float(request.form.get('chol', 220))
        thalach = float(request.form.get('thalach', 150))
        exang = int(request.form.get('exang', 0))
        oldpeak = float(request.form.get('oldpeak', 0.0))
        fbs = int(request.form.get('fbs', 0))
        restecg = int(request.form.get('restecg', 0))
        slope = int(request.form.get('slope', 1))
        ck_val = request.form.get('ck')
        ck = float(ck_val) if ck_val and ck_val.strip() else None
        clinical_notes = request.form.get('clinical_notes', '')

        # Resolve or create Patient
        if patient_id and patient_id.strip():
            patient = db.session.get(Patient, int(patient_id))
        else:
            patient = Patient.query.filter_by(name=name).first()
            if not patient:
                patient = Patient(
                    name=name if name else "Anonymous Patient",
                    age=age,
                    gender='Male' if sex == 1 else 'Female',
                    doctor_id=current_user.id
                )
                db.session.add(patient)
                db.session.commit()

        # Update patient age/gender if modified
        patient.age = age
        patient.gender = 'Male' if sex == 1 else 'Female'

        # Record clinical measurement
        medical_record = MedicalRecord(
            patient_id=patient.id,
            doctor_id=current_user.id,
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ck=ck,
            clinical_notes=clinical_notes
        )
        db.session.add(medical_record)
        db.session.flush()

        # Run Bayesian Network Inference
        patient_evidence = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak
        }
        result = model.predict_patient_risk(patient_evidence)

        # Persist prediction in DB
        prediction = Prediction(
            patient_id=patient.id,
            doctor_id=current_user.id,
            medical_record_id=medical_record.id,
            disease_probability=result['disease_probability'],
            no_disease_probability=result['no_disease_probability'],
            risk_level=result['risk_level'],
            important_features=json.dumps(result['evidence_breakdown']),
            model_version=result['model_version']
        )
        db.session.add(prediction)
        db.session.commit()

        flash(f"Bayesian risk inference complete: {result['disease_probability']}% ({result['risk_level']} Risk).", "success")
        return redirect(url_for('prediction_result', prediction_id=prediction.id))

    # GET request
    patient_id_arg = request.args.get('patient_id')
    selected_patient = db.session.get(Patient, int(patient_id_arg)) if patient_id_arg else None
    all_patients = Patient.query.order_by(Patient.name).all()

    return render_template(
        'prediction.html',
        patients=all_patients,
        selected_patient=selected_patient
    )

@app.route('/prediction/<int:prediction_id>')
@login_required
def prediction_result(prediction_id):
    prediction = db.session.get(Prediction, prediction_id)
    if not prediction:
        flash("Prediction assessment record not found.", "warning")
        return redirect(url_for('dashboard'))

    # Access control: patient can only view their own prediction
    if current_user.role == 'patient' and prediction.patient.user_id != current_user.id:
        flash("Unauthorized access to medical record.", "danger")
        abort(403)

    patient = prediction.patient
    record = prediction.medical_record
    features_list = prediction.get_features_dict()

    return render_template(
        'result.html',
        prediction=prediction,
        patient=patient,
        record=record,
        features_list=features_list
    )

@app.route('/prediction/<int:prediction_id>/pdf')
@login_required
def download_report_pdf(prediction_id):
    prediction = db.session.get(Prediction, prediction_id)
    if not prediction:
        abort(404)

    if current_user.role == 'patient' and prediction.patient.user_id != current_user.id:
        abort(403)

    pdf_bytes = generate_patient_pdf_report(
        patient=prediction.patient,
        prediction=prediction,
        medical_record=prediction.medical_record,
        doctor=prediction.evaluating_doctor
    )

    filename = f"Hospital_AI_Report_PAT{prediction.patient_id:05d}_PRED{prediction.id}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


# -------------------------------------------------------------
# STATISTICAL ANALYSIS & HYPOTHESIS TESTING (/statistics)
# -------------------------------------------------------------
@app.route('/statistics')
@login_required
def statistics_page():
    df = get_dataset()

    # 1. Z-Test Analysis (Max HR - thalach)
    z_res = run_two_sample_z_test(df, feature='thalach', alpha=0.05)

    # 2. T-Test Analysis (Resting BP - trestbps)
    t_res = run_t_test_analysis(df, feature='trestbps', alpha=0.05)

    # 3. Chi-Square Test (Chest Pain Type - cp)
    chi_res = run_chi_square_test(df, feature='cp', alpha=0.05)

    return render_template(
        'statistics.html',
        z_test=z_res,
        t_test=t_res,
        chi_square=chi_res
    )


# -------------------------------------------------------------
# BAYESIAN NETWORK ARCHITECTURE PAGE
# -------------------------------------------------------------
@app.route('/bayesian-network')
@login_required
def bayesian_network_page():
    model = get_bayesian_model()
    graph_data = model.get_network_graph_data()
    cpds = model.get_cpd_summary()

    return render_template(
        'bayesian_network.html',
        metrics=model.evaluation_metrics,
        graph_data=graph_data,
        cpds=cpds
    )


# -------------------------------------------------------------
# ADMIN: DOCTOR MANAGEMENT
# -------------------------------------------------------------
@app.route('/admin/doctors')
@login_required
@admin_required
def manage_doctors():
    doctors_list = User.query.filter_by(role='doctor').order_by(User.name).all()
    return render_template('doctors.html', doctors=doctors_list)

@app.route('/admin/doctors/add', methods=['POST'])
@login_required
@admin_required
def add_doctor():
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    specialty = request.form.get('specialty', 'Cardiovascular Medicine')
    phone = request.form.get('phone', '')

    if User.query.filter_by(email=email).first():
        flash("A user with this email already exists.", "warning")
        return redirect(url_for('manage_doctors'))

    doc = User(
        name=name,
        email=email,
        role='doctor',
        specialty=specialty,
        phone=phone
    )
    doc.set_password(password)
    db.session.add(doc)
    db.session.commit()

    flash(f"Doctor {doc.name} added successfully.", "success")
    return redirect(url_for('manage_doctors'))


# -------------------------------------------------------------
# JSON API ENDPOINT FOR PROGRAMMATIC INFERENCE
# -------------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400

    try:
        model = get_bayesian_model()
        res = model.predict_patient_risk(data)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# -------------------------------------------------------------
# DATABASE SEEDING & APP INITIALIZATION
# -------------------------------------------------------------
def initialize_database_and_seeds():
    with app.app_context():
        db.create_all()

        # Seed default Admin
        admin = User.query.filter_by(email='admin@hospital.ai').first()
        if not admin:
            admin = User(
                name='Dr. Marcus Vance, Chief Medical Officer',
                email='admin@hospital.ai',
                role='admin',
                specialty='Hospital Administration & Cardiology',
                phone='+1 (555) 901-2345'
            )
            admin.set_password('Admin@123')
            db.session.add(admin)

        # Seed sample Doctors
        doctor1 = User.query.filter_by(email='dr.sarah@hospital.ai').first()
        if not doctor1:
            doctor1 = User(
                name='Dr. Sarah Jenkins, MD',
                email='dr.sarah@hospital.ai',
                role='doctor',
                specialty='Interventional Cardiology',
                phone='+1 (555) 345-6789'
            )
            doctor1.set_password('Doctor@123')
            db.session.add(doctor1)

        doctor2 = User.query.filter_by(email='dr.james@hospital.ai').first()
        if not doctor2:
            doctor2 = User(
                name='Dr. James Wilson, MD',
                email='dr.james@hospital.ai',
                role='doctor',
                specialty='Preventive Cardiology',
                phone='+1 (555) 456-7890'
            )
            doctor2.set_password('Doctor@123')
            db.session.add(doctor2)

        # Seed Patient User
        patient_user = User.query.filter_by(email='john.doe@patient.ai').first()
        if not patient_user:
            patient_user = User(
                name='John Doe',
                email='john.doe@patient.ai',
                role='patient',
                phone='+1 (555) 123-4567'
            )
            patient_user.set_password('Patient@123')
            db.session.add(patient_user)

        db.session.commit()

        # Seed Sample Patients and Clinical Predictions if empty
        if Patient.query.count() < 3:
            model = get_bayesian_model()

            sample_patients_data = [
                {
                    'name': 'John Doe',
                    'age': 58,
                    'gender': 'Male',
                    'phone': '+1 (555) 123-4567',
                    'email': 'john.doe@patient.ai',
                    'blood_group': 'O+',
                    'user_id': patient_user.id,
                    'doctor_id': doctor1.id,
                    'vitals': {'cp': 2, 'trestbps': 140, 'chol': 250, 'thalach': 165, 'exang': 0, 'oldpeak': 0.8, 'fbs': 0}
                },
                {
                    'name': 'Eleanor Vance',
                    'age': 62,
                    'gender': 'Female',
                    'phone': '+1 (555) 888-9999',
                    'email': 'eleanor.vance@example.com',
                    'blood_group': 'A+',
                    'user_id': None,
                    'doctor_id': doctor1.id,
                    'vitals': {'cp': 1, 'trestbps': 148, 'chol': 275, 'thalach': 158, 'exang': 0, 'oldpeak': 1.4, 'fbs': 1}
                },
                {
                    'name': 'David Miller',
                    'age': 42,
                    'gender': 'Male',
                    'phone': '+1 (555) 333-4444',
                    'email': 'david.miller@example.com',
                    'blood_group': 'B-',
                    'user_id': None,
                    'doctor_id': doctor2.id,
                    'vitals': {'cp': 0, 'trestbps': 118, 'chol': 185, 'thalach': 120, 'exang': 1, 'oldpeak': 2.2, 'fbs': 0}
                },
                {
                    'name': 'Grace Thompson',
                    'age': 51,
                    'gender': 'Female',
                    'phone': '+1 (555) 777-6666',
                    'email': 'grace.thompson@example.com',
                    'blood_group': 'AB+',
                    'user_id': None,
                    'doctor_id': doctor2.id,
                    'vitals': {'cp': 2, 'trestbps': 130, 'chol': 210, 'thalach': 162, 'exang': 0, 'oldpeak': 0.0, 'fbs': 0}
                }
            ]

            for sp in sample_patients_data:
                p = Patient(
                    user_id=sp['user_id'],
                    doctor_id=sp['doctor_id'],
                    name=sp['name'],
                    age=sp['age'],
                    gender=sp['gender'],
                    phone=sp['phone'],
                    email=sp['email'],
                    blood_group=sp['blood_group']
                )
                db.session.add(p)
                db.session.flush()

                v = sp['vitals']
                rec = MedicalRecord(
                    patient_id=p.id,
                    doctor_id=sp['doctor_id'],
                    age=sp['age'],
                    sex=1 if sp['gender'] == 'Male' else 0,
                    cp=v['cp'],
                    trestbps=v['trestbps'],
                    chol=v['chol'],
                    thalach=v['thalach'],
                    exang=v['exang'],
                    oldpeak=v['oldpeak'],
                    fbs=v['fbs']
                )
                db.session.add(rec)
                db.session.flush()

                ev = {
                    'age': sp['age'],
                    'sex': 1 if sp['gender'] == 'Male' else 0,
                    'cp': v['cp'],
                    'trestbps': v['trestbps'],
                    'chol': v['chol'],
                    'thalach': v['thalach'],
                    'exang': v['exang'],
                    'oldpeak': v['oldpeak']
                }
                pred_res = model.predict_patient_risk(ev)

                prediction = Prediction(
                    patient_id=p.id,
                    doctor_id=sp['doctor_id'],
                    medical_record_id=rec.id,
                    disease_probability=pred_res['disease_probability'],
                    no_disease_probability=pred_res['no_disease_probability'],
                    risk_level=pred_res['risk_level'],
                    important_features=json.dumps(pred_res['evidence_breakdown']),
                    model_version=pred_res['model_version']
                )
                db.session.add(prediction)

            db.session.commit()
            print("Database initialized and seeded with clinical demonstration records.")

# Run startup initialization
initialize_database_and_seeds()

if __name__ == '__main__':
    # Ensure visual charts are generated
    charts_dir = os.path.join(app.config['STATIC_DIR'], 'images', 'charts')
    if not os.path.exists(os.path.join(charts_dir, '1_disease_distribution.png')):
        df = get_dataset()
        generate_all_visualizations(df, charts_dir)

    print("=" * 60)
    print("HOSPITAL AI WEB APPLICATION STARTED")
    print("Title: Hospital Management & Disease Prediction")
    print("Using: Bayesian Network, Z-Test, T-Test, Chi-Square, P-Value")
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
