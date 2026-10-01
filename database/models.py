from datetime import datetime
import json
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='patient')  # 'admin', 'doctor', 'patient'
    phone = db.Column(db.String(30))
    specialty = db.Column(db.String(100), default='General Medicine')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    patient_profile = db.relationship('Patient', foreign_keys='Patient.user_id', backref='user_account', uselist=False)
    assigned_patients = db.relationship('Patient', foreign_keys='Patient.doctor_id', backref='primary_doctor', lazy=True)
    records_created = db.relationship('MedicalRecord', backref='attending_doctor', lazy=True)
    predictions_requested = db.relationship('Prediction', backref='evaluating_doctor', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.role == 'admin'

    def is_doctor(self):
        return self.role == 'doctor'

    def is_patient(self):
        return self.role == 'patient'

    def __repr__(self):
        return f"<User {self.name} ({self.role})>"


class Patient(db.Model):
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)  # 'Male', 'Female', 'Other'
    phone = db.Column(db.String(30))
    email = db.Column(db.String(120))
    address = db.Column(db.Text)
    emergency_contact = db.Column(db.String(100))
    blood_group = db.Column(db.String(10), default='Unknown')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    medical_records = db.relationship('MedicalRecord', backref='patient', cascade='all, delete-orphan', order_by='desc(MedicalRecord.created_at)')
    predictions = db.relationship('Prediction', backref='patient', cascade='all, delete-orphan', order_by='desc(Prediction.prediction_date)')

    @property
    def latest_prediction(self):
        return self.predictions[0] if self.predictions else None

    @property
    def latest_record(self):
        return self.medical_records[0] if self.medical_records else None

    def __repr__(self):
        return f"<Patient {self.name}, Age {self.age}>"


class MedicalRecord(db.Model):
    __tablename__ = 'medical_records'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    # Features from verified dataset
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)  # 1 = male, 0 = female
    cp = db.Column(db.Integer, nullable=False)   # 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic
    trestbps = db.Column(db.Float, nullable=False)  # Resting Blood Pressure (mmHg)
    chol = db.Column(db.Float, nullable=False)      # Serum Cholesterol (mg/dl)
    fbs = db.Column(db.Integer, default=0)          # Fasting blood sugar > 120 mg/dl (1/0)
    restecg = db.Column(db.Integer, default=0)      # Resting ECG (0, 1, 2)
    thalach = db.Column(db.Float, nullable=False)   # Maximum heart rate achieved
    exang = db.Column(db.Integer, default=0)        # Exercise-induced angina (1/0)
    oldpeak = db.Column(db.Float, default=0.0)      # ST depression
    slope = db.Column(db.Integer, default=1)        # Slope of peak exercise ST segment (0, 1, 2)
    ca = db.Column(db.Integer, default=0)           # Major vessels colored by fluoroscopy (0-4)
    thal = db.Column(db.Integer, default=2)         # Thalassemia (0, 1, 2, 3)

    # Optional / Clinical notes
    ck = db.Column(db.Float, nullable=True)         # Creatine Kinase (optional clinical input)
    clinical_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to prediction
    predictions = db.relationship('Prediction', backref='medical_record', cascade='all, delete-orphan')

    @property
    def cp_label(self):
        labels = {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }
        return labels.get(self.cp, f"Type {self.cp}")

    def __repr__(self):
        return f"<MedicalRecord Patient={self.patient_id} BP={self.trestbps} Chol={self.chol}>"


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    medical_record_id = db.Column(db.Integer, db.ForeignKey('medical_records.id'), nullable=True)

    disease_probability = db.Column(db.Float, nullable=False)      # e.g. 78.0 (%)
    no_disease_probability = db.Column(db.Float, nullable=False)   # e.g. 22.0 (%)
    risk_level = db.Column(db.String(30), nullable=False)          # 'Low', 'Moderate', 'High', 'Critical'
    important_features = db.Column(db.Text)                        # JSON serialized string
    model_version = db.Column(db.String(50), default='v1.2.0-DiscreteBayesian')
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)

    def get_features_dict(self):
        try:
            return json.loads(self.important_features) if self.important_features else {}
        except Exception:
            return {}

    def __repr__(self):
        return f"<Prediction Patient={self.patient_id} Risk={self.disease_probability}% ({self.risk_level})>"
