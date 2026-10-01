import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'hospital-ai-secret-key-medical-bayesian-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f"sqlite:///{os.path.join(BASE_DIR, 'database', 'hospital_ai.db')}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATASET_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    MODEL_VERSION = 'v1.2.0-DiscreteBayesian'
    ALPHA = 0.05
