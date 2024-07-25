from flask import Flask
import joblib
from .model.load_model import load_intrusion_model, load_lgbm_model
import os

mlp_model = load_intrusion_model(os.getenv("MLP_MODEL_PATH", "ckpts/model_scripted.pt"))
mlp_model.eval()

lgbm_model = load_lgbm_model(os.getenv("LGBM_MODEL_PATH", "ckpts/lgbm_model.txt"))

encoder = joblib.load(os.getenv("ENCODER_PATH", "ckpts/encoder.pkl"))
scaler = joblib.load(os.getenv("SCALER_PATH", "ckpts/scaler.pkl"))

def create_app():
    app = Flask(__name__)
    
    from app.routes.detect import detect_bp
    app.register_blueprint(detect_bp, url_prefix="/detect")
    
    return app
    