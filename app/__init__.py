# app\__init__.py

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import joblib
from .model.load_model import load_intrusion_model, load_lgbm_model
import os

mlp_model = load_intrusion_model(os.getenv("MLP_MODEL_PATH", "ckpts/model_scripted.pt"))
mlp_model.eval()

lgbm_model = load_lgbm_model(os.getenv("LGBM_MODEL_PATH", "ckpts/lgbm_model.txt"))

encoder = joblib.load(os.getenv("ENCODER_PATH", "ckpts/encoder.pkl"))
scaler = joblib.load(os.getenv("SCALER_PATH", "ckpts/scaler.pkl"))

socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    from app.routes.detect import detect_bp
    from app.routes.data import data_bp
    app.register_blueprint(detect_bp, url_prefix="/detect")
    app.register_blueprint(data_bp, url_prefix="/data")
    
    socketio.init_app(app)
    
    return app
    