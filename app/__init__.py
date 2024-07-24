from flask import Flask
from .model.load_model import load_intrusion_model
import os

model = load_intrusion_model(os.getenv("MODEL_PATH", "ckpts/model_scripted.pt"))
model.eval()

def create_app():
    app = Flask(__name__)
    
    from app.routes.detect import detect_bp
    app.register_blueprint(detect_bp, url_prefix="/detect")
    
    return app
    