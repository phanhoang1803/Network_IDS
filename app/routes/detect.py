from flask import Blueprint, request, jsonify
from app import mlp_model, lgbm_model
import numpy as np

detect_bp = Blueprint("detect", __name__)

@detect_bp.route("/MLP", methods=["POST"])
def detect_intrusion_mlp():
    print(mlp_model)
    
    data = request.get_json()
    print("data", data)
    
    features = np.array(data["features"]).reshape(1, -1)
    prediction = mlp_model(features)
    is_intrusion = prediction[0] == 1
    return jsonify({"intrusion": is_intrusion, "prediction": prediction})

@detect_bp.route("/LGBM", methods=["POST"])
def detect_intrusion_lgbm():
    print(lgbm_model)
    
    data = request.get_json()
    print("data", data)
    
    features = np.array(data["features"]).reshape(1, -1)
    prediction = lgbm_model.predict(features)
    print(prediction)
    is_intrusion = prediction[0] >= 0.5
    print(is_intrusion)
    
    return jsonify({"intrusion": str(is_intrusion)})