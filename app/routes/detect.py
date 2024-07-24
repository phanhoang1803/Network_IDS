from flask import Blueprint, request, jsonify
from app import model
import numpy as np

detect_bp = Blueprint("detect", __name__)

@detect_bp.route("/", methods=["POST", "GET"])
def detect_intrusion():
    print(model)
    
    
    if request.method == "GET":
        return "Hello, World!"
    
    
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model(features)
    is_intrusion = prediction[0] == 1
    return jsonify({"intrusion": is_intrusion, "prediction": prediction})