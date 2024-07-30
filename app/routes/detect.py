from flask import Blueprint, request, jsonify
from app import mlp_model, lgbm_model, encoder, scaler
import numpy as np
import pandas as pd
from DetectorModels.src.data.data_processing import generate_features, process_data
from utils import lgbm_inference

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
    """
    Endpoint for detecting intrusions using LightGBM model.
    Supports both JSON and multipart/form-data content types.

    Returns:
        JSON response with intrusion prediction and status code.
    """
    # Process JSON content type
    if request.content_type == 'application/json':
        data = request.get_json()
        df = pd.DataFrame(data)
    # Process multipart/form-data content type
    elif request.content_type and request.content_type.startswith('multipart/form-data'):
        # Check for file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        # Read CSV file
        df = pd.read_csv(file)
    else:
        # Return error response for unsupported content type
        return jsonify({'error': 'Unsupported content type'}), 400

    # Print dataframe info
    print(df.info())
    
    is_intrusion = lgbm_inference(df, encoder, scaler)

    # Return intrusion prediction as JSON response
    return jsonify({"intrusion": is_intrusion.tolist()})


