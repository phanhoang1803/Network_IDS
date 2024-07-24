import gc
import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from data.data_loading import load_data
from data.data_processing import process_data
from utils.utils import parse_args, set_seed, make_dir
import joblib

def train_svm(X_train, y_train, X_valid, y_valid, CONFIG):
    """
    Train a Support Vector Machine (SVM) model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation labels.
        CONFIG (dict): Configuration dictionary.

    Returns:
        model: Trained SVM model.
    """
    model = SVC(
        C=CONFIG["C"],
        kernel=CONFIG["kernel"],
        gamma=CONFIG["gamma"],
        probability=True,
        shrinking=CONFIG["shrinking"],
        random_state=CONFIG["seed"],
        verbose=True
    )
    
    print("[INFO] Training SVM model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"[INFO] Training completed in {time.time() - start_time:.2f} seconds.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained SVM model on the test set.

    Args:
        model: Trained SVM model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')

    return metrics

def main():
    args, CONFIG = parse_args()
    set_seed(CONFIG["seed"])

    train_csv = os.path.join(CONFIG["data_dir"], "UNSW_NB15_training-set.csv")
    df = load_data(train_csv, CONFIG)
    df, encoder, scaler = process_data(df)

    joblib.dump(os.path.join(CONFIG["save_dir"], "encoder.pkl"))
    joblib.dump(os.path.join(CONFIG["save_dir"], "scaler.pkl"))

    print(df.info())

    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=CONFIG["seed"], stratify=df["label"])

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_valid = df_valid.drop(columns=["label"])
    y_valid = df_valid["label"]

    make_dir(CONFIG["save_dir"])

    model = train_svm(X_train, y_train, X_valid, y_valid, CONFIG)

    print("[INFO] Evaluating model on test set...")
    test_csv = os.path.join(CONFIG["data_dir"], "UNSW_NB15_testing-set.csv")
    df_test = load_data(test_csv, CONFIG)
    df_test, _, _ = process_data(df_test, encoder, scaler)
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    metrics = evaluate_model(model, X_test, y_test)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(CONFIG["save_dir"], "svm_test_metrics.csv"), index=False)

if __name__ == "__main__":
    main()
