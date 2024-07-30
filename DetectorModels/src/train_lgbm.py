import gc
import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from data.data_loading import load_data
from data.data_processing import process_data
from utils.utils import parse_args, set_seed, make_dir

def train_lgbm(X_train, y_train, X_valid, y_valid, CONFIG):
    """
    Train a LightGBM model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation labels.
        CONFIG (dict): Configuration dictionary.

    Returns:
        model: Trained LightGBM model.
        history: Training history with evaluation metrics.
    """
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': CONFIG["num_leaves"],
        'learning_rate': CONFIG["learning_rate"],
        'feature_fraction': CONFIG["feature_fraction"],
        'bagging_fraction': CONFIG["bagging_fraction"],
        'bagging_freq': CONFIG["bagging_freq"],
        'verbose': 1,
    }

    print("[INFO] Training LightGBM model...")
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=CONFIG["num_boost_round"],
                      valid_sets=[lgb_train, lgb_valid])
    print("[INFO] Training completed.")

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained LightGBM model on the test set.

    Args:
        model: Trained LightGBM model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred > 0.505).astype(int)

    # Number of negatives
    print(np.sum(y_test == 0))
    print(np.sum(y_pred == 0))
    print(y_pred[:100])
    print(y_test[:100])
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
    joblib.dump(encoder, os.path.join(CONFIG["save_dir"], "encoder.pkl"))
    joblib.dump(scaler, os.path.join(CONFIG["save_dir"], "scaler.pkl"))

    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=CONFIG["seed"], stratify=df["label"])

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_valid = df_valid.drop(columns=["label"])
    y_valid = df_valid["label"]

    make_dir(CONFIG["save_dir"])

    model = train_lgbm(X_train, y_train, X_valid, y_valid, CONFIG)
    model.save_model(os.path.join(CONFIG["save_dir"], "lgbm_model.txt"))

    print("[INFO] Evaluating model on test set...")
    test_csv = os.path.join(CONFIG["data_dir"], "UNSW_NB15_testing-set.csv")
    df_test = load_data(test_csv, CONFIG)
    df_test, _, _ = process_data(df_test, encoder, scaler)
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    metrics = evaluate_model(model, X_test, y_test)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(CONFIG["save_dir"], "lgbm_test_metrics.csv"), index=False)

if __name__ == "__main__":
    main()
