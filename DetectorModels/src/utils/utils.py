import argparse
import os
import random
import numpy as np
import torch
from model.MLP import MLP

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="Data directory containing UNSW-NB15 dataset", help="Data directory containing UNSW-NB15 dataset")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="MLP", help="Model name")
    parser.add_argument("--save_dir", type=str, default="ckpts", help="Directory to save model")

    # Training parameters (NN-specific)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", help="Scheduler")
    parser.add_argument("--T_max", type=int, default=10, help="T_max for CosineAnnealingLR")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
    parser.add_argument("--T_0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--n_fold", type=int, default=5, help="Number of folds")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to train")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # LightGBM parameters
    parser.add_argument("--num_leaves", type=int, default=31, help="Number of leaves in LightGBM")
    parser.add_argument("--feature_fraction", type=float, default=0.8, help="Feature fraction for LightGBM")
    parser.add_argument("--bagging_fraction", type=float, default=0.8, help="Bagging fraction for LightGBM")
    parser.add_argument("--bagging_freq", type=int, default=5, help="Bagging frequency for LightGBM")
    parser.add_argument("--num_boost_round", type=int, default=100, help="Number of boosting rounds")
    parser.add_argument("--early_stopping_rounds", type=int, default=10, help="Early stopping rounds for LightGBM")

    # XGBoost parameters
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth for base learners")
    parser.add_argument("--eta", type=float, default=0.1, help="Boosting learning rate")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio of the training instance")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Subsample ratio of columns when constructing each tree")
    parser.add_argument("--scale_pos_weight", type=float, default=1, help="Balancing of positive and negative weights")

    # SVM parameters
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter for SVM")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for SVM")
    parser.add_argument("--gamma", type=str, default="scale", help="Kernel coefficient for SVM")
    parser.add_argument("--shrinking", type=bool, default=True, help="Whether to use the shrinking heuristic")
    
    args = parser.parse_args()
    CONFIG = vars(args)
    return args, CONFIG

def fetch_model(CONFIG):
    if CONFIG["model_name"] == "MLP":
        model = MLP(input_dim=CONFIG["input_dim"], hidden_dims=[32], output_dim=1)
    else:
        raise ValueError(f"Model {CONFIG['model_name']} is not recognized.")
    
    return model
