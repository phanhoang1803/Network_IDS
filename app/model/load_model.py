import os
import torch
import lightgbm

def load_intrusion_model(model_path):
    model = torch.jit.load(model_path)
    
    return model

def load_lgbm_model(model_path):
    """"
    Loads the LightGBM model from the specified path. (.txt)
    """
    # Must be .txt
    assert model_path.endswith('.txt')
    
    model = lightgbm.Booster(model_file=model_path)
    return model