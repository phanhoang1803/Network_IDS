import torch

def load_intrusion_model(model_path):
    model = torch.jit.load(model_path)
    
    return model