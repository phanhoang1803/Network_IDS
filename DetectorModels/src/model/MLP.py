import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    A neural network model for binary classification using multiple hidden layers.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=2):
        """
        Parameters:
            input_dim (int): The number of features in the input data.
            hidden_dims (list): The number of neurons in each of the hidden layers.
            output_dim (int): The number of classes in the output data.
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        layers = []
        in_features = input_dim
        for i, out_features in enumerate(hidden_dims):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)