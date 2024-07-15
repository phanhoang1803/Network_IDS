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
        
        self.model = nn.Sequential()
        
        # First hidden layer
        self.model.add_module("linear_0", nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.model.add_module("relu_0", nn.ReLU())
        
        # Add more hidden layers if specified
        for i in range(1, len(self.hidden_dims)):
            self.model.add_module("linear_{}".format(i), nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            self.model.add_module("relu_{}".format(i), nn.ReLU())
        
        # Output layer
        self.model.add_module("linear_{}".format(len(self.hidden_dims)), nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        # Softmax activation for the output layer
        self.model.add_module("softmax", nn.Softmax(dim=1))
    
    def forward(self, x):
        return self.model(x)