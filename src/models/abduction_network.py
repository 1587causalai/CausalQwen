"""
Abduction Network module.

This module implements the abduction network for the causal language model,
which infers the distribution parameters of the latent causal state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbductionNetwork(nn.Module):
    """
    Abduction Network for inferring individual causal representation distribution.
    
    This network takes the feature representation from the Feature Network and outputs
    the parameters (location and scale) of the Cauchy distribution for the latent
    individual causal representation U.
    """
    
    def __init__(self, hidden_size, causal_dim):
        """
        Initialize the Abduction Network.
        
        Args:
            hidden_size (int): Size of the input feature representation
            causal_dim (int): Dimensionality of the individual causal representation
        """
        super().__init__()
        # A simple linear layer to map from feature space to causal space
        # It outputs two values for each dimension: one for loc, one for log_scale
        self.fc = nn.Linear(hidden_size, causal_dim * 2)

    def init_weights(self):
        """
        Initialize weights for identity mapping.
        This is a specific strategy for the case where hidden_size == causal_dim.
        """
        # Initialize the linear layer for an identity-like mapping
        # where loc is the feature and scale is small.
        # This requires hidden_size to be equal to causal_dim.
        hidden_size = self.fc.in_features
        causal_dim = self.fc.out_features // 2
        
        if hidden_size == causal_dim:
            # Create an identity matrix for the location part
            identity_matrix = torch.eye(hidden_size, causal_dim)
            # Create a zero matrix for the scale part
            zero_matrix = torch.zeros(hidden_size, causal_dim)
            
            # Concatenate to form the final weight matrix [causal_dim * 2, hidden_size]
            # The first half of weights for loc, second half for scale.
            final_weight = torch.cat((identity_matrix, zero_matrix), dim=1).t()
            
            with torch.no_grad():
                self.fc.weight.copy_(final_weight)
                # Initialize loc bias to zero
                self.fc.bias.data[:causal_dim].fill_(0.0)
                # Initialize log_scale bias to a value that results in a large scale,
                # reflecting high initial uncertainty about the causal representation.
                self.fc.bias.data[causal_dim:].fill_(2.3) # exp(2.3) ≈ 10
        else:
            # For other cases, use a conservative initialization
            with torch.no_grad():
                # Use Xavier initialization for weights with small gain
                torch.nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
                
                # Initialize loc bias to zero
                self.fc.bias.data[:causal_dim].fill_(0.0)
                # Initialize log_scale bias to a value that results in a large scale,
                # reflecting high initial uncertainty about the causal representation.
                self.fc.bias.data[causal_dim:].fill_(2.3) # exp(2.3) ≈ 10
                
                print(f"  - AbductionNetwork initialized with Xavier (hidden_size={hidden_size} != causal_dim={causal_dim})")


    def forward(self, features):
        """
        Infer the distribution parameters of the individual causal representation.
        
        Args:
            features (torch.Tensor): Input feature representation
                                     Shape: [batch_size, hidden_size]
        
        Returns:
            tuple: (loc, scale) - Parameters of the Cauchy distribution
                   Each has shape: [batch_size, causal_dim]
        """
        # The output has shape [batch_size, causal_dim * 2]
        output = self.fc(features)
        
        # Split the output into loc and log_scale
        loc, log_scale = torch.chunk(output, 2, dim=-1)
        
        # Ensure scale parameter is positive using exponential function
        scale = torch.exp(log_scale)
        
        return loc, scale


class DeepAbductionNetwork(nn.Module):
    """
    Deep Abduction Network with multiple layers.
    
    This is an extension of the basic AbductionNetwork with additional
    hidden layers for more complex inference.
    """
    
    def __init__(self, input_size, causal_dim, hidden_sizes=[512, 256]):
        """
        Initialize the deep abduction network.
        
        Args:
            input_size (int): Size of the input feature representation
            causal_dim (int): Dimensionality of the individual causal representation
            hidden_sizes (list, optional): Sizes of hidden layers. Defaults to [512, 256].
        """
        super().__init__()
        self.input_size = input_size
        self.causal_dim = causal_dim
        
        # Build MLP layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.mlp = nn.Sequential(*layers)
        
        # Final layer to output distribution parameters
        self.causal_inference_layer = nn.Linear(prev_size, causal_dim * 2)
        
    def forward(self, features):
        """
        Infer the distribution parameters of the individual causal representation.
        
        Args:
            features (torch.Tensor): Input feature representation
                                     Shape: [batch_size, input_size]
        
        Returns:
            tuple: (loc, scale) - Parameters of the Cauchy distribution
                  Each has shape: [batch_size, causal_dim]
        """
        # Process features through MLP
        hidden = self.mlp(features)
        
        # Map to distribution parameters
        params = self.causal_inference_layer(hidden)
        
        # Split into location and scale parameters
        loc, log_scale = torch.split(params, self.causal_dim, dim=-1)
        
        # Ensure scale parameter is positive using exponential function
        scale = torch.exp(log_scale)
        
        return loc, scale


class MockAbductionNetwork(nn.Module):
    """
    A mock abduction network for testing and simplified setups.
    This network simply passes through the features, assuming hidden_size equals
    causal_dim.
    """
    
    def __init__(self, hidden_size, causal_dim):
        """
        Initialize the Mock Abduction Network.
        
        Args:
            hidden_size (int): Size of the input feature representation
            causal_dim (int): Dimensionality of the individual causal representation
        """
        super().__init__()
        assert hidden_size == causal_dim, "MockAbductionNetwork requires hidden_size == causal_dim"
        self.causal_dim = causal_dim

    def forward(self, features):
        """
        Infer the distribution parameters of the individual causal representation.
        
        Args:
            features (torch.Tensor): Input feature representation
                                     Shape: [batch_size, hidden_size]
        
        Returns:
            tuple: (loc, scale) - Parameters of the Cauchy distribution
                  Each has shape: [batch_size, causal_dim]
        """
        # Ensure scale parameter is positive using exponential function
        scale = torch.exp(features)
        
        return features, scale

