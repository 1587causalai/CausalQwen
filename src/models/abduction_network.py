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
    Abduction Network for inferring causal state distribution.
    
    Given feature representation z, this network outputs the parameters
    (location and scale) of the Cauchy distribution for the latent causal state U.
    """
    
    def __init__(self, input_size, causal_dim):
        """
        Initialize the abduction network.
        
        Args:
            input_size (int): Size of the input feature representation
            causal_dim (int): Dimensionality of the latent causal state
        """
        super().__init__()
        self.input_size = input_size
        self.causal_dim = causal_dim
        
        # Linear layer to map features to distribution parameters
        # Output has twice the causal_dim: first half for location, second half for scale
        self.causal_inference_layer = nn.Linear(input_size, causal_dim * 2)
        
    def init_weights(self):
        """
        Initialize weights according to the knowledge transfer strategy.
        This sets up the abduction network to be an identity-like transformation
        at the beginning of the training.
        """
        # We assume input_size and causal_dim are the same for identity mapping
        if self.input_size != self.causal_dim:
            raise ValueError(
                f"For identity initialization, input_size ({self.input_size}) "
                f"must equal causal_dim ({self.causal_dim})."
            )

        # Get the weight and bias from the single linear layer
        weight = self.causal_inference_layer.weight
        bias = self.causal_inference_layer.bias
        
        # Split the parameters for loc and scale
        loc_weight, scale_weight = torch.split(weight, self.causal_dim, dim=0)
        loc_bias, scale_bias = torch.split(bias, self.causal_dim, dim=0)
        
        # Initialize loc parameters for identity mapping (loc_U = z)
        # Weight = Identity matrix, Bias = Zero vector
        loc_weight.data.copy_(torch.eye(self.causal_dim))
        loc_bias.data.fill_(0)
        
        # Initialize scale parameters for maximum uncertainty (scale_U = large constant)
        # Weight = Zero matrix, Bias = Large positive value (for log_scale)
        scale_weight.data.fill_(0)
        scale_bias.data.fill_(2.3) # Corresponds to exp(2.3) ≈ 10

    def forward(self, features):
        """
        Infer the distribution parameters of the latent causal state.
        
        Args:
            features (torch.Tensor): Feature representation
                                    Shape: [batch_size, input_size]
        
        Returns:
            tuple: (loc, scale) - Parameters of the Cauchy distribution
                  Each has shape: [batch_size, causal_dim]
        """
        # Map features to distribution parameters
        params = self.causal_inference_layer(features)
        
        # Split into location and scale parameters
        loc, log_scale = torch.split(params, self.causal_dim, dim=-1)
        
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
            causal_dim (int): Dimensionality of the latent causal state
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
        Infer the distribution parameters of the latent causal state.
        
        Args:
            features (torch.Tensor): Feature representation
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

