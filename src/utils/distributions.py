"""
Causal distributions module.

This module implements the Cauchy distribution and related functions for the causal language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def cauchy_pdf(x, loc, scale):
    """
    Compute the probability density function (PDF) of the Cauchy distribution.
    
    Args:
        x (torch.Tensor): Input values
        loc (torch.Tensor): Location parameter (median)
        scale (torch.Tensor): Scale parameter (must be positive)
        
    Returns:
        torch.Tensor: PDF values
    """
    return 1 / (math.pi * scale * (1 + ((x - loc) / scale) ** 2))


def cauchy_cdf(x, loc, scale):
    """
    Compute the cumulative distribution function (CDF) of the Cauchy distribution.
    
    Args:
        x (torch.Tensor): Input values
        loc (torch.Tensor): Location parameter (median)
        scale (torch.Tensor): Scale parameter (must be positive)
        
    Returns:
        torch.Tensor: CDF values
    """
    return 0.5 + (1 / math.pi) * torch.atan((x - loc) / scale)


def cauchy_sample(loc, scale, sample_shape=torch.Size()):
    """
    Sample from a Cauchy distribution using the inverse CDF method.
    
    Args:
        loc (torch.Tensor): Location parameter (median)
        scale (torch.Tensor): Scale parameter (must be positive)
        sample_shape (torch.Size, optional): Shape of the sample. Defaults to torch.Size().
        
    Returns:
        torch.Tensor: Samples from the Cauchy distribution
    """
    uniform = torch.rand(sample_shape + loc.shape, device=loc.device)
    return loc + scale * torch.tan(math.pi * (uniform - 0.5))


def cauchy_sample_reparameterized(loc, scale, epsilon=None):
    """
    Sample from a Cauchy distribution using the reparameterization trick.
    
    This allows gradients to flow through the sampling operation.
    
    Args:
        loc (torch.Tensor): Location parameter (median)
        scale (torch.Tensor): Scale parameter (must be positive)
        epsilon (torch.Tensor, optional): Random noise from Uniform(0, 1). 
                                         If None, it will be generated.
        
    Returns:
        torch.Tensor: Samples from the Cauchy distribution
    """
    if epsilon is None:
        epsilon = torch.rand_like(loc)
    return loc + scale * torch.tan(math.pi * (epsilon - 0.5))


def cauchy_log_prob(x, loc, scale):
    """
    Compute the log probability density of the Cauchy distribution.
    
    Args:
        x (torch.Tensor): Input values
        loc (torch.Tensor): Location parameter (median)
        scale (torch.Tensor): Scale parameter (must be positive)
        
    Returns:
        torch.Tensor: Log probability density values
    """
    return -torch.log(math.pi * scale) - torch.log(1 + ((x - loc) / scale) ** 2)


def cauchy_nll_loss(pred_loc, pred_scale, target, reduction='mean'):
    """
    Compute the negative log-likelihood loss for Cauchy distribution.
    
    Args:
        pred_loc (torch.Tensor): Predicted location parameter
        pred_scale (torch.Tensor): Predicted scale parameter
        target (torch.Tensor): Target values
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. Defaults to 'mean'.
        
    Returns:
        torch.Tensor: Negative log-likelihood loss, shape depends on reduction.
    """
    # Calculate the log probability for each sample
    log_prob = - (math.log(math.pi) + torch.log(pred_scale) + torch.log(1 + ((target - pred_loc) / pred_scale)**2))
    
    # Apply reduction
    if reduction == 'mean':
        return -log_prob.mean()
    elif reduction == 'sum':
        return -log_prob.sum()
    elif reduction == 'none':
        return -log_prob
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


class CauchyLinear(nn.Module):
    """
    Linear layer that preserves Cauchy distribution properties.
    
    When a Cauchy random variable is transformed by a linear function,
    the result is still a Cauchy random variable with transformed parameters.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize the CauchyLinear layer.
        
        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            bias (bool, optional): If set to False, the layer will not learn an additive bias. 
                                  Defaults to True.
        """
        super().__init__()
        # Separate linear layers for location and scale parameters
        self.loc_layer = nn.Linear(in_features, out_features, bias=bias)
        self.scale_layer = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, loc, scale):
        """
        Forward pass that transforms Cauchy distribution parameters.
        
        Args:
            loc (torch.Tensor): Location parameter of input Cauchy distribution
            scale (torch.Tensor): Scale parameter of input Cauchy distribution
            
        Returns:
            tuple: (transformed_loc, transformed_scale)
        """
        # Transform location parameter using the loc linear layer
        transformed_loc = self.loc_layer(loc)
        
        # For scale parameter, we need to use the absolute values of weights
        # Scale transforms as: scale_out = sum(|w_i| * scale_in_i)
        abs_weight = torch.abs(self.scale_layer.weight)
        transformed_scale = F.linear(scale, abs_weight, self.scale_layer.bias)
        
        return transformed_loc, transformed_scale

