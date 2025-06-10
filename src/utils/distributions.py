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
    
    从 mathematical_foundations.md 的正确公式：
    L_cauchy_nll = log(π * scale_Y) + log(1 + ((y_true - loc_Y)/scale_Y)^2)
    
    Args:
        pred_loc (torch.Tensor): Predicted location parameter
        pred_scale (torch.Tensor): Predicted scale parameter  
        target (torch.Tensor): Target values
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. Defaults to 'mean'.
        
    Returns:
        torch.Tensor: Negative log-likelihood loss, shape depends on reduction.
    """
    # 严格按照数学文档中的公式实现：
    # L_cauchy_nll = log(π * scale) + log(1 + ((target - loc)/scale)^2)
    nll_loss = (torch.log(torch.tensor(math.pi, device=pred_scale.device) * pred_scale) + 
                torch.log(1 + ((target - pred_loc) / pred_scale)**2))
    
    # Apply reduction
    if reduction == 'mean':
        return nll_loss.mean()
    elif reduction == 'sum':
        return nll_loss.sum()
    elif reduction == 'none':
        return nll_loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


class CauchyLinear(nn.Module):
    """
    Linear layer that preserves Cauchy distribution properties.
    
    正确的柯西分布线性变换：
    如果 X ~ Cauchy(μ, σ)，那么 Y = AX + B ~ Cauchy(Aμ + B, |A|σ)
    关键：loc 和 scale 必须使用**相同的权重矩阵 A**！
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
        # 共享权重矩阵 A 用于 loc 和 scale 的变换
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
            
        # 使用标准的线性层初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, loc, scale):
        """
        Forward pass that transforms Cauchy distribution parameters.
        
        正确的柯西分布线性变换：
        If X ~ Cauchy(μ, σ), then Y = AX + B ~ Cauchy(Aμ + B, |A|σ)
        
        Args:
            loc (torch.Tensor): Location parameter of input Cauchy distribution
            scale (torch.Tensor): Scale parameter of input Cauchy distribution
            
        Returns:
            tuple: (transformed_loc, transformed_scale)
        """
        # Transform location parameter: loc_out = A * loc + B
        transformed_loc = F.linear(loc, self.weight, self.bias)
        
        # Transform scale parameter: scale_out = |A| * scale (NO BIAS!)
        # 使用权重的绝对值，确保 scale 保持正值
        abs_weight = torch.abs(self.weight)
        transformed_scale = F.linear(scale, abs_weight, bias=None)
        
        # 只确保 scale > 0，不设置上限以保持数学正确性
        transformed_scale = torch.clamp(transformed_scale, min=1e-6)
        
        return transformed_loc, transformed_scale

