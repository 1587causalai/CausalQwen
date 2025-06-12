"""
Distribution utilities for the Causal Language Model.

核心设计原则：
- 所有训练过程基于分布参数的解析计算，无需采样
- 利用柯西分布的线性封闭性进行高效的参数传播
- 仅在推理时的探索性生成中可能需要采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CauchyLinear(nn.Module):
    """
    A linear layer that operates on Cauchy distributions.
    
    This layer transforms Cauchy distribution parameters through a linear transformation,
    maintaining the Cauchy distribution property due to the linear closure of Cauchy distributions.
    
    核心数学原理：
    如果 U ~ Cauchy(μ, γ)，则 Y = aU + b ~ Cauchy(aμ + b, |a|γ)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize the CauchyLinear layer.
        
        Args:
            in_features (int): Size of input features
            out_features (int): Size of output features
            bias (bool): Whether to include bias term. Default: True
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Bias parameter (optional)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, loc, scale):
        """
        Transform Cauchy distribution parameters.
        
        基于柯西分布的线性变换性质：
        如果 U ~ Cauchy(loc_U, scale_U)
        则 Y = W·U + b ~ Cauchy(W·loc_U + b, |W|·scale_U)
        
        Args:
            loc (torch.Tensor): Location parameters of input Cauchy distribution
            scale (torch.Tensor): Scale parameters of input Cauchy distribution
            
        Returns:
            tuple: (new_loc, new_scale) - Parameters of output Cauchy distribution
        """
        # Linear transformation of location parameter
        new_loc = F.linear(loc, self.weight, self.bias)
        
        # Scale transformation (sum of absolute values)
        # For a linear combination: Y = W·U, scale_Y = |W|·scale_U
        weight_abs = self.weight.abs()
        new_scale = F.linear(scale, weight_abs, None)  # No bias for scale
        
        return new_loc, new_scale


def cauchy_cdf(value, loc, scale):
    """
    Calculate the cumulative distribution function of a Cauchy distribution.
    
    数学公式：F(x) = 1/2 + (1/π) * arctan((x - μ) / γ)
    
    这是OvR分类中计算P(S_k > C_k)的核心函数。
    
    Args:
        value (torch.Tensor): Values to evaluate
        loc (torch.Tensor): Location parameters
        scale (torch.Tensor): Scale parameters
        
    Returns:
        torch.Tensor: CDF values
    """
    return 0.5 + (1 / torch.pi) * torch.atan((value - loc) / scale)


def cauchy_log_prob(value, loc, scale, reduction='none'):
    """
    Calculate the log probability density of a value under a Cauchy distribution.
    
    数学公式：log p(x) = -log(π·γ) - log(1 + ((x-μ)/γ)²)
    
    这是回归损失中柯西负对数似然的核心函数。
    
    Args:
        value (torch.Tensor): Values to evaluate
        loc (torch.Tensor): Location parameters
        scale (torch.Tensor): Scale parameters
        reduction (str): Specifies the reduction to apply to the output:
                        'none' | 'mean' | 'sum'. Default: 'none'
        
    Returns:
        torch.Tensor: Log probabilities
    """
    log_prob = -torch.log(torch.pi * scale) - torch.log(1 + ((value - loc) / scale) ** 2)
    
    if reduction == 'none':
        return log_prob
    elif reduction == 'mean':
        return log_prob.mean()
    elif reduction == 'sum':
        return log_prob.sum()
    else:
        raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")


def cauchy_nll_loss(target, loc, scale, reduction='mean'):
    """
    Calculate the negative log-likelihood loss for Cauchy distribution.
    
    修正：参数顺序为 (target, loc, scale) 以符合 PyTorch 惯例
    
    数学公式：L = log(π·scale) + log(1 + ((target - loc)/scale)²)
    
    Args:
        target (torch.Tensor): True values (第一个参数)
        loc (torch.Tensor): Predicted location parameters  
        scale (torch.Tensor): Predicted scale parameters
        reduction (str): Specifies the reduction to apply to the output:
                        'none' | 'mean' | 'sum'. Default: 'mean'
        
    Returns:
        torch.Tensor: NLL loss values
    """
    # 确保scale为正，避免数值问题
    scale = torch.clamp(scale, min=1e-8)
    
    # 计算标准化残差
    z = (target - loc) / scale
    
    # 柯西NLL公式：log(π * scale) + log(1 + z²)
    # 使用 log1p 提高数值稳定性
    import math
    log_scale_term = torch.log(scale) + math.log(math.pi)
    residual_term = torch.log1p(z ** 2)  # log(1 + z²)
    
    loss = log_scale_term + residual_term
    
    # 应用 reduction
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")

# 确保所有必要的函数都被导出
__all__ = ['cauchy_cdf', 'cauchy_pdf', 'cauchy_sample_reparameterized', 'cauchy_nll_loss', 'CauchyLinear']

