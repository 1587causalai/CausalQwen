"""
Initialization utilities for CausalQwen models.

This module provides various weight and bias initialization methods
for the causal language model components.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def init_weights_xavier_gain(module: nn.Module, gain: float = 1.0):
    """
    使用Xavier/Glorot初始化方法初始化线性层权重。
    
    Args:
        module (nn.Module): 要初始化的模块，通常是nn.Linear
        gain (float): 缩放因子，默认为1.0
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)


def init_bias_const(module: nn.Module, value: float = 0.0):
    """
    将偏置初始化为常数值。
    
    Args:
        module (nn.Module): 要初始化的模块，通常是nn.Linear
        value (float): 偏置的常数值，默认为0.0
    """
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.constant_(module.bias, value)


def init_weights_normal(module: nn.Module, mean: float = 0.0, std: float = 0.02):
    """
    使用正态分布初始化权重。
    
    Args:
        module (nn.Module): 要初始化的模块
        mean (float): 正态分布的均值
        std (float): 正态分布的标准差
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=mean, std=std)


def init_weights_kaiming(module: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    """
    使用Kaiming/He初始化方法初始化权重。
    
    Args:
        module (nn.Module): 要初始化的模块
        mode (str): 'fan_in' 或 'fan_out'
        nonlinearity (str): 激活函数类型
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)


def init_weights_uniform(module: nn.Module, a: float = -0.1, b: float = 0.1):
    """
    使用均匀分布初始化权重。
    
    Args:
        module (nn.Module): 要初始化的模块
        a (float): 均匀分布的下界
        b (float): 均匀分布的上界
    """
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, a=a, b=b)


def apply_initialization(module: nn.Module, 
                        weight_init: str = 'xavier', 
                        bias_init: str = 'zero',
                        weight_gain: float = 1.0,
                        bias_value: float = 0.0):
    """
    对模块应用指定的初始化策略。
    
    Args:
        module (nn.Module): 要初始化的模块
        weight_init (str): 权重初始化方法 ['xavier', 'kaiming', 'normal', 'uniform']
        bias_init (str): 偏置初始化方法 ['zero', 'constant']
        weight_gain (float): 权重初始化的增益因子
        bias_value (float): 偏置的常数值（当bias_init='constant'时使用）
    """
    # 权重初始化
    if weight_init == 'xavier':
        init_weights_xavier_gain(module, gain=weight_gain)
    elif weight_init == 'kaiming':
        init_weights_kaiming(module)
    elif weight_init == 'normal':
        init_weights_normal(module, std=0.02)
    elif weight_init == 'uniform':
        init_weights_uniform(module)
    else:
        raise ValueError(f"Unsupported weight initialization: {weight_init}")
    
    # 偏置初始化
    if bias_init == 'zero':
        init_bias_const(module, 0.0)
    elif bias_init == 'constant':
        init_bias_const(module, bias_value)
    else:
        raise ValueError(f"Unsupported bias initialization: {bias_init}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数数量。
    
    Args:
        model (nn.Module): 要统计的模型
        trainable_only (bool): 是否只统计可训练参数
        
    Returns:
        int: 参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_parameters(module: nn.Module):
    """
    冻结模块的所有参数。
    
    Args:
        module (nn.Module): 要冻结的模块
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_parameters(module: nn.Module):
    """
    解冻模块的所有参数。
    
    Args:
        module (nn.Module): 要解冻的模块
    """
    for param in module.parameters():
        param.requires_grad = True


def get_parameter_info(model: nn.Module) -> dict:
    """
    获取模型参数的详细信息。
    
    Args:
        model (nn.Module): 要分析的模型
        
    Returns:
        dict: 包含参数信息的字典
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
    } 