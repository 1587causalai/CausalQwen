"""
Abduction Network module.

This module implements the abduction network for the causal language model,
which infers the distribution parameters of the latent causal state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..utils.initialization_utils import init_weights_xavier_gain, init_bias_const


class AbductionNetwork(nn.Module):
    """
    归因推断网络 (Attributional Abduction Network)
    
    将观测特征 z 映射为个体因果表征 U 的分布参数。
    这是一个从 "果"（观测特征）到 "因"（潜在子群体）的归因过程。
    
    数学描述:
        z -> (loc, scale) where U ~ Cauchy(loc, scale)
        
    参数:
        hidden_size (int): 输入特征的维度
        causal_dim (int): 因果表征的维度
    """
    
    def __init__(self, hidden_size: int, causal_dim: int):
        """
        初始化归因推断网络。

        Args:
            hidden_size (int): 输入特征 `z` 的维度。
            causal_dim (int): 因果表征 `U` 的维度。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.causal_dim = causal_dim
        
        # 单一线性层映射到柯西分布的两个参数
        self.fc = nn.Linear(hidden_size, causal_dim * 2)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行归因推断的前向传播。

        Args:
            features (torch.Tensor): 形状为 `[B, S, H]` 的输入特征。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loc: 形状为 `[B, S, C]` 的位置参数。
                - scale: 形状为 `[B, S, C]` 的尺度参数。
        """
        if features.dim() == 3:
            # 序列输入: [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = features.shape
            features_flat = features.view(-1, hidden_size)
            output_flat = self.fc(features_flat)
            output = output_flat.view(batch_size, seq_len, -1)
        else:
            # 单一输入: [batch_size, hidden_size]
            output = self.fc(features)
            
        # 分离位置参数和尺度参数
        loc, log_scale = torch.chunk(output, 2, dim=-1)
        scale = torch.exp(log_scale)  # 确保尺度为正
        
        return loc, scale

    def initialize_for_identity_mapping(self, scale_bias: float = 2.3):
        """
        初始化网络以实现恒等映射的归因推断
        
        目标: 
            - loc = I * z + 0 (恒等映射，保持特征信息)
            - scale = exp(scale_bias) (高初始不确定性)
            
        Args:
            scale_bias (float): 尺度参数的对数偏置，控制初始不确定性水平
        """
        with torch.no_grad():
            # 权重矩阵设置
            weight = self.fc.weight.data  # [causal_dim * 2, hidden_size]
            
            # 前半部分(loc)设为恒等矩阵的子集或扩展
            loc_weight = weight[:self.causal_dim, :]
            if self.hidden_size == self.causal_dim:
                # C=H: 精确恒等映射
                loc_weight.copy_(torch.eye(self.causal_dim))
            else:
                # C≠H: Xavier初始化作为恒等映射的近似
                nn.init.xavier_uniform_(loc_weight, gain=1.0)
                print(f"  - AbductionNetwork (归因推断网络) initialized with Xavier (hidden_size={self.hidden_size} != causal_dim={self.causal_dim})")
            
            # 后半部分(scale)设为零矩阵（配合偏置实现常数输出）
            scale_weight = weight[self.causal_dim:, :]
            scale_weight.zero_()
            
            # 偏置设置
            bias = self.fc.bias.data  # [causal_dim * 2]
            bias[:self.causal_dim].zero_()  # loc偏置为0
            bias[self.causal_dim:].fill_(scale_bias)  # scale偏置为scale_bias
            
        if self.hidden_size == self.causal_dim:
            print(f"  - AbductionNetwork (归因推断网络) initialized for identity mapping (scale_bias={scale_bias}).")


class DeepAbductionNetwork(nn.Module):
    """
    深度归因推断网络 (Deep Attributional Abduction Network)
    
    这是对基础 `AbductionNetwork` 的扩展，使用了多层感知机（MLP）来增强
    从观测特征到个体因果表征分布的归因能力。
    
    参数:
        hidden_size (int): 输入特征的维度
        causal_dim (int): 因果表征的维度  
        num_layers (int): MLP的层数
        activation (str): 激活函数类型
    """
    
    def __init__(self, hidden_size: int, causal_dim: int, num_layers: int = 3, activation: str = "relu"):
        """
        初始化深度归因推断网络。

        Args:
            hidden_size (int): 输入特征的维度。
            causal_dim (int): 因果表征的维度。
            num_layers (int): MLP的层数。
            activation (str): 激活函数类型。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.causal_dim = causal_dim
        self.num_layers = num_layers
        
        # 选择激活函数
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 构建MLP层
        layers = []
        current_dim = hidden_size
        
        # 隐藏层
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.1))  # 防止过拟合
        
        # 输出层
        layers.append(nn.Linear(hidden_size, causal_dim * 2))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行深度归因推断的前向传播。
        """
        output = self.mlp(features)
        
        # 分离位置参数和尺度参数
        loc, log_scale = torch.chunk(output, 2, dim=-1)
        scale = torch.exp(log_scale)
        
        return loc, scale

    def init_weights(self):
        """
        为深度归因推断网络中的所有线性层应用标准初始化。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_weights_xavier_gain(module, gain=1.0)
                init_bias_const(module, 0.0)


class MockAbductionNetwork(nn.Module):
    """
    模拟归因推断网络 (Mock Attributional Abduction Network)
    
    用于测试和调试的简化版本，要求 hidden_size == causal_dim，
    实现完美的恒等映射归因。
    """
    
    def __init__(self, hidden_size: int, causal_dim: int):
        """
        初始化模拟归因推断网络。

        Args:
            hidden_size (int): 输入特征 `z` 的维度。
            causal_dim (int): 因果表征 `U` 的维度。
        """
        super().__init__()
        assert hidden_size == causal_dim, "MockAbductionNetwork (模拟归因网络) requires hidden_size == causal_dim"
        
        self.hidden_size = hidden_size
        self.causal_dim = causal_dim
        
        # 恒等映射：loc = features, scale = constant
        self.register_buffer('scale_value', torch.tensor(1.0))
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回固定的 loc (恒等映射) 和 scale。
        """
        batch_size = features.shape[0]
        
        # loc 直接等于输入特征（完美恒等映射）
        loc = features
        
        # scale 为常数
        if features.dim() == 3:
            seq_len = features.shape[1]
            scale = self.scale_value.expand(batch_size, seq_len, self.causal_dim)
        else:
            scale = self.scale_value.expand(batch_size, self.causal_dim)
            
        return loc, scale

