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
        
        # 分离的线性层，使逻辑更清晰
        self.loc_net = nn.Linear(hidden_size, causal_dim)
        self.scale_net = nn.Linear(hidden_size, causal_dim)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行归因推断的前向传播。

        Args:
            features (torch.Tensor): 形状为 `[B, S, H]` 或 `[B, H]` 的输入特征。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loc: 形状为 `[B, S, C]` 的位置参数。
                - scale: 形状为 `[B, S, C]` 的尺度参数。
        """
        loc = self.loc_net(features)
        
        # 使用 softplus 替代 exp 来计算 scale，以增强数值稳定性
        # softplus(x) = log(1 + exp(x))
        # 添加一个小的 epsilon 防止 scale 为零
        scale = F.softplus(self.scale_net(features)) + 1e-6
        
        return loc, scale

    def initialize_for_identity_mapping(self, scale_bias: float = 10.0):
        """
        初始化网络以实现恒等映射的归因推断。
        
        目标: 
            - loc_net -> 恒等映射 (loc = I * z + 0)
            - scale_net -> 常量输出 (scale = softplus(scale_bias) ≈ 10)
            
        Args:
            scale_bias (float): 尺度参数的偏置。 softplus(10.0) ≈ 10.
        """
        with torch.no_grad():
            # 初始化 loc_net
            if self.hidden_size == self.causal_dim:
                # C=H: 精确恒等映射
                self.loc_net.weight.copy_(torch.eye(self.causal_dim))
                nn.init.zeros_(self.loc_net.bias)
                print(f"  - AbductionNetwork.loc_net initialized for identity mapping.")
            else:
                # C≠H: 使用更小的增益来减少初始差异
                nn.init.xavier_uniform_(self.loc_net.weight, gain=0.1)
                nn.init.zeros_(self.loc_net.bias)
                print(f"  - AbductionNetwork.loc_net initialized with small Xavier gain=0.1 (H={self.hidden_size}!=C={self.causal_dim})")
            
            # 初始化 scale_net 以输出一个常量
            nn.init.zeros_(self.scale_net.weight)
            nn.init.constant_(self.scale_net.bias, scale_bias)
            print(f"  - AbductionNetwork.scale_net initialized for constant output (bias={scale_bias}).")


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

