"""
CausalEngine Core Networks for sklearn-compatible ML tasks

这个模块包含 CausalEngine 的核心网络，专门适配sklearn分类/回归任务：
1. AbductionNetwork: 从特征推断个体因果表征
2. ActionNetwork: 从个体表征到决策得分

专注于常规ML任务，简化大模型相关复杂性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class AbductionNetwork(nn.Module):
    """
    归因网络：从特征推断个体因果表征
    
    将输入特征 X 映射到个体因果表征分布 U ~ Cauchy(μ_U, γ_U)
    
    数学框架：
    - 输入: X ∈ R^{n_features}
    - 输出: (μ_U, γ_U) ∈ R^{hidden_size} × R^{hidden_size}_+
    - 分布: U ~ Cauchy(μ_U, γ_U)
    
    网络架构：
    - loc_net: X → μ_U (位置网络)
    - scale_net: X → γ_U (尺度网络，通过softplus确保正值)
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层/因果表征维度
        hidden_layers: MLP隐藏层配置 (n_neurons_1, n_neurons_2, ...)
        activation: 激活函数名称
        dropout: dropout比率
        gamma_init: 尺度参数初始化值
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: Tuple[int, ...] = (),
        activation: str = 'relu',
        dropout: float = 0.0,
        gamma_init: float = 10.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma_init = gamma_init
        
        # 构建位置网络 loc_net: X → μ_U
        self.loc_net = self._build_mlp(
            input_size, hidden_size, hidden_layers, activation, dropout
        )
        
        # 构建尺度网络 scale_net: X → log(γ_U) (softplus前)
        self.scale_net = self._build_mlp(
            input_size, hidden_size, hidden_layers, activation, dropout
        )
        
        # 初始化权重
        self._init_weights()
    
    def _build_mlp(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: Tuple[int, ...],
        activation: str,
        dropout: float
    ) -> nn.Module:
        """构建MLP网络"""
        activation_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }.get(activation.lower(), nn.ReLU)
        
        if not hidden_layers:
            # 没有隐藏层，直接线性映射
            return nn.Linear(input_size, output_size)
        
        # 有隐藏层的MLP
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_fn(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 最后一层
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化网络权重"""
        # 位置网络：如果输入输出维度相同且无隐藏层，可以接近恒等初始化
        if (self.input_size == self.hidden_size and 
            isinstance(self.loc_net, nn.Linear)):
            # 接近恒等初始化
            nn.init.eye_(self.loc_net.weight)
            nn.init.zeros_(self.loc_net.bias)
        else:
            # 标准Xavier初始化
            for module in self.loc_net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # 尺度网络：最后一层初始化为常数，确保初始γ_U ≈ gamma_init
        for module in self.scale_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
        # 特别处理尺度网络的最后一层
        if isinstance(self.scale_net, nn.Sequential):
            final_layer = self.scale_net[-1]
        else:
            final_layer = self.scale_net
        
        if isinstance(final_layer, nn.Linear):
            # 初始化为使 softplus(bias) ≈ gamma_init
            init_bias = torch.log(torch.exp(torch.tensor(self.gamma_init)) - 1)
            nn.init.constant_(final_layer.bias, init_bias.item())
            nn.init.zeros_(final_layer.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_size]
            
        Returns:
            mu_U: 位置参数 [batch_size, hidden_size]
            gamma_U: 尺度参数 [batch_size, hidden_size] (保证 > 0)
        """
        # 计算位置参数
        mu_U = self.loc_net(x)
        
        # 计算尺度参数，使用softplus确保正值
        gamma_U = F.softplus(self.scale_net(x))
        
        return mu_U, gamma_U


class ActionNetwork(nn.Module):
    """
    行动网络：从个体表征到决策得分
    
    基于个体表征 U 和五种推理模式，生成决策得分 S
    
    数学框架：
    S = W_A · U' + b_A
    其中 U' 根据推理模式调制：
    - deterministic: U' = μ_U
    - exogenous: U' ~ Cauchy(μ_U, |b_noise|)
    - endogenous: U' ~ Cauchy(μ_U, γ_U)
    - standard: U' ~ Cauchy(μ_U, γ_U + |b_noise|)
    - sampling: U' ~ Cauchy(μ_U + b_noise*ε, γ_U)
    
    Args:
        hidden_size: 输入的因果表征维度
        output_size: 输出决策得分维度（分类类别数或回归维度数）
        b_noise_init: 外生噪声初始化值
        b_noise_trainable: 外生噪声是否可训练
    """
    
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        b_noise_init: float = 0.1,
        b_noise_trainable: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 线性变换权重
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        
        # 外生噪声参数
        if b_noise_trainable:
            self.b_noise = nn.Parameter(torch.full((hidden_size,), b_noise_init))
        else:
            self.register_buffer('b_noise', torch.full((hidden_size,), b_noise_init))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(
        self,
        mu_U: torch.Tensor,
        gamma_U: torch.Tensor,
        mode: str = 'standard'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播 - 五模式的严格数学实现
        
        基于 MATHEMATICAL_FOUNDATIONS_CN.md 的权威定义
        
        Args:
            mu_U: 个体表征位置参数 [batch_size, hidden_size]
            gamma_U: 个体表征尺度参数 [batch_size, hidden_size]
            mode: 推理模式
            
        Returns:
            - deterministic模式: mu_S [batch_size, output_size]
            - 其他模式: (mu_S, gamma_S) 元组
        """
        # 五模式的个体表征调制
        if mode == 'deterministic':
            # Deterministic: U' = μ_U（确定性，只用位置）
            mu_U_final = mu_U
            gamma_U_final = torch.zeros_like(gamma_U)
            
        elif mode == 'exogenous':
            # Exogenous: U' ~ Cauchy(μ_U, |b_noise|)（外生噪声替代内生不确定性）
            mu_U_final = mu_U
            gamma_U_final = torch.abs(self.b_noise).unsqueeze(0).expand_as(gamma_U)
            
        elif mode == 'endogenous':
            # Endogenous: U' ~ Cauchy(μ_U, γ_U)（只用内生不确定性）
            mu_U_final = mu_U
            gamma_U_final = gamma_U
            
        elif mode == 'standard':
            # Standard: U' ~ Cauchy(μ_U, γ_U + |b_noise|)（内生+外生叠加在scale）
            mu_U_final = mu_U
            gamma_U_final = gamma_U + torch.abs(self.b_noise).unsqueeze(0).expand_as(gamma_U)
            
        elif mode == 'sampling':
            # Sampling: 外生噪声影响位置参数
            # 生成标准柯西噪声：ε ~ Cauchy(0, 1)
            uniform = torch.rand_like(mu_U)
            epsilon = torch.tan(torch.pi * (uniform - 0.5))  # ε ~ Cauchy(0, 1)
            
            mu_U_final = mu_U + self.b_noise.unsqueeze(0) * epsilon
            gamma_U_final = gamma_U
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: "
                           "deterministic, exogenous, endogenous, standard, sampling")
        
        # 应用线性因果律: S = W_A * U' + b_A
        # 利用柯西分布的线性稳定性:
        # 如果 U' ~ Cauchy(μ, γ)，则 W*U' + b ~ Cauchy(W*μ + b, |W|*γ)
        mu_S = torch.matmul(mu_U_final, self.weight.T) + self.bias
        
        # 确定性模式直接返回位置参数
        if mode == 'deterministic':
            return mu_S
        
        # 其他模式返回完整分布参数
        # 尺度参数的线性传播：γ_S = |W_A^T| * γ_U'
        gamma_S = torch.matmul(gamma_U_final, torch.abs(self.weight).T)
        
        return mu_S, gamma_S