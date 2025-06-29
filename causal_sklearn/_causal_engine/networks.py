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
from typing import Tuple, Union, Optional


def build_mlp(
    input_size: int,
    output_size: Optional[int] = None,
    hidden_layers: Optional[Tuple[int, ...]] = None,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Module:
    """
    构建一个多层感知机 (MLP).

    参数:
        input_size (int): 输入层的大小.
        output_size (Optional[int]): 输出层的大小. 如果为 None, 默认为 input_size.
        hidden_layers (Optional[Tuple[int, ...]]): 一个元组，指定每个隐藏层的神经元数量。
                                                       如果为 None, 则构建一个从输入到输出的单层线性网络。
        activation (str): 要使用的激活函数.
        dropout (float): Dropout 比率.

    返回:
        nn.Module: 一个 PyTorch MLP 模块.
    """
    if output_size is None:
        output_size = input_size

    if hidden_layers is None:
        hidden_layers = ()

    activation_fn = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }.get(activation.lower(), nn.ReLU)
    
    if not hidden_layers:
        return nn.Linear(input_size, output_size)
    
    layers = []
    current_size = input_size
    
    for layer_size in hidden_layers:
        layers.extend(
            [
                nn.Linear(current_size, layer_size),
                activation_fn(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_size = layer_size
    
    layers.append(nn.Linear(current_size, output_size))
    
    return nn.Sequential(*layers)


class Action(nn.Module):
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
        causal_size: 输入的因果表征维度
        output_size: 输出决策得分维度（分类类别数或回归维度数）
        b_noise_init: 外生噪声初始化值
        b_noise_trainable: 外生噪声是否可训练
    """
    
    def __init__(
        self,
        causal_size: int,
        output_size: int,
        b_noise_init: float = 0.1,
        b_noise_trainable: bool = True
    ):
        super().__init__()
        
        self.causal_size = causal_size
        self.output_size = output_size
        
        # 线性变换权重
        self.weight = nn.Parameter(torch.empty(output_size, causal_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        
        # 外生噪声参数
        if b_noise_trainable:
            self.b_noise = nn.Parameter(torch.full((causal_size,), b_noise_init))
        else:
            self.register_buffer('b_noise', torch.full((causal_size,), b_noise_init))
        
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
            mu_U: 个体表征位置参数 [batch_size, causal_size]
            gamma_U: 个体表征尺度参数 [batch_size, causal_size]
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
            # TODO: epsilon 的似然应该需要用于计算损失， 我们需要加权的损失
            
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


class Perception(nn.Module):
    """
    感知网络：从原始输入 X 中提取高级特征 Z。
    这是因果引擎的第一阶段，负责"看懂世界"。
    """
    def __init__(
        self,
        input_size: int,
        repre_size: int,
        hidden_layers: Optional[Tuple[int, ...]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.repre_size = repre_size

        self.network = build_mlp(
            input_size=input_size,
            output_size=repre_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：X -> Z
        """
        return self.network(x)


class Abduction(nn.Module):
    """
    归因网络：从特征 Z 推断个体的内在因果表征 U。
    这是因果引擎的第二阶段，负责"思考归因"。
    
    数学框架：
    - 输入: Z ∈ R^{repre_size}
    - 输出: (μ_U, γ_U) ∈ R^{causal_size} × R^{causal_size}_+
    - 分布: U ~ Cauchy(μ_U, γ_U)
    """
    def __init__(
        self,
        repre_size: int,
        causal_size: int,
        hidden_layers: Optional[Tuple[int, ...]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        gamma_init: float = 10.0,
    ):
        super().__init__()
        self.repre_size = repre_size
        self.causal_size = causal_size
        self.gamma_init = gamma_init

        # 构建位置网络 loc_net: Z → μ_U
        self.loc_net = build_mlp(
            repre_size, causal_size, hidden_layers, activation, dropout
        )

        # 构建尺度网络 scale_net: Z → log(γ_U) (softplus前)
        self.scale_net = build_mlp(
            repre_size, causal_size, hidden_layers, activation, dropout
        )

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        # 标准Xavier初始化
        for module in self.loc_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 尺度网络：最后一层初始化为常数，确保初始γ_U ≈ gamma_init
        for module in self.scale_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # 特别处理尺度网络的最后一层
        final_layer = self.scale_net[-1] if isinstance(self.scale_net, nn.Sequential) else self.scale_net
        
        if isinstance(final_layer, nn.Linear):
            # 初始化为使 softplus(bias) ≈ gamma_init
            init_bias = torch.log(torch.exp(torch.tensor(self.gamma_init)) - 1)
            nn.init.constant_(final_layer.bias, init_bias.item())
            nn.init.zeros_(final_layer.weight)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: Z -> (μ_U, γ_U)
        
        Args:
            z: 输入特征 [batch_size, repre_size]
            
        Returns:
            mu_U: 位置参数 [batch_size, causal_size]
            gamma_U: 尺度参数 [batch_size, causal_size] (保证 > 0)
        """
        mu_U = self.loc_net(z)
        gamma_U = F.softplus(self.scale_net(z))
        return mu_U, gamma_U