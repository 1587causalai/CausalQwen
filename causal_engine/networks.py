"""
CausalEngine Core Networks

这个模块包含 CausalEngine 的两个核心网络：
1. AbductionNetwork: 从证据推断个体（归因）
2. ActionNetwork: 从个体到决策（行动）

这两个网络构成了因果推理的核心循环。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AbductionNetwork(nn.Module):
    """
    归因网络：从证据推断个体
    
    这是智能的第一步 - 从观察到的证据中推断出"我是谁"。
    网络将上下文特征映射到个体的因果表征分布 U ~ Cauchy(μ, γ)。
    
    数学架构（独立网络版本）：
    1. loc_net: H → C (位置推断)
    2. scale_net: H → C (尺度推断)
    3. 分布生成：(μ, γ) ∈ R^C × R^C_+
    
    构建规则表：
    | 输入参数组合      | loc_net 构建      | scale_net 构建    | 初始化策略         |
    |------------------|------------------|------------------|-------------------|
    | H=C, layers=1    | nn.Linear(H,C)   | nn.Linear(H,C)   | loc:恒等, scale:常数 |
    | H≠C, layers=1    | nn.Linear(H,C)   | nn.Linear(H,C)   | loc:Xavier, scale:常数 |
    | H=C, layers>1    | MLP(H→C*2→C)     | MLP(H→C*2→C)     | loc:Xavier, scale:常数 |
    | H≠C, layers>1    | MLP(H→C*2→C)     | MLP(H→C*2→C)     | loc:Xavier, scale:常数 |
    
    优雅设计：
    - loc_net 和 scale_net 完全独立
    - 当 H = C 且 mlp_layers = 1 时，loc_net 恒等初始化
    - scale_net 总是用 MLP，最后一层初始化为常数
    - 数学上更加灵活和优化
    
    Args:
        input_size: 输入特征维度 (H)
        causal_size: 因果表征维度 (C)
        mlp_layers: MLP 层数（默认为 1）
        mlp_hidden_ratio: MLP 隐藏层大小比例（hidden = C * ratio）
        mlp_activation: MLP 激活函数
        mlp_dropout: MLP dropout 率
        gamma_init: 初始尺度参数
    """
    
    def __init__(
        self, 
        input_size: int,
        causal_size: int,
        mlp_layers: int = 1,
        mlp_hidden_ratio: float = 2.0,
        mlp_activation: str = 'relu',
        mlp_dropout: float = 0.0,
        gamma_init: float = 10.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.causal_size = causal_size
        self.mlp_layers = mlp_layers
        self.mlp_hidden_ratio = mlp_hidden_ratio
        
        # 独立的网络设计：每个网络都有自己的路径
        # loc_net：H → C，独立路径
        if input_size == causal_size and mlp_layers == 1:
            # 特殊情况：直接连接，可以恒等初始化
            self.loc_net = nn.Linear(input_size, causal_size, bias=True)
            self._loc_is_identity_candidate = True
        else:
            # 一般情况：可能需要 MLP + Linear
            if mlp_layers == 1:
                # 单层线性变换
                self.loc_net = nn.Linear(input_size, causal_size, bias=True)
            else:
                # 多层 MLP
                self.loc_net = self._build_mlp(
                    input_size, causal_size, mlp_layers,
                    mlp_hidden_ratio, mlp_activation, mlp_dropout
                )
            self._loc_is_identity_candidate = False
        
        # scale_net：H → C，独立路径（总是用 MLP）
        self.scale_net = self._build_mlp(
            input_size, 
            causal_size,
            mlp_layers,
            mlp_hidden_ratio,
            mlp_activation,
            mlp_dropout
        )
        
        # 初始化
        self._init_weights(gamma_init)
    
    def _build_mlp(
        self,
        input_size: int,
        output_size: int,  # 始终等于 causal_size
        num_layers: int,
        hidden_ratio: float,
        activation: str,
        dropout: float
    ) -> nn.Module:
        """
        构建 MLP 模块（优雅版本）
        
        数学设计：
        - 输入: H 维
        - 输出: C 维 (causal_size)
        - 隐藏层: C * hidden_ratio 维
        
        当 num_layers=1 时：
        - 如果 H=C：单层线性（可以恒等初始化）
        - 如果 H≠C：单层线性（维度转换）
        
        当 num_layers>1 时：深度 MLP
        """
        if num_layers == 1:
            # 单层情况：最简洁的设计
            return nn.Linear(input_size, output_size)
        
        # 多层情况：深度 MLP
        hidden_size = int(output_size * hidden_ratio)
        
        # 选择激活函数
        activation_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }.get(activation.lower(), nn.ReLU)
        
        layers = []
        
        # 第一层
        layers.extend([
            nn.Linear(input_size, hidden_size),
            activation_fn(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # 中间层
        for _ in range(num_layers - 2):  # -2 因为第一层和最后一层已经算了
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                activation_fn(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, gamma_init: float):
        """独立网络的优雅初始化策略"""
        with torch.no_grad():
            # loc_net 初始化
            if self._loc_is_identity_candidate:
                # 特殊情况：H=C 且 layers=1，恒等初始化
                self.loc_net.weight.copy_(torch.eye(self.causal_size))
                self.loc_net.bias.zero_()
            else:
                # 一般情况：Xavier 初始化
                if isinstance(self.loc_net, nn.Linear):
                    nn.init.xavier_uniform_(self.loc_net.weight)
                    nn.init.zeros_(self.loc_net.bias)
                else:
                    # MLP 情况
                    for module in self.loc_net.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.zeros_(module.bias)
            
            # scale_net 初始化（总是 MLP）
            # 修复：正确找到最后一层并初始化
            linear_modules = [m for m in self.scale_net.modules() if isinstance(m, nn.Linear)]
            
            for i, module in enumerate(linear_modules):
                if i == len(linear_modules) - 1:
                    # 最后一层：简化初始化逻辑
                    # 权重：接近0，忽略输入依赖
                    nn.init.uniform_(module.weight, -0.01, 0.01)
                    if module.bias is not None:
                        # 偏置：直接设置为gamma_init
                        # 这样scale_net(input) ≈ 0 * input + gamma_init = gamma_init
                        # 然后gamma_U = softplus(gamma_init) ≈ gamma_init
                        nn.init.constant_(module.bias, gamma_init)
                else:
                    # 中间层：标准Xavier初始化
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor, mode: str = 'standard') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：从证据到个体（统一模式版本）
        
        数学框架：
        X ∈ R^H                    # 输入上下文特征
        μ_U = f_loc(X, mode)       # 位置网络：X → μ_U ∈ R^C (模式自适应)  
        γ_U = f_scale(X, mode)     # 尺度网络：X → γ_U ∈ R^C_+ (模式自适应)
        U ~ Cauchy(μ_U, γ_U)       # 输出个体分布
        
        统一模式处理：
        - deterministic: μ_U = X (恒等映射), γ_U = 0 (确定性)
        - exogenous: μ_U = f_loc(X), γ_U = 0 (无内生不确定性)  
        - 其他模式: μ_U = f_loc(X), γ_U = softplus(f_scale(X))
        
        Args:
            hidden_states: [batch_size, seq_len, input_size] 上下文特征
            mode: 推理模式 ('deterministic', 'exogenous', 'endogenous', 'standard', 'sampling')
            
        Returns:
            loc_U: [batch_size, seq_len, causal_size] 个体位置参数 μ_U
            scale_U: [batch_size, seq_len, causal_size] 个体尺度参数 γ_U
        """
        
        if mode == 'deterministic':
            # Deterministic模式：恒等映射，无不确定性
            # μ_U = X (需要确保维度匹配)
            if self.input_size == self.causal_size:
                # 维度匹配：直接恒等映射
                loc_U = hidden_states
            else:
                # 维度不匹配：使用线性变换但保持确定性意图
                loc_U = self.loc_net(hidden_states)
            # γ_U = 0：完全确定性
            scale_U = torch.zeros_like(loc_U)
            
        elif mode == 'exogenous':
            # Exogenous模式：位置推断，但无内生不确定性
            # μ_U = f_loc(X)：正常位置推断
            loc_U = self.loc_net(hidden_states)
            # γ_U = 0：无内生噪声，只依赖外生噪声
            scale_U = torch.zeros_like(loc_U)
            
        else:
            # 标准因果模式：endogenous, standard, sampling
            # μ_U = f_loc(X)：正常位置推断
            loc_U = self.loc_net(hidden_states)
            # γ_U = softplus(f_scale(X))：正常尺度推断
            scale_U = F.softplus(self.scale_net(hidden_states))
        
        return loc_U, scale_U
    
    @property
    def is_identity_mapping(self) -> bool:
        """loc_net 是否为恒等映射"""
        return self._loc_is_identity_candidate
    
    def get_architecture_description(self) -> str:
        """获取架构描述"""
        if self.mlp_layers == 1:
            if self.is_identity_mapping:
                loc_desc = f"Identity: {self.input_size} → {self.causal_size}"
            else:
                loc_desc = f"Linear: {self.input_size} → {self.causal_size}"
            scale_desc = f"Linear: {self.input_size} → {self.causal_size}"
        else:
            hidden_size = int(self.causal_size * self.mlp_hidden_ratio)
            loc_desc = f"MLP({self.mlp_layers}): {self.input_size} → {hidden_size} → {self.causal_size}"
            scale_desc = f"MLP({self.mlp_layers}): {self.input_size} → {hidden_size} → {self.causal_size}"
        
        return f"loc_net=[{loc_desc}], scale_net=[{scale_desc}]"


class ActionNetwork(nn.Module):
    """
    行动网络：从个体到决策
    
    这是智能的第二步 - 基于"我是谁"来决定"我该做什么"。
    网络应用普适因果律 f(U, ε) 生成决策分布 S ~ Cauchy(μ, γ)。
    
    Args:
        causal_size: 因果表征维度
        output_size: 输出维度（如词汇表大小）
        b_noise_init: 外生噪声初始值
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
        self.b_noise_trainable = b_noise_trainable
        
        # 线性因果律：从个体表征到决策
        self.linear_law = nn.Linear(causal_size, output_size, bias=True)
        
        # 外生噪声参数
        if b_noise_trainable:
            self.b_noise = nn.Parameter(torch.zeros(causal_size))
        else:
            # 注册为buffer，不参与梯度更新
            self.register_buffer('b_noise', torch.zeros(causal_size))
        
        # 初始化
        self._init_weights(b_noise_init)
    
    def _init_weights(self, b_noise_init: float):
        """初始化权重"""
        nn.init.xavier_uniform_(self.linear_law.weight)
        nn.init.zeros_(self.linear_law.bias)
        
        # 初始化b_noise向量 (所有分量设为相同值)
        with torch.no_grad():
            self.b_noise.data.fill_(b_noise_init)
    
    def forward(
        self,
        loc_U: torch.Tensor,
        scale_U: torch.Tensor,
        mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：从个体到决策（模式感知版本）
        
        数学框架：
        U ~ Cauchy(loc_U, scale_U)  # 输入个体分布
        U' = U + b_noise            # 添加外生噪声（模式依赖）
        S = W_A * U' + b_A          # 线性因果律
        
        Args:
            loc_U: 个体位置参数
            scale_U: 个体尺度参数
            mode: 计算模式
                - 'deterministic': 不添加外生噪声
                - 'exogenous': 不添加外生噪声  
                - 'standard': 添加外生噪声
                - 'endogenous': 添加外生噪声
                - 'sampling': 添加外生噪声
            
        Returns:
            loc_S: [batch_size, seq_len, output_size] 决策位置参数
            scale_S: [batch_size, seq_len, output_size] 决策尺度参数
        """
        # 五模式的严格数学实现（基于MATHEMATICAL_FOUNDATIONS_CN.md）
        if mode == 'deterministic':
            # Deterministic: U' = μ_U（确定性，只用位置）
            loc_U_final = loc_U
            scale_U_final = torch.zeros_like(scale_U)
            
        elif mode == 'exogenous':
            # Exogenous: U' ~ Cauchy(μ_U, |b_noise|)（外生噪声替代内生不确定性）
            loc_U_final = loc_U
            scale_U_final = torch.abs(self.b_noise).expand_as(scale_U)
            
        elif mode == 'endogenous':
            # Endogenous: U' ~ Cauchy(μ_U, γ_U)（只用内生不确定性）
            loc_U_final = loc_U
            scale_U_final = scale_U
            
        elif mode == 'standard':
            # Standard: U' ~ Cauchy(μ_U, γ_U + |b_noise|)（内生+外生叠加在scale）
            loc_U_final = loc_U
            scale_U_final = scale_U + torch.abs(self.b_noise)
            
        elif mode == 'sampling':
            # Sampling: U' ~ Cauchy(μ_U + b_noise·E, γ_U)（外生噪声加到位置，然后采样）
            # 生成外生随机噪声E
            exogenous_noise = torch.randn_like(loc_U)  # E ~ N(0,1)
            shifted_loc = loc_U + self.b_noise * exogenous_noise  # μ_U + b_noise·E
            
            # 从修正后的分布采样
            cauchy_samples = torch.distributions.Cauchy(shifted_loc, scale_U).sample()
            loc_U_final = cauchy_samples
            scale_U_final = scale_U
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: deterministic, exogenous, endogenous, standard, sampling")
        
        # 应用线性因果律: S = W_A * U' + b_A
        # 利用柯西分布的线性稳定性:
        # 如果 U' ~ Cauchy(μ, γ)，则 W*U' + b ~ Cauchy(W*μ + b, |W|*γ)
        loc_S = self.linear_law(loc_U_final)  # loc_S = W_A^T * loc_U' + b_A
        
        # 尺度参数的线性传播：scale_S = |W_A^T| * scale_U'
        # 数学验证✅：Cauchy分布线性变换 U ~ Cauchy(μ, γ) => W^T*U ~ Cauchy(W^T*μ, |W^T|*γ)
        # γ_S = γ_U @ |W^T|，其中W_A.weight形状是[output_size, causal_size]
        scale_S = scale_U_final @ torch.abs(self.linear_law.weight).T
        
        return loc_S, scale_S 