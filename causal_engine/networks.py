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
        gamma_init: float = 1.0
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
            for module in self.scale_net.modules():
                if isinstance(module, nn.Linear):
                    # 最后一层特殊初始化为常数输出
                    if module is list(self.scale_net.modules())[-1]:
                        nn.init.zeros_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, gamma_init)
                    else:
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：从证据到个体（独立网络版本）
        
        数学框架：
        X ∈ R^H                    # 输入上下文特征
        μ_U = f_loc(X)             # 位置网络：X → μ_U ∈ R^C  
        γ_U = softplus(g_scale(X)) # 尺度网络：X → γ_U ∈ R^C_+
        U ~ Cauchy(μ_U, γ_U)       # 输出个体分布
        
        网络设计：
        - loc_net: 独立的位置参数推断网络
        - scale_net: 独立的尺度参数推断网络  
        - softplus激活确保尺度参数为正
        
        Args:
            hidden_states: [batch_size, seq_len, input_size] 上下文特征
            
        Returns:
            loc_U: [batch_size, seq_len, causal_size] 个体位置参数 μ_U
            scale_U: [batch_size, seq_len, causal_size] 个体尺度参数 γ_U
        """
        # 独立的网络路径
        # 位置参数推断：μ_U = f_loc(X)
        loc_U = self.loc_net(hidden_states)
        
        # 尺度参数推断：γ_U = softplus(g_scale(X))
        # softplus确保输出为正：softplus(x) = log(1 + exp(x)) > 0
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
        b_noise_init: float = 0.1
    ):
        super().__init__()
        
        self.causal_size = causal_size
        self.output_size = output_size
        
        # 线性因果律：从个体表征到决策
        self.linear_law = nn.Linear(causal_size, output_size, bias=True)
        
        # 外生噪声参数
        self.b_noise = nn.Parameter(torch.zeros(causal_size))
        
        # 初始化
        self._init_weights(b_noise_init)
    
    def _init_weights(self, b_noise_init: float):
        """初始化权重"""
        nn.init.xavier_uniform_(self.linear_law.weight)
        nn.init.zeros_(self.linear_law.bias)
        nn.init.constant_(self.b_noise, b_noise_init)
    
    def forward(
        self,
        loc_U: torch.Tensor,
        scale_U: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：从个体到决策
        
        实现温度统一的噪声控制框架：
        - temperature=0: 纯因果模式（无噪声）
        - temperature>0 & do_sample=False: 噪声增加尺度（不确定性）
        - temperature>0 & do_sample=True: 噪声扰动位置（身份）
        
        数学框架：
        U ~ Cauchy(loc_U, scale_U)  # 输入个体分布
        ε ~ Cauchy(0, 1)            # 标准外生噪声
        U' = f_noise(U, ε, T)       # 噪声注入函数
        S = W_A * U' + b_A          # 线性因果律
        
        Args:
            loc_U: 个体位置参数
            scale_U: 个体尺度参数
            do_sample: 是否采样模式
            temperature: 温度控制
            
        Returns:
            loc_S: [batch_size, seq_len, output_size] 决策位置参数
            scale_S: [batch_size, seq_len, output_size] 决策尺度参数
        """
        if temperature == 0:
            # 纯因果模式：无噪声
            # 数学: U' = U (恒等变换)
            loc_U_final = loc_U
            scale_U_final = scale_U
            
        elif do_sample:
            # 采样模式：噪声扰动位置
            # 数学: U' ~ Cauchy(loc_U + T·|b_noise|·ε, scale_U)
            # 其中 ε ~ Cauchy(0,1) 通过逆变换采样: ε = tan(π(uniform - 0.5))
            uniform_sample = torch.rand_like(loc_U)
            epsilon = torch.tan(torch.pi * (uniform_sample - 0.5))  # 标准柯西分布采样
            loc_U_final = loc_U + temperature * torch.abs(self.b_noise) * epsilon
            scale_U_final = scale_U
            
        else:
            # 标准模式：噪声增加尺度
            # 数学: U' ~ Cauchy(loc_U, scale_U + T·|b_noise|)
            # 利用柯西分布的尺度可加性
            loc_U_final = loc_U
            scale_U_final = scale_U + temperature * torch.abs(self.b_noise)
        
        # 应用线性因果律: S = W_A * U' + b_A
        # 利用柯西分布的线性稳定性:
        # 如果 U' ~ Cauchy(μ, γ)，则 W*U' + b ~ Cauchy(W*μ + b, |W|*γ)
        loc_S = self.linear_law(loc_U_final)  # loc_S = W_A^T * loc_U' + b_A
        
        # 尺度参数的线性传播：scale_S = |W_A^T| * scale_U'
        # 注意：这里使用矩阵乘法而不是逐元素乘法，保持线性变换的完整性
        scale_S = scale_U_final @ torch.abs(self.linear_law.weight).T
        
        return loc_S, scale_S 