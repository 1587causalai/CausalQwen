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
    
    Args:
        input_size: 输入特征维度
        causal_size: 因果表征维度
        gamma_init: 初始尺度参数
    """
    
    def __init__(
        self, 
        input_size: int,
        causal_size: int,
        gamma_init: float = 1.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.causal_size = causal_size
        
        # 位置网络：推断个体群体的中心
        self.loc_net = nn.Linear(input_size, causal_size, bias=True)
        
        # 尺度网络：推断个体群体的多样性
        self.scale_net = nn.Linear(input_size, causal_size, bias=True)
        
        # 初始化
        self._init_weights(gamma_init)
    
    def _init_weights(self, gamma_init: float):
        """智能的初始化策略"""
        with torch.no_grad():
            if self.input_size == self.causal_size:
                # 恒等初始化：保持输入特征
                self.loc_net.weight.copy_(torch.eye(self.causal_size))
                self.loc_net.bias.zero_()
            else:
                # Xavier 初始化
                nn.init.xavier_uniform_(self.loc_net.weight)
                nn.init.zeros_(self.loc_net.bias)
            
            # 尺度网络：初始为常数输出
            nn.init.zeros_(self.scale_net.weight)
            nn.init.constant_(self.scale_net.bias, gamma_init)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：从证据到个体
        
        Args:
            hidden_states: [batch_size, seq_len, input_size] 上下文特征
            
        Returns:
            loc_U: [batch_size, seq_len, causal_size] 个体位置参数
            scale_U: [batch_size, seq_len, causal_size] 个体尺度参数
        """
        loc_U = self.loc_net(hidden_states)
        scale_U = F.softplus(self.scale_net(hidden_states))
        
        return loc_U, scale_U


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
            loc_U_final = loc_U
            scale_U_final = scale_U
            
        elif do_sample:
            # 采样模式：噪声扰动位置
            uniform_sample = torch.rand_like(loc_U)
            epsilon = torch.tan(torch.pi * (uniform_sample - 0.5))
            loc_U_final = loc_U + temperature * torch.abs(self.b_noise) * epsilon
            scale_U_final = scale_U
            
        else:
            # 标准模式：噪声增加尺度
            loc_U_final = loc_U
            scale_U_final = scale_U + temperature * torch.abs(self.b_noise)
        
        # 应用线性因果律
        loc_S = self.linear_law(loc_U_final)
        
        # 尺度参数的线性传播
        scale_S = scale_U_final @ torch.abs(self.linear_law.weight).T
        
        return loc_S, scale_S 