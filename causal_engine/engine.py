"""
CausalEngine Core Implementation

这是因果推理引擎的核心实现，包含了从"归因推断"到"行动决策"的完整因果链条。
本模块完全独立，不依赖任何特定项目或模型架构。

核心数学框架：Y = f(U, ε)
- U: 个体选择变量（从上下文推断）
- ε: 外生噪声
- f: 普适因果机制（线性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CauchyMath:
    """Cauchy分布数学工具类，实现严格的线性稳定性"""
    
    @staticmethod
    def cauchy_linear_stable_loc(loc_input, weight, bias=None):
        """Cauchy分布位置参数的线性变换"""
        # 位置参数变换：直接矩阵乘法
        result = loc_input @ weight.T
        if bias is not None:
            result = result + bias
        return result
    
    @staticmethod  
    def cauchy_linear_stable_scale(scale_input, weight):
        """Cauchy分布尺度参数的线性变换"""
        # 尺度参数变换：直接矩阵乘法
        return scale_input @ torch.abs(weight).T


class CausalEngine(nn.Module):
    """
    因果推理引擎 - 语言模型的因果决策核心
    
    这个引擎实现了革命性的因果生成范式：
    1. 从上下文特征推断个体表征分布 U ~ Cauchy(μ, γ)
    2. 应用普适线性因果律 f(U, ε) 得到决策分布
    3. 支持四种推理模式的统一框架
    
    参数说明：
        hidden_size: 输入特征维度（来自上游模型）
        vocab_size: 词汇表大小（输出维度）
        causal_size: 因果表征维度（默认等于hidden_size）
        b_noise_init: 外生噪声初始值
        gamma_init: 初始尺度参数
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        causal_size: Optional[int] = None,
        b_noise_init: float = 0.1,
        gamma_init: float = 1.0
    ):
        super().__init__()
        
        # 核心维度
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.causal_size = causal_size or hidden_size
        
        # 初始化参数
        self.b_noise_init = b_noise_init
        self.gamma_init = gamma_init
        
        # 构建核心组件
        self._build_abduction_network()
        self._build_action_network()
    
    def _build_abduction_network(self):
        """构建归因推断网络：从证据推断个体"""
        # 位置网络：推断个体群体的中心
        self.abduction_loc = nn.Linear(self.hidden_size, self.causal_size, bias=True)
        # 尺度网络：推断个体群体的多样性
        self.abduction_scale = nn.Linear(self.hidden_size, self.causal_size, bias=True)
        
        # 初始化策略：恒等映射
        with torch.no_grad():
            if self.hidden_size == self.causal_size:
                # 恒等初始化
                self.abduction_loc.weight.copy_(torch.eye(self.causal_size))
                self.abduction_loc.bias.zero_()
            else:
                # Xavier初始化
                nn.init.xavier_uniform_(self.abduction_loc.weight)
                nn.init.zeros_(self.abduction_loc.bias)
            
            # 尺度网络初始化
            nn.init.zeros_(self.abduction_scale.weight)
            nn.init.constant_(self.abduction_scale.bias, self.gamma_init)
    
    def _build_action_network(self):
        """构建行动网络：从个体到决策"""
        # 线性因果律：从个体表征到词汇决策
        self.action_head = nn.Linear(self.causal_size, self.vocab_size, bias=True)
        # 外生噪声参数
        self.b_noise = nn.Parameter(torch.zeros(self.causal_size))
        
        # 初始化
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.constant_(self.b_noise, self.b_noise_init)
    
    def abduction(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        归因推断：从上下文特征推断个体分布
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 上下文特征
            
        Returns:
            loc_U: [batch_size, seq_len, causal_size] 个体位置参数
            scale_U: [batch_size, seq_len, causal_size] 个体尺度参数
        """
        loc_U = self.abduction_loc(hidden_states)
        scale_U = F.softplus(self.abduction_scale(hidden_states))
        return loc_U, scale_U
    
    def action(
        self, 
        loc_U: torch.Tensor, 
        scale_U: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        行动决策：应用因果律得到决策分布
        
        温度统一控制噪声强度：
        - temperature=0: 纯因果模式，无外生噪声
        - temperature>0 & do_sample=False: 噪声增加尺度（不确定性）
        - temperature>0 & do_sample=True: 噪声扰动位置（探索性）
        
        Args:
            loc_U: 个体位置参数
            scale_U: 个体尺度参数  
            do_sample: 是否采样模式
            temperature: 温度参数
            
        Returns:
            loc_S: [batch_size, seq_len, vocab_size] 决策位置参数
            scale_S: [batch_size, seq_len, vocab_size] 决策尺度参数
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
        loc_S = self.action_head(loc_U_final)
        scale_S = scale_U_final @ torch.abs(self.action_head.weight).T
        
        return loc_S, scale_S
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        完整的因果推理流程
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 上下文特征
            do_sample: 是否使用采样模式
            temperature: 温度控制
            return_dict: 是否返回字典格式
            
        Returns:
            包含 loc_S, scale_S, loc_U, scale_U 的字典
        """
        # Step 1: 归因推断
        loc_U, scale_U = self.abduction(hidden_states)
        
        # Step 2: 行动决策
        loc_S, scale_S = self.action(loc_U, scale_U, do_sample, temperature)
        
        if return_dict:
            return {
                "loc_S": loc_S,
                "scale_S": scale_S,
                "loc_U": loc_U,
                "scale_U": scale_U
            }
        else:
            return loc_S, scale_S, loc_U, scale_U
    
    def compute_ovr_probs(
        self, 
        loc_S: torch.Tensor, 
        scale_S: torch.Tensor,
        threshold: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算OvR (One-vs-Rest) 概率
        
        使用柯西分布的CDF计算每个词汇的选择概率：
        P(vocab_k) = P(S_k > threshold) = 0.5 + (1/π)arctan((loc_k - threshold)/scale_k)
        
        Args:
            loc_S: 决策位置参数 [batch_size, seq_len, vocab_size]
            scale_S: 决策尺度参数 [batch_size, seq_len, vocab_size]
            threshold: OvR阈值 (可以是标量或 [vocab_size] 的张量)
            
        Returns:
            probs: [batch_size, seq_len, vocab_size] 概率分布
        """
        if threshold is None:
            threshold = 0.0
        elif isinstance(threshold, torch.Tensor) and threshold.dim() == 1:
            # 如果是词汇级别的阈值，扩展维度
            threshold = threshold.unsqueeze(0).unsqueeze(0)
            
        normalized_diff = (loc_S - threshold) / scale_S
        probs = 0.5 + (1 / torch.pi) * torch.atan(normalized_diff)
        return probs
    
    def get_config(self) -> Dict:
        """获取引擎配置信息"""
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "causal_size": self.causal_size,
            "b_noise_init": self.b_noise_init,
            "gamma_init": self.gamma_init,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "algorithm": "CausalEngine v1.0"
        }
    
    def __repr__(self) -> str:
        config = self.get_config()
        return (
            f"CausalEngine(\n"
            f"  hidden_size={config['hidden_size']},\n"
            f"  vocab_size={config['vocab_size']},\n"
            f"  causal_size={config['causal_size']},\n"
            f"  parameters={config['num_parameters']:,}\n"
            f")"
        ) 