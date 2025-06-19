"""
CausalEngine Core Implementation

CausalEngine 是一个模块化的因果推理引擎，实现了从"归因推断"到"行动决策"
再到"激活输出"的完整因果链条。

核心数学框架：Y = f(U, ε)
- U: 个体选择变量（从上下文推断）
- ε: 外生噪声
- f: 普适因果机制（线性）

模块化架构：
1. AbductionNetwork: 从证据推断个体
2. ActionNetwork: 从个体到决策
3. ActivationHead: 将决策转换为最终输出
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, List

from .networks import AbductionNetwork, ActionNetwork
from .heads import ActivationHead, ActivationMode


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
    因果推理引擎 - 模块化的智能决策系统
    
    CausalEngine 实现了完整的因果推理链条：
    1. 归因（Abduction）：从证据推断个体 U ~ Cauchy(μ, γ)
    2. 行动（Action）：应用因果律 f(U, ε) 得到决策分布
    3. 激活（Activation）：将决策转换为最终输出
    
    这种模块化设计实现了：
    - 清晰的职责分离
    - 灵活的组件替换
    - 统一的分类/回归支持
    
    Args:
        hidden_size: 输入特征维度
        vocab_size: 输出维度（词汇表大小）
        causal_size: 因果表征维度（默认等于hidden_size）
        activation_modes: 激活模式配置（默认全部分类）
        abduction_mlp_layers: AbductionNetwork 的 MLP 层数（0=不使用MLP，除非维度不匹配）
        abduction_mlp_hidden_ratio: MLP 隐藏层大小比例（相对于 causal_size）
        abduction_mlp_activation: MLP 激活函数 ('relu', 'gelu', 'silu', 'tanh', 'sigmoid')
        abduction_mlp_dropout: MLP dropout 率
        b_noise_init: 外生噪声初始值
        gamma_init: 初始尺度参数
        classification_threshold_init: 分类阈值初始值
        regression_scale_init: 回归缩放初始值
        regression_bias_init: 回归偏置初始值
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        causal_size: Optional[int] = None,
        activation_modes: Optional[Union[str, List[str]]] = None,
        # AbductionNetwork MLP 参数
        abduction_mlp_layers: int = 1,
        abduction_mlp_hidden_ratio: float = 2.0,
        abduction_mlp_activation: str = 'relu',
        abduction_mlp_dropout: float = 0.0,
        # 噪声和初始化参数
        b_noise_init: float = 0.1,
        gamma_init: float = 1.0,
        classification_threshold_init: float = 0.0,
        regression_scale_init: float = 1.0,
        regression_bias_init: float = 0.0
    ):
        super().__init__()
        
        # 核心维度
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.causal_size = causal_size or hidden_size
        
        # 构建三大模块
        self.abduction = AbductionNetwork(
            input_size=hidden_size,
            causal_size=self.causal_size,
            mlp_layers=abduction_mlp_layers,
            mlp_hidden_ratio=abduction_mlp_hidden_ratio,
            mlp_activation=abduction_mlp_activation,
            mlp_dropout=abduction_mlp_dropout,
            gamma_init=gamma_init
        )
        
        self.action = ActionNetwork(
            causal_size=self.causal_size,
            output_size=vocab_size,
            b_noise_init=b_noise_init
        )
        
        self.activation = ActivationHead(
            output_size=vocab_size,
            activation_modes=activation_modes,
            classification_threshold_init=classification_threshold_init,
            regression_scale_init=regression_scale_init,
            regression_bias_init=regression_bias_init
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        return_dict: bool = True,
        apply_activation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        完整的因果推理流程
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 上下文特征
            do_sample: 是否使用采样模式
            temperature: 温度控制
            return_dict: 是否返回字典格式
            apply_activation: 是否应用激活头
            
        Returns:
            包含以下内容的字典：
            - output: 最终输出（如果apply_activation=True）
            - loc_S, scale_S: 决策分布参数
            - loc_U, scale_U: 个体分布参数
            - activation_output: 激活头的详细输出
        """
        # Step 1: 归因推断
        loc_U, scale_U = self.abduction(hidden_states)
        
        # Step 2: 行动决策
        loc_S, scale_S = self.action(loc_U, scale_U, do_sample, temperature)
        
        # Step 3: 激活输出（可选）
        if apply_activation:
            activation_output = self.activation(loc_S, scale_S, return_dict=True)
            output = activation_output['output']
        else:
            activation_output = None
            output = None
        
        if return_dict:
            result = {
                "loc_S": loc_S,
                "scale_S": scale_S,
                "loc_U": loc_U,
                "scale_U": scale_U
            }
            
            if apply_activation:
                result["output"] = output
                result["activation_output"] = activation_output
            
            return result
        else:
            if apply_activation:
                return output, loc_S, scale_S, loc_U, scale_U
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
        
        注意：这是一个便捷方法，用于直接计算分类概率。
        更推荐使用 ActivationHead 来处理混合的分类/回归任务。
        
        Args:
            loc_S: 决策位置参数
            scale_S: 决策尺度参数
            threshold: OvR阈值
            
        Returns:
            probs: 概率分布
        """
        if threshold is None:
            threshold = 0.0
        elif isinstance(threshold, torch.Tensor) and threshold.dim() == 1:
            threshold = threshold.unsqueeze(0).unsqueeze(0)
            
        normalized_diff = (loc_S - threshold) / scale_S
        probs = 0.5 + (1 / torch.pi) * torch.atan(normalized_diff)
        return probs
    
    def get_config(self) -> Dict:
        """获取引擎配置信息"""
        config = {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "causal_size": self.causal_size,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "algorithm": "CausalEngine v2.0 (Modular)",
            "modules": {
                "abduction": f"AbductionNetwork({self.hidden_size} → {self.causal_size})",
                "action": f"ActionNetwork({self.causal_size} → {self.vocab_size})",
                "activation": self.activation.get_config()
            }
        }
        return config
    
    def __repr__(self) -> str:
        config = self.get_config()
        return (
            f"CausalEngine(\n"
            f"  hidden_size={config['hidden_size']},\n"
            f"  vocab_size={config['vocab_size']},\n"
            f"  causal_size={config['causal_size']},\n"
            f"  parameters={config['num_parameters']:,}\n"
            f"  modules=[\n"
            f"    {config['modules']['abduction']},\n"
            f"    {config['modules']['action']},\n"
            f"    ActivationHead(modes={config['modules']['activation']['activation_modes'][:5]}...)\n"
            f"  ]\n"
            f")"
        ) 