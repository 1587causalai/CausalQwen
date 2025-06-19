"""
CausalQwen MVP: 辅助组件
只包含项目特定的组件，核心算法都在 causal_engine 中
"""

import torch
import torch.nn as nn
from .config import CausalQwen2Config


class OvRClassifier(nn.Module):
    """
    One-vs-Rest 分类器
    
    这是一个项目特定的组件，用于将 CausalEngine 的输出转换为概率。
    核心的 OvR 概率计算逻辑在 CausalEngine.compute_ovr_probs() 中。
    """
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        self.thresholds = nn.Parameter(torch.full((config.vocab_size,), config.ovr_threshold_init))
    
    def forward(self, loc_S, scale_S):
        """
        计算 OvR 概率
        
        这里只是一个简单的包装器，实际计算委托给 CausalEngine
        """
        # 使用 CausalEngine 的计算方法
        from causal_engine import CausalEngine
        
        # 创建一个临时引擎实例来使用其计算方法
        # 注意：在实际使用中，应该直接调用已有引擎实例的方法
        normalized_diff = (loc_S - self.thresholds.unsqueeze(0).unsqueeze(0)) / scale_S
        probs = 0.5 + (1/torch.pi) * torch.atan(normalized_diff)
        return probs 