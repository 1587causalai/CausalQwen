"""CausalQwen 配置类"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CausalQwenConfig:
    """CausalQwen 模型配置
    
    包含所有模型架构和训练相关的配置参数。
    """
    
    # 模型架构参数
    num_vocab: int = 32001  # 词汇表大小 (Qwen + <NUM>)
    hidden_dim: int = 1024  # 隐藏层维度
    num_layers: int = 24    # Transformer层数
    num_heads: int = 16     # 注意力头数
    
    # 特殊token
    num_token_id: int = 32000  # <NUM> token的ID
    eos_token_id: int = 2      # 结束token
    pad_token_id: int = 0      # 填充token
    
    # 归因推断网络参数
    abduction_hidden_dim: Optional[int] = None  # 默认与hidden_dim相同
    initial_scale: float = 10.0  # 初始scale值
    
    # 行动网络参数
    num_classes: Optional[int] = None  # 分类数，默认与num_vocab相同
    
    # 训练参数
    gate_alpha: float = 0.0  # 门控系数 (0=完全门控, 1=无门控)
    regression_weight: float = 1.0  # 回归损失权重
    
    # 生成参数
    max_position_embeddings: int = 2048  # 最大序列长度
    
    def __post_init__(self):
        """初始化后处理"""
        if self.abduction_hidden_dim is None:
            self.abduction_hidden_dim = self.hidden_dim
            
        if self.num_classes is None:
            self.num_classes = self.num_vocab
