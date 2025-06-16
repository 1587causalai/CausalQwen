#!/usr/bin/env python
"""
数值感知嵌入模块 (Numerical-aware Embedding Module)

本模块负责将混合了文本和数值的输入，转化为一个统一的、
数值感知的特征向量序列 (enhanced_embeddings)。
它严格遵循 `mathematical_foundations.md` 中 "图 2" 的流程。
"""
import torch
import torch.nn as nn
import math

class NumericalAwareEmbedding(nn.Module):
    """
    将 token 嵌入和数值编码融合，生成增强嵌入。
    """
    def __init__(self, base_embedding_layer: nn.Embedding, num_token_id: int, hidden_size: int):
        """
        初始化数值嵌入模块。

        Args:
            base_embedding_layer (nn.Embedding): 预训练模型的词元嵌入层。
            num_token_id (int): <NUM> 特殊词元的 ID。
            hidden_size (int): 模型的隐藏维度。
        """
        super().__init__()
        self.base_embedding_layer = base_embedding_layer
        self.num_token_id = num_token_id
        self.hidden_size = hidden_size
        
        # 定义数值编码所需的可学习方向向量
        self.numerical_direction = nn.Parameter(
            torch.randn(hidden_size) / math.sqrt(hidden_size)
        )

    def forward(self, input_ids: torch.Tensor, numerical_values: torch.Tensor) -> torch.Tensor:
        """
        执行从 `input_ids` 和 `numerical_values` 到 `enhanced_embeddings` 的转换。

        Args:
            input_ids (torch.Tensor): 批处理的词元 ID 序列。
            numerical_values (torch.Tensor): 批处理的对齐数值序列。

        Returns:
            torch.Tensor: 融合了数值信息的增强嵌入张量。
        """
        # 步骤 E, F: 获取基础词元嵌入
        base_embeddings = self.base_embedding_layer(input_ids)

        # 步骤 G: 计算数值编码 φ(v)
        # 公式: φ(v) = sign(v) * ln(1 + |v|) * e_direction
        # 注意：此编码只应应用于 <NUM> 词元所在的位置。
        
        # 1. 创建一个掩码，标记出所有 <NUM> 词元的位置
        #    形状: [batch, seq_len] -> [batch, seq_len, 1]
        num_mask = (input_ids == self.num_token_id).float().unsqueeze(-1)
        
        # 2. 对所有位置的数值应用对数变换
        #    公式: sign(v) * ln(1 + |v|)
        transformed_values = torch.sign(numerical_values) * torch.log1p(torch.abs(numerical_values))
        
        # 3. 将变换后的数值与方向向量相乘以形成编码
        #    - transformed_values 扩展维度: [b, s] -> [b, s, 1]
        #    - numerical_direction 形状: [h]
        #    - 结果 phi_v 形状: [b, s, h]
        phi_v = transformed_values.unsqueeze(-1) * self.numerical_direction

        # 4. 应用掩码，使得编码只在 <NUM> 位置生效
        numerical_encoding = phi_v * num_mask

        # 步骤 H, I: 融合嵌入
        enhanced_embeddings = base_embeddings + numerical_encoding
        
        return enhanced_embeddings 