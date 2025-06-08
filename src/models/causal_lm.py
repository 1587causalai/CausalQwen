"""
因果语言模型核心实现

基于推断-行动范式的因果语言模型，使用柯西分布建模潜在因果状态。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..utils.cauchy import CauchyLinear, threshold_probability, cauchy_sample


class AbductionNetwork(nn.Module):
    """
    推断网络
    
    从输入特征推断潜在因果状态的分布参数。
    输入特征 z -> 因果状态分布 U ~ Cauchy(μ, γ)
    """
    
    def __init__(self, feature_dim: int, causal_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.causal_dim = causal_dim
        
        # 线性层：特征 -> (位置参数, 对数尺度参数)
        self.linear = nn.Linear(feature_dim, causal_dim * 2)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, feature_dim]
        
        Returns:
            (loc, scale): 因果状态分布参数
                loc: 位置参数 [batch_size, causal_dim]
                scale: 尺度参数 [batch_size, causal_dim]
        """
        output = self.linear(features)  # [batch_size, causal_dim * 2]
        
        # 分离位置和对数尺度参数
        loc, log_scale = output.chunk(2, dim=-1)
        
        # 确保尺度参数为正
        scale = torch.exp(log_scale) + 1e-6
        
        return loc, scale


class ActionNetwork(nn.Module):
    """
    行动网络
    
    基于推断的因果状态分布进行预测，包括分类和回归两个头。
    """
    
    def __init__(self, causal_dim: int, vocab_size: int):
        super().__init__()
        self.causal_dim = causal_dim
        self.vocab_size = vocab_size
        
        # 分类头：用于词元预测（OvR策略）
        self.classification_head = CauchyLinear(causal_dim, vocab_size)
        
        # 回归头：用于数值预测
        self.regression_head = CauchyLinear(causal_dim, 1)
        
    def forward(self, causal_loc: torch.Tensor, causal_scale: torch.Tensor, 
                threshold: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            causal_loc: 因果状态位置参数 [batch_size, causal_dim]
            causal_scale: 因果状态尺度参数 [batch_size, causal_dim]
            threshold: OvR分类的决策阈值
        
        Returns:
            预测结果字典，包含分类概率和回归值
        """
        # 分类预测：计算每个词元的决策分数分布
        cls_loc, cls_scale = self.classification_head(causal_loc, causal_scale)
        
        # 计算每个词元超过阈值的概率（OvR策略）
        cls_probs = threshold_probability(cls_loc, cls_scale, threshold)
        
        # 回归预测：使用柯西分布的中位数（位置参数）作为点估计
        reg_loc, reg_scale = self.regression_head(causal_loc, causal_scale)
        reg_pred = reg_loc.squeeze(-1)  # [batch_size]
        
        return {
            'cls_probs': cls_probs,      # [batch_size, vocab_size]
            'cls_loc': cls_loc,          # [batch_size, vocab_size]
            'cls_scale': cls_scale,      # [batch_size, vocab_size]
            'reg_pred': reg_pred,        # [batch_size]
            'reg_loc': reg_loc.squeeze(-1),  # [batch_size]
            'reg_scale': reg_scale.squeeze(-1)  # [batch_size]
        }


class FeatureNetwork(nn.Module):
    """
    特征网络
    
    从输入序列提取特征表示。这里使用简单的实现，
    实际应用中可以替换为预训练的语言模型。
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=2
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入序列 [batch_size, seq_len]
        
        Returns:
            features: 特征表示 [batch_size, embed_dim]
        """
        # 词嵌入
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Transformer编码
        encoded = self.transformer(embeddings)  # [batch_size, seq_len, embed_dim]
        
        # 池化到固定维度
        pooled = self.pooling(encoded.transpose(1, 2)).squeeze(-1)  # [batch_size, embed_dim]
        
        return pooled


class CausalLanguageModel(nn.Module):
    """
    因果语言模型
    
    整合特征网络、推断网络和行动网络的完整模型。
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, 
                 hidden_dim: int = 1024, causal_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.causal_dim = causal_dim
        
        # 三个核心网络
        self.feature_network = FeatureNetwork(vocab_size, embed_dim, hidden_dim)
        self.abduction_network = AbductionNetwork(embed_dim, causal_dim)
        self.action_network = ActionNetwork(causal_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, 
                threshold: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入序列 [batch_size, seq_len]
            threshold: 分类决策阈值
        
        Returns:
            模型预测结果
        """
        # 1. 特征提取
        features = self.feature_network(input_ids)
        
        # 2. 因果状态推断
        causal_loc, causal_scale = self.abduction_network(features)
        
        # 3. 行动预测
        predictions = self.action_network(causal_loc, causal_scale, threshold)
        
        # 添加因果状态信息
        predictions.update({
            'causal_loc': causal_loc,
            'causal_scale': causal_scale,
            'features': features
        })
        
        return predictions
    
    def predict(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测接口
        
        Args:
            input_ids: 输入序列
        
        Returns:
            预测结果，包含最可能的词元和数值
        """
        with torch.no_grad():
            outputs = self.forward(input_ids)
            
            # 获取最可能的词元
            cls_pred = torch.argmax(outputs['cls_probs'], dim=-1)
            
            return {
                'cls_pred': cls_pred,
                'reg_pred': outputs['reg_pred'],
                'cls_probs': outputs['cls_probs']
            }
    
    def sample_causal_state(self, input_ids: torch.Tensor, 
                           num_samples: int = 1) -> torch.Tensor:
        """
        从推断的因果状态分布中采样
        
        Args:
            input_ids: 输入序列
            num_samples: 采样数量
        
        Returns:
            采样的因果状态 [batch_size, num_samples, causal_dim]
        """
        with torch.no_grad():
            features = self.feature_network(input_ids)
            causal_loc, causal_scale = self.abduction_network(features)
            
            samples = []
            for _ in range(num_samples):
                sample = cauchy_sample(causal_loc, causal_scale)
                samples.append(sample)
            
            return torch.stack(samples, dim=1)

