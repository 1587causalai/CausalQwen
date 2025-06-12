"""CausalQwen 核心模块

包含数值嵌入、归因推断和行动网络的实现。
"""

import torch
from torch import nn
from typing import Optional, Tuple, Dict


class NumericalEmbedding(nn.Module):
    """数值感知的嵌入模块"""
    
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.num_vocab + 1, config.hidden_dim)
        self.num_token_id = config.num_token_id
        self.hidden_dim = config.hidden_dim
        
        # 数值编码的可学习向量
        self.num_encoding_vector = nn.Parameter(torch.randn(config.hidden_dim))
        
    def forward(self, input_ids: torch.Tensor, num_values: Optional[torch.Tensor] = None):
        """前向传播
        
        Args:
            input_ids: token序列 [B, S]
            num_values: 数值序列 [B, S]
            
        Returns:
            增强嵌入 [B, S, H]
        """
        # 基础嵌入
        embeddings = self.embedding(input_ids)  # [B, S, H]
        
        if num_values is not None:
            # 数值编码：φ(v) = sign(v) * ln(1 + |v|) * e
            phi_v = self._phi_encoding(num_values)  # [B, S, H]
            
            # 增强嵌入
            embeddings = embeddings + phi_v
            
        return embeddings
    
    def _phi_encoding(self, values: torch.Tensor) -> torch.Tensor:
        """数值编码函数 φ(v)"""
        # φ(v) = sign(v) * ln(1 + |v|) * e
        sign_v = torch.sign(values)  # [B, S]
        log_v = torch.log1p(torch.abs(values))  # [B, S]
        
        # 扩展到隐藏维度
        scale = sign_v * log_v  # [B, S]
        scale = scale.unsqueeze(-1)  # [B, S, 1]
        
        # 应用编码向量
        phi_v = scale * self.num_encoding_vector  # [B, S, H]
        
        return phi_v


class AbductionNetwork(nn.Module):
    """归因推断网络
    
    推断每个位置的因果表征分布 U_i ~ Cauchy(loc, scale)
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # 简化的线性实现
        # loc: 恒等映射初始化
        self.loc_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.loc_proj.weight.data = torch.eye(config.hidden_dim)
        self.loc_proj.bias.data.zero_()
        
        # scale: 确保输出较大的scale
        self.scale_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.scale_proj.bias.data.fill_(2.0)  # 确保 softplus 后 scale ≈ 10
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            features: 特征序列 [B, S, H]
            
        Returns:
            (loc, scale): 柯西分布参数 [B, S, H], [B, S, H]
        """
        loc = self.loc_proj(features)  # [B, S, H]
        
        scale_raw = self.scale_proj(features)  # [B, S, H]
        scale = torch.nn.functional.softplus(scale_raw) + 1e-4  # 确保 > 0
        
        return loc, scale


class ActionNetwork(nn.Module):
    """行动网络
    
    基于因果表征 U 计算分类和回归输出
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_vocab = config.num_vocab + 1  # 包含 <NUM>
        
        # 分类头（可以从Qwen迁移）
        self.class_weights = nn.Parameter(torch.randn(self.num_vocab, config.hidden_dim))
        self.class_bias = nn.Parameter(torch.zeros(self.num_vocab))
        self.class_thresholds = nn.Parameter(torch.zeros(self.num_vocab))
        
        # 回归头
        self.reg_weight = nn.Parameter(torch.randn(config.hidden_dim))
        self.reg_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, u_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            u_samples: 因果表征样本 [B, S, N, H] 或 [B, S, H]
            
        Returns:
            包含分类分数和回归值的字典
        """
        # 处理输入维度
        if u_samples.dim() == 3:
            u_samples = u_samples.unsqueeze(2)  # [B, S, 1, H]
            
        B, S, N, H = u_samples.shape
        
        # 分类分数：S_k = A_k · U + B_k
        # [B, S, N, H] @ [K, H]^T -> [B, S, N, K]
        class_scores = torch.einsum('bsnh,kh->bsnk', u_samples, self.class_weights)
        class_scores = class_scores + self.class_bias  # 广播
        
        # 回归值：Y = W · U + b
        # [B, S, N, H] @ [H] -> [B, S, N]
        regression_values = torch.einsum('bsnh,h->bsn', u_samples, self.reg_weight)
        regression_values = regression_values + self.reg_bias
        
        # 如果N=1，去掉采样维度
        if N == 1:
            class_scores = class_scores.squeeze(2)  # [B, S, K]
            regression_values = regression_values.squeeze(2)  # [B, S]
            
        return {
            'class_scores': class_scores,
            'regression_values': regression_values
        }
    
    def compute_ovr_probs(self, class_scores: torch.Tensor) -> torch.Tensor:
        """计算 OvR 概率（使用sigmoid近似）"""
        # P_k = sigmoid(S_k - C_k)
        return torch.sigmoid(class_scores - self.class_thresholds)
