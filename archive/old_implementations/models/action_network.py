"""
Action Network module.

This module implements the action network for the causal language model,
which transforms the latent causal state into classification and regression outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from ..utils.distributions import CauchyLinear


class ClassificationHead(nn.Module):
    """
    Classification head for token prediction.
    
    This implements the One-vs-Rest (OvR) classification approach,
    where each class has an independent decision score.
    """
    
    def __init__(self, causal_dim, num_classes, threshold=0.0, bias=True):
        """
        Initialize the classification head.
        
        Args:
            causal_dim (int): Dimensionality of the latent causal state
            num_classes (int): Number of classes (vocabulary size)
            threshold (float, optional): Decision threshold. Defaults to 0.0.
            bias (bool, optional): Whether to include bias. Defaults to True.
        """
        super().__init__()
        self.causal_dim = causal_dim
        self.num_classes = num_classes
        self.threshold = threshold
        
        # Linear layer to map causal state to class decision scores
        # 重要：根据 Qwen 的设计，分类头不应该有偏置
        self.causal_linear = CauchyLinear(causal_dim, num_classes, bias=bias)
        
        # Register threshold as a buffer (not a parameter)
        self.register_buffer('thresholds', torch.ones(num_classes) * threshold)
        
    def forward(self, causal_loc, causal_scale):
        """
        Transform causal state distribution to class decision score distributions.
        
        Args:
            causal_loc (torch.Tensor): Location parameter of causal state
                                      Shape: [batch_size, causal_dim] or [batch_size, seq_len, causal_dim]
            causal_scale (torch.Tensor): Scale parameter of causal state
                                        Shape: [batch_size, causal_dim] or [batch_size, seq_len, causal_dim]
        
        Returns:
            tuple: (score_loc, score_scale) - Parameters of the decision score distributions
                  Shape: [batch_size, num_classes] or [batch_size, seq_len, num_classes]
        """
        # Transform causal state distribution to decision score distributions
        score_loc, score_scale = self.causal_linear(causal_loc, causal_scale)
        
        return score_loc, score_scale
    
    def compute_probabilities(self, score_loc, score_scale):
        """
        Compute class probabilities using the Cauchy CDF.
        
        From core-design.md: P(S_k > C_k) = 1/2 + (1/π) * arctan((loc_Sk - C_k)/scale_Sk)
        """
        # Add epsilon for numerical stability, especially if scale can be zero
        stable_scale = score_scale + 1e-9
        
        # Direct implementation of the formula from core-design.md
        # P(S_k > C_k) = 1/2 + (1/π) * arctan((loc_Sk - C_k)/scale_Sk)
        probs = 0.5 + (1 / torch.pi) * torch.atan((score_loc - self.thresholds) / stable_scale)
        
        return probs
    
    def predict(self, score_loc, score_scale):
        """
        Make class predictions based on decision scores.
        
        Args:
            score_loc (torch.Tensor): Location parameters of decision scores
                                     Shape: [batch_size, num_classes]
            score_scale (torch.Tensor): Scale parameters of decision scores
                                       Shape: [batch_size, num_classes]
        
        Returns:
            torch.Tensor: Predicted class indices
                         Shape: [batch_size]
        """
        # Compute probabilities
        probs = self.compute_probabilities(score_loc, score_scale)
        
        # Return class with highest probability
        return torch.argmax(probs, dim=-1)


class RegressionHead(nn.Module):
    """
    Regression head for numerical prediction.
    """
    
    def __init__(self, causal_dim):
        """
        Initialize the regression head.
        
        Args:
            causal_dim (int): Dimensionality of the latent causal state
        """
        super().__init__()
        self.causal_dim = causal_dim
        
        # Linear layer to map causal state to regression value
        self.causal_linear = CauchyLinear(causal_dim, 1)
        
    def forward(self, causal_loc, causal_scale):
        """
        Transform causal state distribution to regression value distribution.
        
        Args:
            causal_loc (torch.Tensor): Location parameter of causal state
                                      Shape: [batch_size, causal_dim] or [batch_size, seq_len, causal_dim]
            causal_scale (torch.Tensor): Scale parameter of causal state
                                        Shape: [batch_size, causal_dim] or [batch_size, seq_len, causal_dim]
        
        Returns:
            tuple: (value_loc, value_scale) - Parameters of the regression value distribution
                  Shape: [batch_size] or [batch_size, seq_len]
        """
        # Transform causal state distribution to regression value distribution
        value_loc, value_scale = self.causal_linear(causal_loc, causal_scale)
        
        # Squeeze the last dimension
        # For sequence inputs: [batch_size, seq_len, 1] -> [batch_size, seq_len]
        # For single inputs: [batch_size, 1] -> [batch_size]
        return value_loc.squeeze(-1), value_scale.squeeze(-1)
    
    def predict(self, value_loc, value_scale):
        """
        Make regression predictions.
        
        For Cauchy distribution, the median (location parameter) is the best
        point estimate.
        
        Args:
            value_loc (torch.Tensor): Location parameter of regression value
                                     Shape: [batch_size]
            value_scale (torch.Tensor): Scale parameter of regression value
                                       Shape: [batch_size]
        
        Returns:
            torch.Tensor: Predicted regression values
                         Shape: [batch_size]
        """
        # For Cauchy distribution, the median (location parameter) is the best point estimate
        return value_loc


class ActionNetwork(nn.Module):
    """
    行动网络 (Action Network)
    
    该网络接收因果表征分布 U 的参数，并将其转换为
    输出分布（分类分数和回归值）的参数。
    它通过组合一个分类头和一个回归头来实现此功能。
    """
    
    def __init__(self, input_dim: int, vocab_size: int, hidden_size: Optional[int] = None, num_token_id: Optional[int] = None):
        """
        初始化行动网络。

        Args:
            input_dim (int): 个体因果表征的维度。
            vocab_size (int): 词汇表的大小。
            hidden_size (Optional[int]): Qwen 模型的隐藏维度 (为兼容性保留)。
            num_token_id (Optional[int]): <NUM> 词元的 ID (为兼容性保留)。
        """
        super().__init__()
        self.input_dim = input_dim
        
        # 直接实例化子模块，逻辑清晰
        self.classification_head = ClassificationHead(
            causal_dim=input_dim,
            num_classes=vocab_size,
            # Qwen 的 lm_head 没有偏置项，所以这里强制为 False
            bias=False
        )
        
        self.regression_head = RegressionHead(causal_dim=input_dim)

    def init_weights(self, qwen_lm_head: Optional[nn.Linear] = None):
        """
        初始化行动网络的权重。
        
        知识传输策略：
        - 分类头：完全复用 Qwen 的 lm_head 的权重。
        - 回归头：使用小的随机值初始化，以实现均匀先验。
        
        Args:
            qwen_lm_head (Optional[nn.Linear]): Qwen 的语言模型头。
        """
        print("🔧 初始化 ActionNetwork...")
        
        # --- 1. 初始化分类头 ---
        if qwen_lm_head is not None:
            print("📚 从 Qwen lm_head 进行知识传输...")
            print(f"   - Qwen lm_head shape: {qwen_lm_head.weight.shape}")
            print(f"   - ActionNet cls head shape: {self.classification_head.causal_linear.weight.shape}")

            # 验证尺寸是否匹配
            if self.classification_head.causal_linear.weight.shape != qwen_lm_head.weight.shape:
                 raise ValueError("分类头与 Qwen lm_head 的权重尺寸不匹配!")

            # 完整复制权重
            self.classification_head.causal_linear.weight.data.copy_(qwen_lm_head.weight.data)
            print("   ✅ 权重完全迁移")

            if hasattr(qwen_lm_head, 'bias') and qwen_lm_head.bias is not None:
                print("   - 警告: Qwen lm_head 包含偏置项，但 ActionNetwork 分类头未使用。")
            else:
                print("   ✅ Qwen lm_head 无偏置项（符合预期）")
        else:
            print("   - 未提供 Qwen lm_head，分类头使用默认 Xavier 初始化。")
            # 如果没有提供预训练权重，则使用标准初始化
            pass

        # --- 2. 初始化回归头 ---
        print("\n🔧 初始化回归头（小随机初始化）...")
        # 使用小的 Xavier增益 (gain=0.01) 来确保初始回归预测接近于零
        nn.init.xavier_uniform_(self.regression_head.causal_linear.weight, gain=0.01)
        # 将偏置初始化为零
        nn.init.zeros_(self.regression_head.causal_linear.bias)
        print("   ✅ 使用 Xavier 初始化 (gain=0.01) 实现均匀先验")
        
        print("   ✅ ActionNetwork 初始化完成")
            
    def forward(self, U_loc: torch.Tensor, U_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        执行行动网络的前向传播。

        Args:
            U_loc (torch.Tensor): 因果表征分布的位置参数。
            U_scale (torch.Tensor): 因果表征分布的尺度参数。

        Returns:
            Dict[str, torch.Tensor]: 包含所有输出分布参数的字典。
        """
        # 直接调用子模块的前向传播
        loc_S, scale_S = self.classification_head(U_loc, U_scale)
        loc_Y, scale_Y = self.regression_head(U_loc, U_scale)
        
        return {
            'loc_S': loc_S,
            'scale_S': scale_S,
            'loc_Y': loc_Y,
            'scale_Y': scale_Y,
        }
    
    def predict(self, causal_loc, causal_scale=None):
        """
        确定性预测。
        
        Args:
            causal_loc: 采样或确定的因果表征
            causal_scale: 如果为 None，则进行完全确定性预测
        
        Returns:
            dict: 包含分类和回归预测的字典
        """
        if causal_scale is None:
            causal_scale = torch.zeros_like(causal_loc)
            
        cls_loc, _ = self.classification_head(causal_loc, causal_scale)
        predicted_tokens = self.classification_head.predict(cls_loc, None)
        
        reg_loc, _ = self.regression_head(causal_loc, causal_scale)
        predicted_values = self.regression_head.predict(reg_loc, None)
        
        return {
            'predicted_tokens': predicted_tokens,
            'predicted_values': predicted_values
        }

def cauchy_sample_reparameterized(loc, scale):
    """
    使用重参数化技巧从柯西分布中采样。
    
    Args:
        loc (torch.Tensor): 位置参数
        scale (torch.Tensor): 尺度参数
        
    Returns:
        torch.Tensor: 采样结果
    """
    # 从标准均匀分布中采样
    uniform_sample = torch.rand_like(loc)
    
    # 通过柯西分布的逆CDF（分位数函数）进行变换
    cauchy_sample = loc + scale * torch.tan(torch.pi * (uniform_sample - 0.5))
    
    return cauchy_sample

