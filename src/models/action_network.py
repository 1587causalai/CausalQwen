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
    Action Network for the Causal Language Model.
    
    This network takes the parameters of the individual causal representation distribution
    and transforms them into the parameters for the output distributions
    (classification scores and regression values).
    """
    
    def __init__(self, input_dim: int, hidden_size: int, num_token_id: int, vocab_size: Optional[int] = None):
        """
        Initialize the Action Network.
        
        Args:
            input_dim (int): Dimensionality of the individual causal representation
            hidden_size (int): Qwen 模型的隐藏维度
            num_token_id (int): The token ID for the <NUM> token.
            vocab_size (Optional[int]): Size of the vocabulary. If None, the classification head will not be initialized.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_token_id = num_token_id
        
        # 分类头 - loc 和 scale
        # 注意：现在分类头可能不会在这里被初始化
        if vocab_size is not None:
            self.classification_head = nn.Linear(input_dim, vocab_size, bias=False)
            self.scale_S_head = nn.Linear(input_dim, vocab_size, bias=True)
        else:
            self.classification_head = None
            self.scale_S_head = None

        # 回归头 - loc 和 scale
        self.regression_head = nn.Linear(input_dim, 1, bias=True)
        self.scale_Y_head = nn.Linear(input_dim, 1, bias=True)

    def init_weights(self, qwen_lm_head: Optional[nn.Parameter] = None):
        """
        Initialize the weights of the action network.
        
        知识传输策略：
        - 分类头：复制 Qwen lm_head 的所有权重（注意：Qwen 没有偏置）
        - 回归头：使用小随机初始化，实现均匀先验
        
        Args:
            qwen_lm_head: Qwen's language model head (nn.Linear)
        """
        print("🔧 初始化 ActionNetwork...")
        
        # --- 1. 初始化/创建分类头 ---
        if qwen_lm_head is not None:
            qwen_vocab_size, qwen_hidden_size = qwen_lm_head.weight.shape
            
            # 如果分类头还未创建，现在根据 qwen_lm_head 的尺寸创建它
            if self.classification_head is None:
                print(f"   创建分类头，尺寸: [{qwen_vocab_size}, {self.input_dim}]")
                self.classification_head = nn.Linear(self.input_dim, qwen_vocab_size, bias=False).to(qwen_lm_head.weight.device)
                self.scale_S_head = nn.Linear(self.input_dim, qwen_vocab_size, bias=True).to(qwen_lm_head.weight.device)

            print("📚 从 Qwen lm_head 进行知识传输...")
            print(f"   Qwen lm_head: {list(qwen_lm_head.weight.shape)}")
            print(f"   CausalQwen 分类头: {list(self.classification_head.weight.shape)}")

            # 验证尺寸是否匹配
            if self.classification_head.weight.shape != qwen_lm_head.weight.shape:
                 raise ValueError(
                    f"尺寸不匹配! CausalQwen head ({self.classification_head.weight.shape}) "
                    f"vs Qwen lm_head ({qwen_lm_head.weight.shape})."
                )

            # 完整复制权重
            self.classification_head.weight.data.copy_(qwen_lm_head.weight.data)
            print("   ✅ 权重完全迁移")

            # 验证 Qwen lm_head 是否有偏置项
            if hasattr(qwen_lm_head, 'bias') and qwen_lm_head.bias is not None:
                print("   - 警告: Qwen lm_head 包含偏置项，但当前模型未使用。")
            else:
                print("   ✅ Qwen lm_head 无偏置项（符合预期）")

        else:
            # 如果没有 qwen_lm_head，则必须在 __init__ 中提供 vocab_size
            if self.classification_head is None:
                raise ValueError("没有提供 qwen_lm_head，无法创建分类头。请在初始化时提供 vocab_size。")
            print("   - 未提供 Qwen lm_head，使用随机初始化分类头。")
            # 这里可以添加随机初始化逻辑，如果需要的话
            pass

        # --- 2. 初始化回归头 ---
        print("\n🔧 初始化回归头（小随机初始化）...")
        # 使用小的随机权重初始化回归头，避免在训练初期产生过大的输出
        nn.init.xavier_uniform_(self.regression_head.weight, gain=0.01)
        nn.init.zeros_(self.regression_head.bias)
        print("   ✅ 使用 Xavier 初始化 (gain=0.01) 实现均匀先验")
        
        # --- 3. 初始化尺度头 ---
        # scale_S 的偏置项初始化为一个较大的正数，确保初始不确定性高
        # scale_Y 的偏置项初始化为一个较小的正数
        if self.scale_S_head is not None:
            nn.init.zeros_(self.scale_S_head.weight)
            nn.init.constant_(self.scale_S_head.bias, 2.3)  # exp(2.3) ≈ 10

        nn.init.zeros_(self.scale_Y_head.weight)
        nn.init.constant_(self.scale_Y_head.bias, 1.0) # exp(1.0) ~ 2.7

        print("   ✅ 知识传输完成")

    def forward(self, U_loc: torch.Tensor, U_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass for the Action Network.
        
        Args:
            U_loc (torch.Tensor): Location parameter of causal state distribution.
            U_scale (torch.Tensor): Scale parameter of causal state distribution.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the parameters of the output distributions.
        """
        # 从 U 中采样一个实例，或直接使用其均值
        # 当前实现：直接使用均值 U_loc 作为下一层的输入
        z_prime = U_loc
        
        # 计算分类和回归的 loc
        loc_S = self.classification_head(z_prime)
        loc_Y = self.regression_head(z_prime).squeeze(-1)  # [B, S, 1] -> [B, S]
        
        # 计算分类和回归的 scale
        # 基础尺度来源于 U_scale，然后通过各自的头进行变换
        # U_scale -> log(U_scale) -> linear -> exp -> scale
        log_U_scale = torch.log(U_scale)
        
        scale_S = torch.exp(self.scale_S_head(log_U_scale))
        scale_Y = torch.exp(self.scale_Y_head(log_U_scale)).squeeze(-1) # [B, S, 1] -> [B, S]

        return {
            "loc_S": loc_S,
            "scale_S": scale_S,
            "loc_Y": loc_Y,
            "scale_Y": scale_Y,
        }
    
    def predict(self, causal_loc, causal_scale=None):
        """
        Make a deterministic prediction based on the output distributions.
        
        Args:
            causal_loc (torch.Tensor): Location parameters of the individual causal representation
            causal_scale (torch.Tensor, optional): Scale parameters. If None, a zero tensor is used
                                                   for a purely deterministic prediction from loc.
        
        Returns:
            dict: A dictionary containing the predicted token, regression value, and <NUM> probability.
        """
        if causal_scale is None:
            # If no scale is provided, assume a deterministic input (zero scale)
            causal_scale = torch.zeros_like(causal_loc)

        # 1. Get output distribution parameters by running the forward pass
        outputs = self.forward(causal_loc, causal_scale)
        loc_S, scale_S = outputs['loc_S'], outputs['scale_S']
        loc_Y, scale_Y = outputs['loc_Y'], outputs['scale_Y']
        
        # 2. Get classification prediction from the distribution
        # This computes probabilities and takes argmax
        cls_pred = self.classification_head.predict(loc_S, scale_S)
        
        # 3. Get regression prediction (point estimate is the location parameter)
        reg_pred = self.regression_head.predict(loc_Y, scale_Y)
        
        # 4. Get probability of <NUM> token
        all_probs = self.classification_head.compute_probabilities(loc_S, scale_S)
        
        # Handle sequence vs. non-sequence input for num_prob
        if all_probs.dim() > 1:
            num_prob = all_probs[..., self.num_token_id]
        else:
            # This case might not be typical, but handle it for robustness
            num_prob = all_probs[self.num_token_id]

        return {
            'cls_pred': cls_pred,
            'reg_pred': reg_pred,
            'num_prob': num_prob
        }

