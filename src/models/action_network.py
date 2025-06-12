"""
Action Network module.

This module implements the action network for the causal language model,
which transforms the latent causal state into classification and regression outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def __init__(self, causal_dim, vocab_size, num_token_id, ovr_threshold=0.0):
        """
        Initialize the Action Network.
        
        Args:
            causal_dim (int): Dimensionality of the individual causal representation
            vocab_size (int): Size of the vocabulary
            num_token_id (int): The token ID for the <NUM> token.
            ovr_threshold (float): Decision threshold for OvR classification
        """
        super().__init__()
        self.causal_dim = causal_dim
        self.vocab_size = vocab_size
        self.num_token_id = num_token_id
        self.ovr_threshold = ovr_threshold
        
        # Classification head for token prediction
        # 重要：设置 bias=False 以匹配 Qwen 的设计
        self.classification_head = ClassificationHead(
            causal_dim, 
            vocab_size, 
            threshold=ovr_threshold,
            bias=False  # Qwen lm_head 没有偏置
        )
        
        # Regression head for numerical prediction
        self.regression_head = RegressionHead(causal_dim)

    def init_weights(self, qwen_lm_head=None, num_target_median=None, num_target_scale=None, num_token_id=None):
        """
        Initialize the weights of the action network.
        
        知识传输策略：
        - 分类头：复制 Qwen lm_head 的所有权重（注意：Qwen 没有偏置）
        - 回归头：使用小随机初始化，实现均匀先验
        
        Args:
            qwen_lm_head: Qwen's language model head (nn.Linear)
            num_target_median: Deprecated, no longer used
            num_target_scale: Deprecated, no longer used
            num_token_id: The token ID for <NUM> (should be 151665)
        """
        print("🔧 初始化 ActionNetwork...")
        
        if qwen_lm_head is not None and num_token_id is not None:
            print("📚 从 Qwen lm_head 进行知识传输...")
            
            # 获取维度信息
            qwen_vocab_size, qwen_hidden_size = qwen_lm_head.weight.shape
            our_vocab_size, our_causal_dim = self.classification_head.causal_linear.weight.shape
            
            print(f"   Qwen lm_head: [{qwen_vocab_size}, {qwen_hidden_size}]")
            print(f"   CausalQwen 分类头: [{our_vocab_size}, {our_causal_dim}]")
            
            # 1. 分类头：知识迁移
            with torch.no_grad():
                if our_vocab_size == qwen_vocab_size:
                    # 完整复制：保持与 Qwen 的完全兼容性
                    self.classification_head.causal_linear.weight.copy_(
                        qwen_lm_head.weight
                    )
                    print(f"   ✅ 完整复制了所有 {qwen_vocab_size} 个权重")
                    print(f"      - Qwen 已用词汇: 151,665 个")
                    print(f"      - 预留位置: 271 个（均已初始化）")
                    print(f"      - <NUM> token (ID: {num_token_id}): 使用第一个预留位置")
                elif our_vocab_size < qwen_vocab_size:
                    # 部分复制（不推荐，但支持）
                    self.classification_head.causal_linear.weight.copy_(
                        qwen_lm_head.weight[:our_vocab_size, :]
                    )
                    print(f"   ⚠️  部分复制了 {our_vocab_size} 个权重（建议使用完整容量 151936）")
                else:
                    raise ValueError(f"CausalQwen vocab size ({our_vocab_size}) > Qwen vocab size ({qwen_vocab_size})")
                
                # 检查 Qwen 是否有偏置
                if hasattr(qwen_lm_head, 'bias') and qwen_lm_head.bias is not None:
                    print("   ⚠️  Qwen lm_head 有偏置项，但 CausalQwen 设计为无偏置")
                else:
                    print("   ✅ Qwen lm_head 无偏置项（符合预期）")
                
                # 确保我们的分类头也没有偏置
                if self.classification_head.causal_linear.bias is not None:
                    print("   ⚠️  警告：CausalQwen 分类头有偏置，将其置零")
                    self.classification_head.causal_linear.bias.zero_()
                
                # 2. 回归头：小随机初始化
                print(f"\n🔧 初始化回归头（小随机初始化）...")
                # 使用小的 Xavier 初始化
                torch.nn.init.xavier_uniform_(self.regression_head.causal_linear.weight, gain=0.01)
                if self.regression_head.causal_linear.bias is not None:
                    self.regression_head.causal_linear.bias.zero_()
                print(f"   ✅ 使用 Xavier 初始化 (gain=0.01) 实现均匀先验")

            print("   ✅ 知识传输完成")
            
        else:
            print("⚠️  未提供 Qwen lm_head 或 num_token_id，使用随机初始化")
            # 分类头：Xavier 初始化（无偏置）
            torch.nn.init.xavier_uniform_(self.classification_head.causal_linear.weight)
            
            # 回归头：小的 Xavier 初始化
            torch.nn.init.xavier_uniform_(self.regression_head.causal_linear.weight, gain=0.01)
            if self.regression_head.causal_linear.bias is not None:
                self.regression_head.causal_linear.bias.zero_()

    def forward(self, causal_loc, causal_scale):
        """
        Transform the individual causal representation distribution to output distributions.
        
        Args:
            causal_loc (torch.Tensor): Location parameters of the individual causal representation
            causal_scale (torch.Tensor): Scale parameters of the individual causal representation
        
        Returns:
            dict: A dictionary containing the parameters for classification and regression.
        """
        # 1. Classification
        cls_loc, cls_scale = self.classification_head(causal_loc, causal_scale)
        
        # 2. Regression
        reg_loc, reg_scale = self.regression_head(causal_loc, causal_scale)
        
        return {
            'cls_loc': cls_loc,
            'cls_scale': cls_scale,
            'reg_loc': reg_loc,
            'reg_scale': reg_scale
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
        cls_loc, cls_scale = outputs['cls_loc'], outputs['cls_scale']
        reg_loc, reg_scale = outputs['reg_loc'], outputs['reg_scale']
        
        # 2. Get classification prediction from the distribution
        # This computes probabilities and takes argmax
        cls_pred = self.classification_head.predict(cls_loc, cls_scale)
        
        # 3. Get regression prediction (point estimate is the location parameter)
        reg_pred = self.regression_head.predict(reg_loc, reg_scale)
        
        # 4. Get probability of <NUM> token
        all_probs = self.classification_head.compute_probabilities(cls_loc, cls_scale)
        
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

