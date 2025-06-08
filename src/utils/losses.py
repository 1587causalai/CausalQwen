"""
损失函数实现

包含OvR分类损失和门控回归损失的实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .cauchy import log_cauchy_pdf, threshold_probability


class OvRClassificationLoss(nn.Module):
    """
    One-vs-Rest (OvR) 分类损失
    
    对每个类别独立计算二元分类损失，而不是使用传统的softmax交叉熵。
    这种方法更适合柯西分布的重尾特性。
    """
    
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, cls_loc: torch.Tensor, cls_scale: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算OvR分类损失
        
        Args:
            cls_loc: 分类决策分数的位置参数 [batch_size, vocab_size]
            cls_scale: 分类决策分数的尺度参数 [batch_size, vocab_size]
            targets: 目标类别索引 [batch_size]
        
        Returns:
            分类损失
        """
        batch_size, vocab_size = cls_loc.shape
        
        # 计算每个类别超过阈值的概率
        probs = threshold_probability(cls_loc, cls_scale, self.threshold)
        
        # 创建one-hot标签
        targets_onehot = F.one_hot(targets, vocab_size).float()
        
        # 对于正类，我们希望概率接近1；对于负类，我们希望概率接近0
        # 使用二元交叉熵损失
        loss = F.binary_cross_entropy(probs, targets_onehot, reduction='mean')
        
        return loss


class GatedRegressionLoss(nn.Module):
    """
    门控回归损失
    
    只有当模型正确预测出<NUM>词元时，才计算回归损失。
    这避免了在不应该预测数值的情况下强制模型学习数值。
    """
    
    def __init__(self, num_token_id: int):
        super().__init__()
        self.num_token_id = num_token_id
        
    def forward(self, reg_loc: torch.Tensor, reg_scale: torch.Tensor,
                cls_probs: torch.Tensor, targets: torch.Tensor, 
                reg_targets: torch.Tensor) -> torch.Tensor:
        """
        计算门控回归损失
        
        Args:
            reg_loc: 回归预测的位置参数 [batch_size]
            reg_scale: 回归预测的尺度参数 [batch_size]
            cls_probs: 分类概率 [batch_size, vocab_size]
            targets: 分类目标 [batch_size]
            reg_targets: 回归目标 [batch_size]
        
        Returns:
            门控回归损失
        """
        # 创建门控掩码：只有当目标是<NUM>词元时才计算回归损失
        gate_mask = (targets == self.num_token_id).float()
        
        # 计算回归损失：使用柯西分布的负对数似然
        reg_loss = -log_cauchy_pdf(reg_targets, reg_loc, reg_scale)
        
        # 应用门控
        gated_loss = reg_loss * gate_mask
        
        # 只对有效样本计算平均损失
        if gate_mask.sum() > 0:
            return gated_loss.sum() / gate_mask.sum()
        else:
            return torch.tensor(0.0, device=reg_loss.device)


class CausalLanguageModelLoss(nn.Module):
    """
    因果语言模型的组合损失函数
    
    结合OvR分类损失和门控回归损失。
    """
    
    def __init__(self, num_token_id: int, cls_weight: float = 1.0, 
                 reg_weight: float = 1.0, threshold: float = 0.0):
        super().__init__()
        self.cls_loss = OvRClassificationLoss(threshold)
        self.reg_loss = GatedRegressionLoss(num_token_id)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                cls_targets: torch.Tensor, reg_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            predictions: 模型预测结果
            cls_targets: 分类目标 [batch_size]
            reg_targets: 回归目标 [batch_size]
        
        Returns:
            损失字典
        """
        # 分类损失
        cls_loss = self.cls_loss(
            predictions['cls_loc'], 
            predictions['cls_scale'], 
            cls_targets
        )
        
        # 回归损失
        reg_loss = self.reg_loss(
            predictions['reg_loc'],
            predictions['reg_scale'],
            predictions['cls_probs'],
            cls_targets,
            reg_targets
        )
        
        # 总损失
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }


class KLDivergenceLoss(nn.Module):
    """
    柯西分布之间的KL散度损失
    
    用于正则化或对比学习。注意柯西分布的KL散度没有闭式解，
    这里使用蒙特卡洛估计。
    """
    
    def __init__(self, num_samples: int = 100):
        super().__init__()
        self.num_samples = num_samples
        
    def forward(self, loc1: torch.Tensor, scale1: torch.Tensor,
                loc2: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
        """
        计算两个柯西分布之间的KL散度
        
        Args:
            loc1, scale1: 第一个分布的参数
            loc2, scale2: 第二个分布的参数
        
        Returns:
            KL散度的蒙特卡洛估计
        """
        # 从第一个分布采样
        samples = cauchy_sample(loc1, scale1, (self.num_samples,) + loc1.shape)
        
        # 计算对数概率
        log_p = log_cauchy_pdf(samples, loc1, scale1)
        log_q = log_cauchy_pdf(samples, loc2, scale2)
        
        # KL散度的蒙特卡洛估计
        kl_div = (log_p - log_q).mean(dim=0)
        
        return kl_div.mean()


def focal_loss(probs: torch.Tensor, targets: torch.Tensor, 
               alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss实现
    
    用于处理类别不平衡问题，特别适用于<NUM>词元相对稀少的情况。
    
    Args:
        probs: 预测概率 [batch_size, num_classes]
        targets: 目标标签 [batch_size]
        alpha: 平衡因子
        gamma: 聚焦参数
    
    Returns:
        Focal loss
    """
    ce_loss = F.cross_entropy(probs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()

