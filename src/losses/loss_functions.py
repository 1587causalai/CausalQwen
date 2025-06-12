"""
损失函数模块

本模块实现了因果语言模型的核心损失函数，包括：
- OvR分类损失
- 柯西回归损失
- 门控机制
- 总损失计算
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Literal


def cauchy_nll_loss(loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor,
                    reduction: Literal['none', 'mean', 'sum'] = 'mean') -> torch.Tensor:
    """
    计算柯西分布的负对数似然损失。
    
    损失公式：
    L = log(π * scale) + log(1 + ((target - loc) / scale)²)
    
    Args:
        loc: 位置参数
        scale: 尺度参数
        target: 目标值
        reduction: 归约方式
        
    Returns:
        损失值
    """
    # 确保scale为正
    scale = torch.abs(scale) + 1e-8
    
    # 计算标准化残差
    z = (target - loc) / scale
    
    # 柯西NLL: log(π * scale) + log(1 + z²)
    loss = torch.log(math.pi * scale) + torch.log1p(z ** 2)
    
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def ovr_classification_loss(probs: torch.Tensor, targets: torch.Tensor,
                           reduction: Literal['none', 'mean', 'sum'] = 'mean') -> torch.Tensor:
    """
    计算OvR（One-vs-Rest）分类损失。
    
    每个类别独立进行二元分类，使用二元交叉熵损失。
    
    Args:
        probs: 各类别的概率 [batch_size, seq_len, vocab_size]
        targets: 目标类别索引 [batch_size, seq_len]
        reduction: 归约方式
        
    Returns:
        OvR分类损失
    """
    batch_size, seq_len, num_classes = probs.shape
    
    # 创建one-hot目标
    targets_flat = targets.view(-1)
    targets_onehot = F.one_hot(targets_flat, num_classes).float()
    targets_onehot = targets_onehot.view(batch_size, seq_len, num_classes)
    
    # 二元交叉熵损失
    # 对于真实类别，我们希望P(S_k > C_k)接近1
    # 对于其他类别，我们希望P(S_k > C_k)接近0
    eps = 1e-9
    bce_loss = -(
        targets_onehot * torch.log(probs + eps) +
        (1 - targets_onehot) * torch.log(1 - probs + eps)
    )
    
    # 对类别维度求和（每个位置的总损失）
    loss_per_position = bce_loss.sum(dim=-1)  # [batch_size, seq_len]
    
    if reduction == 'none':
        return loss_per_position
    elif reduction == 'mean':
        return loss_per_position.mean()
    elif reduction == 'sum':
        return loss_per_position.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def regression_loss(loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor,
                   reduction: Literal['none', 'mean', 'sum'] = 'mean') -> torch.Tensor:
    """
    计算回归损失（柯西负对数似然）。
    
    这是 cauchy_nll_loss 的别名，为了接口一致性。
    """
    return cauchy_nll_loss(loc, scale, target, reduction)


def gated_regression_loss(loc: torch.Tensor, scale: torch.Tensor, y_true: torch.Tensor,
                         gate_prob: torch.Tensor, mask: Optional[torch.Tensor] = None,
                         alpha: float = 1.0, reduction: Literal['none', 'mean', 'sum'] = 'mean') -> torch.Tensor:
    """
    计算门控回归损失（支持混合门控策略）。
    
    数学公式：
    - alpha = 1.0: L_gated = mask * L_cauchy_nll （无门控）
    - alpha = 0.0: L_gated = mask * gate_prob * L_cauchy_nll （完全门控）
    - 0 < alpha < 1: L_gated = mask * (alpha + (1-alpha) * gate_prob) * L_cauchy_nll （混合门控）
    
    Args:
        loc: 回归值的位置参数
        scale: 回归值的尺度参数  
        y_true: 真实回归值
        gate_prob: 门控概率（<NUM> token 的预测概率）
        mask: 指示函数，标识真实标签是否为 <NUM>
        alpha: 门控系数 (1.0 = 无门控, 0.0 = 完全门控)
        reduction: 归约方式
        
    Returns:
        门控后的回归损失
    """
    # 计算基础回归损失（不进行 reduction）
    base_loss = regression_loss(loc, scale, y_true, reduction='none')
    
    # 计算门控权重
    if alpha == 1.0:
        # 无门控：仅使用掩码
        gate_weight = 1.0 if mask is None else mask.float()
    elif alpha == 0.0:
        # 完全门控：使用概率门控
        gate_weight = gate_prob if mask is None else mask.float() * gate_prob
    else:
        # 混合门控：alpha + (1-alpha) * gate_prob
        if mask is None:
            gate_weight = alpha + (1 - alpha) * gate_prob
        else:
            gate_weight = mask.float() * (alpha + (1 - alpha) * gate_prob)
    
    # 应用门控
    gated_loss = gate_weight * base_loss
    
    # 应用 reduction
    if reduction == 'none':
        return gated_loss
    elif reduction == 'mean':
        # 计算有效损失的平均值（忽略权重为0的位置）
        if mask is not None:
            num_valid = mask.float().sum()
            if num_valid > 0:
                return gated_loss.sum() / num_valid
            else:
                return torch.tensor(0.0, device=gated_loss.device)
        else:
            return gated_loss.mean()
    elif reduction == 'sum':
        return gated_loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def compute_total_loss(cls_probs: torch.Tensor, cls_targets: torch.Tensor,
                      reg_loc: torch.Tensor, reg_scale: torch.Tensor, reg_targets: torch.Tensor,
                      num_probs: torch.Tensor, num_mask: torch.Tensor,
                      cls_weight: float = 1.0, reg_weight: float = 1.0,
                      gating_alpha: float = 1.0) -> dict:
    """
    计算总损失函数（支持混合门控策略）。
    
    Args:
        cls_probs: 分类概率 [batch_size, seq_len, vocab_size]
        cls_targets: 分类目标 [batch_size, seq_len]
        reg_loc: 回归位置参数 [batch_size, seq_len]
        reg_scale: 回归尺度参数 [batch_size, seq_len]
        reg_targets: 回归目标 [batch_size, seq_len]
        num_probs: <NUM> token 的预测概率 [batch_size, seq_len]
        num_mask: <NUM> token 的掩码 [batch_size, seq_len]
        cls_weight: 分类损失权重
        reg_weight: 回归损失权重
        gating_alpha: 门控系数 (1.0 = 无门控, 0.0 = 完全门控)
        
    Returns:
        Dict containing total loss and component losses
    """
    # 1. 分类损失（所有位置都计算）
    cls_loss = ovr_classification_loss(cls_probs, cls_targets, reduction='none')
    
    # 2. 门控回归损失
    reg_loss = gated_regression_loss(
        reg_loc, reg_scale, reg_targets, 
        num_probs, num_mask, 
        alpha=gating_alpha,
        reduction='none'
    )
    
    # 3. 计算门控权重（用于监控）
    if gating_alpha == 1.0:
        gate_weights = num_mask.float()
    elif gating_alpha == 0.0:
        gate_weights = num_mask.float() * num_probs
    else:
        gate_weights = num_mask.float() * (gating_alpha + (1 - gating_alpha) * num_probs)
    
    # 4. 聚合损失
    cls_loss_mean = cls_loss.mean()
    reg_loss_mean = reg_loss.mean()
    
    # 5. 总损失
    total_loss = cls_weight * cls_loss_mean + reg_weight * reg_loss_mean
    
    return {
        'total': total_loss,
        'cls': cls_loss_mean,
        'reg': reg_loss_mean,
        # 提供更详细的统计信息
        'cls_per_position': cls_loss,  # [batch_size, seq_len]
        'reg_per_position': reg_loss,  # [batch_size, seq_len]
        'gate_weights': gate_weights,  # [batch_size, seq_len]
        'num_positions': num_mask.sum(),  # 实际的数值位置数量
        'effective_reg_loss': reg_loss.sum() / (num_mask.sum() + 1e-8),  # 每个数值位置的平均损失
        'avg_gate_weight': gate_weights.sum() / (num_mask.sum() + 1e-8),  # 平均门控权重
        'gating_alpha': gating_alpha  # 记录使用的门控系数
    }

# 为了支持更灵活的损失计算，添加一个辅助函数
def masked_mean(tensor, mask, dim=None, keepdim=False):
    """
    计算掩码张量的均值。
    
    Args:
        tensor: 输入张量
        mask: 二元掩码，相同形状
        dim: 沿哪个维度计算均值
        keepdim: 是否保持维度
        
    Returns:
        掩码均值
    """
    if dim is None:
        # 全局均值
        return (tensor * mask).sum() / (mask.sum() + 1e-8)
    else:
        # 沿指定维度的均值
        return (tensor * mask).sum(dim=dim, keepdim=keepdim) / (mask.sum(dim=dim, keepdim=keepdim) + 1e-8)