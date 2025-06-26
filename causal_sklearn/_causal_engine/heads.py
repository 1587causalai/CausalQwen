"""
Activation Heads for CausalEngine

任务激活头，将决策得分转换为任务特定的输出。
支持回归和分类任务，以及不同的推理模式。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Optional
from enum import Enum

from .math_utils import CauchyMath


class TaskType(Enum):
    """任务类型枚举"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ActivationHead(nn.Module):
    """
    通用任务激活头
    
    将决策得分 S 转换为任务特定的输出：
    - 回归任务：直接输出或分布参数
    - 分类任务：One-vs-Rest (OvR) 概率
    
    数学原理：
    - 回归: y = μ_S (deterministic) 或 (μ_S, γ_S) (distributional)
    - 分类: P_k = 1/2 + (1/π)arctan((μ_S - C_k)/γ_S)
    
    Args:
        output_size: 输出维度（回归维度数或分类类别数）
        task_type: 任务类型 ('regression' 或 'classification')
        ovr_threshold: OvR分类阈值 C_k，默认0.0
    """
    
    def __init__(
        self,
        output_size: int,
        task_type: str,
        ovr_threshold: float = 0.0
    ):
        super().__init__()
        
        self.output_size = output_size
        self.task_type = TaskType(task_type)
        
        # OvR分类阈值（可学习参数）
        self.register_parameter(
            'ovr_threshold',
            nn.Parameter(torch.full((output_size,), ovr_threshold))
        )
    
    def forward(
        self,
        decision_scores: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        mode: str = 'standard'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        激活头前向传播
        
        Args:
            decision_scores: 决策得分
                - deterministic模式: μ_S [batch_size, output_size]
                - 其他模式: (μ_S, γ_S) 元组
            mode: 推理模式
            
        Returns:
            - 回归deterministic: 预测值 [batch_size, output_size]
            - 回归其他模式: (位置, 尺度) 分布参数
            - 分类: 类别概率 [batch_size, output_size]
        """
        if mode == 'deterministic':
            return self._deterministic_activation(decision_scores)
        else:
            return self._distributional_activation(decision_scores)
    
    def _deterministic_activation(self, mu_S: torch.Tensor) -> torch.Tensor:
        """确定性激活"""
        if self.task_type == TaskType.REGRESSION:
            # 回归：直接输出位置参数
            return mu_S
        
        elif self.task_type == TaskType.CLASSIFICATION:
            # 分类：使用softmax（与传统ML等价）
            return torch.softmax(mu_S, dim=-1)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _distributional_activation(
        self, 
        score_dist: Tuple[torch.Tensor, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """分布式激活"""
        mu_S, gamma_S = score_dist
        
        if self.task_type == TaskType.REGRESSION:
            # 回归：返回完整分布参数
            return mu_S, gamma_S
        
        elif self.task_type == TaskType.CLASSIFICATION:
            # 分类：计算OvR概率
            # P_k = P(S_k > C_k) = 1/2 + (1/π)arctan((μ_S - C_k)/γ_S)
            
            # 使用柯西分布的生存函数
            ovr_probs = CauchyMath.survival_function(
                self.ovr_threshold.unsqueeze(0).expand_as(mu_S),
                mu_S,
                gamma_S
            )
            
            return ovr_probs
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


class RegressionHead(ActivationHead):
    """
    回归任务专用激活头
    
    简化的回归激活头，专门优化回归任务
    """
    
    def __init__(self, output_size: int):
        super().__init__(output_size, 'regression')
    
    def predict(self, decision_scores, mode: str = 'standard') -> torch.Tensor:
        """
        回归预测（点估计）
        
        Args:
            decision_scores: 决策得分
            mode: 推理模式
            
        Returns:
            predictions: 预测值 [batch_size, output_size]
        """
        if mode == 'deterministic':
            return decision_scores  # μ_S
        else:
            mu_S, gamma_S = decision_scores
            return mu_S  # 使用位置参数作为点预测
    
    def predict_dist(
        self, 
        decision_scores: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        回归分布预测
        
        Args:
            decision_scores: (μ_S, γ_S) 分布参数
            
        Returns:
            (location, scale): 分布参数 [batch_size, output_size, 2]
        """
        mu_S, gamma_S = decision_scores
        return mu_S, gamma_S


class ClassificationHead(ActivationHead):
    """
    分类任务专用激活头
    
    支持二分类和多分类，使用One-vs-Rest策略
    """
    
    def __init__(
        self, 
        n_classes: int, 
        ovr_threshold: float = 0.0
    ):
        super().__init__(n_classes, 'classification', ovr_threshold)
        self.n_classes = n_classes
    
    def predict(self, decision_scores, mode: str = 'standard') -> torch.Tensor:
        """
        分类预测（类别标签）
        
        Args:
            decision_scores: 决策得分
            mode: 推理模式
            
        Returns:
            predictions: 预测类别 [batch_size]
        """
        if mode == 'deterministic':
            # 确定性模式：argmax
            return torch.argmax(decision_scores, dim=-1)
        else:
            # 分布模式：基于OvR概率的argmax
            ovr_probs = self._distributional_activation(decision_scores)
            return torch.argmax(ovr_probs, dim=-1)
    
    def predict_proba(self, decision_scores, mode: str = 'standard') -> torch.Tensor:
        """
        分类概率预测
        
        Args:
            decision_scores: 决策得分
            mode: 推理模式
            
        Returns:
            probabilities: 类别概率 [batch_size, n_classes]
        """
        if mode == 'deterministic':
            return self._deterministic_activation(decision_scores)
        else:
            return self._distributional_activation(decision_scores)
    
    def predict_dist(
        self, 
        decision_scores: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        分类分布预测（OvR激活概率）
        
        Args:
            decision_scores: (μ_S, γ_S) 分布参数
            
        Returns:
            ovr_activations: OvR激活概率 [batch_size, n_classes]
        """
        return self._distributional_activation(decision_scores)


# 损失函数
class CausalLoss:
    """
    CausalEngine统一损失函数
    
    根据任务类型和推理模式选择合适的损失函数：
    - 确定性模式：MSE (回归) / CrossEntropy (分类)
    - 分布模式：CauchyNLL (回归) / OvRBCE (分类)
    """
    
    @staticmethod
    def compute_loss(
        y_true: torch.Tensor,
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        task_type: str,
        mode: str = 'standard'
    ) -> torch.Tensor:
        """
        计算统一损失
        
        Args:
            y_true: 真实标签
            y_pred: 预测结果
            task_type: 任务类型 ('regression' 或 'classification')
            mode: 推理模式
            
        Returns:
            loss: 损失值
        """
        if mode == 'deterministic':
            return CausalLoss._deterministic_loss(y_true, y_pred, task_type)
        else:
            return CausalLoss._distributional_loss(y_true, y_pred, task_type)
    
    @staticmethod
    def _deterministic_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        task_type: str
    ) -> torch.Tensor:
        """确定性模式损失"""
        if task_type == 'regression':
            # MSE损失
            return torch.mean((y_true - y_pred) ** 2)
        
        elif task_type == 'classification':
            # CrossEntropy损失
            return torch.nn.functional.cross_entropy(y_pred, y_true)
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def _distributional_loss(
        y_true: torch.Tensor,
        y_pred: Tuple[torch.Tensor, torch.Tensor],
        task_type: str
    ) -> torch.Tensor:
        """分布模式损失"""
        if task_type == 'regression':
            # 柯西NLL损失
            mu_pred, gamma_pred = y_pred
            return CauchyMath.nll_loss(y_true, mu_pred, gamma_pred)
        
        elif task_type == 'classification':
            # OvR二元交叉熵损失
            ovr_probs = y_pred  # 已经是概率
            
            # 转换为one-hot编码
            if y_true.dim() == 1:
                y_true_onehot = torch.nn.functional.one_hot(
                    y_true, num_classes=ovr_probs.shape[-1]
                ).float()
            else:
                y_true_onehot = y_true.float()
            
            # 二元交叉熵：-[y*log(p) + (1-y)*log(1-p)]
            eps = 1e-8
            ovr_probs = torch.clamp(ovr_probs, eps, 1-eps)
            
            bce_loss = -(y_true_onehot * torch.log(ovr_probs) + 
                        (1 - y_true_onehot) * torch.log(1 - ovr_probs))
            
            return torch.mean(torch.sum(bce_loss, dim=-1))
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")