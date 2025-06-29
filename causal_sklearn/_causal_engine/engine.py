"""
CausalEngine Core Implementation for sklearn-compatible ML tasks

这个模块包含完整的CausalEngine实现，集成三阶段架构：
1. AbductionNetwork: 从特征推断个体因果表征
2. ActionNetwork: 从个体表征到决策得分  
3. ActivationHead: 从决策得分到任务输出

专注于常规ML任务（分类/回归），简化大模型相关复杂性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional

from .networks import AbductionNetwork, ActionNetwork
from .heads import TaskType, create_task_head, TaskHead


class CausalEngine(nn.Module):
    """
    CausalEngine完整实现
    
    基于因果推理的三阶段神经网络架构：
    X → [AbductionNetwork] → U → [ActionNetwork] → S → [ActivationHead] → Y
    
    数学框架：
    1. 归因阶段：X → U ~ Cauchy(μ_U, γ_U) (证据到个体表征)
    2. 行动阶段：U → S ~ Cauchy(μ_S, γ_S) (个体表征到决策得分)  
    3. 激活阶段：S → Y (决策得分到任务输出)
    
    五种推理模式：
    - deterministic: 确定性推理，等价于传统ML
    - exogenous: 外生噪声主导的推理
    - endogenous: 内生不确定性主导的推理
    - standard: 内生+外生混合推理
    - sampling: 采样式因果推理
    
    Args:
        input_size: 输入特征维度
        output_size: 输出维度（回归维度数或分类类别数）
        causal_size: 因果表征维度
        task_type: 任务类型 ('regression' 或 'classification')
        abd_hidden_layers: AbductionNetwork隐藏层配置
        activation: 激活函数名称
        dropout: dropout比率
        gamma_init: 尺度参数初始化值
        b_noise_init: 外生噪声初始化值
        b_noise_trainable: 外生噪声是否可训练
        ovr_threshold: 分类任务的OvR阈值
        learnable_threshold: 是否可学习阈值
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        causal_size: int = None,
        task_type: str = 'regression',
        abd_hidden_layers: Tuple[int, ...] = (),
        activation: str = 'relu',
        dropout: float = 0.0,
        gamma_init: float = 10.0,
        b_noise_init: float = 0.1,
        b_noise_trainable: bool = True,
        ovr_threshold: float = 0.0,
        learnable_threshold: bool = False
    ):
        super().__init__()
        
        # 默认因果维度
        if causal_size is None:
            causal_size = max(input_size, output_size)
        
        self.input_size = input_size
        self.output_size = output_size
        self.causal_size = causal_size
        self.task_type = TaskType(task_type)
        
        # 三阶段网络架构
        
        # 1. 归因网络：X → U ~ Cauchy(μ_U, γ_U)
        self.abduction_net = AbductionNetwork(
            input_size=input_size,
            causal_size=causal_size,
            abd_hidden_layers=abd_hidden_layers,
            activation=activation,
            dropout=dropout,
            gamma_init=gamma_init
        )
        
        # 2. 行动网络：U → S ~ Cauchy(μ_S, γ_S)
        self.action_net = ActionNetwork(
            causal_size=causal_size,
            output_size=output_size,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable
        )
        
        # 3. 任务头：S → Y
        self.task_head = create_task_head(
            output_size=self.output_size,
            task_type=self.task_type.value,
            ovr_threshold=ovr_threshold,
            learnable_threshold=learnable_threshold
        )
    
    def _get_decision_scores(
        self, 
        x: torch.Tensor, 
        mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """内部方法：计算决策得分并确保其格式统一"""
        # 阶段1：归因推断 X → U
        mu_U, gamma_U = self.abduction_net(x)
        
        # 阶段2：行动决策 U → S
        decision_scores_raw = self.action_net(mu_U, gamma_U, mode)
        
        # 确保输出始终是 (mu, gamma) 元组，以统一接口
        if mode == 'deterministic':
            mu_S = decision_scores_raw
            gamma_S = torch.zeros_like(mu_S)
            return mu_S, gamma_S
        else:
            return decision_scores_raw

    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'standard'
    ) -> torch.Tensor:
        """
        CausalEngine前向传播
        
        Args:
            x: 输入特征 [batch_size, input_size]
            mode: 推理模式 ('deterministic', 'exogenous', 'endogenous', 'standard', 'sampling')
            
        Returns:
            任务特定的预测结果（例如，回归值、类别概率）。
        """
        # 获取决策得分
        decision_scores = self._get_decision_scores(x, mode)
        
        # 阶段3：任务激活 S → Y
        output = self.task_head(decision_scores, mode)
        
        return output
    
    def predict(self, x: torch.Tensor, mode: str = 'standard') -> torch.Tensor:
        """
        预测方法（sklearn兼容接口）
        
        Args:
            x: 输入特征 [batch_size, input_size]
            mode: 推理模式
            
        Returns:
            predictions: 预测结果
                - 回归: 预测值 [batch_size, output_size]
                - 分类: 预测类别 [batch_size]
        """
        self.eval()
        with torch.no_grad():
            # forward() now returns the final prediction/probabilities
            output = self.forward(x, mode)

            if self.task_type == TaskType.REGRESSION:
                # 回归: forward() 直接返回点预测值
                return output
            
            elif self.task_type == TaskType.CLASSIFICATION:
                # 分类: forward() 返回概率, 在此基础上 argmax
                return torch.argmax(output, dim=-1)
            
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
    
    def predict_proba(self, x: torch.Tensor, mode: str = 'standard') -> torch.Tensor:
        """
        分类概率预测（仅分类任务）
        
        Args:
            x: 输入特征 [batch_size, input_size]
            mode: 推理模式
            
        Returns:
            probabilities: 类别概率 [batch_size, output_size]
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification tasks")
        
        self.eval()
        with torch.no_grad():
            # forward() in classification mode directly returns probabilities
            return self.forward(x, mode)
    
    def predict_distribution(
        self, 
        x: torch.Tensor, 
        mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分布预测（返回完整分布参数）
        
        Args:
            x: 输入特征 [batch_size, input_size]
            mode: 推理模式（不能是deterministic）
            
        Returns:
            location: 位置参数 [batch_size, output_size]
            scale: 尺度参数 [batch_size, output_size]
        """
        if mode == 'deterministic':
            raise ValueError("Distribution prediction not available in deterministic mode")
        
        self.eval()
        with torch.no_grad():
            # 直接返回决策得分的分布
            return self._get_decision_scores(x, mode)
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        mode: str = 'standard'
    ) -> torch.Tensor:
        """
        计算损失函数
        
        Args:
            x: 输入特征 [batch_size, input_size]
            y: 真实标签 [batch_size, output_size] (回归) 或 [batch_size] (分类)
            mode: 推理模式
            
        Returns:
            loss: 损失值
        """
        # 获取决策得分
        decision_scores = self._get_decision_scores(x, mode)
        
        # 将损失计算委托给任务头
        loss = self.task_head.compute_loss(
            y_true=y,
            decision_scores=decision_scores,
            mode=mode
        )
        
        return loss
    
    def get_causal_representation(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取因果表征（中间表示）
        
        Args:
            x: 输入特征 [batch_size, input_size]
            
        Returns:
            mu_U: 个体表征位置参数 [batch_size, causal_size]
            gamma_U: 个体表征尺度参数 [batch_size, causal_size]
        """
        self.eval()
        with torch.no_grad():
            mu_U, gamma_U = self.abduction_net(x)
            return mu_U, gamma_U
    
    def get_decision_scores(
        self, 
        x: torch.Tensor, 
        mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取决策得分（中间表示），始终返回分布参数。
        
        Args:
            x: 输入特征 [batch_size, input_size]
            mode: 推理模式
            
        Returns:
            (位置, 尺度) 决策得分分布参数
        """
        self.eval()
        with torch.no_grad():
            return self._get_decision_scores(x, mode)
    
    def set_inference_mode(self, mode: str):
        """
        设置默认推理模式（便利方法）
        
        Args:
            mode: 推理模式
        """
        valid_modes = {'deterministic', 'exogenous', 'endogenous', 'standard', 'sampling'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        
        self._default_mode = mode
    
    def get_architecture_info(self) -> dict:
        """
        获取架构信息
        
        Returns:
            info: 架构信息字典
        """
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'causal_size': self.causal_size,
            'task_type': self.task_type.value,
            'task_head': self.task_head.__class__.__name__,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
    def get_mathematical_summary(self) -> str:
        """
        获取数学框架摘要
        
        Returns:
            summary: 数学框架描述
        """
        return f"""
CausalEngine Mathematical Framework:
=====================================

Architecture: {self.input_size} → {self.causal_size} → {self.output_size}
Task Type: {self.task_type.value.title()}
Task Head: {self.task_head.__class__.__name__}

Three-Stage Pipeline:
1. Abduction: X ∈ R^{self.input_size} → U ~ Cauchy(μ_U, γ_U) ∈ R^{self.causal_size}
2. Action: U → S ~ Cauchy(μ_S, γ_S) ∈ R^{self.output_size}  
3. Activation: S → Y ∈ R^{self.output_size}

Mathematical Properties:
- Causal Reasoning: Based on Structural Causal Models
- Analytical Computation: Leverages Cauchy distribution stability
- Heavy-tail Robustness: Natural handling of outliers
- Scale Invariance: Consistent behavior across scales

Inference Modes:
- deterministic: Traditional ML equivalent (μ_U only)
- exogenous: External noise dominated (replaces γ_U with |b_noise|)
- endogenous: Internal uncertainty only (γ_U without b_noise)
- standard: Mixed internal+external (γ_U + |b_noise|)
- sampling: Sampling-based causal inference

Total Parameters: {sum(p.numel() for p in self.parameters())}
"""


# 便利的工厂函数
def create_causal_regressor(
    input_size: int,
    output_size: int = 1,
    causal_size: int = None,
    **kwargs
) -> CausalEngine:
    """
    创建因果回归器
    
    Args:
        input_size: 输入特征维度
        output_size: 输出维度（默认1）
        causal_size: 因果维度
        **kwargs: 其他CausalEngine参数
        
    Returns:
        engine: CausalEngine回归器
    """
    return CausalEngine(
        input_size=input_size,
        output_size=output_size,
        causal_size=causal_size,
        task_type='regression',
        **kwargs
    )


def create_causal_classifier(
    input_size: int,
    n_classes: int,
    causal_size: int = None,
    **kwargs
) -> CausalEngine:
    """
    创建因果分类器
    
    Args:
        input_size: 输入特征维度
        n_classes: 类别数量
        causal_size: 因果维度
        **kwargs: 其他CausalEngine参数 (例如 ovr_threshold, learnable_threshold)
        
    Returns:
        engine: CausalEngine分类器
    """
    return CausalEngine(
        input_size=input_size,
        output_size=n_classes,
        causal_size=causal_size,
        task_type='classification',
        **kwargs
    )
