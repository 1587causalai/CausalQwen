"""
CausalEngine Core Implementation for sklearn-compatible ML tasks

╔════════════════════════════════════════════════════════════════════════════════╗
║                           CausalEngine 四阶段架构流程图                           ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  Input Features (X)                                                            ║
║         │                                                                      ║
║         ▼                                                                      ║
║  ┌─────────────────┐    1. PERCEPTION 感知阶段                                  ║
║  │  Perception     │    ─────────────────────                                  ║
║  │   Network       │    X → Z (特征到高级表征)                                   ║
║  │  (MLP layers)   │    • 多层感知器网络                                          ║
║  └─────────────────┘    • 可配置隐藏层和激活函数                                  ║
║         │                                                                      ║
║         ▼                                                                      ║
║  High-level Representation (Z)                                                ║
║         │                                                                      ║
║         ▼                                                                      ║
║  ┌─────────────────┐    2. ABDUCTION 归因阶段                                   ║
║  │  Abduction      │    ──────────────────────                                 ║
║  │   Network       │    Z → U ~ Cauchy(μ_U, γ_U)                              ║
║  │ (Cauchy Dist)   │    • 推断个体因果表征                                        ║
║  └─────────────────┘    • 输出柯西分布参数                                        ║
║         │                                                                      ║
║         ▼                                                                      ║
║  Causal Representation (U)                                                    ║
║         │                                                                      ║
║         ▼                                                                      ║
║  ┌─────────────────┐    3. ACTION 行动阶段                                      ║
║  │   Action        │    ───────────────────                                    ║
║  │   Network       │    U → S ~ Cauchy(μ_S, γ_S)                              ║
║  │ (Linear + Noise)│    • 个体表征到决策得分                                      ║
║  └─────────────────┘    • 加入可训练外生噪声                                      ║
║         │                                                                      ║
║         ▼                                                                      ║
║  Decision Scores (S)                                                           ║
║         │                                                                      ║
║         ▼                                                                      ║
║  ┌─────────────────┐    4. DECISION 决断阶段                                    ║
║  │  Task Head      │    ─────────────────────                                  ║
║  │ (Regression/    │    S → Y (决策得分到任务输出)                                ║
║  │  Classification)│    • 回归：直接输出 / 柯西NLL                                ║
║  └─────────────────┘    • 分类：Softmax / OvR概率                               ║
║         │                                                                      ║
║         ▼                                                                      ║
║  Final Output (Y)                                                              ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         五种推理模式                                         │  ║
║  │  • deterministic: 确定性模式（U' = μ_U，无随机性）                            │  ║
║  │  • exogenous: 外生模式（U' ~ Cauchy(μ_U, |b_noise|)，外生噪声主导）          │  ║    注：也可以理解成 endogenous 的特例（样本间共享尺度模式）
║  │  • endogenous: 内生模式（U' ~ Cauchy(μ_U, γ_U)，内生不确定性主导）           │  ║
║  │  • standard: 标准模式（U' ~ Cauchy(μ_U, γ_U + |b_noise|)，内生+外生叠加）    │  ║
║  │  • sampling: 采样模式（U' ~ Cauchy(μ_U + b_noise*ε, γ_U)，位置参数扰动）      │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  核心特性：                                                                      ║
║  • 线性稳定性：利用柯西分布的数学性质实现解析计算                                  ║
║  • 可解释性：每个阶段都有明确的因果含义                                           ║
║  • 灵活性：支持回归和分类任务，多种推理模式                                        ║
║  • sklearn兼容：提供fit/predict等标准接口                                        ║
╚════════════════════════════════════════════════════════════════════════════════╝

这个模块包含完整的CausalEngine实现，集成四阶段架构：
1. Perception: 从特征提取高级表征
2. Abduction: 从表征推断个体因果表征
3. Action: 从个体表征到决策得分
4. Decision: 从决策得分到任务输出

专注于常规ML任务（分类/回归），简化大模型相关复杂性。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

from .networks import Perception, Abduction, Action
from .heads import TaskType, create_task_head


class CausalEngine(nn.Module):
    """
    CausalEngine完整实现
    
    基于因果推理的四阶段神经网络架构：
    X → [Perception] → Z → [Abduction] → U → [Action] → S → [Decision] → Y
    
    数学框架：
    1. 感知阶段 (Perception): X → Z (特征到高级表征)
    2. 归因阶段 (Abduction): Z → U ~ Cauchy(μ_U, γ_U) (表征到个体表征)
    3. 行动阶段 (Action): U → S ~ Cauchy(μ_S, γ_S) (个体表征到决策得分)  
    4. 决断阶段 (Decision): S → Y (决策得分到任务输出)
    
    Args:
        input_size: 输入特征维度
        output_size: 输出维度（回归维度数或分类类别数）
        repre_size: 内部表征Z的维度
        causal_size: 因果表征U的维度
        task_type: 任务类型 ('regression' 或 'classification')
        perception_hidden_layers: Perception网络的隐藏层配置
        abduction_hidden_layers: Abduction网络的隐藏层配置
        activation: 激活函数名称
        dropout: dropout比率
        gamma_init: 尺度参数初始化值
        b_noise_init: 外生噪声初始化值
        b_noise_trainable: 外生噪声是否可训练
        ovr_threshold: 分类任务的OvR阈值
        learnable_threshold: 是否可学习阈值
        alpha: 正则化参数
    """
    
    def __init__(
        self,
        input_size: int, # 输入特征的维度
        output_size: int, # 输出维度（回归维度数或分类类别数）
        repre_size: Optional[int] = None, # 表征的维度
        causal_size: Optional[int] = None, # 因果表征的维度
        task_type: str = 'regression', # 任务类型
        perception_hidden_layers: Optional[Tuple[int, ...]] = None, # 感知网络的隐藏层配置
        abduction_hidden_layers: Optional[Tuple[int, ...]] = None, # 归因网络的隐藏层配置
        activation: str = 'relu', 
        dropout: float = 0.0, # dropout比率
        gamma_init: float = 10.0, # 尺度参数初始化值
        b_noise_init: float = 0.1, # 外生噪声初始化值
        b_noise_trainable: bool = True, # 外生噪声是否可训练
        ovr_threshold: float = 0.0, # 分类任务的OvR阈值初始化
        learnable_threshold: bool = False, # 是否可学习阈值
        alpha: float = 0.0 # L2正则化参数
    ):
        super().__init__()
        
        # --- 参数维度处理 ---
        if repre_size is None:
            repre_size = max(input_size, output_size)
        if causal_size is None:
            causal_size = repre_size

        self.input_size = input_size
        self.output_size = output_size
        self.repre_size = repre_size
        self.causal_size = causal_size
        self.task_type = TaskType(task_type)
        
        # 四阶段网络架构
        
        # 1. 感知网络 (Perception): X → Z
        self.perception_net = Perception(
            input_size=input_size,
            repre_size=repre_size,
            hidden_layers=perception_hidden_layers,
            activation=activation,
            dropout=dropout
        )
        
        # 2. 归因网络 (Abduction): Z → U ~ Cauchy(μ_U, γ_U)
        self.abduction_net = Abduction(
            repre_size=repre_size,
            causal_size=causal_size,
            hidden_layers=abduction_hidden_layers,
            activation=activation,
            dropout=dropout,
            gamma_init=gamma_init
        )
        
        # 3. 行动网络 (Action): U → S ~ Cauchy(μ_S, γ_S)
        self.action_net = Action(
            causal_size=causal_size,
            output_size=output_size,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable
        )
        
        # 4. 决断头 (Decision): S → Y
        self.decision_head = create_task_head(
            output_size=self.output_size,
            task_type=self.task_type.value,
            ovr_threshold=ovr_threshold,
            learnable_threshold=learnable_threshold
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=alpha)
    
    def _get_decision_scores(
        self, 
        x: torch.Tensor, 
        mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """内部方法：计算决策得分并确保其格式统一"""
        z = self.perception_net(x)
        mu_U, gamma_U = self.abduction_net(z)
        decision_scores_raw = self.action_net(mu_U, gamma_U, mode)
        
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
        """CausalEngine前向传播"""
        decision_scores = self._get_decision_scores(x, mode)
        output = self.decision_head(decision_scores, mode)
        return output
    
    def predict(self, x: torch.Tensor, mode: str = 'standard') -> torch.Tensor:
        """预测方法（sklearn兼容接口）"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, mode)
            if self.task_type == TaskType.REGRESSION:
                return output
            elif self.task_type == TaskType.CLASSIFICATION:
                return torch.argmax(output, dim=-1)
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
    
    def predict_proba(self, x: torch.Tensor, mode: str = 'standard') -> torch.Tensor:
        """分类概率预测（仅分类任务）"""
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification tasks")
        
        self.eval()
        with torch.no_grad():
            return self.forward(x, mode)
    
    def predict_distribution(
        self, 
        x: torch.Tensor, 
        mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分布预测（返回完整分布参数）"""
        if mode == 'deterministic':
            raise ValueError("Distribution prediction not available in deterministic mode")
        
        self.eval()
        with torch.no_grad():
            return self._get_decision_scores(x, mode)
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        mode: str = 'standard'
    ) -> torch.Tensor:
        """计算损失函数"""
        decision_scores = self._get_decision_scores(x, mode)
        loss = self.decision_head.compute_loss(
            y_true=y,
            decision_scores=decision_scores,
            mode=mode
        )
        return loss
    
    def get_causal_representation(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取因果表征（中间表示）"""
        self.eval()
        with torch.no_grad():
            z = self.perception_net(x)
            mu_U, gamma_U = self.abduction_net(z)
            return mu_U, gamma_U
    
    def get_decision_scores(
        self, 
        x: torch.Tensor, 
        mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取决策得分（中间表示），始终返回分布参数。"""
        self.eval()
        with torch.no_grad():
            return self._get_decision_scores(x, mode)

def create_causal_regressor(
    input_size: int,
    output_size: int = 1,
    repre_size: Optional[int] = None,
    causal_size: Optional[int] = None,
    **kwargs
) -> CausalEngine:
    """创建因果回归器"""
    return CausalEngine(
        input_size=input_size,
        output_size=output_size,
        repre_size=repre_size,
        causal_size=causal_size,
        task_type='regression',
        **kwargs
    )

def create_causal_classifier(
    input_size: int,
    n_classes: int,
    repre_size: Optional[int] = None,
    causal_size: Optional[int] = None,
    **kwargs
) -> CausalEngine:
    """创建因果分类器"""
    return CausalEngine(
        input_size=input_size,
        output_size=n_classes,
        repre_size=repre_size,
        causal_size=causal_size,
        task_type='classification',
        **kwargs
    )
