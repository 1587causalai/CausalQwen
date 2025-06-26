"""
CausalEngine Core Module

因果推理引擎的核心实现，包含三阶段架构：
1. AbductionNetwork: 归因网络（证据→个体表征）
2. ActionNetwork: 行动网络（个体表征→决策得分）
3. ActivationHead: 激活头（决策得分→任务输出）

以及完整的CausalEngine实现和数学工具。
"""

from .engine import (
    CausalEngine,
    create_causal_regressor,
    create_causal_classifier
)

from .networks import (
    AbductionNetwork,
    ActionNetwork
)

from .heads import (
    ActivationHead,
    RegressionHead,
    ClassificationHead,
    CausalLoss,
    TaskType
)

from .math_utils import (
    CauchyMath,
    CauchyMathNumpy
)

__all__ = [
    # Core Engine
    'CausalEngine',
    'create_causal_regressor', 
    'create_causal_classifier',
    
    # Networks
    'AbductionNetwork',
    'ActionNetwork',
    
    # Activation Heads
    'ActivationHead',
    'RegressionHead',
    'ClassificationHead',
    'CausalLoss',
    'TaskType',
    
    # Math Utils
    'CauchyMath',
    'CauchyMathNumpy'
]