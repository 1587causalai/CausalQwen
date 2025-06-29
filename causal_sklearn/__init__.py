"""
Causal-Sklearn: Scikit-learn Compatible Causal Machine Learning

基于 CausalEngine™ 算法的 scikit-learn 兼容实现，实现真正的因果推理机器学习。

CausalEngine 核心创新：
- 🧠 因果推理：理解 Y = f(U, ε) 而非学习 P(Y|X)
- 🎯 三阶段架构：归因推断 → 行动决策 → 任务激活
- 📐 柯西数学：重尾分布 + 线性稳定性 = 解析计算
- 🔧 五种模式：deterministic/exogenous/endogenous/standard/sampling
- ⚡ 无需采样：完全解析化的不确定性传播

理论基础：Distribution-consistency Structural Causal Models (arXiv:2401.15911)
数学文档：docs/MATHEMATICAL_FOUNDATIONS_CN.md
"""

from ._version import __version__
from .regressor import MLPCausalRegressor, MLPPytorchRegressor, MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
from .classifier import MLPCausalClassifier, MLPPytorchClassifier

__all__ = [
    "__version__",
    "MLPCausalRegressor", 
    "MLPPytorchRegressor",
    "MLPCausalClassifier",
    "MLPPytorchClassifier",
    "MLPHuberRegressor",
    "MLPPinballRegressor", 
    "MLPCauchyRegressor"
]

# Package metadata
__author__ = "CausalEngine Team"
__email__ = ""
__license__ = "MIT"
__description__ = "Scikit-learn compatible implementation of CausalEngine for causal machine learning"
__theoretical_foundation__ = "Distribution-consistency Structural Causal Models (arXiv:2401.15911)"
__core_innovation__ = "Three-stage causal reasoning: Abduction → Action → TaskActivation"