"""
CausalEngine Sklearn-Style Interface

这个模块提供了sklearn风格的接口，让用户可以像使用MLPRegressor和MLPClassifier
一样使用CausalEngine的强大功能。

核心设计理念：
- 保持sklearn的简单易用性
- 引入CausalEngine的因果推理能力
- 提供开箱即用的噪声鲁棒性
- 支持统一的predict()接口，多种预测模式

主要组件：
- MLPCausalRegressor: 因果回归器，替代sklearn MLPRegressor
- MLPCausalClassifier: 因果分类器，替代sklearn MLPClassifier
"""

from .base import CausalEstimatorMixin
from .regressor import MLPCausalRegressor
from .classifier import MLPCausalClassifier

__version__ = "1.0.0"
__all__ = [
    # 基础混入类
    "CausalEstimatorMixin",
    
    # 主要估计器
    "MLPCausalRegressor", 
    "MLPCausalClassifier",
]

# sklearn风格接口的版本信息
SKLEARN_INTERFACE_VERSION = {
    "name": "CausalEngine Sklearn Interface",
    "version": __version__,
    "description": "Sklearn-style interface for CausalEngine",
    "core_estimators": ["MLPCausalRegressor", "MLPCausalClassifier"],
    "compatibility": "sklearn 1.0+",
    "design_goal": "Drop-in replacement for MLPRegressor/MLPClassifier with causal inference"
}