"""
Benchmarking tools for comparing causal models with traditional methods.

支持多种基准方法：神经网络、集成方法、SVM、线性方法等。
提供配置驱动的基准测试框架。
"""

from .base import BaselineBenchmark, PyTorchBaseline
from .methods import BaselineMethodFactory, MethodDependencyChecker
from .method_configs import (
    get_method_config, get_method_group, get_task_recommendations,
    list_available_methods, METHOD_GROUPS, TASK_SPECIFIC_RECOMMENDATIONS
)

__all__ = [
    "BaselineBenchmark",
    "PyTorchBaseline",
    "BaselineMethodFactory", 
    "MethodDependencyChecker",
    "get_method_config",
    "get_method_group", 
    "get_task_recommendations",
    "list_available_methods",
    "METHOD_GROUPS",
    "TASK_SPECIFIC_RECOMMENDATIONS"
]