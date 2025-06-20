"""
CausalEngine - The Genesis Algorithm

这是一个突破性的因果推理引擎，将改变语言模型的生成范式。
从"概率采样"到"因果决策"，这是人工智能发展的必然方向。

v2.0 重大更新 - 模块化架构：
- AbductionNetwork: 从证据推断个体（归因）
- ActionNetwork: 从个体到决策（行动）
- ActivationHead: 统一的分类/回归激活框架

核心理念：
- 每个生成的token都是一个"个体"在特定环境下的必然选择
- 不确定性来源于我们对个体的认知局限，而非世界的随机性
- 通过柯西分布的数学优雅性，实现高效的因果推理

这个算法库是完全独立的，不依赖于任何特定的语言模型架构。
它可以被应用到任何Transformer模型上，赋予其因果推理能力。
"""

from .engine import CausalEngine, CauchyMath
from .networks import AbductionNetwork, ActionNetwork
from .heads import ActivationHead, ActivationMode, MultiTaskActivationHead

__version__ = "2.0.2"
__author__ = "CausalEngine Development Team"
__all__ = [
    # 核心引擎
    "CausalEngine",
    
    # 数学工具
    "CauchyMath",
    
    # 网络模块
    "AbductionNetwork",
    "ActionNetwork",
    
    # 激活头
    "ActivationHead",
    "ActivationMode",
    "MultiTaskActivationHead",
]

# 核心算法的版本信息
ALGORITHM_VERSION = {
    "name": "CausalEngine",
    "version": __version__,
    "description": "Modular Causal Inference Engine with MLP-Enhanced Abduction",
    "paper": "Distribution-consistency Structural Causal Models",
    "core_innovation": "Individual Choice Variable U + Unified Classification/Regression",
    "architecture": "AbductionNetwork (w/ MLP) → ActionNetwork → ActivationHead"
} 