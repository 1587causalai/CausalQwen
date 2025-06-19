"""
CausalEngine - The Genesis Algorithm

这是一个革命性的因果推理引擎，将改变语言模型的生成范式。
从"概率采样"到"因果决策"，这是人工智能发展的必然方向。

核心理念：
- 每个生成的token都是一个"个体"在特定环境下的必然选择
- 不确定性来源于我们对个体的认知局限，而非世界的随机性
- 通过柯西分布的数学优雅性，实现高效的因果推理

这个算法库是完全独立的，不依赖于任何特定的语言模型架构。
它可以被应用到任何Transformer模型上，赋予其因果推理能力。
"""

from .engine import CausalEngine, CauchyMath

__version__ = "1.0.0"
__author__ = "CausalEngine Development Team"
__all__ = ["CausalEngine", "CauchyMath"]

# 核心算法的版本信息
ALGORITHM_VERSION = {
    "name": "CausalEngine",
    "version": __version__,
    "description": "Universal Causal Inference Engine for Language Models",
    "paper": "Distribution-consistency Structural Causal Models",
    "core_innovation": "Individual Choice Variable U in Language Generation"
} 