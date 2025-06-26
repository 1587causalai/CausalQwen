"""
Causal-Sklearn: Scikit-learn Compatible Causal Machine Learning

This package provides scikit-learn compatible implementations of CausalEngine
for causal machine learning tasks.
"""

from ._version import __version__
from .regressor import MLPCausalRegressor
from .classifier import MLPCausalClassifier

__all__ = [
    "__version__",
    "MLPCausalRegressor", 
    "MLPCausalClassifier"
]

# Package metadata
__author__ = "CausalEngine Team"
__email__ = ""
__license__ = "MIT"
__description__ = "Scikit-learn compatible implementation of CausalEngine"