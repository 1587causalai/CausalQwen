#!/usr/bin/env python3
"""
CausalSklearn工具函数模块 - 最优版本
=====================================

提供通用的工具函数。
此模块在重构后已被精简。核心数据处理逻辑已移至 `causal_sklearn.data_processing`。
"""

import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Optional, Tuple

