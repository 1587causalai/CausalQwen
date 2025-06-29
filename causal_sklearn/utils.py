#!/usr/bin/env python3
"""
CausalSklearn工具函数模块 - 最优版本
=====================================

提供通用的工具函数，包括标签异常处理、数据处理等功能。
基于UltraThink重构思想，采用最优雅和简洁的设计。
"""

import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Optional, Tuple


def add_label_anomalies(y, anomaly_ratio=0.1, anomaly_type='regression', 
                       regression_anomaly_strategy='shuffle', 
                       classification_anomaly_strategy='shuffle'):
    """
    给标签添加异常 - 用于测试模型的鲁棒性
    
    这是一个通用的标签噪声注入工具，支持回归和分类任务的多种异常策略。
    
    Args:
        y: 原始标签数组
        anomaly_ratio: 异常比例 (0.0-1.0)
        anomaly_type: 任务类型 ('regression' 或 'classification')
        regression_anomaly_strategy: 回归异常策略 ('shuffle' 或 'outlier')
        classification_anomaly_strategy: 分类异常策略 ('flip' 或 'shuffle')
    
    Returns:
        numpy.ndarray: 添加异常后的标签数组
    
    Examples:
        >>> y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_noisy = add_label_anomalies(y_reg, 0.4, 'regression', 'shuffle')
        
        >>> y_cls = np.array([0, 1, 2, 0, 1, 2])
        >>> y_noisy = add_label_anomalies(y_cls, 0.3, 'classification', 'flip')
    """
    y_noisy = y.copy()
    n_anomalies = int(len(y) * anomaly_ratio)
    
    if n_anomalies == 0:
        return y_noisy
        
    # 随机选择异常样本索引
    anomaly_indices = np.random.choice(len(y), n_anomalies, replace=False)
    
    if anomaly_type == 'regression':
        if regression_anomaly_strategy == 'shuffle':
            # 策略1: 通过随机排序一部分标签来创建错误的X-y配对
            labels_to_shuffle = y_noisy[anomaly_indices]
            np.random.shuffle(labels_to_shuffle)
            y_noisy[anomaly_indices] = labels_to_shuffle
        
        elif regression_anomaly_strategy == 'outlier':
            # 策略2: 极端离群值异常
            y_std = np.std(y)
            for idx in anomaly_indices:
                if np.random.random() < 0.5:
                    # 3倍标准差偏移
                    sign = np.random.choice([-1, 1])
                    y_noisy[idx] = y[idx] + sign * 3.0 * y_std
                else:
                    # 10倍缩放
                    scale_factor = np.random.choice([0.1, 10.0])
                    y_noisy[idx] = y[idx] * scale_factor
            
    elif anomaly_type == 'classification':
        unique_labels = np.unique(y)
        
        if classification_anomaly_strategy == 'flip':
            # 策略1: 标签翻转到其他类别
            for idx in anomaly_indices:
                other_labels = unique_labels[unique_labels != y[idx]]
                if len(other_labels) > 0:
                    y_noisy[idx] = np.random.choice(other_labels)
        
        elif classification_anomaly_strategy == 'shuffle':
            # 策略2: 标签打乱（保持类别分布）
            labels_to_shuffle = y_noisy[anomaly_indices].copy()
            np.random.shuffle(labels_to_shuffle)
            y_noisy[anomaly_indices] = labels_to_shuffle
    
    return y_noisy


@dataclass
class SplitConfig:
    """数据分割配置类 - 统一管理所有分割参数"""
    # 基础分割参数
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = None
    shuffle: bool = True
    stratify: Optional[np.ndarray] = None
    
    # 异常注入配置（默认比例为0，即正常分割）
    anomaly_ratio: float = 0.0
    anomaly_type: str = 'regression'
    regression_strategy: str = 'shuffle'
    classification_strategy: str = 'shuffle'
    
    # 输出配置
    verbose: bool = False


class CausalSplitter:
    """
    因果数据分割器 - 简化的分割逻辑实现
    
    核心设计原则：
    1. 只支持2分割（train/test）
    2. 分离关注点 - 异常注入与分割逻辑解耦
    3. 简化接口 - 配置类管理参数
    """
    
    def __init__(self, *arrays, config: Optional[SplitConfig] = None):
        self.arrays = list(arrays)
        self.config = config or SplitConfig()
        
    def split(self) -> Tuple[np.ndarray, ...]:
        """执行分割并返回结果"""
        # 执行2分割
        result = self._two_way_split()
        
        # 应用异常注入（仅当异常比例 > 0时）
        if self.config.anomaly_ratio > 0:
            result = self._apply_anomalies(result)
        
        # 打印信息
        if self.config.verbose:
            self._print_summary(result)
        
        return self._to_sklearn_format(result)
    
    def _two_way_split(self) -> dict:
        """2分割实现"""
        split_arrays = train_test_split(
            *self.arrays,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=self.config.stratify
        )
        
        # 重新组织数组
        n_arrays = len(self.arrays)
        train_arrays = [split_arrays[i * 2] for i in range(n_arrays)]
        test_arrays = [split_arrays[i * 2 + 1] for i in range(n_arrays)]
        
        return {
            'train': train_arrays,
            'test': test_arrays
        }
    
    
    def _apply_anomalies(self, result: dict) -> dict:
        """应用异常注入 - 仅对训练集，测试集始终保持纯净"""
        # 创建副本避免修改原数据
        new_result = {}
        for key, arrays in result.items():
            new_result[key] = [arr.copy() for arr in arrays]
        
        # 对训练集注入异常（假设y是第二个数组）
        if len(new_result['train']) >= 2:
            new_result['train'][1] = add_label_anomalies(
                new_result['train'][1],
                anomaly_ratio=self.config.anomaly_ratio,
                anomaly_type=self.config.anomaly_type,
                regression_anomaly_strategy=self.config.regression_strategy,
                classification_anomaly_strategy=self.config.classification_strategy
            )
        
        # 测试集始终保持纯净，不注入异常
        
        return new_result
    
    def _print_summary(self, result: dict):
        """打印分割摘要"""
        print(f"🔄 CausalSklearn数据分割")
        print(f"   模式: 2分割 (train/test)")
        print(f"   样本数: {len(self.arrays[0])}")
        
        if self.config.anomaly_ratio > 0:
            print(f"   异常注入: {self.config.anomaly_type}, 比例={self.config.anomaly_ratio:.1%}")
            strategy = self.config.regression_strategy if self.config.anomaly_type == 'regression' else self.config.classification_strategy
            print(f"   异常策略: {strategy}")
            print(f"   测试集: 保持纯净")
        
        print(f"   分割结果: train={len(result['train'][0])}, test={len(result['test'][0])}")
    
    def _to_sklearn_format(self, result: dict) -> Tuple[np.ndarray, ...]:
        """转换为sklearn风格的tuple"""
        # 2分割: X_train, X_test, y_train, y_test, ...
        output = []
        for i in range(len(self.arrays)):
            output.extend([result['train'][i], result['test'][i]])
        return tuple(output)


def causal_split(*arrays, **kwargs) -> Tuple[np.ndarray, ...]:
    """
    因果数据分割函数 - 简洁高效的实现
    
    核心特性：
    1. 只支持2分割模式（train/test）
    2. 训练集可选异常注入，测试集始终纯净
    3. 异常比例默认0.0（正常分割）
    4. 验证集分割由各估计器内部处理（early stopping）
    
    Args:
        *arrays: 要分割的数组（X, y等）
        test_size: 测试集大小 (默认0.2)
        random_state: 随机种子
        shuffle: 是否打乱数据 (默认True)
        stratify: 分层分割的目标数组
        
        anomaly_ratio: 异常比例 (默认0.0，即正常分割)
        anomaly_type: 'regression' 或 'classification' (默认'regression')
        regression_anomaly_strategy: 回归异常策略 ('shuffle' 或 'outlier')
        classification_anomaly_strategy: 分类异常策略 ('flip' 或 'shuffle')
        
        verbose: 是否显示详细信息
        
    Returns:
        X_train, X_test, y_train, y_test (以及更多数组如果提供)
        
    Examples:
        >>> # 正常分割（无异常）
        >>> X_train, X_test, y_train, y_test = causal_split(X, y)
        
        >>> # 带异常注入的分割
        >>> X_train, X_test, y_train, y_test = causal_split(
        ...     X, y, anomaly_ratio=0.1, anomaly_type='regression', verbose=True
        ... )
        
        >>> # 分类任务的分层分割
        >>> X_train, X_test, y_train, y_test = causal_split(
        ...     X, y, stratify=y, anomaly_ratio=0.2, anomaly_type='classification'
        ... )
    """
    # 参数映射
    config = SplitConfig()
    
    # 基础参数
    if 'test_size' in kwargs: config.test_size = kwargs['test_size']
    if 'random_state' in kwargs: config.random_state = kwargs['random_state']
    if 'shuffle' in kwargs: config.shuffle = kwargs['shuffle']
    if 'stratify' in kwargs: config.stratify = kwargs['stratify']
    
    # 异常注入参数
    if 'anomaly_ratio' in kwargs: config.anomaly_ratio = kwargs['anomaly_ratio']
    if 'anomaly_type' in kwargs: config.anomaly_type = kwargs['anomaly_type']
    if 'regression_anomaly_strategy' in kwargs: config.regression_strategy = kwargs['regression_anomaly_strategy']
    if 'classification_anomaly_strategy' in kwargs: config.classification_strategy = kwargs['classification_anomaly_strategy']
    
    # 输出参数
    if 'verbose' in kwargs: config.verbose = kwargs['verbose']
    
    # 参数验证
    if len(arrays) < 2:
        raise ValueError("至少需要提供2个数组（通常是X和y）")
    
    # 执行分割
    splitter = CausalSplitter(*arrays, config=config)
    return splitter.split()


