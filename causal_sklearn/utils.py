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


def add_label_anomalies(y, ratio: float = 0.1, task_type: str = 'regression', strategy: str = 'shuffle'):
    """
    给标签添加异常 - 用于测试模型的鲁棒性
    
    Args:
        y: 原始标签数组
        ratio: 异常比例 (0.0-1.0)
        task_type: 任务类型 ('regression' 或 'classification')
        strategy: 异常策略。
                  回归: 'shuffle', 'outlier'.
                  分类: 'flip', 'shuffle'.
    
    Returns:
        numpy.ndarray: 添加异常后的标签数组
    
    Examples:
        >>> y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_noisy = add_label_anomalies(y_reg, 0.4, 'regression', 'shuffle')
        
        >>> y_cls = np.array([0, 1, 2, 0, 1, 2])
        >>> y_noisy = add_label_anomalies(y_cls, 0.3, 'classification', 'flip')
    """
    y_noisy = y.copy()
    n_anomalies = int(len(y) * ratio)
    
    if n_anomalies == 0:
        return y_noisy
        
    # 随机选择异常样本索引
    anomaly_indices = np.random.choice(len(y), n_anomalies, replace=False)
    
    if task_type == 'regression':
        if strategy == 'shuffle':
            # 策略1: 通过随机排序一部分标签来创建错误的X-y配对
            labels_to_shuffle = y_noisy[anomaly_indices]
            np.random.shuffle(labels_to_shuffle)
            y_noisy[anomaly_indices] = labels_to_shuffle
        
        elif strategy == 'outlier':
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
        else:
            raise ValueError(f"Invalid strategy '{strategy}' for regression task. Use 'shuffle' or 'outlier'.")
            
    elif task_type == 'classification':
        unique_labels = np.unique(y)
        
        if strategy == 'flip':
            # 策略1: 标签翻转到其他类别
            for idx in anomaly_indices:
                other_labels = unique_labels[unique_labels != y[idx]]
                if len(other_labels) > 0:
                    y_noisy[idx] = np.random.choice(other_labels)
        
        elif strategy == 'shuffle':
            # 策略2: 标签打乱（保持类别分布）
            labels_to_shuffle = y_noisy[anomaly_indices].copy()
            np.random.shuffle(labels_to_shuffle)
            y_noisy[anomaly_indices] = labels_to_shuffle
        else:
            raise ValueError(f"Invalid strategy '{strategy}' for classification task. Use 'flip' or 'shuffle'.")
    
    return y_noisy


def causal_split(*arrays, 
                 test_size: Optional[float] = 0.2, 
                 random_state: Optional[int] = None, 
                 shuffle: bool = True, 
                 stratify: Optional[np.ndarray] = None,
                 anomaly_ratio: float = 0.0,
                 anomaly_type: str = 'regression',
                 anomaly_strategy: str = 'shuffle',
                 verbose: bool = False) -> Tuple[np.ndarray, ...]:
    """
    因果数据分割函数 - 简洁高效的实现
    
    核心特性：
    1. 基于 sklearn.model_selection.train_test_split
    2. 对训练集中的y标签可选注入异常，测试集始终纯净
    
    Args:
        *arrays: 要分割的数组（X, y等）
        test_size: 测试集大小 (默认0.2)
        random_state: 随机种子
        shuffle: 是否打乱数据 (默认True)
        stratify: 分层分割的目标数组
        
        anomaly_ratio: 异常比例 (默认0.0，即正常分割)
        anomaly_type: 'regression' 或 'classification' (默认'regression')
        anomaly_strategy: 异常策略。回归: 'shuffle' 或 'outlier'。分类: 'flip' 或 'shuffle'。
        
        verbose: 是否显示详细信息
        
    Returns:
        X_train, X_test, y_train, y_test (以及更多数组如果提供)
    """

    if anomaly_ratio > 0 and len(arrays) < 2:
        raise ValueError("Anomaly injection requires at least two arrays (e.g., X and y).")

    # 1. Perform standard train_test_split
    split_results = list(train_test_split(
        *arrays,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify
    ))
    
    # 2. Apply anomalies to the training set's y-values if requested
    if anomaly_ratio > 0:
        y_train_index = 2  # Corresponds to y_train in [X_train, X_test, y_train, y_test]
        
        y_train_noisy = add_label_anomalies(
            split_results[y_train_index],
            ratio=anomaly_ratio,
            task_type=anomaly_type,
            strategy=anomaly_strategy
        )
        
        split_results[y_train_index] = y_train_noisy

    # 3. Print summary if verbose
    if verbose:
        train_size = len(split_results[0])
        test_size_val = len(split_results[1])
        print(f"🔄 Causal data split completed.")
        print(f"   Total samples: {len(arrays[0])}, Train: {train_size}, Test: {test_size_val}")
        if anomaly_ratio > 0:
            print(f"   Anomaly Injection on training set:")
            print(f"     Type: {anomaly_type}, Ratio: {anomaly_ratio:.1%}")
            print(f"     Strategy: {anomaly_strategy}")
        print(f"   Test set remains clean.")

    return tuple(split_results)


