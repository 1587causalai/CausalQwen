#!/usr/bin/env python3
"""
CausalSklearn工具函数模块
=========================

提供通用的工具函数，包括标签异常处理、数据处理等功能。
这些工具函数可以在项目的多个模块中复用。
"""

import numpy as np


def add_label_anomalies(y, anomaly_ratio=0.1, anomaly_type='regression', 
                       regression_anomaly_strategy='shuffle', 
                       classification_anomaly_strategy='shuffle'):
    """
    给标签添加异常 - 用于测试模型的鲁棒性
    
    这是一个通用的标签噪声注入工具，支持回归和分类任务的多种异常策略。
    
    Args:
        y: 原始标签数组
        anomaly_ratio: 异常比例 (0.0-1.0)
        anomaly_type: 任务类型
            - 'regression': 回归异常
            - 'classification': 分类翻转
        regression_anomaly_strategy: 回归异常策略
            - 'shuffle': 打乱标签 (保持标签分布)
            - 'outlier': 极端离群值
        classification_anomaly_strategy: 分类异常策略
            - 'flip': 翻转到其他类别 (避免自翻转)
            - 'shuffle': 打乱异常标签 (保持类别分布)
    
    Returns:
        numpy.ndarray: 添加异常后的标签数组
    
    Examples:
        >>> # 回归任务 - shuffle策略
        >>> y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_noisy = add_label_anomalies(y_reg, 0.4, 'regression', 'shuffle')
        
        >>> # 分类任务 - flip策略
        >>> y_cls = np.array([0, 1, 2, 0, 1, 2])
        >>> y_noisy = add_label_anomalies(y_cls, 0.3, 'classification', 'flip')
        
        >>> # 分类任务 - shuffle策略
        >>> y_noisy = add_label_anomalies(y_cls, 0.3, 'classification', 'shuffle')
    """
    y_noisy = y.copy()
    n_anomalies = int(len(y) * anomaly_ratio)
    
    if n_anomalies == 0:
        return y_noisy
        
    # 随机选择异常样本索引
    anomaly_indices = np.random.choice(len(y), n_anomalies, replace=False)
    
    if anomaly_type == 'regression':
        if regression_anomaly_strategy == 'shuffle':
            # 策略1: 通过随机排序一部分标签来创建错误的X-y配对 (默认)
            # 获取这些异常索引对应的原始标签
            labels_to_shuffle = y_noisy[anomaly_indices]
            
            # 将这些标签随机排序（in-place）
            np.random.shuffle(labels_to_shuffle)
            
            # 将打乱后的标签重新赋给y_noisy
            y_noisy[anomaly_indices] = labels_to_shuffle
        
        elif regression_anomaly_strategy == 'outlier':
            # 策略2: 简单而强烈的离群值异常
            y_std = np.std(y)
            
            for idx in anomaly_indices:
                # 随机选择异常类型
                if np.random.random() < 0.5:
                    # 策略1: 3倍标准差偏移
                    sign = np.random.choice([-1, 1])
                    y_noisy[idx] = y[idx] + sign * 3.0 * y_std
                else:
                    # 策略2: 10倍缩放
                    scale_factor = np.random.choice([0.1, 10.0])  # 极端缩放
                    y_noisy[idx] = y[idx] * scale_factor
            
    elif anomaly_type == 'classification':
        unique_labels = np.unique(y)
        
        if classification_anomaly_strategy == 'flip':
            # 策略1: 传统标签翻转 - 翻转到其他类别 (避免自翻转)
            for idx in anomaly_indices:
                other_labels = unique_labels[unique_labels != y[idx]]
                if len(other_labels) > 0:
                    y_noisy[idx] = np.random.choice(other_labels)
        
        elif classification_anomaly_strategy == 'shuffle':
            # 策略2: 分类shuffle - 类似回归的shuffle策略，完全保持类别分布
            # 获取这些异常索引对应的原始标签
            labels_to_shuffle = y_noisy[anomaly_indices].copy()
            
            # 将这些标签随机排序（in-place）
            np.random.shuffle(labels_to_shuffle)
            
            # 将打乱后的标签重新赋给y_noisy
            y_noisy[anomaly_indices] = labels_to_shuffle
    
    return y_noisy


def get_anomaly_strategy_info():
    """
    获取所有可用异常策略的信息
    
    Returns:
        dict: 包含策略信息的字典
    """
    return {
        'regression': {
            'shuffle': {
                'name': '打乱标签',
                'description': '通过随机排序一部分标签来创建错误的X-y配对',
                'preserves_distribution': True,
                'noise_intensity': 'medium'
            },
            'outlier': {
                'name': '极端离群值',
                'description': '添加3倍标准差偏移或10倍缩放的极端值',
                'preserves_distribution': False,
                'noise_intensity': 'high'
            }
        },
        'classification': {
            'flip': {
                'name': '标签翻转',
                'description': '翻转到其他类别，确保实际翻转发生',
                'preserves_distribution': False,
                'noise_intensity': 'high'
            },
            'shuffle': {
                'name': '标签打乱',
                'description': '类似回归shuffle，完全保持类别分布',
                'preserves_distribution': True,
                'noise_intensity': 'medium'
            }
        }
    }


def validate_anomaly_parameters(anomaly_ratio, anomaly_type, regression_anomaly_strategy, classification_anomaly_strategy):
    """
    验证异常策略参数的有效性
    
    Args:
        anomaly_ratio: 异常比例
        anomaly_type: 任务类型
        regression_anomaly_strategy: 回归异常策略
        classification_anomaly_strategy: 分类异常策略
    
    Raises:
        ValueError: 参数无效时抛出异常
    """
    # 验证异常比例
    if not 0.0 <= anomaly_ratio <= 1.0:
        raise ValueError(f"anomaly_ratio必须在[0.0, 1.0]范围内，得到: {anomaly_ratio}")
    
    # 验证任务类型
    valid_types = ['regression', 'classification']
    if anomaly_type not in valid_types:
        raise ValueError(f"anomaly_type必须是{valid_types}之一，得到: {anomaly_type}")
    
    # 验证回归策略
    valid_regression_strategies = ['shuffle', 'outlier']
    if regression_anomaly_strategy not in valid_regression_strategies:
        raise ValueError(f"regression_anomaly_strategy必须是{valid_regression_strategies}之一，得到: {regression_anomaly_strategy}")
    
    # 验证分类策略
    valid_classification_strategies = ['flip', 'shuffle']
    if classification_anomaly_strategy not in valid_classification_strategies:
        raise ValueError(f"classification_anomaly_strategy必须是{valid_classification_strategies}之一，得到: {classification_anomaly_strategy}")


# 为了向后兼容，提供一个简化版本
def add_regression_anomalies(y, anomaly_ratio=0.1, strategy='shuffle'):
    """
    回归任务专用的异常标签处理 (简化接口)
    
    Args:
        y: 原始标签
        anomaly_ratio: 异常比例
        strategy: 'shuffle' 或 'outlier'
    
    Returns:
        numpy.ndarray: 添加异常后的标签
    """
    return add_label_anomalies(y, anomaly_ratio, 'regression', strategy)


def add_classification_anomalies(y, anomaly_ratio=0.1, strategy='flip'):
    """
    分类任务专用的异常标签处理 (简化接口)
    
    Args:
        y: 原始标签
        anomaly_ratio: 异常比例
        strategy: 'flip' 或 'shuffle'
    
    Returns:
        numpy.ndarray: 添加异常后的标签
    """
    return add_label_anomalies(y, anomaly_ratio, 'classification', classification_anomaly_strategy=strategy)


# 便于导入的别名
label_noise = add_label_anomalies
regression_noise = add_regression_anomalies
classification_noise = add_classification_anomalies