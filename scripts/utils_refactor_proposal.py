#!/usr/bin/env python3
"""
utils.py 重构提议：统一 shuffle 策略
=====================================

问题：当前 shuffle 策略在回归和分类中重复实现，逻辑基本相同
解决：提取通用的 shuffle 函数，消除代码重复
"""

import numpy as np

def apply_shuffle_strategy(y_noisy, anomaly_indices, task_type, original_values):
    """
    通用的 shuffle 策略实现
    
    Args:
        y_noisy: 需要修改的标签数组
        anomaly_indices: 异常样本的索引
        task_type: 任务类型 ('regression' 或 'classification')
        original_values: 原始标签值
    
    Returns:
        tuple: (changes_made, new_values, unchanged_ratio)
    """
    # 核心逻辑：随机打乱异常样本的标签
    labels_to_shuffle = y_noisy[anomaly_indices].copy()
    np.random.shuffle(labels_to_shuffle)
    y_noisy[anomaly_indices] = labels_to_shuffle
    
    # 获取新值
    new_values = y_noisy[anomaly_indices]
    
    # 根据任务类型选择合适的变化检测方法
    if task_type == 'regression':
        # 回归：使用数值容差检测变化
        changes_made = np.sum(np.abs(new_values - original_values) > 1e-10)
    else:
        # 分类：使用精确相等检测变化
        changes_made = np.sum(new_values != original_values)
    
    # 计算未改变的比例
    unchanged_ratio = 1.0 - (changes_made / len(anomaly_indices)) if len(anomaly_indices) > 0 else 0.0
    
    return int(changes_made), new_values.tolist(), unchanged_ratio

def add_label_anomalies_refactored(y, ratio: float = 0.1, task_type: str = 'regression', 
                                  strategy: str = 'shuffle', return_info: bool = False, 
                                  random_state: Optional[int] = None):
    """
    重构后的异常注入函数 - 统一了 shuffle 策略
    """
    # 设置随机种子确保可重现性
    if random_state is not None:
        np.random.seed(random_state)
    
    y_noisy = y.copy()
    n_anomalies = int(len(y) * ratio)
    
    # 初始化异常信息
    anomaly_info = {
        'requested_ratio': ratio,
        'actual_ratio': n_anomalies / len(y) if len(y) > 0 else 0.0,
        'n_anomalies': n_anomalies,
        'n_total': len(y),
        'strategy': strategy,
        'task_type': task_type,
        'anomaly_indices': [],
        'changes_made': 0,
        'original_values': [],
        'new_values': []
    }
    
    if n_anomalies == 0:
        if return_info:
            return y_noisy, anomaly_info
        return y_noisy
        
    # 随机选择异常样本索引
    anomaly_indices = np.random.choice(len(y), n_anomalies, replace=False)
    anomaly_info['anomaly_indices'] = anomaly_indices.tolist()
    
    # 获取原始值
    original_values = y_noisy[anomaly_indices].copy()
    anomaly_info['original_values'] = original_values.tolist()
    
    if strategy == 'shuffle':
        # ✅ 统一的 shuffle 策略 - 适用于回归和分类
        changes_made, new_values, unchanged_ratio = apply_shuffle_strategy(
            y_noisy, anomaly_indices, task_type, original_values
        )
        
        anomaly_info['changes_made'] = changes_made
        anomaly_info['new_values'] = new_values
        anomaly_info['unchanged_ratio'] = unchanged_ratio
    
    elif task_type == 'regression' and strategy == 'outlier':
        # 回归特有：极端离群值异常
        y_std = np.std(y)
        y_mean = np.mean(y)
        changes_made = 0
        
        for idx in anomaly_indices:
            original_val = y_noisy[idx]
            if np.random.random() < 0.5:
                # 3-5倍标准差偏移
                sign = np.random.choice([-1, 1])
                multiplier = np.random.uniform(3.0, 5.0)
                y_noisy[idx] = y_mean + sign * multiplier * y_std
            else:
                # 极值缩放
                scale_factor = np.random.choice([0.1, 0.2, 5.0, 10.0])
                y_noisy[idx] = y_mean + (original_val - y_mean) * scale_factor
            changes_made += 1
        
        anomaly_info['changes_made'] = changes_made
        anomaly_info['new_values'] = y_noisy[anomaly_indices].tolist()
    
    elif task_type == 'classification' and strategy == 'flip':
        # 分类特有：标签翻转到其他类别
        unique_labels = np.unique(y)
        changes_made = 0
        
        for idx in anomaly_indices:
            original_label = y_noisy[idx]
            other_labels = unique_labels[unique_labels != original_label]
            if len(other_labels) > 0:
                y_noisy[idx] = np.random.choice(other_labels)
                changes_made += 1
        
        anomaly_info['changes_made'] = changes_made
        anomaly_info['new_values'] = y_noisy[anomaly_indices].tolist()
    
    else:
        # 错误的策略组合
        valid_strategies = {
            'regression': ['shuffle', 'outlier'],
            'classification': ['shuffle', 'flip']
        }
        raise ValueError(f"Invalid strategy '{strategy}' for {task_type} task. "
                        f"Valid strategies: {valid_strategies[task_type]}")
    
    # 验证异常注入效果（统一逻辑）
    if len(anomaly_info['original_values']) > 0 and len(anomaly_info['new_values']) > 0:
        original_vals = np.array(anomaly_info['original_values'])
        new_vals = np.array(anomaly_info['new_values'])
        
        if task_type == 'regression':
            # 计算变化的程度
            changes = np.abs(new_vals - original_vals)
            anomaly_info['avg_change'] = float(np.mean(changes))
            anomaly_info['max_change'] = float(np.max(changes))
        else:
            # 分类任务：计算实际改变的比例
            actually_changed = np.sum(new_vals != original_vals)
            anomaly_info['actual_change_ratio'] = actually_changed / len(original_vals)
    
    if return_info:
        return y_noisy, anomaly_info
    return y_noisy

# 示例用法对比
def demonstrate_improvement():
    """展示重构前后的差异"""
    print("🔧 utils.py 重构提议演示")
    print("=" * 50)
    
    # 测试数据
    y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_cls = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    
    print("\n📊 重构前的问题:")
    print("   - shuffle 策略在回归和分类中重复实现")
    print("   - 73-89行（回归）和133-141行（分类）基本相同")
    print("   - 维护成本高，容易不一致")
    
    print("\n✅ 重构后的优势:")
    print("   - shuffle 策略提取为通用函数")
    print("   - 消除代码重复")
    print("   - 更易维护和测试")
    print("   - 保持完全相同的行为")
    
    print("\n🧪 功能验证:")
    
    # 回归 shuffle
    y_reg_noisy, info_reg = add_label_anomalies_refactored(
        y_reg, 0.5, 'regression', 'shuffle', return_info=True, random_state=42
    )
    print(f"   回归 shuffle: {info_reg['changes_made']}/{info_reg['n_anomalies']} 改变")
    
    # 分类 shuffle
    y_cls_noisy, info_cls = add_label_anomalies_refactored(
        y_cls, 0.5, 'classification', 'shuffle', return_info=True, random_state=42
    )
    print(f"   分类 shuffle: {info_cls['changes_made']}/{info_cls['n_anomalies']} 改变")
    
    print("\n💡 建议:")
    print("   1. 将 apply_shuffle_strategy 函数添加到 utils.py")
    print("   2. 重构 add_label_anomalies 使用统一的 shuffle 实现")
    print("   3. 保持向后兼容性")
    print("   4. 添加单元测试验证行为一致性")

if __name__ == "__main__":
    demonstrate_improvement()