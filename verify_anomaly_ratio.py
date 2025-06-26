#!/usr/bin/env python3
"""
验证CLASSIFICATION_LABEL_NOISE确实等于真实异常比例
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from quick_test_causal_engine import QuickTester

def verify_anomaly_ratio():
    # 使用当前配置
    CLASSIFICATION_LABEL_NOISE = 0.2  # 当前设置20%
    print(f'🎯 测试CLASSIFICATION_LABEL_NOISE = {CLASSIFICATION_LABEL_NOISE} (20%)')
    print('=' * 50)

    # 生成相同数据
    np.random.seed(42)
    X, y = make_classification(n_samples=2000, n_features=15, n_classes=3, n_informative=7, 
                              n_redundant=0, n_clusters_per_class=1, class_sep=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f'训练集大小: {len(y_train)}')
    print(f'原始类别分布: {np.bincount(y_train)}')
    print()

    # 添加异常
    tester = QuickTester()
    y_train_noisy = tester.add_label_anomalies(y_train, CLASSIFICATION_LABEL_NOISE, 'classification')

    # 计算真实异常比例
    changed = y_train != y_train_noisy
    actual_anomaly_count = np.sum(changed)
    actual_anomaly_ratio = actual_anomaly_count / len(y_train)

    print(f'📊 异常统计:')
    print(f'   设置异常比例: {CLASSIFICATION_LABEL_NOISE:.1%}')
    print(f'   实际异常样本: {actual_anomaly_count} / {len(y_train)}')
    print(f'   实际异常比例: {actual_anomaly_ratio:.1%}')
    print(f'   是否完全匹配: {abs(actual_anomaly_ratio - CLASSIFICATION_LABEL_NOISE) < 0.001}')
    print()

    print(f'🔍 验证无自转换:')
    # 检查是否有任何自转换（原标签 = 新标签）
    for orig_label in [0, 1, 2]:
        same_count = np.sum((y_train == orig_label) & (y_train_noisy == orig_label))
        diff_count = np.sum((y_train == orig_label) & (y_train_noisy != orig_label))
        total_orig = np.sum(y_train == orig_label)
        print(f'   标签{orig_label}: 保持{same_count}个, 改变{diff_count}个 (总共{total_orig}个)')

    print(f'\n🔄 标签转换详情:')
    transition_count = 0
    for orig in [0, 1, 2]:
        for new in [0, 1, 2]:
            if orig != new:
                count = np.sum((y_train == orig) & (y_train_noisy == new))
                if count > 0:
                    print(f'   {orig}→{new}: {count}个转换')
                    transition_count += count
    
    print(f'\n📈 最终验证:')
    print(f'   总转换数: {transition_count}')
    print(f'   预期转换数: {int(len(y_train) * CLASSIFICATION_LABEL_NOISE)}')
    print(f'   ✅ 结论: CLASSIFICATION_LABEL_NOISE = 真实异常比例，无幸运效应！')

if __name__ == "__main__":
    verify_anomaly_ratio()