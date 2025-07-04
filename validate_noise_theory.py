#!/usr/bin/env python3
"""
验证100%噪声理论分析

测试三种情况：
1. 正常情况：正确训练集 + 正确测试集
2. 当前实现：噪声训练集 + 正确测试集 
3. 理论期望：噪声训练集 + 噪声测试集

这将帮助我们理解为什么100%噪声结果不符合理论期望。
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def apply_noise_to_labels(y, noise_level, n_classes=3, random_state=42):
    """对标签应用噪声"""
    np.random.seed(random_state)
    
    if noise_level == 0:
        return y.copy()
    
    n_noisy = int(len(y) * noise_level)
    y_noisy = y.copy()
    noisy_indices = np.random.choice(len(y), n_noisy, replace=False)
    
    for idx in noisy_indices:
        possible_labels = [i for i in range(n_classes) if i != y[idx]]
        y_noisy[idx] = np.random.choice(possible_labels)
    
    return y_noisy

def test_noise_scenarios():
    """测试不同噪声场景"""
    print("🔬 验证100%噪声理论分析")
    print("=" * 60)
    
    # 生成数据
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=42
    )
    
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集大小: {len(y_train_clean)}")
    print(f"测试集大小: {len(y_test_clean)}")
    print()
    
    # 测试三种场景
    scenarios = [
        ("正常情况", y_train_clean, y_test_clean),
        ("当前实现", apply_noise_to_labels(y_train_clean, 1.0), y_test_clean),
        ("理论期望", apply_noise_to_labels(y_train_clean, 1.0), apply_noise_to_labels(y_test_clean, 1.0))
    ]
    
    results = []
    
    for scenario_name, y_train, y_test in scenarios:
        print(f"📊 测试场景: {scenario_name}")
        print("-" * 40)
        
        # 计算标签一致性
        if scenario_name == "正常情况":
            train_consistency = 1.0
            test_consistency = 1.0
        elif scenario_name == "当前实现":
            train_consistency = np.mean(y_train_clean == y_train)
            test_consistency = np.mean(y_test_clean == y_test)
        else:  # 理论期望
            train_consistency = np.mean(y_train_clean == y_train)
            test_consistency = np.mean(y_test_clean == y_test)
        
        print(f"训练集标签一致性: {train_consistency:.1%}")
        print(f"测试集标签一致性: {test_consistency:.1%}")
        
        # 训练模型
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"模型准确率: {accuracy:.1%}")
        results.append((scenario_name, accuracy))
        print()
    
    print("🎯 结果分析")
    print("=" * 40)
    for scenario_name, accuracy in results:
        print(f"{scenario_name}: {accuracy:.1%}")
    
    print(f"\n💡 理论分析:")
    print(f"- 随机基线 (1/3): {1/3:.1%}")
    print(f"- 当前实现结果: {results[1][1]:.1%}")
    print(f"- 理论期望结果: {results[2][1]:.1%}")
    
    # 验证随机预测基线
    print(f"\n🎲 随机预测验证:")
    np.random.seed(42)
    y_random = np.random.choice(3, size=len(y_test_clean))
    random_accuracy = accuracy_score(y_test_clean, y_random)
    print(f"随机预测准确率: {random_accuracy:.1%}")

if __name__ == '__main__':
    test_noise_scenarios()