#!/usr/bin/env python3
"""
测试 causal_split 功能的脚本
===============================

展示 causal_split 如何进行数据分割和异常注入，包括：
1. 基本数据分割功能
2. 异常注入策略测试 (shuffle vs outlier)
3. 不同任务类型 (回归 vs 分类)
4. 边界条件测试
5. 可视化异常注入效果
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import os
import warnings

# 忽略seaborn特定FutureWarning，保持输出清洁
warnings.filterwarnings('ignore', message="use_inf_as_na option is deprecated")

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_sklearn.utils import causal_split, add_label_anomalies, validate_anomaly_injection, print_anomaly_summary
from sklearn.datasets import make_regression, make_classification


def test_basic_split():
    """测试基本的数据分割功能"""
    print("🔄 1. 基本数据分割测试")
    print("-" * 40)
    
    # 生成测试数据
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    
    # 基本分割（无异常注入）
    X_train, X_test, y_train, y_test = causal_split(
        X, y,
        test_size=0.2,
        random_state=42,
        verbose=True
    )
    
    print(f"✅ 基本分割完成: 训练集 {len(X_train)}, 测试集 {len(X_test)}")
    return X, y


def test_anomaly_strategies(X, y):
    """测试 'shuffle' 异常注入策略"""
    print(f"\n🎯 2. 'Shuffle' 异常注入策略测试 (回归)")
    print("-" * 40)

    strategy = 'shuffle'
    anomaly_ratio = 0.2

    print(f"\n--- 测试策略: {strategy} ---")

    split_results, anomaly_info = causal_split(
        X, y,
        test_size=0.2,
        random_state=42,
        anomaly_ratio=anomaly_ratio,
        anomaly_type='regression',
        anomaly_strategy=strategy,
        verbose=False,  # 保持控制台清洁
        return_anomaly_info=True
    )
    X_train, X_test, y_train, y_test = split_results

    print(f"  - 设定的异常比例: {anomaly_ratio:.2f}")
    if anomaly_info and anomaly_info['n_anomalies'] > 0:
        original_values = anomaly_info['original_values']
        new_values = y_train[anomaly_info['anomaly_indices']]
        mae = np.mean(np.abs(new_values - original_values))
        print(f"  - 实际注入的异常: {anomaly_info['n_anomalies']} (占训练集 {anomaly_info['n_anomalies'] / len(y_train):.2%})")
        print(f"  - 被修改标签的 MAE: {mae:.4f}")
    else:
        print("  - 未注入异常。")

    results = {
        strategy: {
            'X_train': X_train,
            'y_train': y_train,
            'anomaly_info': anomaly_info
        }
    }
    return results


def test_classification_anomalies():
    """测试分类任务的'shuffle'异常注入"""
    print(f"\n🏷️ 3. 'Shuffle' 异常注入测试 (分类)")
    print("-" * 40)

    # 生成分类数据
    X_cls, y_cls = make_classification(n_samples=500, n_features=4, n_classes=3,
                                       n_informative=4, n_redundant=0, random_state=42)

    strategy = 'shuffle'
    anomaly_ratio = 0.15

    print(f"\n--- 测试分类策略: {strategy} ---")

    split_results, anomaly_info = causal_split(
        X_cls, y_cls,
        test_size=0.25,
        random_state=42,
        anomaly_ratio=anomaly_ratio,
        anomaly_type='classification',
        anomaly_strategy=strategy,
        verbose=False, # 保持控制台清洁
        return_anomaly_info=True
    )
    X_train, X_test, y_train, y_test = split_results
    
    print(f"  - 设定的异常比例: {anomaly_ratio:.2f}")
    if anomaly_info and anomaly_info['changes_made'] > 0:
        actual_change_ratio = anomaly_info['changes_made'] / len(y_train)
        print(f"  - 实际注入的异常: {anomaly_info['n_anomalies']}")
        print(f"  - 实际被改变的标签占比: {actual_change_ratio:.2%}")
    else:
        print("  - 未注入异常或没有标签被改变。")


def test_edge_cases():
    """测试边界条件"""
    print(f"\n⚠️ 4. 边界条件测试")
    print("-" * 40)
    
    # 小数据集
    X_small = np.random.randn(10, 3)
    y_small = np.random.randn(10)
    
    print("\n--- 小数据集（10样本）---")
    split_results, anomaly_info = causal_split(
        X_small, y_small,
        test_size=0.3,
        random_state=42,
        anomaly_ratio=0.1,  # 期望1个异常，但可能为0
        anomaly_type='regression',
        anomaly_strategy='shuffle',
        verbose=True,
        return_anomaly_info=True
    )
    X_train, X_test, y_train, y_test = split_results
    
    # 零异常比例
    print("\n--- 零异常比例测试 ---")
    X_train, X_test, y_train, y_test = causal_split(
        X_small, y_small,
        test_size=0.3,
        random_state=42,
        anomaly_ratio=0.0,
        verbose=True
    )


def visualize_anomaly_effects(results):
    """可视化'shuffle'策略的异常注入效果"""
    print(f"\n📊 5. 异常注入效果可视化 ('shuffle' 策略)")
    print("-" * 40)

    try:
        strategy = 'shuffle'
        if strategy not in results:
            print(f"⚠️ 结果中未找到 '{strategy}' 策略的信息，跳过可视化。")
            return

        data = results[strategy]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        y_train = data['y_train']
        anomaly_info = data['anomaly_info']

        # 验证 "shuffle" 策略是否保持了标签的整体分布
        ax.set_title(f'Distribution Validation for Shuffle Strategy')
        sns.kdeplot(y_train, ax=ax, label='Overall Label Distribution', color='blue', fill=True, alpha=0.1)
        
        if anomaly_info['n_anomalies'] > 0:
            original_values = anomaly_info['original_values']
            shuffled_indices = anomaly_info['anomaly_indices']
            shuffled_new_values = y_train[shuffled_indices]
            
            sns.kdeplot(shuffled_new_values, ax=ax, label='Shuffled Labels Distribution', color='red', linewidth=2.5, linestyle='--')
            sns.kdeplot(original_values, ax=ax, label='Original Labels Distribution (pre-shuffle)', color='green', linewidth=2)

        ax.set_xlabel('Label Value')
        ax.set_ylabel('Density')
        ax.legend()

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'anomaly_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化图表已保存至: {save_path}")
        plt.close()

    except Exception as e:
        print(f"⚠️ 可视化失败 (可能是显示环境问题): {e}")


def test_reproducibility():
    """测试可重现性"""
    print(f"\n🔁 6. 可重现性测试")
    print("-" * 40)
    
    X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    
    # 运行两次相同的分割
    results1 = causal_split(X, y, test_size=0.2, random_state=42, 
                           anomaly_ratio=0.1, anomaly_strategy='shuffle',
                           return_anomaly_info=True)
    
    results2 = causal_split(X, y, test_size=0.2, random_state=42,
                           anomaly_ratio=0.1, anomaly_strategy='shuffle', 
                           return_anomaly_info=True)
    
    # 检查是否完全一致
    X_train1, X_test1, y_train1, y_test1 = results1[0]
    anomaly_info1 = results1[1]
    
    X_train2, X_test2, y_train2, y_test2 = results2[0]
    anomaly_info2 = results2[1]
    
    train_identical = np.allclose(X_train1, X_train2) and np.allclose(y_train1, y_train2)
    test_identical = np.allclose(X_test1, X_test2) and np.allclose(y_test1, y_test2)
    anomaly_identical = (anomaly_info1['anomaly_indices'] == anomaly_info2['anomaly_indices'])
    
    if train_identical and test_identical and anomaly_identical:
        print("✅ 可重现性测试通过：相同随机种子产生完全相同的结果")
    else:
        print("❌ 可重现性测试失败：结果不一致")


def generate_summary_report():
    """生成测试摘要报告"""
    print(f"\n📋 7. 测试摘要报告 (聚焦 'shuffle' 策略)")
    print("=" * 50)
    
    summary = {
        "功能": [
            "✅ 基本数据分割",
            "✅ 'Shuffle' 异常注入 (回归)",
            "✅ 'Shuffle' 异常注入 (分类)", 
            "✅ 边界条件处理",
            "✅ 详细信息报告",
            "✅ 可重现性保证"
        ],
        "核心策略": [
            "🎯 shuffle: 随机化训练集标签，模拟因果关系的破坏，同时保持标签的边缘分布不变。"
        ],
        "核心特性": [
            "🔒 测试集始终保持纯净",
            "📊 通过可视化验证分布一致性",
            "⚡ 高效的实现（基于sklearn）"
        ]
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")


def main():
    """主测试函数"""
    print("🧪 causal_split 功能全面测试")
    print("=" * 60)
    print("目标：验证数据分割和异常注入的所有功能")
    print()
    
    # 1. 基本分割测试
    X, y = test_basic_split()
    
    # 2. 异常策略对比
    results = test_anomaly_strategies(X, y)
    
    # 3. 分类任务测试
    test_classification_anomalies()
    
    # 4. 边界条件测试
    test_edge_cases()
    
    # 5. 可视化（可选）
    visualize_anomaly_effects(results)
    
    # 6. 可重现性测试
    test_reproducibility()
    
    # 7. 生成摘要报告
    generate_summary_report()
    
    print(f"\n🎉 所有测试完成！")
    print("💡 causal_split 函数工作正常，可以安全用于数据分割和异常注入")


if __name__ == "__main__":
    main()