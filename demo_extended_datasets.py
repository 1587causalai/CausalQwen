#!/usr/bin/env python3
"""
扩展数据集演示脚本

🎯 目标：演示如何使用新增的扩展回归数据集功能
🔬 核心：展示多个真实回归数据集的加载和基础信息

使用方法：
python demo_extended_datasets.py
"""

import sys
import os
sys.path.append('.')

from causal_sklearn.data_processing import (
    list_available_regression_datasets,
    load_extended_regression_dataset,
    EXTENDED_REGRESSION_DATASETS
)
from scripts.regression_robustness_real_datasets import CONFIG

def demo_dataset_loading():
    """演示数据集加载功能"""
    
    print("🚀 扩展数据集功能演示")
    print("=" * 70)
    
    # 显示所有可用数据集
    list_available_regression_datasets()
    
    print("\n📊 数据集加载演示:")
    print("-" * 50)
    
    # 演示几个代表性数据集
    demo_datasets = [
        'california_housing',  # 大数据集
        'diabetes',           # 中等数据集  
        'auto_mpg',          # OpenML数据集
    ]
    
    for dataset_name in demo_datasets:
        try:
            print(f"\n🔧 加载数据集: {dataset_name}")
            X, y, info = load_extended_regression_dataset(
                dataset_name=dataset_name,
                random_state=42,
                return_info=True
            )
            
            print(f"✅ 成功加载: {info['name']}")
            print(f"   📐 数据形状: X={X.shape}, y={y.shape}")
            print(f"   📊 目标变量: min={info['y_min']:.3f}, max={info['y_max']:.3f}, mean={info['y_mean']:.3f}")
            print(f"   🔗 数据源: {info['source']}")
            print(f"   📝 描述: {info['description']}")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    
    print(f"\n💡 如何在回归脚本中使用:")
    print("1. 修改 CONFIG['dataset_name'] 为任意支持的数据集名称")
    print("2. 确保 CONFIG['use_extended_datasets'] = True")
    print("3. 运行 scripts/regression_robustness_real_datasets.py")

def demo_quick_comparison():
    """演示在多个数据集上的快速比较"""
    
    print("\n\n🎯 多数据集快速比较:")
    print("=" * 70)
    
    quick_test_datasets = ['diabetes', 'auto_mpg']
    
    for dataset_name in quick_test_datasets:
        print(f"\n📊 数据集: {dataset_name}")
        
        # 修改配置
        test_config = CONFIG.copy()
        test_config['dataset_name'] = dataset_name
        test_config['use_extended_datasets'] = True
        test_config['n_runs'] = 1  # 快速演示
        test_config['max_iter'] = 100  # 减少迭代次数
        test_config['noise_levels'] = [0.0, 0.5, 1.0]  # 只测试3个噪声级别
        test_config['verbose'] = False  # 减少输出
        test_config['save_plots'] = False  # 不保存图表
        test_config['save_data'] = False  # 不保存数据
        
        print(f"   配置: 噪声级别 {test_config['noise_levels']}, 最大迭代 {test_config['max_iter']}")
        print(f"   模式: 快速演示 (不保存结果)")
        print(f"   说明: 如需完整测试，请运行主脚本")

if __name__ == '__main__':
    # 基础功能演示
    demo_dataset_loading()
    
    # 快速比较演示
    demo_quick_comparison()
    
    print("\n✅ 扩展数据集功能演示完成!")
    print("💡 提示: 现在可以在回归鲁棒性脚本中使用这些数据集进行完整测试")