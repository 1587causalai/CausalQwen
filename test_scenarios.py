#!/usr/bin/env python3
"""
CausalEngine 常用测试场景
提供一些预定义的测试场景，方便快速验证算法效果
"""

import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from quick_test_causal_engine import QuickTester

def scenario_clean_data():
    """场景1: 干净数据 - 基线性能测试"""
    print("🧼 场景1: 干净数据测试")
    print("=" * 50)
    
    tester = QuickTester()
    
    # 回归测试
    print("\n📈 回归任务:")
    tester.test_regression(
        n_samples=1000,
        n_features=8,
        noise=0.1,
        label_noise_ratio=0.0,  # 无标签噪声
        hidden_layer_sizes=(64, 32),
        gamma_init=10.0,
        b_noise_init=0.1,
        max_iter=800
    )
    
    # 分类测试
    print("\n🎯 分类任务:")
    tester.test_classification(
        n_samples=1000,
        n_features=8,
        n_classes=2,
        class_sep=1.0,
        label_noise_ratio=0.0,  # 无标签噪声
        hidden_layer_sizes=(64, 32),
        gamma_init=10.0,
        b_noise_init=0.1,
        max_iter=800
    )

def scenario_label_noise():
    """场景2: 标签噪声 - 鲁棒性测试"""
    print("\n🔊 场景2: 标签噪声测试")
    print("=" * 50)
    
    tester = QuickTester()
    
    # 回归 - 高斯噪声
    print("\n📈 回归任务 (20%高斯噪声):")
    tester.test_regression(
        n_samples=800,
        n_features=10,
        noise=0.1,
        label_noise_ratio=0.2,  # 20%噪声
        label_noise_type='gaussian',
        hidden_layer_sizes=(128, 64),
        gamma_init=15.0,  # 更大的初始尺度
        b_noise_init=0.2,  # 更大的噪声项
        max_iter=1000
    )
    
    # 分类 - 标签翻转
    print("\n🎯 分类任务 (15%标签翻转):")
    tester.test_classification(
        n_samples=800,
        n_features=10,
        n_classes=3,
        class_sep=0.8,  # 较难分离
        label_noise_ratio=0.15,  # 15%标签翻转
        label_noise_type='flip',
        hidden_layer_sizes=(128, 64),
        gamma_init=15.0,
        b_noise_init=0.2,
        ovr_threshold_init=0.0,
        max_iter=1000
    )

def scenario_network_comparison():
    """场景3: 网络结构对比"""
    print("\n🏗️ 场景3: 网络结构对比")
    print("=" * 50)
    
    tester = QuickTester()
    
    # 小网络
    print("\n🔸 小网络 (32,16):")
    tester.test_regression(
        n_samples=600,
        n_features=6,
        hidden_layer_sizes=(32, 16),
        gamma_init=10.0,
        b_noise_init=0.1,
        max_iter=600,
        verbose=True
    )
    
    # 大网络
    print("\n🔹 大网络 (256,128,64):")
    tester.test_regression(
        n_samples=600,
        n_features=6,
        hidden_layer_sizes=(256, 128, 64),
        gamma_init=10.0,
        b_noise_init=0.1,
        max_iter=600,
        verbose=True
    )

def scenario_parameter_sensitivity():
    """场景4: 参数敏感性测试"""
    print("\n🎛️ 场景4: 参数敏感性测试")
    print("=" * 50)
    
    tester = QuickTester()
    
    # gamma_init敏感性
    for gamma_val in [1.0, 5.0, 10.0, 20.0]:
        print(f"\n🔧 gamma_init = {gamma_val}:")
        tester.test_regression(
            n_samples=500,
            n_features=8,
            hidden_layer_sizes=(64, 32),
            gamma_init=gamma_val,
            b_noise_init=0.1,
            max_iter=400,
            verbose=False  # 简化输出
        )
        
        # 只显示关键指标
        results = tester.results['regression']
        print(f"   deterministic R²: {results['deterministic']['R²']:.4f}")
        print(f"   standard R²:      {results['standard']['R²']:.4f}")

def scenario_extreme_noise():
    """场景5: 极端噪声环境"""
    print("\n💥 场景5: 极端噪声环境")
    print("=" * 50)
    
    tester = QuickTester()
    
    print("\n📈 回归 (50%高斯噪声):")
    tester.test_regression(
        n_samples=800,
        n_features=12,
        noise=0.2,  # 数据噪声
        label_noise_ratio=0.5,  # 50%标签噪声!
        label_noise_type='gaussian',
        hidden_layer_sizes=(128, 64, 32),
        gamma_init=20.0,  # 大尺度应对噪声
        b_noise_init=0.5,  # 大噪声项
        max_iter=1200
    )
    
    print("\n🎯 分类 (30%标签翻转):")
    tester.test_classification(
        n_samples=800,
        n_features=12,
        n_classes=3,
        class_sep=0.6,  # 难分离
        label_noise_ratio=0.3,  # 30%标签翻转!
        label_noise_type='flip',
        hidden_layer_sizes=(128, 64, 32),
        gamma_init=20.0,
        b_noise_init=0.5,
        ovr_threshold_init=0.0,
        max_iter=1200
    )

def scenario_multi_class():
    """场景6: 多分类挑战"""
    print("\n🌈 场景6: 多分类挑战")
    print("=" * 50)
    
    tester = QuickTester()
    
    for n_classes in [2, 5, 8]:
        print(f"\n🎯 {n_classes}分类:")
        tester.test_classification(
            n_samples=1000,
            n_features=15,
            n_classes=n_classes,
            class_sep=0.8,
            label_noise_ratio=0.1,
            hidden_layer_sizes=(128, 64),
            gamma_init=10.0,
            b_noise_init=0.1,
            ovr_threshold_init=0.0,
            max_iter=800,
            verbose=False
        )
        
        # 简化显示
        results = tester.results['classification']
        print(f"   sklearn Acc:      {results['sklearn']['Acc']:.4f}")
        print(f"   deterministic Acc: {results['deterministic']['Acc']:.4f}")
        print(f"   standard Acc:     {results['standard']['Acc']:.4f}")

if __name__ == "__main__":
    print("🚀 CausalEngine 测试场景合集")
    print("=" * 60)
    
    # 运行所有场景
    scenario_clean_data()
    scenario_label_noise()
    scenario_network_comparison()
    scenario_parameter_sensitivity()
    scenario_extreme_noise()
    scenario_multi_class()
    
    print("\n✅ 所有测试场景完成!")
    print("📝 使用提示:")
    print("   - 单独运行场景: python test_scenarios.py")
    print("   - 自定义测试: from quick_test_causal_engine import QuickTester")
    print("   - 参数调节: 修改场景函数中的参数值")