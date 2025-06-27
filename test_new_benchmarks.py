#!/usr/bin/env python3
"""
测试新的基准测试框架

验证所有新增的基准方法是否正常工作。
"""

import numpy as np
from sklearn.datasets import make_regression
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_sklearn.benchmarks import (
    BaselineBenchmark, 
    MethodDependencyChecker, 
    get_method_group,
    list_available_methods
)

def test_dependency_availability():
    """测试依赖可用性检查"""
    print("🔍 测试依赖可用性检查")
    print("=" * 50)
    
    checker = MethodDependencyChecker()
    checker.print_dependency_status()
    
    return True

def test_method_listing():
    """测试方法列表功能"""
    print("\n📝 测试方法列表功能")
    print("=" * 50)
    
    # 列出所有可用方法
    methods = list_available_methods()
    print(f"可用方法总数: {len(methods)}")
    print(f"前5个方法: {methods[:5]}")
    
    # 测试方法组合
    basic_group = get_method_group('basic')
    print(f"基础组合: {basic_group}")
    
    comprehensive_group = get_method_group('comprehensive')
    print(f"全面组合: {comprehensive_group}")
    
    return True

def test_benchmark_creation():
    """测试基准测试实例创建"""
    print("\n🏗️ 测试基准测试实例创建")
    print("=" * 50)
    
    try:
        benchmark = BaselineBenchmark()
        print("✅ BaselineBenchmark 创建成功")
        
        # 打印方法可用性
        benchmark.print_method_availability()
        
        return True
    except Exception as e:
        print(f"❌ BaselineBenchmark 创建失败: {e}")
        return False

def test_small_dataset_benchmark():
    """在小数据集上测试基准测试"""
    print("\n🧪 在小数据集上测试基准测试")
    print("=" * 50)
    
    # 创建小型合成数据集
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    
    try:
        benchmark = BaselineBenchmark()
        
        # 测试轻量级方法组合
        print("测试轻量级方法组合...")
        results = benchmark.compare_models(
            X=X, 
            y=y,
            task_type='regression',
            baseline_methods=['sklearn_mlp', 'random_forest'],  # 只测试2个简单方法
            causal_modes=['deterministic', 'standard'],          # 只测试2个CausalEngine模式
            anomaly_ratio=0.1,
            verbose=True
        )
        
        print(f"\n✅ 基准测试完成，得到 {len(results)} 个结果")
        
        # 打印简单的结果摘要
        for method, metrics in results.items():
            r2 = metrics['test']['R²']
            mae = metrics['test']['MAE']
            print(f"   {method}: R² = {r2:.3f}, MAE = {mae:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 小数据集基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_availability_filtering():
    """测试方法可用性过滤"""
    print("\n🔍 测试方法可用性过滤")
    print("=" * 50)
    
    try:
        from causal_sklearn.benchmarks.methods import filter_available_methods
        
        # 测试混合的方法列表（包含可用和不可用的）
        test_methods = [
            'sklearn_mlp',      # 应该可用
            'random_forest',    # 应该可用
            'xgboost',         # 可能不可用
            'lightgbm',        # 可能不可用
            'nonexistent_method'  # 不存在的方法
        ]
        
        available, unavailable = filter_available_methods(test_methods)
        
        print(f"可用方法: {available}")
        print(f"不可用方法: {unavailable}")
        
        # 基本验证
        assert 'sklearn_mlp' in available, "sklearn_mlp应该是可用的"
        assert 'random_forest' in available, "random_forest应该是可用的"
        assert 'nonexistent_method' in unavailable, "nonexistent_method应该是不可用的"
        
        print("✅ 方法可用性过滤测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 方法可用性过滤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("🧪 新基准测试框架验证")
    print("=" * 60)
    
    tests = [
        ("依赖可用性检查", test_dependency_availability),
        ("方法列表功能", test_method_listing),
        ("基准测试实例创建", test_benchmark_creation),
        ("方法可用性过滤", test_method_availability_filtering),
        ("小数据集基准测试", test_small_dataset_benchmark),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 出现异常: {e}")
            results[test_name] = False
    
    # 总结报告
    print("\n📊 测试总结报告")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name:<25} {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！新基准测试框架工作正常。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)