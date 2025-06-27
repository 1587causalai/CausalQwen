#!/usr/bin/env python3
"""
使用新基准测试模块的简化示例
演示如何使用 causal_sklearn.benchmarks 进行快速基准对比
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_sklearn.benchmarks import BaselineBenchmark

def quick_regression_benchmark():
    """快速回归基准测试"""
    print("🚀 快速回归基准测试")
    print("=" * 60)
    
    benchmark = BaselineBenchmark()
    
    # 运行基准测试
    results = benchmark.benchmark_synthetic_data(
        task_type='regression',
        n_samples=1000,
        n_features=20,
        causal_modes=['deterministic', 'standard'],  # CausalEngine模式
        anomaly_ratio=0.1,  # 10%标签异常
        verbose=True
    )
    
    return results

def quick_classification_benchmark():
    """快速分类基准测试"""
    print("\n🚀 快速分类基准测试")
    print("=" * 60)
    
    benchmark = BaselineBenchmark()
    
    # 运行基准测试
    results = benchmark.benchmark_synthetic_data(
        task_type='classification',
        n_samples=1000,
        n_features=20,
        causal_modes=['deterministic', 'standard'],  # CausalEngine模式
        anomaly_ratio=0.05,  # 5%标签异常
        verbose=True
    )
    
    return results

if __name__ == "__main__":
    print("🧪 CausalEngine基准测试演示")
    print("使用新的模块化基准测试框架\n")
    
    # 回归基准测试
    reg_results = quick_regression_benchmark()
    
    # 分类基准测试  
    cls_results = quick_classification_benchmark()
    
    print("\n🎉 基准测试完成！")
    print("基准测试模块位置: causal_sklearn/benchmarks/")
    print("- BaselineBenchmark: 完整基准测试类")
    print("- PyTorchBaseline: PyTorch基线模型")