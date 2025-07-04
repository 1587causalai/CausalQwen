#!/usr/bin/env python3
"""
使用改进后的扩展脚本运行分析

特点：
- 11个噪声级别（0%-100%）
- 多次运行取平均值（无误差条）
- 更稳定的超参数
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.quick_test_causal_engine_extended import run_robustness_analysis, EXTENDED_CONFIG

def main():
    """运行改进后的鲁棒性分析"""
    print("🚀 运行改进后的噪声鲁棒性分析")
    print("=" * 60)
    
    # 使用默认配置，这里有所有的改进：
    # - 11个噪声级别 (0%-100%)
    # - 3000样本
    # - 学习率 0.001
    # - 3次运行取平均
    # - 无误差条，只显示平均值线条
    
    print("🔧 当前配置:")
    print(f"  - 噪声级别: {len(EXTENDED_CONFIG['noise_levels'])}个 (0%-100%)")
    print(f"  - 样本数: {EXTENDED_CONFIG['n_samples']}")
    print(f"  - 运行次数: {EXTENDED_CONFIG['n_runs']}")
    print(f"  - 学习率: {EXTENDED_CONFIG['learning_rate']}")
    print(f"  - 最大迭代: {EXTENDED_CONFIG['max_iter']}")
    print(f"  - 早停耐心: {EXTENDED_CONFIG['patience']}")
    
    # 运行分析
    regression_results, classification_results = run_robustness_analysis()
    
    print("\n✅ 分析完成！")
    print(f"📊 结果保存在: {EXTENDED_CONFIG['output_dir']}")

if __name__ == '__main__':
    main()