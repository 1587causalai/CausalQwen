#!/usr/bin/env python3
"""
验证集大小设置演示脚本

🎯 目标：演示如何在两个鲁棒性分析脚本中设置验证集大小
🔬 核心：展示validation_fraction参数的使用和影响

使用方法：
python demo_validation_settings.py
"""

import sys
import os
sys.path.append('.')

from scripts.regression_robustness_real_datasets import CONFIG as REGRESSION_CONFIG
from scripts.classification_robustness_real_datasets import REAL_DATASETS_CONFIG as CLASSIFICATION_CONFIG

def demo_validation_settings():
    """演示验证集设置功能"""
    
    print("🔧 验证集大小设置演示")
    print("=" * 60)
    
    print("\n📊 回归脚本当前验证集设置:")
    print(f"  validation_fraction: {REGRESSION_CONFIG['validation_fraction']}")
    print(f"  early_stopping: {REGRESSION_CONFIG['early_stopping']}")
    print(f"  n_iter_no_change: {REGRESSION_CONFIG['n_iter_no_change']}")
    
    print("\n📊 分类脚本当前验证集设置:")
    print(f"  validation_fraction: {CLASSIFICATION_CONFIG['validation_fraction']}")
    print(f"  early_stopping: {CLASSIFICATION_CONFIG['early_stopping']}")
    print(f"  n_iter_no_change: {CLASSIFICATION_CONFIG['n_iter_no_change']}")
    
    print("\n🎯 验证集大小的影响:")
    print("  - validation_fraction 控制从训练集中分出多少作为验证集")
    print("  - 验证集用于早停机制，防止过拟合")
    print("  - 较大的验证集 (0.2) 提供更可靠的早停信号")
    print("  - 较小的验证集 (0.1) 保留更多数据用于训练")
    print("  - 默认值 0.15 是一个平衡的选择")
    
    print("\n🔧 如何修改验证集大小:")
    print("1. 回归脚本: 修改 scripts/regression_robustness_real_datasets.py 中的 CONFIG")
    print("2. 分类脚本: 修改 scripts/classification_robustness_real_datasets.py 中的 REAL_DATASETS_CONFIG")
    print("3. 或者在运行时动态修改配置字典")
    
    print("\n💡 示例配置:")
    example_config = {
        'validation_fraction': 0.2,     # 20%作为验证集
        'early_stopping': True,         # 开启早停
        'n_iter_no_change': 30,         # 30轮无改善则停止
    }
    
    for key, value in example_config.items():
        print(f"  '{key}': {value}")
    
    print("\n✅ 验证集设置演示完成!")
    print("💡 建议: 对于小数据集使用较小的validation_fraction (0.1-0.15)")
    print("💡 建议: 对于大数据集可以使用较大的validation_fraction (0.15-0.2)")

if __name__ == '__main__':
    demo_validation_settings()