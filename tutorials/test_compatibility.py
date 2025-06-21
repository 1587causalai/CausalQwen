"""
测试兼容层是否正常工作
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutorials.utils.ablation_networks import (
    create_full_causal_classifier, 
    create_full_causal_regressor,
    create_ablated_classifier,
    create_ablated_regressor
)

print("🧪 测试兼容层...")

# 测试分类器
print("\n1. 测试分类器兼容层")
try:
    clf = create_full_causal_classifier(input_size=20, num_classes=3)
    print("   ✅ 创建分类器成功")
    
    # 测试前向传播
    x = torch.randn(32, 20)  # batch_size=32, features=20
    output = clf(x)
    print(f"   ✅ 前向传播成功，输出形状: {output.shape}")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试回归器
print("\n2. 测试回归器兼容层")
try:
    reg = create_full_causal_regressor(input_size=15, output_size=1)
    print("   ✅ 创建回归器成功")
    
    # 测试前向传播
    x = torch.randn(32, 15)  # batch_size=32, features=15
    output = reg(x)
    print(f"   ✅ 前向传播成功，输出形状: {output.shape}")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

print("\n✨ 兼容层测试完成！") 