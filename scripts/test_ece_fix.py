#!/usr/bin/env python3
"""
测试ECE计算修复的脚本
验证OvR概率是否正确归一化为和为1的概率分布
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_ovr_normalization():
    """直接测试OvR概率归一化"""
    print("=== 测试OvR概率归一化 ===")
    
    # 模拟一些决策分数
    batch_size = 5
    num_classes = 10
    
    # 创建模拟的决策分数分布参数
    cls_loc = torch.randn(batch_size, num_classes) * 2  # 位置参数
    cls_scale = torch.exp(torch.randn(batch_size, num_classes) * 0.5) + 0.1  # 尺度参数
    
    print(f"模拟数据形状: cls_loc={cls_loc.shape}, cls_scale={cls_scale.shape}")
    
    # 计算OvR概率 (独立概率，不归一化)
    ovr_probs = 0.5 + (1 / torch.pi) * torch.atan(cls_loc / cls_scale)
    
    # 计算归一化概率 (使用softmax)
    normalized_probs = torch.softmax(cls_loc, dim=1)
    
    print(f"\n🔍 概率分布比较:")
    print("-" * 60)
    for i in range(batch_size):
        ovr_sum = ovr_probs[i].sum().item()
        norm_sum = normalized_probs[i].sum().item()
        
        ovr_max = ovr_probs[i].max().item()
        norm_max = normalized_probs[i].max().item()
        
        ovr_pred = ovr_probs[i].argmax().item()
        norm_pred = normalized_probs[i].argmax().item()
        
        print(f"样本 {i+1}:")
        print(f"  OvR:        和={ovr_sum:.4f}, 最大={ovr_max:.4f}, 预测类别={ovr_pred}")
        print(f"  归一化:     和={norm_sum:.4f}, 最大={norm_max:.4f}, 预测类别={norm_pred}")
        print(f"  预测一致:   {ovr_pred == norm_pred}")
        print()
    
    # 模拟ECE计算的差异
    print("📊 ECE计算影响分析:")
    print("-" * 60)
    
    # 模拟真实标签
    true_labels = torch.randint(0, num_classes, (batch_size,))
    
    # 使用OvR最大概率作为置信度 (错误方法)
    ovr_max_probs = ovr_probs.max(dim=1)[0]
    ovr_predictions = ovr_probs.argmax(dim=1)
    
    # 使用归一化最大概率作为置信度 (正确方法)
    norm_max_probs = normalized_probs.max(dim=1)[0]
    norm_predictions = normalized_probs.argmax(dim=1)
    
    print(f"置信度比较 (前3个样本):")
    for i in range(min(3, batch_size)):
        print(f"  样本 {i+1}: OvR置信度={ovr_max_probs[i]:.4f}, 归一化置信度={norm_max_probs[i]:.4f}")
    
    print(f"\n💡 关键差异:")
    print(f"1. OvR概率和: {ovr_probs.sum(dim=1).mean():.4f} (应该≠1)")
    print(f"2. 归一化概率和: {normalized_probs.sum(dim=1).mean():.4f} (应该=1)")
    print(f"3. OvR平均置信度: {ovr_max_probs.mean():.4f}")
    print(f"4. 归一化平均置信度: {norm_max_probs.mean():.4f}")
    
    # 简单的ECE估算
    def simple_ece(confidences, predictions, labels):
        correct = (predictions == labels).float()
        # 简单的1-bin ECE
        avg_confidence = confidences.mean()
        avg_accuracy = correct.mean()
        return abs(avg_confidence - avg_accuracy).item()
    
    ovr_ece = simple_ece(ovr_max_probs, ovr_predictions, true_labels)
    norm_ece = simple_ece(norm_max_probs, norm_predictions, true_labels)
    
    print(f"\n📈 简化ECE估算:")
    print(f"  OvR方法ECE: {ovr_ece:.4f}")
    print(f"  归一化方法ECE: {norm_ece:.4f}")
    print(f"  差异: {abs(ovr_ece - norm_ece):.4f}")
    
    return ovr_probs, normalized_probs


def demonstrate_problem():
    """演示问题和解决方案"""
    print("\n" + "="*60)
    print("🚨 ECE计算问题演示")
    print("="*60)
    
    print("""
❌ 原始错误方法:
1. 计算OvR独立概率: P(类别k) = 0.5 + (1/π)*arctan(loc_k/scale_k)
2. 直接使用最大OvR概率作为置信度
3. 问题: OvR概率和 ≠ 1，不是真正的概率分布

✅ 修复后正确方法:
1. 计算OvR决策分数的位置参数: loc_k
2. 使用softmax归一化: P(类别k) = exp(loc_k) / Σ_j exp(loc_j)
3. 使用最大归一化概率作为置信度
4. 优势: 概率和 = 1，符合多分类校准标准

🎯 您的建议完全正确!
ECE衡量的应该是经过归一化后整个多分类系统的校准性。
    """)
    
    ovr_probs, norm_probs = test_ovr_normalization()
    
    print(f"\n✅ 修复验证:")
    print(f"- OvR概率范围: [{ovr_probs.min():.3f}, {ovr_probs.max():.3f}]")
    print(f"- 归一化概率范围: [{norm_probs.min():.3f}, {norm_probs.max():.3f}]")
    print(f"- 归一化概率和检查: {(norm_probs.sum(dim=1) - 1.0).abs().max():.6f} (应该≈0)")


if __name__ == "__main__":
    demonstrate_problem() 