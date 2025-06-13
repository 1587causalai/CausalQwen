#!/usr/bin/env python
"""
分类损失 (Classification Loss) 流程图式验证脚本

本脚本严格按照 `mathematical_foundations.md` 中的 "图 5.1" 流程图，
旨在白盒测试核心损失函数 `ovr_classification_loss` 的计算逻辑是否精确。
"""
import torch
import torch.nn.functional as F
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses.loss_functions import ovr_classification_loss

def print_step(step_name, description):
    """打印流程图步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  步骤 {step_name}: {description}")
    print(f"{'-'*70}")

def main():
    print("🚀 CausalQwen - 分类损失 (L_cls) 核心逻辑深度验证")

    # --- 参数定义 ---
    B, S, V = 4, 16, 128 # 使用较小的 V 以加速测试
    C_OVR = 100.0
    print(f"\n设定参数: B={B}, S={S}, V={V}, C_OvR={C_OVR}")

    # --- 步骤 1: 准备模拟输入 ---
    print_step("1", "准备模拟输入: loc_S, scale_S, 和真实标签")
    loc_S = torch.randn(B, S, V) * 50
    scale_S = torch.rand(B, S, V) * 5 + 1
    labels = torch.randint(0, V, (B, S))
    
    # 根据 loc 和 scale 计算概率 P，这是损失函数的直接输入
    prob_P = 0.5 + (1 / torch.pi) * torch.atan((loc_S - C_OVR) / scale_S)

    # --- 步骤 2: 手动计算损失 (Ground Truth) ---
    print_step("2", "手动计算损失 (Ground Truth) - 精确模仿底层函数")
    
    epsilon = 1e-8
    logits_manual = torch.log(prob_P / (1 - prob_P + epsilon) + epsilon)
    y_one_hot = F.one_hot(labels, num_classes=V).float()
    
    bce_loss_unreduced = F.binary_cross_entropy_with_logits(
        logits_manual, y_one_hot, reduction='none'
    )
    L_cls_per_token = bce_loss_unreduced.sum(dim=-1)
    expected_scalar_loss = L_cls_per_token.mean()
    
    print(f"   - 预期的标量损失 (手动计算): {expected_scalar_loss.item():.6f}")

    # --- 步骤 3: 使用 `ovr_classification_loss` 函数计算 ---
    print_step("3", "使用 `ovr_classification_loss` 函数计算")
    actual_scalar_loss = ovr_classification_loss(
        probs=prob_P, 
        targets=labels,
        reduction='mean'
    )
    print(f"   - 实际的标量损失 (函数计算): {actual_scalar_loss.item():.6f}")
    
    # --- 步骤 4: 核心数学逻辑验证 ---
    print_step("4", "核心数学逻辑验证")
    
    loss_match = torch.allclose(expected_scalar_loss, actual_scalar_loss, atol=1e-5)
    
    print(f"\n   --- 验证: 手动计算 vs. 函数计算 ---")
    print(f"     - 结论: {'✅ 通过' if loss_match else '❌ 失败'}")
    if not loss_match:
        diff = torch.abs(expected_scalar_loss - actual_scalar_loss)
        print(f"     - 绝对差异: {diff.item():.8f}")

    print(f"\n\n{'='*80}")
    if loss_match:
        print("🎉 验证成功！`ovr_classification_loss` 的实现完全符合数学设计。")
    else:
        print("❌ 验证失败！请检查 `ovr_classification_loss` 的内部逻辑。")

if __name__ == '__main__':
    main() 