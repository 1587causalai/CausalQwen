#!/usr/bin/env python
"""
行动网络 (Action Network) 流程图式验证脚本

本脚本旨在验证重构后的 `ActionNetwork` 的核心数学行为，
确保其初始化和前向传播完全符合 `mathematical_foundations.md` 中
关于知识迁移和柯西分布线性稳定性的设计。
"""
import os
import sys
import torch
import torch.nn as nn

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.action_network import ActionNetwork

def print_step(step_name, description):
    """打印流程图步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  步骤 {step_name}: {description}")
    print(f"{'-'*70}")

def print_tensor_stats(name, tensor, is_cauchy_param: bool = False):
    """打印张量的详细统计信息。
    
    Args:
        name (str): 张量的名称。
        tensor (torch.Tensor): 要分析的张量。
        is_cauchy_param (bool): 如果为 True，则仅使用对柯西分布具有鲁棒性的统计量（中位数和IQR）。
                                否则，使用标准统计量。
    """
    tensor_flat = tensor.detach().cpu().to(torch.float32).flatten()
    print(f"   - {name}:")
    print(f"     - Shape: {tensor.shape}")
    if is_cauchy_param:
        median = torch.median(tensor_flat).item()
        q1 = torch.quantile(tensor_flat, 0.25).item()
        q3 = torch.quantile(tensor_flat, 0.75).item()
        iqr = q3 - q1
        print(f"     - Median (中位数): {median:.4f}")
        print(f"     - IQR (四分位距): {iqr:.4f}")
    else:
        print(f"     - Mean (均值): {tensor_flat.mean().item():.4f}")
        print(f"     - Std (标准差):  {tensor_flat.std().item():.4f}")
        print(f"     - Min:  {tensor_flat.min().item():.4f}")
        print(f"     - Max:  {tensor_flat.max().item():.4f}")

def main():
    print("🚀 CausalQwen - Action Network 深度验证")
    
    # --- 参数定义 ---
    H = C = 896  # 隐藏维度 = 因果维度
    V = 151936   # 词汇表大小
    BATCH_SIZE = 4
    SEQ_LEN = 16
    
    print(f"\n设定参数: B={BATCH_SIZE}, S={SEQ_LEN}, H=C={H}, V={V}")

    # --- 步骤 1: 准备输入数据和模拟的Qwen lm_head ---
    print_step("1", "准备输入分布 U ~ Cauchy(loc_U, scale_U) 和 Qwen lm_head")
    
    # 模拟 AbductionNetwork 的输出
    # loc_U 是随机的，模拟 Qwen 最后一层的输出 z
    loc_U = torch.randn(BATCH_SIZE, SEQ_LEN, C)
    # scale_U 是一个大常数，模拟"无知先验"
    scale_U = torch.full_like(loc_U, 10.0)
    
    print("\n--- 输入分布 U ---")
    print_tensor_stats("loc_U (模拟z)", loc_U, is_cauchy_param=False)
    print_tensor_stats("scale_U (因果尺度)", scale_U, is_cauchy_param=True)

    # 创建一个模拟的 Qwen lm_head
    qwen_lm_head = nn.Linear(H, V, bias=False)
    print("\n--- 模拟的 Qwen lm_head ---")
    print(f"   - Shape: {qwen_lm_head.weight.shape}")

    # --- 步骤 2: 初始化 ActionNetwork 并进行知识迁移 ---
    print_step("2", "初始化 ActionNetwork 并进行知识迁移")
    action_net = ActionNetwork(input_dim=C, vocab_size=V)
    action_net.init_weights(qwen_lm_head=qwen_lm_head)
    print("   ✅ ActionNetwork 初始化并加载权重成功。")

    # --- 步骤 3: 执行前向传播 ---
    print_step("3", "执行前向传播: (S, Y) = ActionNetwork(U)")
    outputs = action_net(loc_U, scale_U)
    loc_S, scale_S = outputs['loc_S'], outputs['scale_S']
    loc_Y, scale_Y = outputs['loc_Y'], outputs['scale_Y']
    
    print("\n--- 输出 ---")
    print_tensor_stats("分类 loc_S", loc_S, is_cauchy_param=True)
    print_tensor_stats("分类 scale_S", scale_S, is_cauchy_param=True)
    print_tensor_stats("回归 loc_Y", loc_Y, is_cauchy_param=True)
    print_tensor_stats("回归 scale_Y", scale_Y, is_cauchy_param=True)

    # --- 步骤 4: 核心数学逻辑验证 ---
    print_step("4", "核心数学逻辑验证")
    
    # 验证 1: 分类知识迁移 (loc_S)
    # ActionNetwork的分类loc头权重已经复制了qwen_lm_head
    # 因此，对同一个输入z (即loc_U)，两者的输出应该完全相同
    qwen_logits = qwen_lm_head(loc_U)
    cls_knowledge_transfer_ok = torch.allclose(loc_S, qwen_logits, atol=1e-6)
    print(f"\n   --- 验证 1: 分类知识迁移 (loc_S vs Qwen logits) ---")
    print(f"     - 理论: loc_S 应与 qwen_lm_head(loc_U) 的输出完全相同。")
    print(f"     - 结论: {'✅ 通过' if cls_knowledge_transfer_ok else '❌ 失败'}")
    if not cls_knowledge_transfer_ok:
        print(f"     - 差异 (L1 Loss): {torch.nn.functional.l1_loss(loc_S, qwen_logits).item():.8f}")

    # 验证 2: 回归无偏先验 (loc_Y)
    # 回归头的权重被初始化为非常小的值，偏置为0，所以输出应该接近于0
    # 由于 loc_U 的值域在 [-4, 4] 之间，我们将容忍度放宽
    reg_prior_ok = loc_Y.abs().max() < 1e-1
    print(f"\n   --- 验证 2: 回归无偏先验 (loc_Y ≈ 0) ---")
    print(f"     - 理论: 回归头的权重被初始化为接近0，因此 loc_Y 输出应接近0。")
    print(f"     - 实际最大绝对值: {loc_Y.abs().max().item():.6f}")
    print(f"     - 结论: {'✅ 通过' if reg_prior_ok else '❌ 失败'}")
        
    # 验证 3: 尺度参数传递
    # scale_S = |W_cls| * scale_U, scale_Y = |W_reg| * scale_U
    # 由于 scale_U 是常数10，W_cls不为0，W_reg接近0，我们预期：
    # scale_S 是一个较大的常数，scale_Y 是一个较小的正数。
    scale_S_ok = scale_S.median() > 1.0 # 期望是一个显著大于0的数
    # scale_Y 是 |W_reg| 与 scale_U 的矩阵乘积，应该是一个较小的正数，
    # 且应远小于 scale_S。我们验证它是否小于 scale_S 的十分之一。
    scale_Y_ok = scale_Y.median() < (scale_S.median() / 10.0)
    print(f"\n   --- 验证 3: 尺度参数传递 (scale_S, scale_Y) ---")
    print(f"     - 理论: scale_S = |W_cls|*scale_U (大), scale_Y = |W_reg|*scale_U (小)")
    print(f"     - 实际 scale_S (中位数): {scale_S.median().item():.4f}")
    print(f"     - 实际 scale_Y (中位数): {scale_Y.median().item():.4f}")
    print(f"     - 结论 (分类): {'✅ 通过' if scale_S_ok else '❌ 失败'}")
    print(f"     - 结论 (回归): {'✅ 通过' if scale_Y_ok else '❌ 失败'}")

    print(f"\n\n{'='*80}")
    final_success = cls_knowledge_transfer_ok and reg_prior_ok and scale_S_ok and scale_Y_ok
    if final_success:
        print("🎉 验证成功！ActionNetwork 的实现完全符合数学设计。")
    else:
        print("❌ 验证失败！请检查初始化或前向传播逻辑。")

if __name__ == '__main__':
    main() 