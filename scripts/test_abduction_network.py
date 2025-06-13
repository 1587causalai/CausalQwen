#!/usr/bin/env python
"""
归因推断网络 (Abduction Network) 流程图式验证脚本

本脚本旨在验证重构后的 `AbductionNetwork` 的核心数学行为，
确保其初始化和前向传播完全符合 `mathematical_foundations.md` 的设计。
"""
import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.abduction_network import AbductionNetwork

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
        is_cauchy_param (bool): 如果为 True，则使用对柯西分布更具鲁棒性的统计量（中位数和IQR）。
                                否则，使用标准统计量（均值和标准差）。
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
    print("🚀 CausalQwen - Abduction Network 深度验证")
    
    # --- 参数定义 ---
    H = C = 896  # 隐藏维度 = 因果维度
    BATCH_SIZE = 4
    SEQ_LEN = 16
    SCALE_BIAS = 10.0  # softplus(10.0) ≈ 10.0
    
    print(f"\n设定参数: B={BATCH_SIZE}, S={SEQ_LEN}, H=C={H}")

    # --- 步骤 1: 初始化网络 ---
    print_step("1", "初始化 Abduction Network")
    abduction_net = AbductionNetwork(hidden_size=H, causal_dim=C)
    print("   ✅ 网络实例化成功。")
    
    # --- 步骤 2: 应用恒等映射初始化 ---
    print_step("2", "应用恒等映射初始化")
    print(f"   - 目标: loc=z, scale=softplus({SCALE_BIAS})≈10")
    abduction_net.initialize_for_identity_mapping(scale_bias=SCALE_BIAS)
    print("   ✅ 初始化方法调用成功。")

    # --- 步骤 3: 准备输入数据 ---
    print_step("3", "准备随机输入张量 z")
    z = torch.randn(BATCH_SIZE, SEQ_LEN, H)
    print_tensor_stats("输入 z", z, is_cauchy_param=False)

    # --- 步骤 4: 执行前向传播 ---
    print_step("4", "执行前向传播: (loc, scale) = AbductionNetwork(z)")
    loc, scale = abduction_net(z)
    
    print("\n   --- 输出 ---")
    print_tensor_stats("输出 loc (柯西位置参数)", loc, is_cauchy_param=True)
    print_tensor_stats("输出 scale (柯西尺度参数)", scale, is_cauchy_param=True)

    # --- 步骤 5: 核心数学逻辑验证 ---
    print_step("5", "核心数学逻辑验证")
    
    # 验证 1: loc 是否实现了恒等映射
    loc_is_identity = torch.allclose(loc, z, atol=1e-6)
    print(f"\n   --- 验证 1: 恒等映射 (loc ≈ z) ---")
    print(f"     - 结论: {'✅ 通过' if loc_is_identity else '❌ 失败'}")
    if not loc_is_identity:
        # 对于柯西分布的位置参数，L1损失比L2损失（MSE）更具数学意义
        print(f"     - 差异 (L1 Loss): {torch.nn.functional.l1_loss(loc, z).item():.8f}")
        
    # 验证 2: scale 是否为接近 10 的常数
    expected_scale_val = torch.nn.functional.softplus(torch.tensor(SCALE_BIAS)).item()
    scale_is_constant = torch.allclose(
        scale, 
        torch.full_like(scale, expected_scale_val), 
        atol=1e-5
    )
    scale_flat = scale.flatten()
    scale_iqr = (torch.quantile(scale_flat, 0.75) - torch.quantile(scale_flat, 0.25)).item()
    
    print(f"\n   --- 验证 2: 高不确定性柯西先验 (scale ≈ large constant) ---")
    print(f"     - 目标: scale为一个大常数，形成宽泛的'无知先验'")
    print(f"     - 理论值: {expected_scale_val:.4f}")
    print(f"     - 实际中位数: {torch.median(scale).item():.4f}")
    print(f"     - 实际IQR: {scale_iqr:.4f}")
    print(f"     - 结论: {'✅ 通过' if scale_is_constant and scale_iqr < 1e-6 else '❌ 失败'}")

    print(f"\n\n{'='*80}")
    if loc_is_identity and scale_is_constant:
        print("🎉 验证成功！AbductionNetwork 的实现完全符合数学设计。")
    else:
        print("❌ 验证失败！请检查初始化或前向传播逻辑。")

if __name__ == '__main__':
    main() 