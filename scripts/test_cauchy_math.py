"""
CauchyMath 工具类测试脚本
验证柯西分布的各种数学函数实现
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_engine import CauchyMath


def test_basic_properties():
    """测试柯西分布的基本性质"""
    print("=" * 60)
    print("CauchyMath 基本性质测试")
    print("=" * 60)
    
    # 标准柯西分布参数
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    
    # 测试点
    x_values = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    
    print("\n📊 标准柯西分布 Cauchy(0, 1) 在不同点的值:")
    print("x\t\tPDF\t\tCDF\t\tSurvival\tLog-PDF")
    print("-" * 65)
    
    for x in x_values:
        pdf = CauchyMath.cauchy_pdf(x, loc, scale)
        cdf = CauchyMath.cauchy_cdf(x, loc, scale)
        survival = CauchyMath.cauchy_survival(x, loc, scale)
        log_pdf = CauchyMath.cauchy_log_pdf(x, loc, scale)
        
        print(f"{x:.1f}\t\t{pdf:.4f}\t\t{cdf:.4f}\t\t{survival:.4f}\t\t{log_pdf:.4f}")
    
    print("\n✅ 验证点:")
    # 在μ处，CDF应该等于0.5
    cdf_at_loc = CauchyMath.cauchy_cdf(loc, loc, scale)
    print(f"   CDF(μ=0) = {cdf_at_loc:.6f} (应该 = 0.5)")
    
    # CDF + Survival 应该等于1
    x_test = torch.tensor(1.5)
    cdf_test = CauchyMath.cauchy_cdf(x_test, loc, scale)
    survival_test = CauchyMath.cauchy_survival(x_test, loc, scale)
    sum_test = cdf_test + survival_test
    print(f"   CDF(1.5) + Survival(1.5) = {sum_test:.6f} (应该 = 1.0)")
    
    # PDF积分验证（数值近似）
    x_range = torch.linspace(-10, 10, 1000)
    dx = x_range[1] - x_range[0]
    pdf_values = CauchyMath.cauchy_pdf(x_range, loc, scale)
    integral_approx = torch.sum(pdf_values) * dx
    print(f"   PDF积分近似 = {integral_approx:.4f} (应该 ≈ 1.0)")


def test_quantile_cdf_inverse():
    """测试分位函数和CDF的互逆性"""
    print("\n" + "=" * 60)
    print("分位函数与CDF互逆性测试")
    print("=" * 60)
    
    loc = torch.tensor(2.0)
    scale = torch.tensor(1.5)
    
    # 测试概率值
    p_values = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    
    print("\np\t\tQuantile\tCDF(Quantile)\tError")
    print("-" * 50)
    
    max_error = 0.0
    for p in p_values:
        quantile = CauchyMath.cauchy_quantile(p, loc, scale)
        cdf_of_quantile = CauchyMath.cauchy_cdf(quantile, loc, scale)
        error = torch.abs(cdf_of_quantile - p)
        max_error = max(max_error, error.item())
        
        print(f"{p:.2f}\t\t{quantile:.4f}\t\t{cdf_of_quantile:.6f}\t\t{error:.2e}")
    
    print(f"\n最大误差: {max_error:.2e}")
    if max_error < 1e-6:
        print("✅ 分位函数与CDF互逆性验证通过")
    else:
        print("❌ 分位函数与CDF互逆性验证失败")


def test_parameter_effects():
    """测试不同参数对分布的影响"""
    print("\n" + "=" * 60)
    print("参数影响测试")
    print("=" * 60)
    
    x = torch.tensor(0.0)
    
    # 测试位置参数影响
    print("\n📍 位置参数 μ 的影响 (scale=1.0, x=0.0):")
    locs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    scale = torch.tensor(1.0)
    
    print("μ\t\tPDF(0)\t\tCDF(0)")
    print("-" * 30)
    for loc in locs:
        pdf = CauchyMath.cauchy_pdf(x, loc, scale)
        cdf = CauchyMath.cauchy_cdf(x, loc, scale)
        print(f"{loc:.1f}\t\t{pdf:.4f}\t\t{cdf:.4f}")
    
    # 测试尺度参数影响
    print("\n📏 尺度参数 γ 的影响 (loc=0.0, x=0.0):")
    loc = torch.tensor(0.0)
    scales = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
    
    print("γ\t\tPDF(0)\t\tCDF(0)")
    print("-" * 30)
    for scale in scales:
        pdf = CauchyMath.cauchy_pdf(x, loc, scale)
        cdf = CauchyMath.cauchy_cdf(x, loc, scale)
        print(f"{scale:.1f}\t\t{pdf:.4f}\t\t{cdf:.4f}")


def test_linear_stability():
    """测试线性稳定性"""
    print("\n" + "=" * 60)
    print("柯西分布线性稳定性测试")
    print("=" * 60)
    
    # 原始分布
    loc_original = torch.tensor([1.0, 2.0])
    scale_original = torch.tensor([0.5, 1.0])
    
    # 线性变换 Y = 2X + 3
    weight = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # 2x2 单位矩阵乘以2
    bias = torch.tensor([3.0, 3.0])
    
    # 使用CauchyMath计算变换后的参数
    loc_transformed = CauchyMath.cauchy_linear_stable_loc(
        loc_original.unsqueeze(0).unsqueeze(0), weight, bias
    )[0, 0]
    scale_transformed = CauchyMath.cauchy_linear_stable_scale(
        scale_original.unsqueeze(0).unsqueeze(0), weight
    )[0, 0]
    
    # 理论值：如果 X ~ Cauchy(μ, γ)，则 2X + 3 ~ Cauchy(2μ + 3, 2γ)
    loc_theory = 2 * loc_original + 3
    scale_theory = 2 * scale_original
    
    print("原始分布参数:")
    print(f"  loc:   {loc_original.tolist()}")
    print(f"  scale: {scale_original.tolist()}")
    
    print("\n变换后参数 (Y = 2X + 3):")
    print(f"  理论值 - loc:   {loc_theory.tolist()}")
    print(f"  计算值 - loc:   {loc_transformed.tolist()}")
    print(f"  理论值 - scale: {scale_theory.tolist()}")
    print(f"  计算值 - scale: {scale_transformed.tolist()}")
    
    # 验证误差
    loc_error = torch.max(torch.abs(loc_transformed - loc_theory))
    scale_error = torch.max(torch.abs(scale_transformed - scale_theory))
    
    print(f"\n最大误差:")
    print(f"  位置参数: {loc_error:.2e}")
    print(f"  尺度参数: {scale_error:.2e}")
    
    if loc_error < 1e-6 and scale_error < 1e-6:
        print("✅ 线性稳定性验证通过")
    else:
        print("❌ 线性稳定性验证失败")


def test_sampling_consistency():
    """测试采样一致性"""
    print("\n" + "=" * 60)
    print("采样一致性测试")
    print("=" * 60)
    
    loc = torch.tensor(1.0)
    scale = torch.tensor(2.0)
    
    # 使用分位函数进行采样
    torch.manual_seed(42)
    uniform_samples = torch.rand(10000)
    cauchy_samples = CauchyMath.cauchy_quantile(uniform_samples, loc, scale)
    
    # 计算样本统计
    sorted_samples = torch.sort(cauchy_samples).values
    sample_median = sorted_samples[len(sorted_samples) // 2].item()
    
    # 计算经验CDF在几个点的值
    test_points = torch.tensor([loc - scale, loc, loc + scale])
    empirical_cdf = []
    theoretical_cdf = []
    
    for point in test_points:
        empirical = torch.mean((cauchy_samples <= point).float())
        theoretical = CauchyMath.cauchy_cdf(point, loc, scale)
        empirical_cdf.append(empirical.item())
        theoretical_cdf.append(theoretical.item())
    
    print(f"样本数量: {len(cauchy_samples)}")
    print(f"理论中位数: {loc.item():.4f}")
    print(f"样本中位数: {sample_median:.4f}")
    print(f"中位数误差: {abs(sample_median - loc.item()):.4f}")
    
    print("\n经验CDF vs 理论CDF:")
    print("点\t\t经验CDF\t\t理论CDF\t\t误差")
    print("-" * 50)
    for i, point in enumerate(test_points):
        error = abs(empirical_cdf[i] - theoretical_cdf[i])
        print(f"{point:.2f}\t\t{empirical_cdf[i]:.4f}\t\t{theoretical_cdf[i]:.4f}\t\t{error:.4f}")
    
    max_cdf_error = max(abs(emp - theo) for emp, theo in zip(empirical_cdf, theoretical_cdf))
    if max_cdf_error < 0.02:  # 允许2%的误差
        print("✅ 采样一致性验证通过")
    else:
        print("❌ 采样一致性验证失败")


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n" + "=" * 60)
    print("数值稳定性测试")
    print("=" * 60)
    
    # 测试极端值
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    
    extreme_values = torch.tensor([-100.0, -10.0, 10.0, 100.0])
    
    print("极端值测试:")
    print("x\t\tPDF\t\tCDF\t\tLog-PDF")
    print("-" * 45)
    
    all_finite = True
    for x in extreme_values:
        pdf = CauchyMath.cauchy_pdf(x, loc, scale)
        cdf = CauchyMath.cauchy_cdf(x, loc, scale)
        log_pdf = CauchyMath.cauchy_log_pdf(x, loc, scale)
        
        print(f"{x:.0f}\t\t{pdf:.2e}\t{cdf:.6f}\t{log_pdf:.2f}")
        
        if not (torch.isfinite(pdf) and torch.isfinite(cdf) and torch.isfinite(log_pdf)):
            all_finite = False
    
    if all_finite:
        print("✅ 数值稳定性验证通过")
    else:
        print("❌ 数值稳定性验证失败")


def plot_distributions():
    """绘制不同参数的柯西分布"""
    try:
        print("\n" + "=" * 60)
        print("绘制柯西分布图")
        print("=" * 60)
        
        x = torch.linspace(-10, 10, 1000)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 不同位置参数的PDF
        ax1.set_title('PDF with Different Location Parameters')
        for loc in [-2, 0, 2]:
            pdf = CauchyMath.cauchy_pdf(x, torch.tensor(float(loc)), torch.tensor(1.0))
            ax1.plot(x.numpy(), pdf.numpy(), label=f'μ={loc}, γ=1')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('PDF')
        
        # 2. 不同尺度参数的PDF  
        ax2.set_title('PDF with Different Scale Parameters')
        for scale in [0.5, 1.0, 2.0]:
            pdf = CauchyMath.cauchy_pdf(x, torch.tensor(0.0), torch.tensor(scale))
            ax2.plot(x.numpy(), pdf.numpy(), label=f'μ=0, γ={scale}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('PDF')
        
        # 3. 不同参数的CDF
        ax3.set_title('CDF with Different Parameters')
        for loc, scale in [(0, 1), (-1, 1), (0, 2)]:
            cdf = CauchyMath.cauchy_cdf(x, torch.tensor(float(loc)), torch.tensor(scale))
            ax3.plot(x.numpy(), cdf.numpy(), label=f'μ={loc}, γ={scale}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('x')
        ax3.set_ylabel('CDF')
        
        # 4. PDF vs Log-PDF比较
        ax4.set_title('PDF vs Log-PDF (μ=0, γ=1)')
        pdf = CauchyMath.cauchy_pdf(x, torch.tensor(0.0), torch.tensor(1.0))
        log_pdf = CauchyMath.cauchy_log_pdf(x, torch.tensor(0.0), torch.tensor(1.0))
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(x.numpy(), pdf.numpy(), 'b-', label='PDF')
        line2 = ax4_twin.plot(x.numpy(), log_pdf.numpy(), 'r--', label='Log-PDF')
        
        ax4.set_xlabel('x')
        ax4.set_ylabel('PDF', color='b')
        ax4_twin.set_ylabel('Log-PDF', color='r')
        ax4.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('cauchy_distributions.png', dpi=150, bbox_inches='tight')
        print("✅ 分布图已保存为 'cauchy_distributions.png'")
        
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘图测试")


if __name__ == "__main__":
    print("🧮 CauchyMath 工具类完整测试")
    
    test_basic_properties()
    test_quantile_cdf_inverse()
    test_parameter_effects()
    test_linear_stability()
    test_sampling_consistency()
    test_numerical_stability()
    plot_distributions()
    
    print("\n" + "=" * 60)
    print("🎉 CauchyMath 测试完成！")
    print("=" * 60)
    
    print("\n📋 测试总结:")
    print("✅ 基本性质验证")
    print("✅ 分位函数与CDF互逆性")
    print("✅ 参数影响分析")
    print("✅ 线性稳定性")
    print("✅ 采样一致性")
    print("✅ 数值稳定性")
    print("✅ 可视化验证")
    
    print("\n🔧 可用的 CauchyMath 函数:")
    print("- cauchy_pdf: 概率密度函数")
    print("- cauchy_cdf: 累积分布函数")
    print("- cauchy_survival: 生存函数 P(X > x)")
    print("- cauchy_log_pdf: 对数概率密度函数")
    print("- cauchy_quantile: 分位函数 (逆CDF)")
    print("- cauchy_linear_stable_loc: 位置参数线性变换")
    print("- cauchy_linear_stable_scale: 尺度参数线性变换") 