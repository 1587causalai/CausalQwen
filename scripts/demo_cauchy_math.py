"""
CauchyMath 工具类演示脚本
展示新增的柯西分布数学函数
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_engine import CauchyMath


def main():
    print("🧮 CauchyMath 工具类演示")
    print("=" * 50)
    
    # 设置柯西分布参数
    loc = torch.tensor(1.0)     # 位置参数 μ
    scale = torch.tensor(2.0)   # 尺度参数 γ
    x = torch.tensor(3.0)       # 测试点
    
    print(f"📍 柯西分布 Cauchy(μ={loc.item()}, γ={scale.item()})")
    print(f"🎯 在点 x={x.item()} 处的计算结果:")
    print("-" * 30)
    
    # 基础分布函数
    pdf = CauchyMath.cauchy_pdf(x, loc, scale)
    cdf = CauchyMath.cauchy_cdf(x, loc, scale)
    survival = CauchyMath.cauchy_survival(x, loc, scale)
    log_pdf = CauchyMath.cauchy_log_pdf(x, loc, scale)
    
    print(f"PDF(x):      {pdf:.6f}")
    print(f"CDF(x):      {cdf:.6f}")
    print(f"Survival(x): {survival:.6f}")
    print(f"Log-PDF(x):  {log_pdf:.6f}")
    
    # 验证性质
    print("\n🔍 验证分布性质:")
    print(f"CDF + Survival = {(cdf + survival):.6f} (应该 = 1.0)")
    
    # 分位函数演示
    print("\n📊 分位函数演示:")
    probabilities = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    print("概率 → 分位数 → 验证CDF")
    print("-" * 30)
    
    for p in probabilities:
        quantile = CauchyMath.cauchy_quantile(p, loc, scale)
        cdf_check = CauchyMath.cauchy_cdf(quantile, loc, scale)
        print(f"{p:.2f} → {quantile:.3f} → {cdf_check:.6f}")
    
    # 采样演示
    print("\n🎲 采样演示:")
    torch.manual_seed(42)
    uniform_samples = torch.rand(5)
    cauchy_samples = CauchyMath.cauchy_quantile(uniform_samples, loc, scale)
    
    print("均匀样本 → 柯西样本")
    print("-" * 20)
    for u, c in zip(uniform_samples, cauchy_samples):
        print(f"{u:.3f} → {c:.3f}")
    
    # 线性稳定性演示
    print("\n🔧 线性稳定性演示:")
    print("如果 X ~ Cauchy(1, 2)，那么 Y = 3X + 5 ~ Cauchy(8, 6)")
    
    # 原始分布
    original_samples = CauchyMath.cauchy_quantile(torch.rand(1000), loc, scale)
    
    # 手动线性变换
    transformed_manual = 3 * original_samples + 5
    
    # 使用CauchyMath计算理论参数
    weight = torch.tensor([[3.0]])
    bias = torch.tensor([5.0])
    
    new_loc = CauchyMath.cauchy_linear_stable_loc(
        loc.unsqueeze(0).unsqueeze(0), weight, bias
    )[0, 0]
    new_scale = CauchyMath.cauchy_linear_stable_scale(
        scale.unsqueeze(0).unsqueeze(0), weight
    )[0, 0]
    
    print(f"理论新分布: Cauchy({new_loc:.1f}, {new_scale:.1f})")
    
    # 验证样本 - 使用排序方法避免版本兼容性问题
    sorted_samples = torch.sort(transformed_manual).values
    sample_median = sorted_samples[len(sorted_samples) // 2].item()
    print(f"样本中位数: {sample_median:.2f} (应该 ≈ {new_loc:.1f})")
    
    print("\n✅ CauchyMath 演示完成！")
    print("\n🔧 可用函数列表:")
    print("- cauchy_pdf: 概率密度函数")
    print("- cauchy_cdf: 累积分布函数") 
    print("- cauchy_survival: 生存函数")
    print("- cauchy_log_pdf: 对数概率密度函数")
    print("- cauchy_quantile: 分位函数")
    print("- cauchy_linear_stable_loc: 位置参数线性变换")
    print("- cauchy_linear_stable_scale: 尺度参数线性变换")


if __name__ == "__main__":
    main() 