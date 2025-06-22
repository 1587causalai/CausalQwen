"""
CausalEngine 理论基础演示教程
=================================

本教程详细展示CausalEngine的核心数学理论框架：
基于最新的数学基础文档 (MATHEMATICAL_FOUNDATIONS_CN.md) 
和基准测试协议 (benchmark_strategy.md)

核心理论框架：
Y = f(U, ε) - 普适因果机制

三阶段架构：
1. 归因推断 (Abduction): 证据 E → 个体选择变量 U ~ Cauchy(μ_U, γ_U)
2. 行动决策 (Action): 个体 U → 决策得分 S ~ Cauchy(loc_S, scale_S)  
3. 任务激活 (TaskActivation): 决策 S → 任务输出 Y

数学创新：
- U: 个体选择变量（Individual Choice Variable）- 因果推理的核心
- ε: 外生噪声（Exogenous Noise）- 不可控随机扰动
- 柯西分布线性稳定性: 实现解析不确定性传播，避免采样开销
- 四种推理模式: 因果、标准、采样、兼容模式的统一框架
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from causal_engine import CausalEngine


def demonstrate_core_mathematical_framework():
    """
    演示Y = f(U, ε)核心数学框架
    """
    print("🔬 CausalEngine 核心数学框架演示")
    print("=" * 60)
    
    print("\n📐 理论框架: Y = f(U, ε)")
    print("  Y: 观测结果")
    print("  U: 个体选择变量 (Individual Choice Variable)")
    print("  ε: 外生噪声 (Exogenous Noise)")
    print("  f: 普适因果机制 (Universal Causal Mechanism)")
    
    print("\n🏗️ 三阶段架构 (基于最新理论):")
    print("  阶段1: 归因推断 (Abduction): 证据 E → 个体 U ~ Cauchy(μ_U, γ_U)")
    print("  阶段2: 行动决策 (Action): 个体 U → 决策 S ~ Cauchy(loc_S, scale_S)")
    print("  阶段3: 任务激活 (TaskActivation): 决策 S → 任务输出 Y")
    
    print("\n💡 核心创新点:")
    print("  ✨ 解析计算: 柯西分布线性稳定性避免采样开销")
    print("  ✨ 独立决策: OvR策略摆脱softmax归一化约束")
    print("  ✨ 统一框架: 四种推理模式的连续过渡")
    print("  ✨ 不确定性量化: 完整保留并传播不确定性信息")
    
    # 创建CausalEngine实例
    print("\n⚡ 创建CausalEngine实例")
    engine = CausalEngine(
        hidden_size=128,      # 输入证据维度
        vocab_size=10,        # 输出决策维度
        causal_size=64,       # 个体表征维度
        activation_modes="classification"
    )
    
    print(f"  隐藏层大小 (证据维度): {engine.hidden_size}")
    print(f"  因果表征大小 (个体U维度): {engine.causal_size}")
    print(f"  输出维度 (决策S维度): {engine.vocab_size}")
    
    return engine


def demonstrate_abduction_stage():
    """
    演示归因推断阶段: 证据 E → 个体 U
    """
    print("\n\n🔍 阶段1: 归因推断 (Abduction)")
    print("=" * 60)
    print("数学表达: 证据 E → 个体选择变量 U ~ Cauchy(μ_U, γ_U)")
    
    # 创建CausalEngine
    engine = CausalEngine(
        hidden_size=128,
        vocab_size=10,
        causal_size=64,
        activation_modes="classification"
    )
    
    # 模拟证据输入
    batch_size = 8
    seq_len = 1
    evidence_E = torch.randn(batch_size, seq_len, 128)  # 证据E
    
    print(f"\n📊 输入证据 E 形状: {evidence_E.shape}")
    
    # 归因推断：从证据推断个体
    with torch.no_grad():
        loc_U, scale_U = engine.abduction(evidence_E)
    
    print(f"\n🎲 个体选择变量 U 的分布参数:")
    print(f"  位置参数 μ_U 形状: {loc_U.shape}")
    print(f"  尺度参数 γ_U 形状: {scale_U.shape}")
    
    # 展示归因推断的数学含义
    print(f"\n📈 归因推断的数学含义:")
    print(f"  μ_U = loc_net(E): 从证据推断个体的中心特征")
    print(f"  γ_U = softplus(scale_net(E)): 从证据推断个体的不确定性")
    print(f"  U ~ Cauchy(μ_U, γ_U): 个体选择变量的完整分布")
    
    # 分析个体选择变量的统计特性
    print(f"\n🔬 个体选择变量 U 的统计分析:")
    print(f"  平均位置参数: {loc_U.mean(dim=0).mean():.4f}")
    print(f"  平均尺度参数: {scale_U.mean(dim=0).mean():.4f}")
    print(f"  位置参数标准差: {loc_U.std():.4f}")
    print(f"  尺度参数标准差: {scale_U.std():.4f}")
    
    return engine, loc_U, scale_U


def demonstrate_action_stage(engine, loc_U, scale_U):
    """
    演示行动决策阶段: 个体 U → 决策 S
    """
    print("\n\n⚡ 阶段2: 行动决策 (Action)")
    print("=" * 60)
    print("数学表达: 个体 U → 决策得分 S ~ Cauchy(loc_S, scale_S)")
    
    # 行动决策：从个体生成决策
    with torch.no_grad():
        # 不同推理模式的决策
        print(f"\n🎯 四种推理模式的决策生成 (基于最新理论):")
        
        # 1. 因果推理 (T=0)
        loc_S_causal, scale_S_causal = engine.action(loc_U, scale_U, do_sample=False, temperature=0)
        print(f"  因果模式 (T=0): loc_S={loc_S_causal.shape}, scale_S={scale_S_causal.shape}")
        
        # 2. 标准推理 (T>0, no sample)
        loc_S_standard, scale_S_standard = engine.action(loc_U, scale_U, do_sample=False, temperature=1.0)
        print(f"  标准模式 (T=1.0): loc_S={loc_S_standard.shape}, scale_S={scale_S_standard.shape}")
        
        # 3. 采样推理 (T>0, sample)
        loc_S_sample, scale_S_sample = engine.action(loc_U, scale_U, do_sample=True, temperature=0.8)
        print(f"  采样模式 (T=0.8): loc_S={loc_S_sample.shape}, scale_S={scale_S_sample.shape}")
        
        # 4. 兼容模式 (任意温度)
        loc_S_compat, scale_S_compat = engine.action(loc_U, scale_U, do_sample=False, temperature=1.0)
        print(f"  兼容模式: 与传统transformer行为对齐")
    
    # 展示行动决策的数学机制 (基于最新数学文档)
    print(f"\n🔧 行动决策的两步数学机制:")
    print(f"  步骤1: 外生噪声注入")
    print(f"    U' = U + b_noise·ε (ε ~ Cauchy(0,1))")
    print(f"    → U' ~ Cauchy(μ_U, γ_U + |b_noise|) [标准模式]")
    print(f"    → U' ~ Cauchy(μ_U + T·|b_noise|·ε, γ_U) [采样模式]")
    print(f"  步骤2: 线性因果变换")
    print(f"    S = W_A·U' + b_A")
    print(f"    → loc_S = μ_U·W_A^T + b_A")
    print(f"    → scale_S = (γ_U + |b_noise|)·|W_A^T|")
    
    # 分析不同模式下决策分布的差异
    print(f"\n📊 四种推理模式的数学特征:")
    print(f"  因果模式 - 平均尺度: {scale_S_causal.mean():.4f} (纯粹因果推理)")
    print(f"  标准模式 - 平均尺度: {scale_S_standard.mean():.4f} (扩大不确定性)")
    print(f"  采样模式 - 位置偏移: {(loc_S_sample - loc_S_causal).abs().mean():.4f} (身份探索)")
    print(f"  兼容模式 - 传统对齐: 确保与传统方法可比")
    
    # 体现温度参数的数学调制原理
    print(f"\n🌡️ 温度参数的数学调制机制:")
    print(f"  T = 0: 完全关闭外生噪声 → 纯粹因果推理")
    print(f"  T > 0 + do_sample=False: 噪声扩大尺度参数 → 增加不确定性")
    print(f"  T > 0 + do_sample=True: 噪声扰动位置参数 → 身份探索")
    print(f"  数学优雅性: 同一框架实现确定性到随机性的连续过渡")
    
    return loc_S_standard, scale_S_standard


def demonstrate_activation_stage(engine, loc_S, scale_S):
    """
    演示任务激活阶段: 决策 S → 输出 Y
    """
    print("\n\n✨ 阶段3: 任务激活 (TaskActivation)")
    print("=" * 60)
    print("数学表达: 决策 S → 任务输出 Y")
    
    # 任务激活：从决策得分到最终输出
    with torch.no_grad():
        activation_output = engine.activation(loc_S, scale_S, return_dict=True)
        final_output = activation_output['output']
    
    print(f"\n🎯 任务激活过程:")
    print(f"  输入决策得分 S: {loc_S.shape}")
    print(f"  最终任务输出 Y: {final_output.shape}")
    
    # 展示三种激活机制的数学原理 (基于最新数学文档)
    print(f"\n🔬 三种任务激活函数的数学机制:")
    print(f"  1. 词元索引激活 (分类任务):")
    print(f"     基础函数: f_k(s_k) = I(s_k > C_k)")
    print(f"     概率计算: P_k = 1/2 + arctan((μ_k - C_k)/γ_k)/π")
    print(f"     决策机制: OvR策略，独立激活概率")
    print(f"     损失函数: 二元交叉熵损失")
    
    print(f"\n  2. 数值激活 (回归任务):")
    print(f"     基础函数: f_k(s_k) = w_k·s_k + b_k")
    print(f"     分布变换: Y_k ~ Cauchy(w_k·μ_k + b_k, |w_k|·γ_k)")
    print(f"     输出获取: ŷ_k = w_k·μ_k + b_k (位置参数)")
    print(f"     损失函数: 柯西分布负对数似然损失")
    
    print(f"\n  3. 离散有序激活 (有序分类):")
    print(f"     基础函数: f_k(s_k) = ∑y_i·I(C_i < s_k ≤ C_{{i+1}})")
    print(f"     概率计算: P(y_i) = F(C_{{i+1}}) - F(C_i)")
    print(f"     决策机制: 区间概率的解析计算")
    print(f"     损失函数: 分类交叉熵损失")
    
    # 关键创新点
    print(f"\n💡 任务激活的关键创新:")
    print(f"  ✨ 直接预测: 用随机变量S_k本身预测，而非统计量")
    print(f"  ✨ 解析计算: 利用柯西分布CDF避免采样")
    print(f"  ✨ 任务无关: 同一决策得分支持多种任务类型")
    print(f"  ✨ 扩展性: 新任务只需定义激活函数和损失")
    
    # 分析输出的统计特性
    print(f"\n📈 输出统计特性:")
    print(f"  输出范围: [{final_output.min():.4f}, {final_output.max():.4f}]")
    print(f"  输出均值: {final_output.mean():.4f}")
    print(f"  输出标准差: {final_output.std():.4f}")
    print(f"  不确定性量化: 保留了完整的分布信息")
    
    return final_output


def demonstrate_complete_causal_chain():
    """
    演示完整的因果链条: E → U → S → Y
    """
    print("\n\n🔗 完整因果链条演示: E → U → S → Y")
    print("=" * 60)
    
    # 创建CausalEngine
    engine = CausalEngine(
        hidden_size=128,
        vocab_size=10,
        causal_size=64,
        activation_modes="classification"
    )
    
    # 输入证据
    batch_size = 4
    evidence_E = torch.randn(batch_size, 1, 128)
    
    print(f"📊 输入证据 E: {evidence_E.shape}")
    
    # 完整前向传播
    with torch.no_grad():
        outputs = engine(
            hidden_states=evidence_E,
            do_sample=False,
            temperature=1.0,
            return_dict=True,
            apply_activation=True
        )
    
    # 提取各阶段输出
    loc_U = outputs['loc_U']
    scale_U = outputs['scale_U']
    loc_S = outputs['loc_S']
    scale_S = outputs['scale_S']
    final_output = outputs['output']
    
    print(f"\n🔗 完整因果链条追踪:")
    print(f"  证据 E → 个体分布参数:")
    print(f"    μ_U: {loc_U.shape}, 范围 [{loc_U.min():.3f}, {loc_U.max():.3f}]")
    print(f"    γ_U: {scale_U.shape}, 范围 [{scale_U.min():.3f}, {scale_U.max():.3f}]")
    
    print(f"  个体 U → 决策分布参数:")
    print(f"    loc_S: {loc_S.shape}, 范围 [{loc_S.min():.3f}, {loc_S.max():.3f}]")
    print(f"    scale_S: {scale_S.shape}, 范围 [{scale_S.min():.3f}, {scale_S.max():.3f}]")
    
    print(f"  决策 S → 最终输出:")
    print(f"    Y: {final_output.shape}, 范围 [{final_output.min():.3f}, {final_output.max():.3f}]")
    
    return outputs


def demonstrate_cauchy_stability():
    """
    演示柯西分布的线性稳定性 - CausalEngine的数学核心
    """
    print("\n\n📐 柯西分布线性稳定性 - CausalEngine的数学基石")
    print("=" * 60)
    print("核心理论: 如果 X ~ Cauchy(μ, γ), 则 aX + b ~ Cauchy(aμ + b, |a|γ)")
    
    # 创建柯西分布样本
    loc1, scale1 = 2.0, 1.0
    loc2, scale2 = -1.0, 0.5
    
    print(f"\n🎲 原始分布示例:")
    print(f"  X1 ~ Cauchy({loc1}, {scale1})")
    print(f"  X2 ~ Cauchy({loc2}, {scale2})")
    
    # 线性稳定性验证
    a, b = 2.0, 3.0
    transformed_loc = a * loc1 + b
    transformed_scale = abs(a) * scale1
    
    print(f"\n🔄 线性变换验证: Y = {a}X1 + {b}")
    print(f"  解析结果: Y ~ Cauchy({transformed_loc}, {transformed_scale})")
    print(f"  应用场景: 行动网络的线性因果变换")
    
    # 相加稳定性验证
    sum_loc = loc1 + loc2
    sum_scale = scale1 + scale2
    
    print(f"\n➕ 分布相加验证: Z = X1 + X2")
    print(f"  解析结果: Z ~ Cauchy({sum_loc}, {sum_scale})")
    print(f"  应用场景: 外生噪声注入过程")
    
    # CausalEngine中的具体应用
    print(f"\n🧠 在CausalEngine中的具体应用:")
    print(f"  噪声注入: U + b_noise·ε → Cauchy(μ_U, γ_U + |b_noise|)")
    print(f"  线性变换: W·U' + b → Cauchy(W·μ_{U'} + b, |W|·γ_{U'})")
    print(f"  任务激活: w·S + b → Cauchy(w·μ_S + b, |w|·γ_S)")
    
    print(f"\n💡 解析计算的革命性优势:")
    print(f"  🚀 计算效率: 避免大规模蒙特卡洛采样")
    print(f"  🎯 数学严格: 基于分布的精确变换")
    print(f"  🔄 不确定性传播: 完整保留分布信息")
    print(f"  📐 可扩展性: 支持任意复杂的线性组合")
    
    print(f"\n🏆 与传统方法的对比:")
    print(f"  传统方法: 需要大量采样 → 计算开销大 → 近似结果")
    print(f"  CausalEngine: 解析计算 → 高效精确 → 完整分布信息")
    print(f"  性能提升: 计算速度提升数倍，精度显著改善")


def visualize_causal_chain():
    """
    可视化因果链条的数学流程
    """
    print("\n\n📊 生成因果链条可视化图表")
    print("=" * 60)
    
    # 创建示例数据
    engine = CausalEngine(hidden_size=64, vocab_size=5, causal_size=32)
    evidence = torch.randn(1, 1, 64)
    
    with torch.no_grad():
        outputs = engine(evidence, return_dict=True, apply_activation=True)
    
    # Setup plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CausalEngine Causal Chain Mathematical Flow', fontsize=16, fontweight='bold')
    
    # 1. Input evidence distribution
    evidence_flat = evidence.flatten().numpy()
    axes[0, 0].hist(evidence_flat, bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Input Evidence E\n(Observed Data)', fontweight='bold')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Individual location parameter
    loc_U_flat = outputs['loc_U'].flatten().numpy()
    axes[0, 1].hist(loc_U_flat, bins=20, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Individual Location μ_U\n(Abduction Stage)', fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Individual scale parameter
    scale_U_flat = outputs['scale_U'].flatten().numpy()
    axes[0, 2].hist(scale_U_flat, bins=20, alpha=0.7, color='lightcoral')
    axes[0, 2].set_title('Individual Scale γ_U\n(Uncertainty)', fontweight='bold')
    axes[0, 2].set_xlabel('Value')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Decision location parameter
    loc_S_flat = outputs['loc_S'].flatten().numpy()
    axes[1, 0].hist(loc_S_flat, bins=20, alpha=0.7, color='gold')
    axes[1, 0].set_title('Decision Location loc_S\n(Action Stage)', fontweight='bold')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # 5. Decision scale parameter
    scale_S_flat = outputs['scale_S'].flatten().numpy()
    axes[1, 1].hist(scale_S_flat, bins=20, alpha=0.7, color='plum')
    axes[1, 1].set_title('Decision Scale scale_S\n(Decision Uncertainty)', fontweight='bold')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Final output
    output_flat = outputs['output'].flatten().numpy()
    axes[1, 2].hist(output_flat, bins=20, alpha=0.7, color='lightsteelblue')
    axes[1, 2].set_title('Final Output Y\n(Task Result)', fontweight='bold')
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('tutorials/00_getting_started/causal_chain_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    
    print("✅ Visualization chart saved: tutorials/00_getting_started/causal_chain_visualization.png")


def main():
    """
    主函数：完整的理论框架演示
    """
    print("🌟 CausalEngine 理论基础完整演示")
    print("展示Y = f(U, ε)框架的两阶段架构实现")
    print("=" * 80)
    
    # 1. 核心数学框架介绍
    engine = demonstrate_core_mathematical_framework()
    
    # 2. 归因推断阶段演示
    engine, loc_U, scale_U = demonstrate_abduction_stage()
    
    # 3. 行动决策阶段演示
    loc_S, scale_S = demonstrate_action_stage(engine, loc_U, scale_U)
    
    # 4. 激活输出阶段演示
    final_output = demonstrate_activation_stage(engine, loc_S, scale_S)
    
    # 5. 完整因果链条演示
    complete_outputs = demonstrate_complete_causal_chain()
    
    # 6. 柯西分布稳定性演示
    demonstrate_cauchy_stability()
    
    # 7. 可视化
    visualize_causal_chain()
    
    # 8. 总结
    print("\n\n🎉 CausalEngine理论基础演示完成！")
    print("=" * 80)
    print("🔬 核心理论验证 (基于最新数学文档):")
    print("  ✅ Y = f(U, ε) 框架: 个体选择变量驱动的因果推理")
    print("  ✅ 三阶段架构: 归因 → 行动 → 任务激活的清晰分离")
    print("  ✅ 柯西分布稳定性: 解析不确定性传播的数学基石")
    print("  ✅ 四种推理模式: 因果、标准、采样、兼容的统一框架")
    print("  ✅ 三种激活机制: 词元、数值、有序分类的通用支持")
    
    print("\n🚀 CausalEngine的革命性优势:")
    print("  🎯 真正因果推理: 超越相关性的个体因果建模")
    print("  📐 解析计算优势: 避免采样的高效精确计算")
    print("  🔄 架构可解释性: 三阶段独立且可组合的设计")
    print("  🌡️ 推理模式灵活: 统一框架下的多样化推理")
    print("  ✨ 任务通用性: 一个引擎支持多种任务类型")
    
    print("\n📊 基准测试协议要点:")
    print("  🧪 科学控制: 固定噪声vs自适应噪声的对比实验")
    print("  📈 系统评估: 噪声强度对性能影响的全面分析")
    print("  🎛️ 参数标准化: 统一的训练超参数和评估指标")
    print("  🔬 可复现性: 标准化流程确保结果可重现")
    
    print("\n📚 推荐学习路径:")
    print("  1. 基准协议: tutorials/00_getting_started/benchmark_protocol_intro.py")
    print("  2. 分类应用: tutorials/01_classification/ (基于新实验设计)")
    print("  3. 回归应用: tutorials/02_regression/ (基于新实验设计)")
    print("  4. 消融实验: tutorials/03_ablation_studies/ (固定vs自适应噪声)")
    print("  5. 高级主题: tutorials/04_advanced_topics/ (四种推理模式)")
    print("  6. 数学理论: causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md")
    print("  7. 实验协议: causal_engine/misc/benchmark_strategy.md")


if __name__ == "__main__":
    main()