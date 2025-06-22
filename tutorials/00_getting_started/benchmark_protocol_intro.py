"""
CausalEngine 基准测试协议介绍
=============================

本教程详细介绍CausalEngine的官方基准测试协议，包括：
1. 实验设计原理和科学控制方法
2. 固定噪声vs自适应噪声两组实验
3. 四种推理模式的完整演示
4. 参数配置和评估指标体系

基于文档: causal_engine/misc/benchmark_strategy.md
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from causal_engine import CausalEngine


def introduce_benchmark_protocol():
    """
    介绍CausalEngine基准测试协议的核心设计理念
    """
    print("🔬 CausalEngine 基准测试协议")
    print("=" * 60)
    
    print("\n📋 实验目标:")
    print("  1. 性能评估: 在标准数据集上系统性评估CausalEngine相对于传统基线的性能")
    print("  2. 建立基线: 为后续模型改进、消融实验建立可靠的性能基准")
    print("  3. 假设验证: 科学验证'让模型自主学习全局噪声'的核心假设")
    
    print("\n🎯 指导原则:")
    print("  科学对照原则: 将机制参数(如b_noise)视为固定的实验条件或明确的探索变量")
    print("  因果归因: 实现清晰的因果归因和可解释的结果")
    print("  可复现性: 所有实验配置标准化，确保结果可复现")


def demonstrate_model_architecture():
    """
    演示CausalEngine的三阶段架构和参数体系
    """
    print("\n\n🏗️ CausalEngine 架构与参数定义")
    print("=" * 60)
    
    print("\n📐 三阶段架构:")
    print("  阶段1: 归因网络 (AbductionNetwork)")
    print("    - 功能: 从证据z推断个体柯西分布参数")
    print("    - 参数: W_loc, b_loc (位置), W_scale, b_scale (尺度)")
    print("    - 数学: μ_U = W_loc·z + b_loc")
    print("            γ_U = softplus(W_scale·z + b_scale)")
    
    print("\n  阶段2: 行动网络 (ActionNetwork)")
    print("    - 功能: 应用普适因果律，U → S")
    print("    - 参数: W_cls, b_cls (可学习), b_noise (可配置)")
    print("    - 数学: loc_S = W_cls·loc_U + b_cls")
    print("            scale_S = (scale_U + |b_noise|)·|W_cls|")
    
    print("\n  阶段3: 任务激活头 (TaskActivationHeads)")
    print("    - 分类头: P_k = P(S_k > C_ovr), C_ovr为全局决策阈值")
    print("    - 回归头: Y_k ~ Cauchy(w_k·loc_S + b_k, |w_k|·scale_S)")
    print("    - 有序分类: P(y_i) = P(C_{k,i} < S_k ≤ C_{k,i+1})")
    
    # 创建示例模型展示参数
    print("\n⚡ 实际架构演示:")
    engine = CausalEngine(
        hidden_size=128,
        vocab_size=10,
        causal_size=64,
        activation_modes="classification"
    )
    
    print(f"  模型参数统计:")
    total_params = sum(p.numel() for p in engine.parameters())
    trainable_params = sum(p.numel() for p in engine.parameters() if p.requires_grad)
    print(f"    总参数量: {total_params:,}")
    print(f"    可训练参数: {trainable_params:,}")
    
    # 展示各个网络的参数
    print(f"  网络结构:")
    print(f"    归因网络 - 位置网络: {engine.abduction.loc_net}")
    print(f"    归因网络 - 尺度网络: {engine.abduction.scale_net}")
    print(f"    行动网络维度: {engine.causal_size} → {engine.vocab_size}")


def demonstrate_experimental_groups():
    """
    演示两组平行对比实验的设计
    """
    print("\n\n🧪 实验组设计: 固定噪声 vs 自适应噪声")
    print("=" * 60)
    
    print("\n📊 实验组A: 固定噪声强度实验")
    print("  目标: 测试不同固定噪声强度对模型性能的影响")
    print("  控制: b_noise.requires_grad = False")
    print("  测试噪声值: [0.0, 0.1, 1.0, 10.0]")
    
    # 演示不同噪声强度的效果
    noise_values = [0.0, 0.1, 1.0, 10.0]  # 基准协议更新：更有区分度的关键值
    print(f"\n  噪声强度分析:")
    
    for noise in noise_values:
        if noise == 0.01:
            effect = "低噪声 - 高置信度，可能过拟合"
        elif noise == 0.05:
            effect = "中低噪声 - 较高置信度"
        elif noise == 0.1:
            effect = "标准噪声(默认) - 平衡的不确定性表达"
        elif noise == 0.2:
            effect = "中高噪声 - 保守预测"
        else:  # 0.5
            effect = "高噪声 - 高度不确定，可能欠拟合"
        
        print(f"    b_noise = {noise}: {effect}")
    
    print("\n📊 实验组B: 自适应噪声学习实验")
    print("  目标: 验证让模型自主学习全局环境噪声的有效性")
    print("  控制: b_noise.requires_grad = True")
    print("  初始化: b_noise = nn.Parameter(torch.tensor([0.1]))")
    print("  优化: 与主网络同时训练，可使用不同学习率")


def demonstrate_training_hyperparameters():
    """
    演示标准化的训练超参数配置
    """
    print("\n\n⚙️ 标准化训练超参数")
    print("=" * 60)
    
    config = {
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "learning_schedule": "线性warm-up(前10% steps) + cosine decay",
        "weight_decay": 0.01,
        "epochs": 100,
        "early_stopping": True,
        "patience": 10,
        "batch_size": 64,
        "monitor_metric": "验证集损失"
    }
    
    print("  训练配置:")
    for key, value in config.items():
        print(f"    {key}: {value}")
    
    # 固定参数配置
    print("\n  固定超参数:")
    print("    C_ovr (分类阈值): 0.0")
    print("    w_k, b_k (回归头): 1.0, 0.0")
    print("    随机种子: 42 (可复现性)")
    
    # 评估指标
    print("\n  评估指标:")
    print("    分类任务: Accuracy, F1-Score(Macro), Precision, Recall")
    print("    回归任务: MAE, RMSE, MdAE, MSE, R²")
    print("    统计检验: t-test, Wilcoxon signed-rank test")


def demonstrate_inference_modes():
    """
    演示四种推理模式的数学原理和应用场景
    """
    print("\n\n🎯 四种推理模式详解")
    print("=" * 60)
    
    # 创建示例模型
    engine = CausalEngine(
        hidden_size=64,
        vocab_size=8,
        causal_size=32,
        activation_modes="classification"
    )
    
    # 准备示例输入
    batch_size = 4
    evidence = torch.randn(batch_size, 1, 64)
    
    print("\n🎲 输入证据:")
    print(f"  形状: {evidence.shape}")
    print(f"  示例值范围: [{evidence.min():.3f}, {evidence.max():.3f}]")
    
    with torch.no_grad():
        # 模式1: 因果推理 (T=0)
        print("\n  模式1: 因果推理 (Causal Mode)")
        print("    设置: temperature=0, do_sample=any")
        print("    机制: 完全关闭外生噪声影响")
        print("    数学: U' ~ Cauchy(μ_U, γ_U)")
        print("    应用: 确定性推理、硬决策、点估计")
        
        causal_output = engine(evidence, temperature=0, do_sample=False, return_dict=True)
        causal_uncertainty = causal_output['scale_S'].mean().item()
        print(f"    平均不确定性: {causal_uncertainty:.4f}")
        
        # 模式2: 标准推理 (T>0, no sample)
        print("\n  模式2: 标准推理 (Standard Mode)")
        print("    设置: temperature>0, do_sample=False")
        print("    机制: 噪声缩放后增加尺度参数，扩大不确定性")
        print("    数学: U' ~ Cauchy(μ_U, γ_U + T·|b_noise|)")
        print("    应用: 稳定生成、软决策、置信区间")
        
        standard_output = engine(evidence, temperature=1.0, do_sample=False, return_dict=True)
        standard_uncertainty = standard_output['scale_S'].mean().item()
        print(f"    平均不确定性: {standard_uncertainty:.4f} (↑ 增加了 {standard_uncertainty - causal_uncertainty:.4f})")
        
        # 模式3: 采样推理 (T>0, sample)
        print("\n  模式3: 采样推理 (Sampling Mode)")
        print("    设置: temperature>0, do_sample=True")
        print("    机制: 噪声缩放后扰动位置参数，改变个体身份")
        print("    数学: ε~Cauchy(0,1), U' ~ Cauchy(μ_U + T·|b_noise|·ε, γ_U)")
        print("    应用: 创造性生成、探索边界、蒙特卡洛")
        
        sampling_output = engine(evidence, temperature=0.8, do_sample=True, return_dict=True)
        sampling_loc_shift = (sampling_output['loc_U'] - causal_output['loc_U']).abs().mean().item()
        print(f"    位置参数平均偏移: {sampling_loc_shift:.4f}")
        
        # 模式4: 兼容模式
        print("\n  模式4: 兼容推理 (Compatible Mode)")
        print("    设置: 任意temperature, 特殊兼容标志")
        print("    机制: 兼容传统transformer行为")
        print("    应用: 与传统模型对比基准")


def demonstrate_scientific_control():
    """
    演示科学控制的实验设计
    """
    print("\n\n🔬 科学控制实验设计")
    print("=" * 60)
    
    print("\n🎛️ 优雅的实验控制:")
    print("  核心设计: 通过b_noise.requires_grad的True/False控制实验组")
    print("  优势1: 只需一个布尔开关切换两种模式")
    print("  优势2: 除噪声学习方式外，其他设置完全相同")
    print("  优势3: 确保对比的公平性和科学性")
    
    # 代码示例
    print("\n💻 实现示例:")
    print("  # 实验组A: 固定噪声")
    print("  engine.action.b_noise.requires_grad = False")
    print("  engine.action.b_noise.data.fill_(0.1)  # 固定为0.1")
    print("")
    print("  # 实验组B: 自适应噪声")
    print("  engine.action.b_noise.requires_grad = True")
    print("  # b_noise作为可学习参数包含在优化器中")
    
    print("\n📊 预期实验结果:")
    print("  噪声强度-性能曲线: 揭示非线性关系和最优区间")
    print("  固定vs自适应对比: 验证自适应学习的有效性")
    print("  跨数据集分析: 不同任务的最优噪声特性")


def demonstrate_evaluation_framework():
    """
    演示完整的评估与分析框架
    """
    print("\n\n📊 评估与分析框架")
    print("=" * 60)
    
    print("\n🎯 核心对比分析:")
    print("  1. 噪声强度影响分析:")
    print("     - 对比不同固定噪声值(0.01-0.5)的性能")
    print("     - 绘制性能-噪声曲线")
    print("     - 分析最优噪声区间")
    
    print("\n  2. 固定vs自适应噪声:")
    print("     - 对比最优固定噪声vs自适应学习")
    print("     - 分析模型学到的噪声值合理性")
    print("     - 评估自适应学习的性能提升")
    
    print("\n  3. CausalEngine vs传统基线:")
    print("     - CausalEngine最佳配置vs标准MLP")
    print("     - 验证因果架构的有效性")
    print("     - 量化因果推理的优势")
    
    print("\n📈 统计分析方法:")
    print("  - 多次运行(≥5次)确保结果稳定性")
    print("  - 统计显著性检验(p < 0.05)")
    print("  - 效应量分析(Cohen's d)")
    print("  - 置信区间计算")


def visualize_benchmark_protocol():
    """
    可视化基准测试协议的实验流程
    """
    print("\n\n📊 生成基准协议流程图")
    print("=" * 60)
    
    # 创建实验流程可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CausalEngine Benchmark Protocol Overview', fontsize=16, fontweight='bold')
    
    # 1. 噪声强度对比实验
    noise_values = [0.0, 0.1, 1.0, 10.0]  # 基准协议更新：更有区分度的关键值
    # 模拟性能数据（实际应该从真实实验获得）
    performance = [0.75, 0.85, 0.83, 0.78]
    
    ax1.plot(noise_values, performance, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Strength (b_noise)')
    ax1.set_ylabel('Performance')
    ax1.set_title('Fixed Noise Experiment (Group A)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 标注最优点
    best_idx = np.argmax(performance)
    ax1.annotate(f'Optimal: {noise_values[best_idx]}', 
                xy=(noise_values[best_idx], performance[best_idx]),
                xytext=(noise_values[best_idx]*2, performance[best_idx]+0.02),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 2. 推理模式对比
    modes = ['Causal\n(T=0)', 'Standard\n(T=1.0)', 'Sampling\n(T=0.8)', 'Compatible']
    mode_performance = [0.83, 0.85, 0.84, 0.79]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax2.bar(modes, mode_performance, color=colors, alpha=0.7)
    ax2.set_ylabel('Performance')
    ax2.set_title('Inference Modes Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, perf in zip(bars, mode_performance):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{perf:.3f}', ha='center', va='bottom')
    
    # 3. 架构对比
    architectures = ['Traditional\nMLP', 'CausalEngine\n(Ablated)', 'CausalEngine\n(Full)']
    arch_performance = [0.79, 0.81, 0.85]
    colors_arch = ['lightcoral', 'lightblue', 'lightgreen']
    
    bars_arch = ax3.bar(architectures, arch_performance, color=colors_arch, alpha=0.8)
    ax3.set_ylabel('Performance')
    ax3.set_title('Architecture Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, perf in zip(bars_arch, arch_performance):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{perf:.3f}', ha='center', va='bottom')
    
    # 4. 实验组A vs B对比
    experiments = ['Fixed Noise\n(Best)', 'Adaptive Noise\n(Learned)']
    exp_performance = [0.85, 0.87]
    learned_noise = 0.12  # 模拟学到的噪声值
    
    bars_exp = ax4.bar(experiments, exp_performance, color=['skyblue', 'orange'], alpha=0.8)
    ax4.set_ylabel('Performance')
    ax4.set_title('Fixed vs Adaptive Noise Learning')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加标签
    ax4.text(0, exp_performance[0] + 0.01, f'{exp_performance[0]:.3f}', ha='center')
    ax4.text(1, exp_performance[1] + 0.01, f'{exp_performance[1]:.3f}\n(Learned: {learned_noise})', ha='center')
    
    plt.tight_layout()
    plt.savefig('tutorials/00_getting_started/benchmark_protocol_overview.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 基准协议流程图已保存: tutorials/00_getting_started/benchmark_protocol_overview.png")


def main():
    """
    主函数：完整的基准协议介绍
    """
    print("🌟 CausalEngine 基准测试协议完整介绍")
    print("基于官方文档: causal_engine/misc/benchmark_strategy.md")
    print("=" * 80)
    
    # 1. 协议概述
    introduce_benchmark_protocol()
    
    # 2. 模型架构介绍
    demonstrate_model_architecture()
    
    # 3. 实验组设计
    demonstrate_experimental_groups()
    
    # 4. 训练超参数
    demonstrate_training_hyperparameters()
    
    # 5. 推理模式演示
    demonstrate_inference_modes()
    
    # 6. 科学控制设计
    demonstrate_scientific_control()
    
    # 7. 评估框架
    demonstrate_evaluation_framework()
    
    # 8. 可视化
    visualize_benchmark_protocol()
    
    # 9. 总结
    print("\n\n🎉 基准测试协议介绍完成！")
    print("=" * 80)
    print("🔬 核心设计理念:")
    print("  ✅ 科学对照: 通过requires_grad控制变量")
    print("  ✅ 系统评估: 固定噪声vs自适应噪声全面对比")
    print("  ✅ 标准配置: 统一的超参数和评估指标")
    print("  ✅ 可复现性: 固定随机种子和标准流程")
    
    print("\n🚀 实验价值:")
    print("  🎯 理论验证: 验证'自主学习全局噪声'假设")
    print("  📐 机制理解: 揭示噪声强度与性能的关系")
    print("  🔄 基线建立: 为后续研究提供标准基准")
    print("  🌡️ 模式优化: 优化四种推理模式的应用")
    
    print("\n📚 下一步:")
    print("  1. 理论基础: tutorials/00_getting_started/theoretical_foundations.py")
    print("  2. 实际应用: tutorials/01_classification/ & tutorials/02_regression/")
    print("  3. 消融实验: tutorials/03_ablation_studies/comprehensive_comparison.py")
    print("  4. 数学文档: causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md")


if __name__ == "__main__":
    main()