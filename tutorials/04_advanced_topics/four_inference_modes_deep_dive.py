"""
CausalEngine 四种推理模式深度解析
=================================

基于最新数学理论 (MATHEMATICAL_FOUNDATIONS_CN.md) 的完整推理模式分析
深入探讨每种模式的数学原理、哲学意义和实际应用场景

四种推理模式:
1. 因果模式 (Causal): T=0 - 纯粹因果推理，无外生干扰
2. 标准模式 (Standard): T>0, do_sample=False - 扩大不确定性，承认认识局限
3. 采样模式 (Sampling): T>0, do_sample=True - 个体身份探索，多样性分析
4. 兼容模式 (Compatible): 任意T - 与传统方法对齐，便于对比

核心创新: 通过温度参数的数学调制实现从确定性到随机性的统一框架
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from causal_engine import CausalEngine


class InferenceModeAnalyzer:
    """
    推理模式分析器 - 深度分析四种推理模式的特性
    """
    
    def __init__(self, input_size=64, vocab_size=10, causal_size=32):
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.causal_size = causal_size
        
        # 创建CausalEngine实例
        self.engine = CausalEngine(
            hidden_size=input_size,
            vocab_size=vocab_size,
            causal_size=causal_size,
            activation_modes="classification"
        )
        
        # 推理模式配置
        self.modes = {
            'causal': {'temperature': 0, 'do_sample': False, 'name': '因果模式'},
            'standard': {'temperature': 1.0, 'do_sample': False, 'name': '标准模式'},
            'sampling': {'temperature': 0.8, 'do_sample': True, 'name': '采样模式'},
            'compatible': {'temperature': 1.0, 'do_sample': False, 'name': '兼容模式'}
        }
    
    def demonstrate_mathematical_principles(self):
        """
        演示四种推理模式的数学原理
        """
        print("🔬 四种推理模式的数学原理分析")
        print("=" * 60)
        
        # 创建示例输入
        batch_size = 8
        evidence = torch.randn(batch_size, 1, self.input_size)
        
        print(f"\n📊 输入证据: {evidence.shape}")
        print(f"  批次大小: {batch_size}")
        print(f"  特征维度: {self.input_size}")
        
        with torch.no_grad():
            # 先获取基础的个体表征
            loc_U, scale_U = self.engine.abduction(evidence)
            print(f"\n🎲 个体表征分布参数:")
            print(f"  位置参数 μ_U: {loc_U.shape}, 范围 [{loc_U.min():.3f}, {loc_U.max():.3f}]")
            print(f"  尺度参数 γ_U: {scale_U.shape}, 范围 [{scale_U.min():.3f}, {scale_U.max():.3f}]")
            
            # 分析各种推理模式
            mode_results = {}
            
            for mode_key, config in self.modes.items():
                print(f"\n{config['name']} - 数学变换分析:")
                
                # 执行推理
                result = self.engine(
                    evidence, 
                    temperature=config['temperature'],
                    do_sample=config['do_sample'],
                    return_dict=True
                )
                
                loc_S = result['loc_S']
                scale_S = result['scale_S']
                output = result['output']
                
                # 数学分析
                print(f"  温度参数: T = {config['temperature']}")
                print(f"  采样标志: do_sample = {config['do_sample']}")
                
                if config['temperature'] == 0:
                    print(f"  数学变换: U' = U (无噪声注入)")
                    print(f"  决策分布: S ~ Cauchy(loc_S, scale_S)")
                    print(f"  哲学含义: 纯粹因果推理，个体在无外生干扰下的必然选择")
                elif not config['do_sample']:
                    print(f"  数学变换: U' ~ Cauchy(μ_U, γ_U + T·|b_noise|)")
                    print(f"  噪声效应: 扩大尺度参数，增加决策不确定性")
                    print(f"  哲学含义: 承认环境不确定性对决策的影响")
                else:
                    print(f"  数学变换: ε~Cauchy(0,1), U' ~ Cauchy(μ_U + T·|b_noise|·ε, γ_U)")
                    print(f"  噪声效应: 扰动位置参数，探索不同个体身份")
                    print(f"  哲学含义: 探索个体在随机扰动下的多样表现")
                
                # 统计特性
                uncertainty = scale_S.mean().item()
                diversity = loc_S.std().item()
                
                print(f"  平均不确定性: {uncertainty:.4f}")
                print(f"  位置多样性: {diversity:.4f}")
                
                mode_results[mode_key] = {
                    'config': config,
                    'loc_S': loc_S,
                    'scale_S': scale_S,
                    'output': output,
                    'uncertainty': uncertainty,
                    'diversity': diversity
                }
            
            return mode_results
    
    def analyze_temperature_effects(self):
        """
        分析温度参数对不同推理模式的影响
        """
        print("🌡️ 温度参数效应深度分析")
        print("=" * 60)
        
        # 测试不同温度值
        temperatures = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
        evidence = torch.randn(4, 1, self.input_size)
        
        print("\n📊 温度扫描实验:")
        
        with torch.no_grad():
            results = {'standard': {}, 'sampling': {}}
            
            for temp in temperatures:
                print(f"\n  T = {temp}:")
                
                # 标准模式 (扩大尺度)
                standard_result = self.engine(
                    evidence, temperature=temp, do_sample=False, return_dict=True
                )
                
                # 采样模式 (扰动位置)
                sampling_result = self.engine(
                    evidence, temperature=temp, do_sample=True, return_dict=True
                )
                
                # 分析效应
                std_uncertainty = standard_result['scale_S'].mean().item()
                smp_diversity = sampling_result['loc_S'].std().item()
                
                print(f"    标准模式不确定性: {std_uncertainty:.4f}")
                print(f"    采样模式位置多样性: {smp_diversity:.4f}")
                
                results['standard'][temp] = std_uncertainty
                results['sampling'][temp] = smp_diversity
            
            return results, temperatures
    
    def demonstrate_philosophical_meanings(self):
        """
        演示四种推理模式的哲学意义和应用场景
        """
        print("\n🧠 推理模式的哲学意义与应用场景")
        print("=" * 60)
        
        philosophical_analysis = {
            'causal': {
                'name': '因果模式 (Causal Mode)',
                'philosophy': '在无外生干扰的理想条件下，个体基于其内在特征的必然决策',
                'key_insight': '揭示个体的本质特征和决策倾向',
                'applications': [
                    '科学研究中的因果关系分析',
                    '理论模型的验证和解释',
                    '基准测试和性能评估',
                    '决策系统的核心逻辑验证'
                ],
                'strengths': [
                    '结果稳定、可重现',
                    '便于理论分析和解释',
                    '计算效率高',
                    '适合精确性要求高的场景'
                ],
                'limitations': [
                    '忽略了现实中的不确定性',
                    '可能过于自信',
                    '缺乏鲁棒性分析',
                    '难以处理噪声数据'
                ]
            },
            
            'standard': {
                'name': '标准模式 (Standard Mode)',
                'philosophy': '承认认识的局限性，在决策中体现对环境不确定性的敬畏',
                'key_insight': '平衡确定性与不确定性，提供校准的置信度',
                'applications': [
                    '风险评估和管理',
                    '医疗诊断辅助系统',
                    '金融投资决策支持',
                    '安全关键系统的决策'
                ],
                'strengths': [
                    '提供不确定性量化',
                    '决策更加谨慎合理',
                    '便于风险控制',
                    '适合需要置信度的场景'
                ],
                'limitations': [
                    '计算复杂度略高',
                    '需要不确定性校准',
                    '可能过于保守',
                    '对超参数敏感'
                ]
            },
            
            'sampling': {
                'name': '采样模式 (Sampling Mode)',
                'philosophy': '探索个体在不同情境扰动下的多样化表现，发现潜在可能性',
                'key_insight': '理解个体行为的多样性和适应性',
                'applications': [
                    '创造性内容生成',
                    '多样化推荐系统',
                    '鲁棒性测试和验证',
                    '蒙特卡洛方法应用'
                ],
                'strengths': [
                    '探索多样化可能性',
                    '增强模型鲁棒性',
                    '适合创造性任务',
                    '提供丰富的候选结果'
                ],
                'limitations': [
                    '结果随机性较强',
                    '需要多次采样',
                    '计算开销较大',
                    '不适合精确性要求高的场景'
                ]
            },
            
            'compatible': {
                'name': '兼容模式 (Compatible Mode)',
                'philosophy': '与传统统计学习方法对齐，便于渐进式技术迁移',
                'key_insight': '在保持因果架构优势的同时确保向后兼容',
                'applications': [
                    '与传统系统的集成',
                    '基准对比和验证',
                    '渐进式技术升级',
                    '性能评估和分析'
                ],
                'strengths': [
                    '易于集成和部署',
                    '便于性能对比',
                    '学习成本低',
                    '风险可控'
                ],
                'limitations': [
                    '未充分利用因果架构优势',
                    '可能限制创新应用',
                    '长期技术债务风险',
                    '哲学价值相对有限'
                ]
            }
        }
        
        for mode_key, analysis in philosophical_analysis.items():
            print(f"\n🎯 {analysis['name']}")
            print(f"  哲学内涵: {analysis['philosophy']}")
            print(f"  核心洞察: {analysis['key_insight']}")
            
            print(f"\n  💼 应用场景:")
            for app in analysis['applications']:
                print(f"    • {app}")
            
            print(f"\n  ✅ 主要优势:")
            for strength in analysis['strengths']:
                print(f"    • {strength}")
            
            print(f"\n  ⚠️ 局限性:")
            for limitation in analysis['limitations']:
                print(f"    • {limitation}")
    
    def practical_decision_guide(self):
        """
        提供实际应用中的推理模式选择指南
        """
        print("\n🎯 推理模式选择实用指南")
        print("=" * 60)
        
        decision_matrix = {
            '精确性要求': {
                '极高': 'causal',
                '高': 'standard', 
                '中等': 'standard',
                '低': 'sampling'
            },
            '不确定性量化需求': {
                '不需要': 'causal',
                '需要': 'standard',
                '重要': 'standard',
                '关键': 'standard'
            },
            '创造性要求': {
                '不需要': 'causal',
                '一般': 'standard',
                '重要': 'sampling',
                '关键': 'sampling'
            },
            '计算资源': {
                '受限': 'causal',
                '充足': 'sampling',
                '中等': 'standard',
                '无限制': 'sampling'
            },
            '风险承受度': {
                '极低': 'causal',
                '低': 'standard',
                '中等': 'standard',
                '高': 'sampling'
            }
        }
        
        print("\n📋 决策矩阵:")
        print("\n维度\t\t\t因果模式\t标准模式\t采样模式")
        print("-" * 60)
        
        criteria_examples = {
            '精确性要求': {'极高': '科学计算', '高': '医疗诊断', '中等': '推荐系统', '低': '内容生成'},
            '不确定性量化': {'不需要': '分类任务', '需要': '风险评估', '重要': '投资决策', '关键': '安全系统'},
            '创造性要求': {'不需要': '数据分析', '一般': '个性推荐', '重要': '艺术创作', '关键': '游戏AI'},
            '计算资源': {'受限': '移动端', '充足': '云计算', '中等': '边缘计算', '无限制': '超算中心'},
            '风险承受度': {'极低': '金融交易', '低': '医疗决策', '中等': '商业应用', '高': '研发实验'}
        }
        
        # 实际场景示例
        scenarios = [
            {
                'name': '🏥 医疗诊断辅助系统',
                'requirements': {
                    '精确性要求': '高',
                    '不确定性量化需求': '关键',
                    '创造性要求': '不需要',
                    '风险承受度': '极低'
                },
                'recommended_mode': 'standard',
                'reasoning': '需要高精确性和不确定性量化，风险承受度极低'
            },
            {
                'name': '🎨 创意内容生成系统',
                'requirements': {
                    '精确性要求': '低',
                    '创造性要求': '关键',
                    '计算资源': '充足',
                    '风险承受度': '高'
                },
                'recommended_mode': 'sampling',
                'reasoning': '重视创造性和多样性，可接受一定随机性'
            },
            {
                'name': '📊 科学数据分析',
                'requirements': {
                    '精确性要求': '极高',
                    '不确定性量化需求': '不需要',
                    '创造性要求': '不需要',
                    '风险承受度': '极低'
                },
                'recommended_mode': 'causal',
                'reasoning': '要求最高精确性和可重现性，不需要随机性'
            },
            {
                'name': '💰 金融风险评估',
                'requirements': {
                    '精确性要求': '高',
                    '不确定性量化需求': '关键',
                    '风险承受度': '低',
                    '计算资源': '中等'
                },
                'recommended_mode': 'standard',
                'reasoning': '需要精确的风险量化和保守的决策策略'
            }
        ]
        
        print("\n🌟 实际应用场景示例:")
        for scenario in scenarios:
            print(f"\n{scenario['name']}")
            print(f"  选择理由: {scenario['reasoning']}")
            requirements_str = ', '.join([f"{k}({v})" for k, v in scenario['requirements'].items()])
            print(f"  关键需求: {requirements_str}")
            print(f"  推荐模式: {self.modes[scenario['recommended_mode']]['name']}")
            print("-" * 20)
    
    def visualize_mode_characteristics(self):
        """
        可视化四种推理模式的特征对比
        """
        print("\n📊 生成推理模式特征对比图")
        
        # 创建示例数据
        evidence = torch.randn(100, 1, self.input_size)
        
        mode_data = {}
        
        with torch.no_grad():
            for mode_key, config in self.modes.items():
                results = []
                for _ in range(10):  # 多次采样以观察变异性
                    result = self.engine(
                        evidence,
                        temperature=config['temperature'],
                        do_sample=config['do_sample'],
                        return_dict=True
                    )
                    
                    # 计算关键指标
                    uncertainty = result['scale_S'].mean().item()
                    diversity = result['loc_S'].std().item()
                    max_prob = torch.softmax(result['output'], dim=-1).max(dim=-1)[0].mean().item()
                    
                    results.append({
                        'uncertainty': uncertainty,
                        'diversity': diversity,
                        'confidence': max_prob
                    })
                
                mode_data[mode_key] = results
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('四种推理模式特征对比分析', fontsize=16, fontweight='bold')
        
        # 1. 不确定性对比
        uncertainties = {mode: [r['uncertainty'] for r in results] for mode, results in mode_data.items()}
        
        ax1 = axes[0, 0]
        ax1.boxplot(uncertainties.values(), labels=[self.modes[k]['name'] for k in uncertainties.keys()])
        ax1.set_title('决策不确定性分布')
        ax1.set_ylabel('平均尺度参数')
        ax1.grid(True, alpha=0.3)
        
        # 2. 多样性对比
        diversities = {mode: [r['diversity'] for r in results] for mode, results in mode_data.items()}
        
        ax2 = axes[0, 1]
        ax2.boxplot(diversities.values(), labels=[self.modes[k]['name'] for k in diversities.keys()])
        ax2.set_title('位置参数多样性')
        ax2.set_ylabel('标准差')
        ax2.grid(True, alpha=0.3)
        
        # 3. 置信度对比
        confidences = {mode: [r['confidence'] for r in results] for mode, results in mode_data.items()}
        
        ax3 = axes[1, 0]
        ax3.boxplot(confidences.values(), labels=[self.modes[k]['name'] for k in confidences.keys()])
        ax3.set_title('预测置信度分布')
        ax3.set_ylabel('最大概率')
        ax3.grid(True, alpha=0.3)
        
        # 4. 综合特征雷达图
        ax4 = axes[1, 1]
        
        # 计算每种模式的平均特征
        mode_features = {}
        for mode, results in mode_data.items():
            mode_features[mode] = {
                'stability': 1 - np.std([r['uncertainty'] for r in results]),  # 稳定性
                'confidence': np.mean([r['confidence'] for r in results]),     # 置信度
                'diversity': np.mean([r['diversity'] for r in results]),       # 多样性
                'uncertainty': np.mean([r['uncertainty'] for r in results])    # 不确定性
            }
        
        # 绘制特征条形图
        features = ['稳定性', '置信度', '多样性', '不确定性']
        x = np.arange(len(features))
        width = 0.2
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (mode, color) in enumerate(zip(mode_features.keys(), colors)):
            values = list(mode_features[mode].values())
            # 归一化到0-1范围
            values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 
                     for v in values]
            
            ax4.bar(x + i * width, values, width, label=self.modes[mode]['name'], 
                   color=color, alpha=0.8)
        
        ax4.set_xlabel('特征维度')
        ax4.set_ylabel('归一化得分')
        ax4.set_title('综合特征对比')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(features)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('tutorials/04_advanced_topics/four_modes_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 可视化图表已保存: tutorials/04_advanced_topics/four_modes_comparison.png")


def main():
    """
    主函数: 完整的四种推理模式深度分析
    """
    print("🌟 CausalEngine 四种推理模式深度解析")
    print("基于最新数学理论的完整推理机制分析")
    print("=" * 80)
    
    # 创建分析器
    analyzer = InferenceModeAnalyzer()
    
    # 1. 数学原理演示
    print("\n🔬 步骤1: 数学原理演示")
    mode_results = analyzer.demonstrate_mathematical_principles()
    
    # 2. 温度效应分析
    print("\n🌡️ 步骤2: 温度参数效应分析")
    temp_results, temperatures = analyzer.analyze_temperature_effects()
    
    # 3. 哲学意义解析
    print("\n🧠 步骤3: 哲学意义与应用场景")
    analyzer.demonstrate_philosophical_meanings()
    
    # 4. 实用决策指南
    print("\n🎯 步骤4: 实用选择指南")
    analyzer.practical_decision_guide()
    
    # 5. 可视化特征对比
    print("\n📊 步骤5: 特征可视化")
    analyzer.visualize_mode_characteristics()
    
    # 6. 总结和建议
    print("\n🎉 四种推理模式深度分析完成！")
    print("=" * 80)
    print("🔬 关键发现:")
    print("  ✅ 因果模式: 最稳定、最可解释，适合科学分析")
    print("  ✅ 标准模式: 平衡性能与不确定性，适合大多数应用")
    print("  ✅ 采样模式: 最具创造性，适合探索性任务")
    print("  ✅ 兼容模式: 便于迁移和对比，适合渐进升级")
    
    print("\n💡 核心洞察:")
    print("  🎯 温度参数提供了从确定性到随机性的连续调节")
    print("  🧠 每种模式都有其独特的哲学价值和应用场景")
    print("  🔄 统一框架实现了多样化推理需求的灵活支持")
    print("  📐 数学严格性确保了推理过程的可靠性")
    
    print("\n📚 进一步学习:")
    print("  1. 任务激活机制: tutorials/04_advanced_topics/task_activation_mechanisms.py")
    print("  2. 多任务学习: tutorials/04_advanced_topics/multi_task_learning_framework.py")
    print("  3. 不确定性量化: tutorials/04_advanced_topics/uncertainty_quantification.py")
    print("  4. 数学理论: causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md")


if __name__ == "__main__":
    main()