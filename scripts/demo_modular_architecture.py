"""
演示 CausalEngine v2.0 的模块化架构

展示新架构的强大功能：
1. 模块化设计
2. 混合激活（分类 + 回归）
3. 灵活的应用场景
"""

import torch
import sys
sys.path.insert(0, '/Users/gongqian/DailyLog/CausalQwen')

from causal_engine import (
    CausalEngine,
    AbductionNetwork,
    ActionNetwork, 
    ActivationHead,
    MultiTaskActivationHead
)


def demo_basic_modular_usage():
    """演示基本的模块化使用"""
    print("=" * 60)
    print("1. 基本模块化使用")
    print("=" * 60)
    
    # 独立使用各个模块
    hidden_size = 256
    causal_size = 128
    vocab_size = 1000
    
    # 创建独立模块
    abduction = AbductionNetwork(hidden_size, causal_size)
    action = ActionNetwork(causal_size, vocab_size)
    activation = ActivationHead(vocab_size, activation_modes="classification")
    
    # 模拟输入
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    print("\n📊 前向传播流程:")
    
    # Step 1: 归因
    loc_U, scale_U = abduction(hidden_states)
    print(f"  1️⃣ 归因: {hidden_states.shape} → U~Cauchy({loc_U.shape}, {scale_U.shape})")
    
    # Step 2: 行动
    loc_S, scale_S = action(loc_U, scale_U, temperature=1.0)
    print(f"  2️⃣ 行动: U → S~Cauchy({loc_S.shape}, {scale_S.shape})")
    
    # Step 3: 激活
    output = activation(loc_S, scale_S, return_dict=False)
    print(f"  3️⃣ 激活: S → 输出 {output.shape}")
    
    print(f"\n✅ 模块化架构允许灵活组合和替换组件")


def demo_mixed_activation():
    """演示混合激活功能"""
    print("\n" + "=" * 60)
    print("2. 混合激活功能")
    print("=" * 60)
    
    # 场景：语言模型 + 情感分析
    vocab_size = 50000  # 词汇预测
    sentiment_dims = 3  # 情感维度：积极、中性、消极
    confidence_dim = 1  # 置信度分数
    
    total_output = vocab_size + sentiment_dims + confidence_dim
    
    # 配置激活模式
    modes = (
        ["classification"] * vocab_size +      # 词汇预测（分类）
        ["classification"] * sentiment_dims +   # 情感分类
        ["regression"] * confidence_dim        # 置信度（回归）
    )
    
    print(f"\n📊 混合输出配置:")
    print(f"  - 词汇预测: {vocab_size} 维（分类）")
    print(f"  - 情感分析: {sentiment_dims} 维（分类）")
    print(f"  - 置信度: {confidence_dim} 维（回归）")
    
    # 创建引擎
    engine = CausalEngine(
        hidden_size=768,
        vocab_size=total_output,
        activation_modes=modes
    )
    
    # 模拟输入
    hidden_states = torch.randn(1, 5, 768)
    output = engine(hidden_states)
    
    # 分析输出
    vocab_probs = output['output'][:, :, :vocab_size]
    sentiment_probs = output['output'][:, :, vocab_size:vocab_size+sentiment_dims]
    confidence = output['output'][:, :, -confidence_dim:]
    
    print(f"\n📈 输出分析:")
    print(f"  - 词汇概率范围: [{vocab_probs.min():.3f}, {vocab_probs.max():.3f}]")
    print(f"  - 情感概率: {sentiment_probs[0, -1].tolist()}")
    print(f"  - 置信度分数: {confidence[0, -1].item():.3f}")
    
    print(f"\n✅ 统一框架支持混合的分类和回归任务")


def demo_multi_task_head():
    """演示多任务激活头"""
    print("\n" + "=" * 60)
    print("3. 多任务激活头")
    print("=" * 60)
    
    # 场景：多模态模型
    print("\n📊 多模态任务配置:")
    print("  - 文本生成：下一个词预测")
    print("  - 图像理解：物体分类 + 边界框回归")
    
    # 配置多任务头
    heads_config = {
        "text": {
            "output_size": 30000,
            "activation_modes": "classification"
        },
        "vision_class": {
            "output_size": 1000,  # ImageNet 类别
            "activation_modes": "classification"
        },
        "vision_bbox": {
            "output_size": 4,  # x, y, w, h
            "activation_modes": "regression"
        }
    }
    
    multi_head = MultiTaskActivationHead(heads_config)
    
    # 模拟多任务输入
    batch_size = 2
    seq_len = 10
    
    # 假设 ActionNetwork 为每个任务生成了不同的 S 分布
    loc_S_dict = {
        "text": torch.randn(batch_size, seq_len, 30000),
        "vision_class": torch.randn(batch_size, 1, 1000),
        "vision_bbox": torch.randn(batch_size, 1, 4)
    }
    
    scale_S_dict = {
        "text": torch.rand(batch_size, seq_len, 30000) + 0.1,
        "vision_class": torch.rand(batch_size, 1, 1000) + 0.1,
        "vision_bbox": torch.rand(batch_size, 1, 4) + 0.1
    }
    
    # 前向传播
    outputs = multi_head(loc_S_dict, scale_S_dict)
    
    print(f"\n📈 多任务输出:")
    for task, output in outputs.items():
        print(f"  - {task}: {output['output'].shape}")
    
    print(f"\n✅ 多任务头支持复杂的多模态应用")


def demo_custom_activation():
    """演示自定义激活模式"""
    print("\n" + "=" * 60)
    print("4. 自定义激活模式")
    print("=" * 60)
    
    # 场景：科学计算 - 预测分子性质
    print("\n📊 分子性质预测:")
    print("  - 前100维：原子类型（分类）")
    print("  - 后20维：物理性质（回归）")
    print("    - 能量、键长、键角等")
    
    # 创建自定义模式
    atom_types = 100
    properties = 20
    modes = ["classification"] * atom_types + ["regression"] * properties
    
    engine = CausalEngine(
        hidden_size=512,
        vocab_size=atom_types + properties,
        activation_modes=modes,
        classification_threshold_init=0.5,  # 更高的分类阈值
        regression_scale_init=10.0,        # 物理量的尺度
        regression_bias_init=-5.0          # 能量偏置
    )
    
    # 模拟分子表示
    hidden_states = torch.randn(1, 50, 512)  # 50个原子
    output = engine(hidden_states)
    
    # 提取不同类型的输出
    atom_probs = output['activation_output']['classification_probs']
    properties = output['activation_output']['regression_values']
    
    print(f"\n📈 预测结果:")
    print(f"  - 原子类型概率: {atom_probs.shape}")
    print(f"  - 物理性质预测: {properties.shape}")
    print(f"  - 能量预测示例: {properties[0, 0, 0].item():.3f} eV")
    
    print(f"\n✅ 灵活的激活配置支持各种科学计算应用")


def main():
    print("\n" + "="*80)
    print("CausalEngine v2.0 - 模块化架构演示")
    print("="*80 + "\n")
    
    # 运行所有演示
    demo_basic_modular_usage()
    demo_mixed_activation()
    demo_multi_task_head()
    demo_custom_activation()
    
    print("\n" + "="*80)
    print("🎉 CausalEngine v2.0 - 从单一算法到通用智能框架")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 