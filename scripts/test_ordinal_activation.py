"""
测试离散有序激活功能
演示如何使用 CausalEngine 的三种激活模式：分类、回归、离散有序
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_engine import CausalEngine, ActivationMode


def test_mixed_activation_modes():
    """测试混合激活模式：分类 + 回归 + 离散有序"""
    
    # 配置：10个输出维度，混合使用三种激活模式
    hidden_size = 64
    output_size = 10
    
    # 定义每个维度的激活模式
    activation_modes = [
        "classification",  # 0: 二分类
        "classification",  # 1: 二分类
        "regression",      # 2: 数值回归
        "regression",      # 3: 数值回归
        "ordinal",         # 4: 5级评分（1-5星）
        "ordinal",         # 5: 3级情感（负面/中性/正面）
        "classification",  # 6: 二分类
        "ordinal",         # 7: 10级置信度
        "regression",      # 8: 数值回归
        "classification"   # 9: 二分类
    ]
    
    # 离散有序维度的类别数配置
    ordinal_num_classes = [5, 3, 10]  # 对应维度 4, 5, 7
    
    # 创建 CausalEngine
    engine = CausalEngine(
        hidden_size=hidden_size,
        vocab_size=output_size,
        activation_modes=activation_modes,
        ordinal_num_classes=ordinal_num_classes,
        ordinal_threshold_init=1.0  # 初始阈值范围
    )
    
    # 创建测试输入
    batch_size = 2
    seq_len = 3
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    outputs = engine(hidden_states, temperature=0.5, apply_activation=True)
    
    # 打印结果
    print("=== CausalEngine 混合激活模式测试 ===\n")
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {outputs['output'].shape}\n")
    
    # 详细展示激活输出
    activation_output = outputs['activation_output']
    
    print("1. 分类维度 (0, 1, 6, 9):")
    print(f"   概率值: {activation_output['classification_probs'][0, 0, :].tolist()}")
    print(f"   (每个值表示 P(S > C) 的概率)\n")
    
    print("2. 回归维度 (2, 3, 8):")
    print(f"   数值输出: {activation_output['regression_values'][0, 0, :].tolist()}")
    print(f"   (线性变换后的值: a*S + b)\n")
    
    print("3. 离散有序维度:")
    ordinal_preds = activation_output['ordinal_predictions'][0, 0, :].tolist()
    print(f"   - 维度4 (5级评分): 预测级别 = {int(ordinal_preds[0]) + 1}/5")
    print(f"   - 维度5 (3级情感): 预测级别 = {['负面', '中性', '正面'][int(ordinal_preds[1])]}")
    print(f"   - 维度7 (10级置信): 预测级别 = {int(ordinal_preds[2]) + 1}/10\n")
    
    # 展示决策分布参数
    print("4. 底层决策分布参数 S ~ Cauchy(loc_S, scale_S):")
    print(f"   loc_S[0, 0, :5]: {outputs['loc_S'][0, 0, :5].tolist()}")
    print(f"   scale_S[0, 0, :5]: {outputs['scale_S'][0, 0, :5].tolist()}\n")
    
    # 展示个体分布参数
    print("5. 个体表征分布 U ~ Cauchy(loc_U, scale_U):")
    print(f"   loc_U[0, 0, :5]: {outputs['loc_U'][0, 0, :5].tolist()}")
    print(f"   scale_U[0, 0, :5]: {outputs['scale_U'][0, 0, :5].tolist()}")


def test_ordinal_only():
    """测试纯离散有序激活"""
    
    hidden_size = 32
    output_size = 3  # 三个离散有序输出
    
    # 全部使用离散有序激活
    engine = CausalEngine(
        hidden_size=hidden_size,
        vocab_size=output_size,
        activation_modes="ordinal",  # 所有维度都是离散有序
        ordinal_num_classes=4        # 所有维度都是4级
    )
    
    # 测试输入
    hidden_states = torch.randn(1, 1, hidden_size)
    
    # 前向传播
    outputs = engine(hidden_states, temperature=0.0)  # 纯因果推理
    
    print("\n=== 纯离散有序激活测试 ===\n")
    print(f"所有{output_size}个维度都是4级离散有序输出")
    print(f"预测结果: {outputs['output'][0, 0, :].tolist()}")
    print("(0=第1级, 1=第2级, 2=第3级, 3=第4级)")


if __name__ == "__main__":
    test_mixed_activation_modes()
    test_ordinal_only()
    
    print("\n✅ 离散有序激活功能测试完成！") 