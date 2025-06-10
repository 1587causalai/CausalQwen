#!/usr/bin/env python3
"""
柯西分布线性变换数学验证脚本

本脚本详细验证 CauchyLinear 层的数学正确性，包括：
1. 柯西分布线性变换公式的正确实现
2. 分类头和回归头的scale计算验证
3. 修复后的回归头初始化验证
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper
from src.utils.distributions import CauchyLinear

def verify_cauchy_linear_math():
    """验证 CauchyLinear 的数学正确性"""
    print("🧮 柯西分布线性变换数学验证")
    print("=" * 60)
    
    # 创建一个简单的CauchyLinear层进行测试
    in_features = 4
    out_features = 3
    cauchy_layer = CauchyLinear(in_features, out_features)
    
    # 设置已知的权重和偏置用于验证
    with torch.no_grad():
        cauchy_layer.weight.data = torch.tensor([
            [1.0, -2.0, 0.5, 1.5],    # 第一个输出
            [0.0, 1.0, -1.0, 0.5],    # 第二个输出  
            [-0.5, 0.0, 2.0, -1.0]    # 第三个输出
        ])
        cauchy_layer.bias.data = torch.tensor([1.0, -0.5, 2.0])
    
    # 创建测试输入
    batch_size = 2
    input_loc = torch.tensor([
        [1.0, 2.0, -1.0, 0.5],   # 第一个样本
        [0.0, 1.5, 1.0, -0.5]    # 第二个样本
    ])
    input_scale = torch.tensor([
        [0.5, 1.0, 0.8, 1.2],    # 第一个样本
        [1.5, 0.5, 1.0, 0.8]     # 第二个样本
    ])
    
    print(f"输入 loc: {input_loc}")
    print(f"输入 scale: {input_scale}")
    print(f"权重矩阵:\n{cauchy_layer.weight}")
    print(f"偏置: {cauchy_layer.bias}")
    
    # 使用CauchyLinear进行变换
    output_loc, output_scale = cauchy_layer(input_loc, input_scale)
    
    print(f"\nCauchyLinear 输出:")
    print(f"输出 loc: {output_loc}")
    print(f"输出 scale: {output_scale}")
    
    # 手工验证第一个样本的第一个输出
    print(f"\n📐 手工验证第一个样本的第一个输出:")
    sample_idx = 0
    output_idx = 0
    
    weight_row = cauchy_layer.weight[output_idx]  # [1.0, -2.0, 0.5, 1.5]
    bias_val = cauchy_layer.bias[output_idx]      # 1.0
    
    # 理论计算 loc: W * input_loc + bias
    theoretical_loc = torch.dot(weight_row, input_loc[sample_idx]) + bias_val
    print(f"理论 loc = {weight_row} · {input_loc[sample_idx]} + {bias_val}")
    print(f"         = {torch.dot(weight_row, input_loc[sample_idx]):.4f} + {bias_val}")
    print(f"         = {theoretical_loc:.4f}")
    print(f"实际 loc = {output_loc[sample_idx, output_idx]:.4f}")
    print(f"loc 一致性: {'✅' if abs(theoretical_loc - output_loc[sample_idx, output_idx]) < 1e-5 else '❌'}")
    
    # 理论计算 scale: |W| * input_scale
    abs_weight_row = torch.abs(weight_row)
    theoretical_scale = torch.dot(abs_weight_row, input_scale[sample_idx])
    print(f"\n理论 scale = |{weight_row}| · {input_scale[sample_idx]}")
    print(f"          = {abs_weight_row} · {input_scale[sample_idx]}")
    print(f"          = {theoretical_scale:.4f}")
    print(f"实际 scale = {output_scale[sample_idx, output_idx]:.4f}")
    print(f"scale 一致性: {'✅' if abs(theoretical_scale - output_scale[sample_idx, output_idx]) < 1e-5 else '❌'}")

def verify_model_cauchy_math():
    """验证模型中的柯西数学实现"""
    print("\n🏗️ 模型中的柯西分布数学验证")
    print("=" * 60)
    
    # 创建模型和分词器
    tokenizer = QwenTokenizerWrapper(
        model_path="~/models/Qwen2.5-0.5B", 
        use_real_tokenizer=True
    )
    
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        causal_dim=896,
        use_real_qwen=True,
        qwen_model_path="~/models/Qwen2.5-0.5B"
    )
    
    model = CausalLanguageModel(config)
    
    # 执行知识传输初始化
    print("执行知识传输初始化...")
    model.init_weights(num_target_median=50.0, num_target_scale=10.0)
    
    # 创建测试输入
    text = "The price is 42.5 dollars."
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
    
    # 前向传播
    print("执行前向传播...")
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['numerical_values'], inputs['attention_mask'])
    
    # 验证分类头的数学
    print(f"\n🔍 分类头数学验证:")
    cls_head = model.action_network.classification_head.causal_linear
    sample_pos = 0  # 第一个样本，第一个位置
    
    # 获取输入的因果表征
    causal_loc = outputs['causal_loc'][0, sample_pos]  # [causal_dim]
    causal_scale = outputs['causal_scale'][0, sample_pos]  # [causal_dim]
    
    print(f"输入因果表征统计:")
    print(f"  causal_loc 均值: {causal_loc.mean():.4f}, 标准差: {causal_loc.std():.4f}")
    print(f"  causal_scale 均值: {causal_scale.mean():.4f}, 标准差: {causal_scale.std():.4f}")
    
    # 获取分类头的权重
    cls_weight = cls_head.weight  # [vocab_size, causal_dim]
    cls_bias = cls_head.bias if cls_head.bias is not None else torch.zeros(cls_weight.shape[0])
    
    print(f"分类头权重统计:")
    print(f"  权重形状: {cls_weight.shape}")
    print(f"  权重均值: {cls_weight.mean():.6f}, 标准差: {cls_weight.std():.6f}")
    print(f"  |权重| 均值: {torch.abs(cls_weight).mean():.6f}")
    
    # 验证几个特定位置的计算
    test_indices = [0, 1, tokenizer.num_token_id]  # 包括<NUM> token
    
    for idx in test_indices:
        if idx >= cls_weight.shape[0]:
            continue
            
        weight_row = cls_weight[idx]
        bias_val = cls_bias[idx]
        
        # 理论计算
        theoretical_loc = torch.dot(weight_row, causal_loc) + bias_val
        theoretical_scale = torch.dot(torch.abs(weight_row), causal_scale)
        
        # 实际输出
        actual_loc = outputs['cls_loc'][0, sample_pos, idx]
        actual_scale = outputs['cls_scale'][0, sample_pos, idx]
        
        print(f"\n  Token {idx} {'(<NUM>)' if idx == tokenizer.num_token_id else ''}:")
        print(f"    理论 loc: {theoretical_loc:.6f}")
        print(f"    实际 loc: {actual_loc:.6f}")
        print(f"    理论 scale: {theoretical_scale:.6f}")
        print(f"    实际 scale: {actual_scale:.6f}")
        
        loc_match = abs(theoretical_loc - actual_loc) < 1e-5
        scale_match = abs(theoretical_scale - actual_scale) < 1e-5
        print(f"    数学一致性: loc {'✅' if loc_match else '❌'}, scale {'✅' if scale_match else '❌'}")
        
        if not loc_match or not scale_match:
            print(f"    ❌ 数学公式可能存在实现错误！")
    
    # 验证回归头的数学 
    print(f"\n🔍 回归头数学验证:")
    reg_head = model.action_network.regression_head.causal_linear
    
    reg_weight = reg_head.weight[0]  # [causal_dim] - 只有一个输出
    reg_bias = reg_head.bias[0] if reg_head.bias is not None else torch.tensor(0.0)
    
    print(f"回归头权重统计:")
    print(f"  权重形状: {reg_weight.shape}")
    print(f"  权重均值: {reg_weight.mean():.6f}, 标准差: {reg_weight.std():.6f}")
    print(f"  |权重| 均值: {torch.abs(reg_weight).mean():.6f}")
    print(f"  偏置值: {reg_bias:.4f}")
    
    # 理论计算
    theoretical_reg_loc = torch.dot(reg_weight, causal_loc) + reg_bias
    theoretical_reg_scale = torch.dot(torch.abs(reg_weight), causal_scale)
    
    # 实际输出
    actual_reg_loc = outputs['reg_loc'][0, sample_pos]
    actual_reg_scale = outputs['reg_scale'][0, sample_pos]
    
    print(f"\n回归输出验证:")
    print(f"  理论 reg_loc: {theoretical_reg_loc:.6f}")
    print(f"  实际 reg_loc: {actual_reg_loc:.6f}")
    print(f"  理论 reg_scale: {theoretical_reg_scale:.6f}")
    print(f"  实际 reg_scale: {actual_reg_scale:.6f}")
    
    reg_loc_match = abs(theoretical_reg_loc - actual_reg_loc) < 1e-5
    reg_scale_match = abs(theoretical_reg_scale - actual_reg_scale) < 1e-5
    print(f"  数学一致性: loc {'✅' if reg_loc_match else '❌'}, scale {'✅' if reg_scale_match else '❌'}")
    
    if not reg_loc_match or not reg_scale_match:
        print(f"  ❌ 回归头数学公式可能存在实现错误！")
    else:
        print(f"  ✅ 回归头修复后的初始化和数学实现正确！")
    
    # 验证修复效果
    print(f"\n📊 修复效果验证:")
    print(f"  修复前问题: reg_loc=50.0 (忽略输入), reg_scale=0.001 (最小值)")
    print(f"  修复后结果: reg_loc={actual_reg_loc:.4f} (响应输入), reg_scale={actual_reg_scale:.4f} (合理值)")
    
    if abs(actual_reg_loc - 50.0) > 0.01:
        print(f"  ✅ 回归头现在正确响应输入的因果表征！")
    else:
        print(f"  ❌ 回归头仍然只依赖偏置，可能需要进一步调试")
        
    if actual_reg_scale > 0.01:
        print(f"  ✅ 回归头的scale现在有合理的值！")
    else:
        print(f"  ❌ 回归头的scale仍然太小")

def main():
    """主函数"""
    print("🔬 柯西分布线性变换数学验证脚本")
    print("=" * 80)
    
    try:
        # 1. 验证基础的CauchyLinear数学
        verify_cauchy_linear_math()
        
        # 2. 验证模型中的数学实现
        verify_model_cauchy_math()
        
        print("\n" + "=" * 80)
        print("✅ 数学验证完成！检查上述输出以确认所有公式的正确性。")
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 