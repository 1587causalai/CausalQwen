#!/usr/bin/env python3
"""
基于第一性原理的初始化策略测试脚本

本脚本测试修改后的ActionNetwork初始化，验证：
1. 所有偏置都被正确设置为0（移除魔法数字）
2. 初始概率分布更加均匀和公平
3. <NUM> token不再被人为抑制
4. 回归头不再使用数据依赖的偏置
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper

def test_first_principles_initialization():
    """测试基于第一性原理的初始化策略"""
    
    print("=" * 80)
    print("=   基于第一性原理的初始化策略测试   =")
    print("=" * 80)
    
    # 1. 设置模型和配置
    print("\n[步骤 1] 设置模型和配置...")
    
    # 加载分词器
    tokenizer = QwenTokenizerWrapper(model_path="/Users/gongqian/models/Qwen2.5-0.5B", use_real_tokenizer=True)
    print(f"加载分词器完成，词汇表大小: {tokenizer.vocab_size}")
    print(f"<NUM> token ID: {tokenizer.num_token_id}")
    
    # 创建配置
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        causal_dim=896,  # C=H 约束
        use_mock_feature_network=False,  # 使用真实Qwen模型
        ovr_threshold=10.0
    )
    
    # 创建模型
    model = CausalLanguageModel(config)
    print(f"模型创建完成")
    
    # 2. 执行初始化并验证
    print("\n[步骤 2] 执行第一性原理初始化...")
    
    # 使用虚拟的数据统计值（在新策略下这些值会被忽略）
    dummy_median = 50.0
    dummy_scale = 25.0
    
    model.init_weights(dummy_median, dummy_scale)
    
    # 3. 验证初始化结果
    print("\n[步骤 3] 验证第一性原理初始化结果...")
    
    # 3.1 验证分类头的偏置
    cls_head = model.action_network.classification_head.causal_linear
    cls_bias = cls_head.bias.data if cls_head.bias is not None else None
    
    print(f"\n🔍 分类头偏置验证:")
    if cls_bias is not None:
        bias_zero_check = torch.allclose(cls_bias, torch.zeros_like(cls_bias), atol=1e-6)
        print(f"  所有偏置是否为0: {'✅' if bias_zero_check else '❌'}")
        print(f"  偏置统计: 均值={cls_bias.mean().item():.6f}, 最大绝对值={cls_bias.abs().max().item():.6f}")
        
        # 特别检查<NUM> token的偏置
        num_bias = cls_bias[config.num_token_id].item()
        print(f"  <NUM> token (ID: {config.num_token_id}) 偏置: {num_bias:.6f} (应该是0.0)")
        
        if bias_zero_check:
            print(f"  ✅ 第一性原理验证通过: 无人为偏见，公平起点")
        else:
            print(f"  ❌ 检测到非零偏置，可能仍有魔法数字")
    else:
        print(f"  ❌ 分类头没有偏置层")
    
    # 3.2 验证回归头的偏置
    reg_head = model.action_network.regression_head.causal_linear
    reg_bias = reg_head.bias.data if reg_head.bias is not None else None
    
    print(f"\n🔍 回归头偏置验证:")
    if reg_bias is not None:
        reg_bias_zero = torch.allclose(reg_bias, torch.zeros_like(reg_bias), atol=1e-6)
        print(f"  回归偏置是否为0: {'✅' if reg_bias_zero else '❌'}")
        print(f"  回归偏置值: {reg_bias.item():.6f} (应该是0.0，而非数据中位数50.0)")
        
        if reg_bias_zero:
            print(f"  ✅ 第一性原理验证通过: 无数据依赖偏置，纯净学习")
        else:
            print(f"  ❌ 检测到非零回归偏置，可能仍使用数据统计量")
    else:
        print(f"  ❌ 回归头没有偏置层")
    
    # 4. 测试前向传播，验证初始概率分布
    print(f"\n[步骤 4] 测试初始概率分布...")
    
    # 创建测试输入
    test_input = torch.tensor([[1, 2, 3, config.num_token_id, 5]])  # 包含<NUM> token
    test_nums = torch.tensor([[0.0, 0.0, 0.0, 99.99, 0.0]])
    
    with torch.no_grad():
        outputs = model(test_input, test_nums)
        
        # 计算OvR概率
        cls_loc = outputs['cls_loc']
        cls_scale = outputs['cls_scale']
        probs = model.action_network.classification_head.compute_probabilities(cls_loc, cls_scale)
        
        # 检查概率分布的均匀性
        sample_pos = 3  # <NUM> token的位置
        probs_at_num = probs[0, sample_pos, :]  # 在<NUM>位置的概率
        
        prob_mean = probs_at_num.mean().item()
        prob_std = probs_at_num.std().item()
        num_token_prob = probs_at_num[config.num_token_id].item()
        
        print(f"\n🔍 初始概率分布分析 (位置 {sample_pos}, <NUM> token):")
        print(f"  平均概率: {prob_mean:.4f}")
        print(f"  概率标准差: {prob_std:.4f}")
        print(f"  <NUM> token概率: {num_token_prob:.4f}")
        print(f"  概率范围: [{probs_at_num.min().item():.4f}, {probs_at_num.max().item():.4f}]")
        
        # 检查是否接近均匀分布（在高阈值下，所有概率应该都比较小且相近）
        expected_prob_near_threshold = 0.5  # 当loc=0, scale较大时，概率应该接近0.5
        uniform_check = abs(prob_mean - 0.5) < 0.1  # 允许一定的偏差
        
        if uniform_check:
            print(f"  ✅ 第一性原理验证通过: 初始分布接近均匀，无人为偏好")
        else:
            print(f"  ⚠️  初始分布可能不够均匀，需要进一步调查")
    
    # 5. 总结
    print(f"\n" + "=" * 80)
    print(f"=   第一性原理初始化测试总结   =")
    print(f"=" * 80)
    
    all_checks_passed = True
    
    if cls_bias is not None and bias_zero_check:
        print(f"✅ 分类头偏置: 全部为0，移除魔法数字偏见")
    else:
        print(f"❌ 分类头偏置: 检测到问题")
        all_checks_passed = False
    
    if reg_bias is not None and reg_bias_zero:
        print(f"✅ 回归头偏置: 为0，移除数据依赖偏置")
    else:
        print(f"❌ 回归头偏置: 检测到问题")
        all_checks_passed = False
    
    if uniform_check:
        print(f"✅ 初始概率分布: 接近均匀，体现第一性原理")
    else:
        print(f"⚠️  初始概率分布: 需要进一步验证")
    
    if all_checks_passed:
        print(f"\n🎉 第一性原理初始化策略验证通过!")
        print(f"   模型现在从纯净、无偏见的状态开始学习")
        print(f"   所有不确定性通过AbductionNetwork的scale_U表达")
        print(f"   移除了所有启发式魔法数字")
    else:
        print(f"\n⚠️  发现问题，需要进一步修复初始化策略")
    
    return all_checks_passed

if __name__ == "__main__":
    success = test_first_principles_initialization()
    exit(0 if success else 1) 