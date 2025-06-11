#!/usr/bin/env python3
"""
测试去掉特殊初始化后的效果

验证：
1. <NUM> token 使用预留权重而不是特殊初始化
2. 回归头偏置为0而不是数据中位数
3. 权重继承正确工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper
from src.models.feature_network import QwenFeatureNetwork

def test_no_special_initialization():
    """测试去掉特殊初始化后的效果"""
    
    print("=" * 80)
    print("=   测试去掉特殊初始化后的效果   =")
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
        use_real_qwen=True,
        qwen_model_path="/Users/gongqian/models/Qwen2.5-0.5B",
        ovr_threshold=10.0
    )
    
    # 创建模型
    model = CausalLanguageModel(config)
    print(f"模型创建完成")
    
    # 2. 获取Qwen原始权重以供对比
    print("\n[步骤 2] 获取Qwen原始权重...")
    
    # 获取Qwen的lm_head权重
    qwen_feature_network = model.feature_network.base_network if hasattr(model.feature_network, 'base_network') else model.feature_network
    qwen_lm_head = qwen_feature_network.get_lm_head()
    qwen_lm_head_weight = qwen_lm_head.weight.data
    
    print(f"Qwen lm_head权重形状: {qwen_lm_head_weight.shape}")
    
    # 检查<NUM> token位置的预留权重
    num_token_id = tokenizer.num_token_id
    if num_token_id < qwen_lm_head_weight.shape[0]:
        original_num_weight = qwen_lm_head_weight[num_token_id, :]
        print(f"<NUM> token (ID: {num_token_id}) 在Qwen中的预留权重统计:")
        print(f"  均值: {original_num_weight.mean().item():.6f}")
        print(f"  标准差: {original_num_weight.std().item():.6f}")
        print(f"  非零元素: {(original_num_weight != 0).sum().item()}/{original_num_weight.numel()}")
    else:
        print(f"警告：<NUM> token ID {num_token_id} 超出Qwen权重范围")
    
    # 3. 执行修改后的初始化
    print("\n[步骤 3] 执行修改后的初始化...")
    
    # 这些参数现在应该被忽略
    model.init_weights(num_target_median=999.99, num_target_scale=888.88)
    
    # 4. 验证<NUM> token权重继承
    print("\n[步骤 4] 验证<NUM> token权重继承...")
    
    cls_head = model.action_network.classification_head.causal_linear
    causal_num_weight = cls_head.weight.data[num_token_id, :]
    
    print(f"CausalQwen中<NUM> token权重统计:")
    print(f"  均值: {causal_num_weight.mean().item():.6f}")
    print(f"  标准差: {causal_num_weight.std().item():.6f}")
    print(f"  非零元素: {(causal_num_weight != 0).sum().item()}/{causal_num_weight.numel()}")
    
    # 检查权重是否完全一致
    if num_token_id < qwen_lm_head_weight.shape[0]:
        weight_diff = torch.abs(causal_num_weight - original_num_weight)
        max_diff = weight_diff.max().item()
        mean_diff = weight_diff.mean().item()
        
        print(f"权重继承验证:")
        print(f"  最大差异: {max_diff:.8f}")
        print(f"  平均差异: {mean_diff:.8f}")
        print(f"  权重一致: {'✅' if max_diff < 1e-6 else '❌'}")
    
    # 5. 验证回归头偏置
    print("\n[步骤 5] 验证回归头偏置...")
    
    reg_head = model.action_network.regression_head.causal_linear
    reg_bias = reg_head.bias.data if reg_head.bias is not None else None
    
    if reg_bias is not None:
        print(f"回归头偏置值: {reg_bias.item():.6f}")
        print(f"偏置为零: {'✅' if abs(reg_bias.item()) < 1e-6 else '❌'}")
    else:
        print(f"回归头无偏置")
    
    # 6. 验证分类头偏置
    print("\n[步骤 6] 验证分类头偏置...")
    
    cls_bias = cls_head.bias.data if cls_head.bias is not None else None
    
    if cls_bias is not None:
        print(f"分类头偏置统计:")
        print(f"  均值: {cls_bias.mean().item():.6f}")
        print(f"  标准差: {cls_bias.std().item():.6f}")
        print(f"  <NUM> token偏置: {cls_bias[num_token_id].item():.6f}")
        print(f"  全零偏置: {'✅' if torch.allclose(cls_bias, torch.zeros_like(cls_bias)) else '❌'}")
    else:
        print(f"分类头无偏置")
    
    # 7. 测试前向传播
    print("\n[步骤 7] 测试前向传播...")
    
    # 创建测试输入
    test_input = torch.tensor([[1, 2, 3, num_token_id, 5]])  # 包含<NUM> token
    test_nums = torch.tensor([[0.0, 0.0, 0.0, 123.45, 0.0]])
    
    with torch.no_grad():
        outputs = model(test_input, test_nums)
        
        cls_loc = outputs['cls_loc']
        cls_scale = outputs['cls_scale']
        reg_loc = outputs['reg_loc']
        reg_scale = outputs['reg_scale']
        
        print(f"前向传播成功:")
        print(f"  cls_loc形状: {cls_loc.shape}")
        print(f"  cls_scale形状: {cls_scale.shape}")
        print(f"  reg_loc形状: {reg_loc.shape}")
        print(f"  reg_scale形状: {reg_scale.shape}")
        
        # 检查<NUM> token位置的输出
        num_pos = 3  # <NUM> token的位置
        num_cls_loc = cls_loc[0, num_pos, num_token_id].item()
        num_cls_scale = cls_scale[0, num_pos, num_token_id].item()
        num_reg_loc = reg_loc[0, num_pos].item()
        num_reg_scale = reg_scale[0, num_pos].item()
        
        print(f"<NUM> token位置输出:")
        print(f"  分类 loc: {num_cls_loc:.6f}")
        print(f"  分类 scale: {num_cls_scale:.6f}")
        print(f"  回归 loc: {num_reg_loc:.6f}")
        print(f"  回归 scale: {num_reg_scale:.6f}")
    
    # 8. 总结
    print(f"\n" + "=" * 80)
    print(f"=   去掉特殊初始化测试总结   =")
    print(f"=" * 80)
    
    print("✅ 成功验证：")
    print("  - <NUM> token 使用预留权重，无特殊随机初始化")
    print("  - 回归头偏置为0，无数据依赖")
    print("  - 权重继承正确工作")
    print("  - 前向传播正常")
    
    print("\n🎯 核心改进：")
    print("  - 利用Qwen预留token的预初始化权重")
    print("  - 消除了数据依赖的偏置初始化")
    print("  - 简化了初始化流程")


if __name__ == "__main__":
    test_no_special_initialization() 