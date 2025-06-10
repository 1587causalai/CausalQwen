#!/usr/bin/env python3
"""
ActionNetwork初始化专项测试

专门测试ActionNetwork的第一性原理初始化是否正确实施
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.models.action_network import ActionNetwork

def test_action_network_init():
    """测试ActionNetwork的第一性原理初始化"""
    
    print("=" * 60)
    print("=   ActionNetwork第一性原理初始化测试   =")
    print("=" * 60)
    
    # 1. 创建ActionNetwork
    causal_dim = 896
    vocab_size = 151666
    num_token_id = 151665  # <NUM> token ID
    
    action_net = ActionNetwork(causal_dim, vocab_size, num_token_id)
    print(f"创建ActionNetwork: causal_dim={causal_dim}, vocab_size={vocab_size}")
    
    # 2. 创建一个模拟的Qwen lm_head用于测试
    mock_qwen_lm_head = nn.Linear(causal_dim, vocab_size)
    print(f"创建模拟Qwen lm_head用于测试")
    
    # 3. 执行第一性原理初始化
    print("\n执行第一性原理初始化...")
    dummy_median = 50.0  # 这个值应该被忽略
    dummy_scale = 25.0   # 这个值应该被忽略
    
    action_net.init_weights(mock_qwen_lm_head, dummy_median, dummy_scale, num_token_id)
    
    # 4. 验证分类头偏置
    print(f"\n🔍 验证分类头偏置:")
    cls_head = action_net.classification_head.causal_linear
    cls_bias = cls_head.bias.data if cls_head.bias is not None else None
    
    if cls_bias is not None:
        bias_zero_check = torch.allclose(cls_bias, torch.zeros_like(cls_bias), atol=1e-6)
        print(f"  所有偏置是否为0: {'✅' if bias_zero_check else '❌'}")
        print(f"  偏置统计: 均值={cls_bias.mean().item():.6f}, 最大绝对值={cls_bias.abs().max().item():.6f}")
        
        # 检查<NUM> token
        num_bias = cls_bias[num_token_id].item()
        print(f"  <NUM> token偏置: {num_bias:.6f} (应该是0.0)")
        
        if bias_zero_check:
            print(f"  ✅ 第一性原理验证通过")
        else:
            print(f"  ❌ 发现非零偏置")
            # 输出一些非零偏置的例子
            non_zero_indices = torch.nonzero(cls_bias.abs() > 1e-6).flatten()
            if len(non_zero_indices) > 0:
                print(f"  非零偏置样本: {[(i.item(), cls_bias[i].item()) for i in non_zero_indices[:5]]}")
    else:
        print(f"  ❌ 分类头没有偏置层")
    
    # 5. 验证回归头偏置
    print(f"\n🔍 验证回归头偏置:")
    reg_head = action_net.regression_head.causal_linear
    reg_bias = reg_head.bias.data if reg_head.bias is not None else None
    
    if reg_bias is not None:
        reg_bias_zero = torch.allclose(reg_bias, torch.zeros_like(reg_bias), atol=1e-6)
        print(f"  回归偏置是否为0: {'✅' if reg_bias_zero else '❌'}")
        print(f"  回归偏置值: {reg_bias.item():.6f} (应该是0.0)")
        
        if reg_bias_zero:
            print(f"  ✅ 第一性原理验证通过")
        else:
            print(f"  ❌ 发现非零回归偏置")
    else:
        print(f"  ❌ 回归头没有偏置层")
    
    # 6. 总结
    print(f"\n" + "=" * 60)
    print(f"=   测试总结   =")
    print(f"=" * 60)
    
    success = True
    if cls_bias is not None:
        if bias_zero_check:
            print(f"✅ 分类头偏置: 第一性原理初始化成功")
        else:
            print(f"❌ 分类头偏置: 初始化失败，存在非零偏置")
            success = False
    
    if reg_bias is not None:
        if reg_bias_zero:
            print(f"✅ 回归头偏置: 第一性原理初始化成功")
        else:
            print(f"❌ 回归头偏置: 初始化失败，存在非零偏置")
            success = False
    
    if success:
        print(f"\n🎉 第一性原理初始化测试通过!")
        print(f"   所有偏置都正确设置为0")
        print(f"   移除了魔法数字偏见")
    else:
        print(f"\n⚠️  测试失败: 初始化没有按照第一性原理执行")
    
    return success

if __name__ == "__main__":
    success = test_action_network_init()
    exit(0 if success else 1) 