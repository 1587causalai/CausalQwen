#!/usr/bin/env python3
"""
测试更新后的run_experiments.py脚本

验证第一性原理初始化策略是否正确集成到实验框架中
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper

def test_updated_config():
    """测试更新后的配置是否正确工作"""
    
    print("=" * 60)
    print("=   测试更新后的实验配置   =")
    print("=" * 60)
    
    # 1. 测试新的配置参数
    print("\n🔧 测试CausalLMConfig新参数...")
    
    config = CausalLMConfig(
        vocab_size=151666,
        num_token_id=151665,
        hidden_size=896,
        causal_dim=896,
        ovr_threshold=10.0,
        reg_loss_weight=1.0,
        initial_scale_bias=2.3,  # 新参数
        use_real_qwen=True,
        qwen_model_path="/Users/gongqian/models/Qwen2.5-0.5B"
    )
    
    print(f"✅ CausalLMConfig创建成功:")
    print(f"   ovr_threshold: {config.ovr_threshold}")
    print(f"   reg_loss_weight: {config.reg_loss_weight}")
    print(f"   initial_scale_bias: {config.initial_scale_bias}")
    
    # 2. 测试模型创建和初始化
    print(f"\n🚀 测试模型创建和第一性原理初始化...")
    
    model = CausalLanguageModel(config)
    print(f"✅ 模型创建成功")
    
    # 3. 测试初始化
    print(f"\n🧮 测试第一性原理初始化...")
    
    # 使用虚拟数据统计值
    dummy_median = 50.0
    dummy_scale = 25.0
    
    model.init_weights(dummy_median, dummy_scale)
    
    # 4. 验证归因推断网络 (AbductionNetwork) 的scale偏置
    print(f"\n🔍 验证归因推断网络 (AbductionNetwork) 的scale偏置...")
    
    abduction_bias = model.abduction_network.fc.bias.data
    causal_dim = model.causal_dim
    scale_bias = abduction_bias[causal_dim:]  # 后半部分是scale偏置
    
    expected_bias = config.initial_scale_bias
    actual_bias = scale_bias[0].item()  # 所有scale偏置应该相同
    
    print(f"   期望的scale偏置: {expected_bias}")
    print(f"   实际的scale偏置: {actual_bias:.6f}")
    print(f"   exp({actual_bias:.1f}) ≈ {torch.exp(torch.tensor(actual_bias)).item():.1f}")
    
    bias_correct = abs(actual_bias - expected_bias) < 1e-6
    print(f"   scale偏置正确: {'✅' if bias_correct else '❌'}")
    
    # 5. 验证ActionNetwork的第一性原理初始化
    print(f"\n🔍 验证ActionNetwork的第一性原理初始化...")
    
    cls_bias = model.action_network.classification_head.causal_linear.bias.data
    reg_bias = model.action_network.regression_head.causal_linear.bias.data
    
    cls_bias_zero = torch.allclose(cls_bias, torch.zeros_like(cls_bias), atol=1e-6)
    reg_bias_zero = torch.allclose(reg_bias, torch.zeros_like(reg_bias), atol=1e-6)
    
    print(f"   分类头偏置全为0: {'✅' if cls_bias_zero else '❌'}")
    print(f"   回归头偏置全为0: {'✅' if reg_bias_zero else '❌'}")
    
    # 6. 测试不同的initial_scale_bias值
    print(f"\n🔬 测试不同的initial_scale_bias值...")
    
    test_biases = [1.0, 2.3, 3.0]
    for test_bias in test_biases:
        test_config = CausalLMConfig(
            vocab_size=151666,
            num_token_id=151665,
            hidden_size=896,
            causal_dim=896,
            initial_scale_bias=test_bias,
            use_real_qwen=False  # 使用mock以加快测试
        )
        
        test_model = CausalLanguageModel(test_config)
        test_model.init_weights(50.0, 25.0)
        
        test_scale_bias = test_model.abduction_network.fc.bias.data[896:][0].item()
        expected_scale = torch.exp(torch.tensor(test_bias)).item()
        
        print(f"   bias={test_bias} → scale≈{expected_scale:.1f} (实际bias: {test_scale_bias:.1f})")
    
    # 7. 总结
    print(f"\n" + "=" * 60)
    print(f"=   测试总结   =")
    print(f"=" * 60)
    
    all_passed = bias_correct and cls_bias_zero and reg_bias_zero
    
    if all_passed:
        print(f"🎉 所有测试通过!")
        print(f"   ✅ 新配置参数正确工作")
        print(f"   ✅ 归因推断网络 (AbductionNetwork) 支持可配置的初始不确定性")
        print(f"   ✅ ActionNetwork使用第一性原理初始化")
        print(f"   ✅ 整个框架准备好进行实验")
    else:
        print(f"⚠️  部分测试失败，需要进一步检查")
    
    return all_passed

if __name__ == "__main__":
    success = test_updated_config()
    exit(0 if success else 1) 