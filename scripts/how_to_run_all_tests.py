#!/usr/bin/env python3
"""
CausalQwen MVP 框架快速测试脚本

测试目标：
1. 验证模型能够正确初始化
2. 验证前向传播能够正常进行
3. 验证三种推理模式都能工作
4. 验证基础的梯度计算

这是一个占位式测试，主要确保框架架构正确，具体数学实现后续完善。
"""

import sys
import os
import torch
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from causal_qwen_mvp import (
        CausalQwenMVPForCausalLM, 
        CausalQwen2Config,
        CausalInferenceEngine,
        InferenceValidator,
        get_model_info
    )
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


def test_model_initialization():
    """测试模型初始化"""
    print("\n🔧 测试模型初始化...")
    
    try:
        # 创建小型配置进行测试
        config = CausalQwen2Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,  # 添加这个参数
            max_position_embeddings=512,
            causal_size=128,  # 因果模型参数
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            use_cache=True,
            rope_theta=10000.0,
            hidden_act="silu",
        )
        
        model = CausalQwenMVPForCausalLM(config)
        
        # 检查模型结构
        assert hasattr(model, 'abduction_network')
        assert hasattr(model, 'action_network') 
        assert hasattr(model, 'ovr_classifier')
        
        print(f"✅ 模型初始化成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print(traceback.format_exc())
        return None


def test_forward_pass(model):
    """测试前向传播和数学正确性"""
    print("\n🚀 测试前向传播...")
    
    try:
        # 创建测试输入
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        
        # 检查输出结构
        assert outputs.loss is not None
        assert outputs.loc_S is not None
        assert outputs.scale_S is not None
        assert outputs.loc_U is not None
        assert outputs.scale_U is not None
        
        print(f"✅ 前向传播成功")
        print(f"   - 损失: {outputs.loss.item():.4f}")
        print(f"   - loc_S shape: {outputs.loc_S.shape}")
        print(f"   - scale_S shape: {outputs.scale_S.shape}")
        print(f"   - loc_U shape: {outputs.loc_U.shape}")
        print(f"   - scale_U shape: {outputs.scale_U.shape}")
        
        # 🔍 数学正确性检查
        errors = []
        
        # 检查1: Cauchy分布参数约束
        if not torch.all(outputs.scale_U > 0):
            errors.append("❌ scale_U包含非正值，违反Cauchy分布约束")
        if not torch.all(outputs.scale_S > 0):
            errors.append("❌ scale_S包含非正值，违反Cauchy分布约束")
        
        # 检查2: 数值稳定性
        if torch.any(torch.isnan(outputs.loc_U)) or torch.any(torch.isinf(outputs.loc_U)):
            errors.append("❌ loc_U包含NaN或Inf值")
        if torch.any(torch.isnan(outputs.scale_U)) or torch.any(torch.isinf(outputs.scale_U)):
            errors.append("❌ scale_U包含NaN或Inf值")
        if torch.any(torch.isnan(outputs.loc_S)) or torch.any(torch.isinf(outputs.loc_S)):
            errors.append("❌ loc_S包含NaN或Inf值")
        if torch.any(torch.isnan(outputs.scale_S)) or torch.any(torch.isinf(outputs.scale_S)):
            errors.append("❌ scale_S包含NaN或Inf值")
        
        if errors:
            print("\n⚠️  发现数学错误:")
            for error in errors:
                print(f"   {error}")
            return False
        
        print("✅ 数学正确性检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        print(traceback.format_exc())
        return False


def test_inference_modes(model):
    """测试三种推理模式"""
    print("\n🎯 测试推理模式...")
    
    # 创建测试输入
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    
    modes = ['standard', 'causal', 'compatible']
    results = {}
    
    for mode in modes:
        try:
            with torch.no_grad():
                output = model.inference(input_ids, mode=mode)
            results[mode] = output
            print(f"✅ {mode}推理模式: 输出shape {output.shape}")
        except Exception as e:
            print(f"❌ {mode}推理模式失败: {e}")
            results[mode] = None
    
    # 验证结果一致性
    successful_modes = [mode for mode, result in results.items() if result is not None]
    if len(successful_modes) >= 2:
        print(f"✅ {len(successful_modes)}/3 推理模式成功")
    else:
        print(f"⚠️  只有 {len(successful_modes)}/3 推理模式成功")
    
    return results


def test_generation(model):
    """测试序列生成"""
    print("\n📝 测试序列生成...")
    
    try:
        input_ids = torch.randint(0, model.config.vocab_size, (1, 3))
        
        # 测试生成
        with torch.no_grad():
            generated = model.generate_step_by_step(
                input_ids, 
                max_length=10, 
                mode='standard'
            )
        
        print(f"✅ 序列生成成功")
        print(f"   - 输入长度: {input_ids.shape[-1]}")
        print(f"   - 生成长度: {generated.shape[-1]}")
        print(f"   - 输入tokens: {input_ids[0].tolist()}")
        print(f"   - 生成tokens: {generated[0].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 序列生成失败: {e}")
        print(traceback.format_exc())
        return False


def test_gradient_computation(model):
    """测试梯度计算"""
    print("\n📈 测试梯度计算...")
    
    try:
        # 创建测试数据
        input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
        labels = torch.randint(0, model.config.vocab_size, (2, 8))
        
        # 设置为训练模式
        model.train()
        
        # 前向传播
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "没有找到任何梯度"
        
        print(f"✅ 梯度计算成功")
        print(f"   - 训练损失: {loss.item():.4f}")
        
        # 清理梯度
        model.zero_grad()
        
        return True
        
    except Exception as e:
        print(f"❌ 梯度计算失败: {e}")
        print(traceback.format_exc())
        return False


def test_mathematical_implementation(model):
    """测试数学实现的严格性"""
    print("\n🧮 测试数学实现符合性...")
    
    errors = []
    
    # 检查1: AbductionNetwork的bias项
    abduction = model.abduction_network
    if abduction.loc_net.bias is None:
        errors.append("❌ AbductionNetwork.loc_net缺少bias项，违反设计文档")
    if abduction.scale_net.bias is None:
        errors.append("❌ AbductionNetwork.scale_net缺少bias项，违反设计文档")
    
    # 检查2: ActionNetwork的bias项
    action = model.action_network
    if action.lm_head.bias is None:
        errors.append("❌ ActionNetwork.lm_head缺少bias项，违反设计文档")
    
    # 检查3: b_noise的维度
    expected_b_noise_size = model.config.causal_size
    actual_b_noise_size = action.b_noise.size(0)
    if actual_b_noise_size != expected_b_noise_size:
        errors.append(f"❌ ActionNetwork.b_noise维度错误: 应该是[{expected_b_noise_size}], 实际是[{actual_b_noise_size}]")
    
    # 检查4: 恒等映射初始化
    if model.config.causal_size == model.config.hidden_size:
        expected_identity = torch.eye(model.config.causal_size)
        if not torch.allclose(abduction.loc_net.weight, expected_identity, atol=1e-5):
            errors.append("❌ AbductionNetwork.loc_net未正确初始化为恒等映射")
    
    # 检查5: 测试softplus激活函数
    test_input = torch.randn(2, 10, model.config.hidden_size)
    with torch.no_grad():
        # 检查是否在forward中使用了正确的softplus
        scale_raw = abduction.scale_net(test_input)
        loc_U, scale_U = abduction(test_input)
        expected_softplus = torch.nn.functional.softplus(scale_raw)
        
        # 正确的检查：scale_U应该等于softplus(scale_raw)
        if not torch.allclose(scale_U, expected_softplus, atol=1e-5):
            errors.append("❌ AbductionNetwork未使用softplus激活函数")
        
        # 检查是否错误使用了abs+eps (需要更严格的检查)
        wrong_abs_eps = torch.abs(scale_raw) + 1e-6
        if torch.allclose(scale_U, wrong_abs_eps, atol=1e-7, rtol=1e-6):
            errors.append("❌ AbductionNetwork错误使用torch.abs+1e-6而非softplus激活")
    
    # 检查6: 线性稳定性实现
    try:
        loc_U = torch.randn(2, 5, model.config.causal_size)
        scale_U = torch.abs(torch.randn(2, 5, model.config.causal_size)) + 0.1
        
        loc_S, scale_S = action(loc_U, scale_U)
        
        # 验证位置参数变换是否正确（应该是标准线性变换）
        expected_loc_S = action.lm_head(loc_U)
        if not torch.allclose(loc_S, expected_loc_S, atol=1e-5):
            errors.append("❌ ActionNetwork位置参数变换不正确")
            
        # 验证尺度参数变换是否使用了线性稳定性
        scale_U_noisy = scale_U + torch.abs(action.b_noise)
        expected_scale_S = torch.matmul(scale_U_noisy, torch.abs(action.lm_head.weight).T)
        if not torch.allclose(scale_S, expected_scale_S, atol=1e-5):
            errors.append("❌ ActionNetwork尺度参数变换不符合线性稳定性")
            
    except Exception as e:
        errors.append(f"❌ 线性稳定性测试失败: {e}")
    
    if errors:
        print("🚨 发现数学实现错误:")
        for error in errors:
            print(f"   {error}")
        print("\n💡 这些错误说明当前实现不符合design-docs/causal_qwen.md的数学要求")
        return False
    else:
        print("✅ 数学实现检查通过")
        return True


def test_inference_validator(model):
    """测试推理验证器"""
    print("\n🔍 测试推理验证器...")
    
    try:
        validator = InferenceValidator(model)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
        
        # 测试一致性验证
        results = validator.validate_inference_consistency(input_ids, num_samples=3)
        
        print(f"✅ 推理验证器测试成功")
        print(f"   - 测试了 {len(results)} 种推理模式")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理验证器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🎬 开始CausalQwen MVP框架测试")
    print("="*50)
    
    # 显示模型信息
    info = get_model_info()
    print(f"📋 {info['name']} v{info['version']}")
    print(f"📄 {info['description']}")
    print(f"⚡ 状态: {info['status']}")
    
    # 测试序列
    test_results = {}
    
    # 1. 模型初始化测试
    model = test_model_initialization()
    test_results['initialization'] = model is not None
    
    if model is None:
        print("\n❌ 模型初始化失败，跳过后续测试")
        return
    
    # 2. 前向传播测试
    test_results['forward_pass'] = test_forward_pass(model)
    
    # 3. 推理模式测试
    inference_results = test_inference_modes(model)
    test_results['inference_modes'] = any(r is not None for r in inference_results.values())
    
    # 4. 序列生成测试
    test_results['generation'] = test_generation(model)
    
    # 5. 梯度计算测试
    test_results['gradient_computation'] = test_gradient_computation(model)
    
    # 6. 数学实现符合性测试
    test_results['mathematical_implementation'] = test_mathematical_implementation(model)
    
    # 7. 推理验证器测试
    test_results['inference_validator'] = test_inference_validator(model)
    
    # 汇总结果
    print("\n" + "="*50)
    print("📊 测试结果汇总:")
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name:20}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！MVP框架基础功能正常")
    elif passed >= total * 0.8:
        print("⚠️  大部分测试通过，框架基本可用")
    else:
        print("❌ 多个测试失败，需要检查框架实现")
    
    # 下一步建议
    print(f"\n📋 下一步建议:")
    for step in info['next_steps']:
        print(f"   - {step}")


if __name__ == "__main__":
    main() 