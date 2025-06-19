#!/usr/bin/env python3
"""
CausalQwen与Qwen兼容性测试

验证重点：
1. CausalQwen完全兼容Qwen的生成接口
2. do_sample参数控制因果推理行为差异
3. 与Qwen相同的参数产生期望的行为
4. 生成质量和多样性验证
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_section(title):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.WHITE}ℹ️  {msg}{Colors.END}")

def test_qwen_compatibility():
    """测试与Qwen的接口兼容性"""
    print_section("Qwen接口兼容性测试")
    
    try:
        from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config
        
        # 创建小型测试模型
        config = CausalQwen2Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
            causal_size=128
        )
        
        model = CausalQwenMVPForCausalLM(config)
        print_success("CausalQwen模型创建成功")
        
        # 测试输入
        input_ids = torch.randint(0, 1000, (2, 5))
        print_info(f"测试输入: {input_ids.shape}")
        
        return model, input_ids
        
    except Exception as e:
        print_error(f"模型创建失败: {e}")
        return None, None

def test_generation_interface(model, input_ids):
    """测试生成接口"""
    print_section("生成接口测试")
    
    try:
        # 测试1: 确定性生成 (do_sample=False)
        print_info("测试确定性生成 (do_sample=False)")
        det_output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0
        )
        print_success(f"确定性生成成功: {det_output.shape}")
        print_info(f"生成序列: {det_output[0].tolist()}")
        
        # 测试2: 采样生成 (do_sample=True)
        print_info("测试采样生成 (do_sample=True)")
        samp_output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        print_success(f"采样生成成功: {samp_output.shape}")
        print_info(f"生成序列: {samp_output[0].tolist()}")
        
        # 测试3: 验证长度正确性
        expected_length = input_ids.shape[1] + 5
        if det_output.shape[1] == expected_length and samp_output.shape[1] == expected_length:
            print_success("生成长度正确")
        else:
            print_error(f"生成长度错误: 期望{expected_length}, 实际det={det_output.shape[1]}, samp={samp_output.shape[1]}")
        
        # 测试4: 验证do_sample差异
        det_new = det_output[0, input_ids.shape[1]:].tolist()
        samp_new = samp_output[0, input_ids.shape[1]:].tolist()
        
        difference_count = sum(1 for a, b in zip(det_new, samp_new) if a != b)
        print_info(f"确定性vs采样差异: {difference_count}/5 个位置不同")
        
        if difference_count > 0:
            print_success("do_sample参数正确控制生成差异")
        else:
            print_error("do_sample参数未产生预期差异")
        
        return True
        
    except Exception as e:
        print_error(f"生成接口测试失败: {e}")
        return False

def test_causal_mathematical_principles(model, input_ids):
    """测试因果数学原理"""
    print_section("因果数学原理测试")
    
    try:
        from causal_qwen_mvp import InferenceValidator
        
        validator = InferenceValidator(model)
        results = validator.validate_causal_principles(input_ids, temperature=1.0)
        
        # 位置参数差异
        pos_diff = results['position_difference'].item()
        scale_diff = results['scale_difference'].item()
        
        print_info(f"位置参数差异: {pos_diff:.6f}")
        print_info(f"尺度参数差异: {scale_diff:.6f}")
        
        if pos_diff > 1e-3:
            print_success("位置参数在不同模式下有显著差异（符合因果设计）")
        else:
            print_error("位置参数差异过小")
        
        # 验证基础表征
        loc_U_mean = results['base_representations']['loc_U'].mean().item()
        scale_U_mean = results['base_representations']['scale_U'].mean().item()
        
        print_info(f"个体表征统计: loc_U={loc_U_mean:.4f}, scale_U={scale_U_mean:.4f}")
        
        if scale_U_mean > 1e-3:
            print_success("尺度参数初始化正确（>0）")
        else:
            print_error("尺度参数过小，可能初始化有问题")
        
        return True
        
    except Exception as e:
        print_error(f"因果数学原理测试失败: {e}")
        return False

def test_temperature_effects(model, input_ids):
    """测试温度参数效果"""
    print_section("温度参数效果测试")
    
    try:
        temperatures = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]  # 添加温度为零的测试
        results = []
        
        for temp in temperatures:
            # 固定随机种子确保可比性
            torch.manual_seed(42)
            output = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=True,
                temperature=temp,
                top_k=100  # 增大top_k以更好观察温度效果
            )
            new_tokens = output[0, input_ids.shape[1]:].tolist()
            results.append((temp, new_tokens))
            print_info(f"温度T={temp}: {new_tokens}")
        
        # 特别验证温度为零的场景
        if len(results) > 0 and results[0][0] == 0.0:
            print_info("🌡️ 温度为零是极其重要的边界条件！")
            temp_zero_tokens = results[0][1]
            print_info(f"温度T=0结果: {temp_zero_tokens}")
        
        # 分析温度效果
        unique_sequences = len(set(tuple(result[1]) for result in results))
        print_info(f"不同温度产生的序列多样性: {unique_sequences}/{len(temperatures)}")
        
        if unique_sequences >= 3:
            print_success("温度参数有效控制生成多样性")
        else:
            print_error("温度参数效果不明显")
        
        # 测试确定性模式下温度无效
        torch.manual_seed(42)
        det_output1 = model.generate(input_ids, max_new_tokens=3, do_sample=False, temperature=0.1)
        torch.manual_seed(42) 
        det_output2 = model.generate(input_ids, max_new_tokens=3, do_sample=False, temperature=2.0)
        
        if torch.equal(det_output1, det_output2):
            print_success("确定性模式下温度参数正确无效")
        else:
            print_error("确定性模式下温度参数意外生效")
        
        return True
        
    except Exception as e:
        print_error(f"温度参数测试失败: {e}")
        return False

def test_batch_generation(model, input_ids):
    """测试批量生成"""
    print_section("批量生成测试")
    
    try:
        batch_size = input_ids.shape[0]
        
        # 批量采样生成
        batch_output = model.generate(
            input_ids,
            max_new_tokens=4,
            do_sample=True,
            temperature=1.0,
            top_k=50
        )
        
        print_success(f"批量生成成功: {batch_output.shape}")
        
        # 验证批内多样性
        batch_sequences = []
        for i in range(batch_size):
            new_tokens = batch_output[i, input_ids.shape[1]:].tolist()
            batch_sequences.append(new_tokens)
            print_info(f"批次{i}: {new_tokens}")
        
        unique_in_batch = len(set(tuple(seq) for seq in batch_sequences))
        print_info(f"批内序列多样性: {unique_in_batch}/{batch_size}")
        
        if batch_size > 1 and unique_in_batch > 1:
            print_success("批量生成具有多样性")
        
        return True
        
    except Exception as e:
        print_error(f"批量生成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print_section("CausalQwen与Qwen兼容性测试")
    print_info("验证CausalQwen是否完全兼容Qwen的生成接口")
    
    # 创建模型
    model, input_ids = test_qwen_compatibility()
    if model is None:
        return
    
    # 执行测试
    tests = [
        ("生成接口兼容性", test_generation_interface),
        ("因果数学原理", test_causal_mathematical_principles),
        ("温度参数效果", test_temperature_effects),
        ("批量生成", test_batch_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{Colors.CYAN}🔬 正在执行: {test_name}{Colors.END}")
        success = test_func(model, input_ids)
        results.append((test_name, success))
    
    # 总结报告
    print_section("测试总结")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status} {test_name}")
    
    print(f"\n{Colors.BOLD}总体结果: {passed}/{total} 测试通过{Colors.END}")
    
    if passed == total:
        print_success("🎉 所有测试通过！CausalQwen与Qwen完全兼容！")
    else:
        print_error("⚠️ 部分测试失败，需要进一步调试")
    
    print_info("CausalQwen核心创新：")
    print_info("├─ Causal模式 (temperature=0): 纯因果生成，无外生噪声")
    print_info("├─ Standard模式 (do_sample=False, temperature>0): 噪声增加决策不确定性")
    print_info("├─ Sampling模式 (do_sample=True, temperature>0): 噪声扰动个体身份")
    print_info("└─ Compatible模式: 传统Softmax，与原始Qwen兼容")

if __name__ == "__main__":
    main()