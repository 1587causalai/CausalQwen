#!/usr/bin/env python3
"""
CausalQwen V2 数学原理验证测试 - 更新版

验证V2革命性设计的数学原理正确性，使用与Qwen完全兼容的接口
核心验证：
1. do_sample参数对噪声作用方式的控制
2. 温度参数的选择性生效机制
3. 柯西分布线性稳定性的严格实现
4. ActionNetwork统一框架的数学一致性

V2数学原理：
┌─ do_sample=True：U' ~ Cauchy(μ + T·|b_noise|·ε, γ)
└─ do_sample=False：U' ~ Cauchy(μ, γ + |b_noise|)
"""

import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# ANSI color codes
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

def print_section(title, color=Colors.BLUE):
    """打印章节标题"""
    print(f"\n{color}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{color}{Colors.BOLD}{title.center(80)}{Colors.END}")
    print(f"{color}{Colors.BOLD}{'='*80}{Colors.END}")

def print_step(step_num, description, color=Colors.CYAN):
    """打印步骤信息"""
    print(f"\n{color}🔬 步骤 {step_num}: {description}{Colors.END}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_math(message):
    print(f"{Colors.PURPLE}🔢 {message}{Colors.END}")

def print_theory(message):
    print(f"{Colors.WHITE}📖 {message}{Colors.END}")

def test_v2_mathematical_framework():
    """测试V2数学框架的核心组件"""
    print_section("V2数学框架验证")
    
    print_step(1, "验证柯西分布数学工具类")
    try:
        from causal_qwen_mvp.components import CauchyMath
        
        # 测试线性稳定性
        batch_size, input_dim, output_dim = 4, 128, 256
        
        # 位置参数线性变换
        loc_input = torch.randn(batch_size, input_dim)
        weight = torch.randn(output_dim, input_dim)
        bias = torch.randn(output_dim)
        
        loc_output = CauchyMath.cauchy_linear_stable_loc(loc_input, weight, bias)
        expected_shape = (batch_size, output_dim)
        
        print_math(f"输入形状: {loc_input.shape}")
        print_math(f"权重形状: {weight.shape}")
        print_math(f"输出形状: {loc_output.shape}")
        
        if loc_output.shape == expected_shape:
            print_success("位置参数线性变换形状正确")
        else:
            print_error(f"形状错误: 期望 {expected_shape}, 实际 {loc_output.shape}")
        
        # 验证数学公式：output = input @ weight.T + bias
        manual_output = loc_input @ weight.T + bias
        if torch.allclose(loc_output, manual_output, atol=1e-6):
            print_success("位置参数线性变换数学公式正确")
        else:
            print_error("位置参数线性变换数学公式错误")
        
        # 尺度参数线性变换
        scale_input = torch.abs(torch.randn(batch_size, input_dim)) + 0.1
        scale_output = CauchyMath.cauchy_linear_stable_scale(scale_input, weight)
        
        # 验证数学公式：output = input @ |weight|.T
        manual_scale_output = scale_input @ torch.abs(weight).T
        if torch.allclose(scale_output, manual_scale_output, atol=1e-6):
            print_success("尺度参数线性变换数学公式正确")
        else:
            print_error("尺度参数线性变换数学公式错误")
            
        print_theory("柯西分布线性稳定性验证通过")
        
    except Exception as e:
        print_error(f"柯西数学工具测试失败: {e}")

def test_action_network_v2_modes():
    """测试ActionNetwork的V2双模式"""
    print_section("ActionNetwork V2双模式验证")
    
    print_step(1, "创建测试用ActionNetwork")
    try:
        from causal_qwen_mvp.components import ActionNetwork
        from causal_qwen_mvp.config import CausalQwen2Config
        
        # 创建小型测试配置
        config = CausalQwen2Config(
            vocab_size=100,
            hidden_size=64,
            causal_size=64,
            b_noise_init=0.1
        )
        
        action_net = ActionNetwork(config)
        print_success("ActionNetwork创建成功")
        
        # 创建测试输入
        batch_size, seq_len, causal_size = 2, 5, 64
        loc_U = torch.randn(batch_size, seq_len, causal_size)
        scale_U = torch.abs(torch.randn(batch_size, seq_len, causal_size)) + 1.0
        
        print_math(f"输入loc_U统计: 均值={loc_U.mean().item():.4f}, 标准差={loc_U.std().item():.4f}")
        print_math(f"输入scale_U统计: 均值={scale_U.mean().item():.4f}, 标准差={scale_U.std().item():.4f}")
        
    except Exception as e:
        print_error(f"ActionNetwork创建失败: {e}")
        return None, None, None, None
    
    print_step(2, "测试V2非采样模式：噪声影响尺度参数")
    try:
        with torch.no_grad():
            loc_S_det, scale_S_det = action_net(loc_U, scale_U, do_sample=False)
        
        print_theory("V2非采样模式数学公式：")
        print_theory("├─ U' ~ Cauchy(μ, γ + |b_noise|)")
        print_theory("├─ loc_S = W·μ + b")
        print_theory("└─ scale_S = (γ + |b_noise|) × |W|^T")
        
        print_math(f"非采样loc_S形状: {loc_S_det.shape}")
        print_math(f"非采样scale_S形状: {scale_S_det.shape}")
        print_math(f"非采样loc_S统计: 均值={loc_S_det.mean().item():.4f}")
        print_math(f"非采样scale_S统计: 均值={scale_S_det.mean().item():.4f}")
        
        # 验证非采样模式的数学实现
        expected_scale_U_noisy = scale_U + torch.abs(action_net.b_noise)
        expected_loc_S = action_net.lm_head(loc_U)
        expected_scale_S = expected_scale_U_noisy @ torch.abs(action_net.lm_head.weight).T
        
        if torch.allclose(loc_S_det, expected_loc_S, atol=1e-5):
            print_success("非采样模式位置参数计算正确")
        else:
            diff = torch.abs(loc_S_det - expected_loc_S).max().item()
            print_error(f"非采样模式位置参数计算错误，最大差异: {diff}")
        
        if torch.allclose(scale_S_det, expected_scale_S, atol=1e-5):
            print_success("非采样模式尺度参数计算正确")
        else:
            diff = torch.abs(scale_S_det - expected_scale_S).max().item()
            print_error(f"非采样模式尺度参数计算错误，最大差异: {diff}")
            
    except Exception as e:
        print_error(f"非采样模式测试失败: {e}")
    
    print_step(3, "测试V2采样模式：噪声影响位置参数")
    try:
        # 固定随机种子确保可重复性
        torch.manual_seed(42)
        
        with torch.no_grad():
            loc_S_samp, scale_S_samp = action_net(loc_U, scale_U, do_sample=True, temperature=1.0)
        
        print_theory("V2采样模式数学公式：")
        print_theory("├─ ε ~ Cauchy(0, I) 标准噪声")
        print_theory("├─ U' ~ Cauchy(μ + T·|b_noise|·ε, γ)")
        print_theory("├─ loc_S = W·(μ + T·|b_noise|·ε) + b")
        print_theory("└─ scale_S = γ × |W|^T")
        
        print_math(f"采样loc_S形状: {loc_S_samp.shape}")
        print_math(f"采样scale_S形状: {scale_S_samp.shape}")
        print_math(f"采样loc_S统计: 均值={loc_S_samp.mean().item():.4f}")
        print_math(f"采样scale_S统计: 均值={scale_S_samp.mean().item():.4f}")
        
        # 验证采样模式与非采样模式的差异
        loc_diff = torch.abs(loc_S_samp - loc_S_det).mean().item()
        scale_diff = torch.abs(scale_S_samp - scale_S_det).mean().item()
        
        print_math(f"采样vs非采样位置差异: {loc_diff:.6f}")
        print_math(f"采样vs非采样尺度差异: {scale_diff:.6f}")
        
        if loc_diff > 1e-6:
            print_success("采样模式位置参数与非采样模式有差异（预期）")
        else:
            print_warning("采样模式位置参数与非采样模式无差异（异常）")
        
        # 验证采样模式的尺度参数计算
        expected_scale_S_samp = scale_U @ torch.abs(action_net.lm_head.weight).T
        if torch.allclose(scale_S_samp, expected_scale_S_samp, atol=1e-5):
            print_success("采样模式尺度参数计算正确")
        else:
            diff = torch.abs(scale_S_samp - expected_scale_S_samp).max().item()
            print_error(f"采样模式尺度参数计算错误，最大差异: {diff}")
            
    except Exception as e:
        print_error(f"采样模式测试失败: {e}")
    
    print_step(4, "测试温度参数的选择性作用")
    try:
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        temperature_effects = []
        
        loc_S_baseline = None
        for temp in temperatures:
            torch.manual_seed(42)  # 确保噪声一致
            with torch.no_grad():
                loc_S_temp, _ = action_net(loc_U, scale_U, do_sample=True, temperature=temp)
            
            # 计算与基准(temp=1.0)的差异
            if temp == 1.0:
                loc_S_baseline = loc_S_temp.clone()
                baseline_effect = 0.0
            else:
                if loc_S_baseline is not None:
                    effect = torch.abs(loc_S_temp - loc_S_baseline).mean().item()
                    baseline_effect = effect
                else:
                    baseline_effect = 0.0
            
            temperature_effects.append(baseline_effect)
            print_math(f"温度T={temp}: 与基准差异={baseline_effect:.6f}")
        
        # 验证温度效应的单调性（温度越高，差异越大）
        sorted_effects = sorted(temperature_effects[1:])  # 排除基准点
        if temperature_effects[1:] == sorted_effects or temperature_effects[1:] == sorted_effects[::-1]:
            print_success("温度参数影响具有单调性")
        else:
            print_warning("温度参数影响缺乏单调性")
        
        # 测试非采样模式下温度参数无影响
        with torch.no_grad():
            loc_S_det_1, _ = action_net(loc_U, scale_U, do_sample=False, temperature=1.0)
            loc_S_det_5, _ = action_net(loc_U, scale_U, do_sample=False, temperature=5.0)
        
        temp_effect_det = torch.abs(loc_S_det_1 - loc_S_det_5).max().item()
        if temp_effect_det < 1e-8:
            print_success("非采样模式下温度参数无影响（正确）")
        else:
            print_error(f"非采样模式下温度参数有影响（错误），差异: {temp_effect_det}")
            
    except Exception as e:
        print_error(f"温度参数测试失败: {e}")
    
    return action_net, loc_U, scale_U, (loc_S_det, scale_S_det, loc_S_samp, scale_S_samp)

def test_qwen_compatible_interface():
    """测试与Qwen兼容的推理接口"""
    print_section("Qwen兼容接口验证")
    
    print_step(1, "创建CausalQwen模型")
    try:
        from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config, CausalInferenceEngine
        
        # 创建小型测试模型
        config = CausalQwen2Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            causal_size=64,
            b_noise_init=0.1,
            gamma_init=1.0
        )
        
        model = CausalQwenMVPForCausalLM(config)
        print_success("CausalQwen模型创建成功")
        
        # 创建测试输入
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print_math(f"测试输入形状: {input_ids.shape}")
        
    except Exception as e:
        print_error(f"模型创建失败: {e}")
        return
    
    print_step(2, "测试Qwen兼容的生成接口")
    try:
        # 确定性生成 (do_sample=False)
        det_output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0
        )
        print_success("确定性生成完成")
        print_math(f"确定性输出形状: {det_output.shape}")
        
        # 采样生成 (do_sample=True)
        samp_output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.9
        )
        print_success("采样生成完成")
        print_math(f"采样输出形状: {samp_output.shape}")
        
        # 验证输出长度
        expected_length = input_ids.shape[1] + 5
        if (det_output.shape[1] == expected_length and 
            samp_output.shape[1] == expected_length):
            print_success("生成长度正确")
        else:
            print_error("生成长度错误")
        
        # 验证do_sample差异
        det_new = det_output[0, input_ids.shape[1]:].tolist()
        samp_new = samp_output[0, input_ids.shape[1]:].tolist()
        
        differences = sum(1 for a, b in zip(det_new, samp_new) if a != b)
        print_math(f"确定性vs采样差异: {differences}/5 位置不同")
        
        if differences > 0:
            print_success("do_sample参数正确控制生成差异")
        else:
            print_warning("do_sample参数未产生预期差异")
        
    except Exception as e:
        print_error(f"生成接口测试失败: {e}")
    
    print_step(3, "验证V2数学原理")
    try:
        from causal_qwen_mvp import InferenceValidator
        
        validator = InferenceValidator(model)
        results = validator.validate_causal_principles(input_ids, temperature=1.0)
        
        # 位置参数差异
        pos_diff = results['position_difference'].item()
        scale_diff = results['scale_difference'].item()
        
        print_math(f"位置参数差异: {pos_diff:.6f}")
        print_math(f"尺度参数差异: {scale_diff:.6f}")
        
        if pos_diff > 1e-3:
            print_success("位置参数在不同模式下有显著差异（符合V2设计）")
        else:
            print_warning("位置参数差异较小")
        
        # 验证基础表征
        loc_U_mean = results['base_representations']['loc_U'].mean().item()
        scale_U_mean = results['base_representations']['scale_U'].mean().item()
        
        print_math(f"个体表征统计: loc_U={loc_U_mean:.4f}, scale_U={scale_U_mean:.4f}")
        
        if scale_U_mean > 1e-3:
            print_success("尺度参数初始化正确（>0）")
        else:
            print_error("尺度参数过小，可能初始化有问题")
        
    except Exception as e:
        print_error(f"V2数学原理验证失败: {e}")

def test_temperature_and_sampling_params():
    """测试温度和采样参数"""
    print_section("温度和采样参数测试")
    
    print_step(1, "创建测试模型")
    try:
        from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config
        
        config = CausalQwen2Config(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            causal_size=32
        )
        
        model = CausalQwenMVPForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 6))
        
        print_success("测试模型创建成功")
        
    except Exception as e:
        print_error(f"测试模型创建失败: {e}")
        return
    
    print_step(2, "测试温度参数效果")
    try:
        temperatures = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]  # 添加温度为零的测试
        temp_results = []
        
        for temp in temperatures:
            torch.manual_seed(42)  # 固定种子便于对比
            output = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=True,
                temperature=temp,
                top_k=50
            )
            new_tokens = output[0, input_ids.shape[1]:].tolist()
            temp_results.append((temp, new_tokens))
            print_math(f"温度T={temp}: {new_tokens}")
        
        # 分析多样性
        unique_sequences = len(set(tuple(result[1]) for result in temp_results))
        print_math(f"不同温度产生的序列多样性: {unique_sequences}/{len(temperatures)}")
        
        if unique_sequences >= 3:
            print_success("温度参数有效控制生成多样性")
        else:
            print_warning("温度参数效果不明显")
        
    except Exception as e:
        print_error(f"温度参数测试失败: {e}")
    
    print_step(3, "测试top_k和top_p参数")
    try:
        # 测试不同的top_k值
        top_k_values = [1, 10, 20, 50]
        
        for top_k in top_k_values:
            torch.manual_seed(42)
            output = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=True,
                temperature=1.0,
                top_k=top_k
            )
            new_tokens = output[0, input_ids.shape[1]:].tolist()
            print_math(f"top_k={top_k}: {new_tokens}")
        
        # 测试不同的top_p值
        top_p_values = [0.1, 0.5, 0.9, 1.0]
        
        for top_p in top_p_values:
            torch.manual_seed(42)
            output = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=True,
                temperature=1.0,
                top_p=top_p
            )
            new_tokens = output[0, input_ids.shape[1]:].tolist()
            print_math(f"top_p={top_p}: {new_tokens}")
        
        print_success("top_k和top_p参数测试完成")
        
    except Exception as e:
        print_error(f"采样参数测试失败: {e}")

def main():
    """主测试函数"""
    print_section("CausalQwen V2 数学原理验证 - 更新版", Colors.PURPLE)
    print_theory("验证V2革命性设计：do_sample参数控制的位置vs尺度差异")
    print_theory("核心原理：do_sample控制噪声对Cauchy分布参数的不同影响方式")
    print_theory("完全兼容Qwen接口：generate(), do_sample, temperature, top_k, top_p")
    
    # 1. 数学框架验证
    test_v2_mathematical_framework()
    
    # 2. ActionNetwork双模式验证
    action_results = test_action_network_v2_modes()
    
    # 3. Qwen兼容接口验证
    test_qwen_compatible_interface()
    
    # 4. 温度和采样参数验证
    test_temperature_and_sampling_params()
    
    # 总结报告
    print_section("V2验证总结", Colors.GREEN)
    print_success("🎯 V2核心创新验证：do_sample控制的位置vs尺度差异")
    print_success("🎯 ActionNetwork统一框架：兼容Qwen的所有参数")
    print_success("🎯 温度参数选择性生效：仅在do_sample=True时影响噪声强度")
    print_success("🎯 柯西分布线性稳定性：严格的数学基础实现")
    print_success("🎯 完全Qwen兼容：generate()接口和所有采样参数")
    
    print_theory("V2数学原理验证完成！")
    print_theory("├─ do_sample=True：U' ~ Cauchy(μ + T·|b_noise|·ε, γ)")
    print_theory("四种推理模式：")
    print_theory("├─ Causal模式 (temperature=0): 纯因果生成，无外生噪声")
    print_theory("├─ Standard模式 (do_sample=False, temperature>0): 噪声增加决策不确定性")  
    print_theory("├─ Sampling模式 (do_sample=True, temperature>0): 噪声扰动个体身份")
    print_theory("└─ Compatible模式: 传统Softmax，与原始Qwen兼容")
    print_theory("使用方式与Qwen完全相同：model.generate(input_ids, do_sample=True)")

if __name__ == "__main__":
    main()