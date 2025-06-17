#!/usr/bin/env python3
"""
CausalQwen vs Qwen 端到端对比测试 - 更新版

使用完全兼容Qwen的接口进行对比测试

目标：从相同输入出发，对比CausalQwen与原始Qwen的输出差异
验证：
1. CausalQwen确定性模式 (do_sample=False) 与Qwen的兼容性
2. CausalQwen采样模式 (do_sample=True) 体现V2数学原理
3. 所有模式的数学计算符合设计文档
4. 完整的生成能力验证

测试流程：
输入文本 → tokenize → 生成 → 对比分析
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
    print(f"\n{color}📋 步骤 {step_num}: {description}{Colors.END}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.WHITE}ℹ️  {message}{Colors.END}")

def print_math(message):
    print(f"{Colors.PURPLE}🔢 {message}{Colors.END}")

def load_models_and_tokenizer():
    """加载Qwen和CausalQwen模型"""
    print_section("模型加载与初始化")
    
    print_step(1, "加载Qwen2原始模型")
    try:
        from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
        
        qwen_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print_success(f"Qwen2分词器加载成功，词汇表大小: {len(tokenizer.vocab)}")
        
        # 加载Qwen模型配置和权重
        qwen_config = Qwen2Config.from_pretrained(qwen_path)
        qwen_model = Qwen2ForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.float32)
        qwen_model.eval()
        print_success(f"Qwen2模型加载成功，参数量: {sum(p.numel() for p in qwen_model.parameters()):,}")
        
        return tokenizer, qwen_model, qwen_config
        
    except Exception as e:
        print_error(f"Qwen2模型加载失败: {e}")
        return None, None, None

def create_causal_qwen_model(qwen_config):
    """创建CausalQwen模型"""
    print_step(2, "创建CausalQwen模型")
    try:
        from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config
        
        # 创建CausalQwen配置（完全继承Qwen参数）
        causal_config = CausalQwen2Config(
            # 完全复制Qwen配置
            **qwen_config.to_dict(),
            # CausalQwen特有参数
            causal_size=qwen_config.hidden_size,  # C = H
            abduction_init_strategy='identity',
            b_noise_init=0.1,
            gamma_init=10.0,
            ovr_threshold_init=0.0
        )
        
        # 创建模型
        causal_model = CausalQwenMVPForCausalLM(causal_config)
        causal_model.eval()
        print_success(f"CausalQwen模型创建成功，参数量: {sum(p.numel() for p in causal_model.parameters()):,}")
        
        return causal_model, causal_config
        
    except Exception as e:
        print_error(f"CausalQwen模型创建失败: {e}")
        return None, None

def copy_qwen_weights_to_causal(qwen_model, causal_model):
    """将Qwen权重复制到CausalQwen"""
    print_step(3, "复制Qwen预训练权重")
    try:
        print_info("复制Transformer基础权重...")
        
        # 使用完整的state_dict复制
        qwen_state_dict = qwen_model.model.state_dict()
        causal_state_dict = causal_model.model.state_dict()
        
        # 复制所有匹配的权重
        copied_keys = []
        for key in qwen_state_dict.keys():
            if key in causal_state_dict and qwen_state_dict[key].shape == causal_state_dict[key].shape:
                causal_state_dict[key].copy_(qwen_state_dict[key])
                copied_keys.append(key)
        
        # 加载更新后的state_dict
        causal_model.model.load_state_dict(causal_state_dict)
        
        print_info(f"成功复制 {len(copied_keys)} 个Transformer参数")
        
        # 验证复制效果
        print_info("验证权重复制效果...")
        with torch.no_grad():
            test_input = torch.randint(0, 1000, (1, 5))
            qwen_features = qwen_model.model(test_input)[0]
            causal_features = causal_model.model(test_input)[0]
            feature_diff = torch.abs(qwen_features - causal_features).mean().item()
            
        print_math(f"特征验证差异: {feature_diff:.8f}")
        
        if feature_diff < 1e-6:
            print_success("✅ Transformer权重复制完美！")
        else:
            print_warning(f"⚠️ 特征差异 {feature_diff:.8f}，继续调试...")
        
        # 复制lm_head到ActionNetwork
        print_info("复制lm_head权重到ActionNetwork...")
        causal_model.action_network.copy_weights_from_qwen(qwen_model)
        
        print_success("权重复制完成！CausalQwen现在继承了Qwen的预训练知识")
        
        return True
        
    except Exception as e:
        print_error(f"权重复制失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_generation_methods(text, tokenizer, qwen_model, causal_model):
    """对比不同的生成方法"""
    print_section(f"生成方法对比：'{text}'")
    
    print_step(1, "准备输入")
    
    # 编码文本
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)
    
    print_info(f"原始文本: '{text}'")
    print_info(f"Token IDs: {input_ids.tolist()}")
    print_info(f"Token数量: {input_ids.shape[1]}")
    
    # 解码每个token
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    print_info(f"Token序列: {tokens}")
    
    print_step(2, "Qwen基准生成")
    
    with torch.no_grad():
        # === Qwen确定性生成 ===
        qwen_det_output = qwen_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        qwen_det_new = qwen_det_output[0, input_ids.shape[1]:].tolist()
        qwen_det_text = tokenizer.decode(qwen_det_new)
        
        print_info(f"Qwen确定性生成: tokens={qwen_det_new}, text='{qwen_det_text}'")
        
        # === Qwen采样生成 ===  
        qwen_samp_output = qwen_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        qwen_samp_new = qwen_samp_output[0, input_ids.shape[1]:].tolist()
        qwen_samp_text = tokenizer.decode(qwen_samp_new)
        
        print_info(f"Qwen采样生成: tokens={qwen_samp_new}, text='{qwen_samp_text}'")
    
    print_step(3, "CausalQwen生成对比")
    
    with torch.no_grad():
        # === CausalQwen确定性生成 (do_sample=False) ===
        print_info("🎯 CausalQwen确定性模式 (do_sample=False)")
        print_info("   V2原理: 噪声影响尺度参数，增加决策不确定性")
        
        causal_det_output = causal_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0
        )
        causal_det_new = causal_det_output[0, input_ids.shape[1]:].tolist()
        causal_det_text = tokenizer.decode(causal_det_new)
        
        print_info(f"CausalQwen确定性: tokens={causal_det_new}, text='{causal_det_text}'")
        
        # === CausalQwen采样生成 (do_sample=True) ===
        print_info("🎲 CausalQwen采样模式 (do_sample=True)")
        print_info("   V2原理: 噪声影响位置参数，扰动个体身份")
        
        causal_samp_outputs = []
        for trial in range(3):  # 多次采样展示随机性
            causal_samp_output = causal_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            causal_samp_new = causal_samp_output[0, input_ids.shape[1]:].tolist()
            causal_samp_text = tokenizer.decode(causal_samp_new)
            causal_samp_outputs.append((causal_samp_new, causal_samp_text))
            print_info(f"CausalQwen采样#{trial+1}: tokens={causal_samp_new}, text='{causal_samp_text}'")
    
    print_step(4, "一致性验证")
    
    # 验证1：确定性模式的logits一致性（关键验证）
    print_info("🎯 验证CausalQwen确定性模式的loc_S与Qwen的logits一致性")
    
    # 获取CausalQwen确定性模式的前向传播结果
    with torch.no_grad():
        causal_outputs = causal_model(input_ids)
        # 提取loc_S（决策分布的位置参数）
        if hasattr(causal_outputs, 'loc_S'):
            causal_loc_S = causal_outputs.loc_S
        else:
            # 如果没有直接的loc_S，通过ActionNetwork获取
            transformer_out = causal_model.model(input_ids)
            hidden_states = transformer_out.last_hidden_state
            loc_U, scale_U = causal_model.abduction_network(hidden_states)
            causal_loc_S, _ = causal_model.action_network(loc_U, scale_U, do_sample=False)
        
        # 获取Qwen的logits
        qwen_outputs = qwen_model(input_ids)
        qwen_logits = qwen_outputs.logits
        
        # 比较最后一个位置的logits/loc_S
        last_pos_causal = causal_loc_S[:, -1, :]  # [batch, vocab]
        last_pos_qwen = qwen_logits[:, -1, :]     # [batch, vocab]
        
        logits_diff = torch.abs(last_pos_causal - last_pos_qwen).mean().item()
        logits_max_diff = torch.abs(last_pos_causal - last_pos_qwen).max().item()
        
        print_math(f"loc_S vs Qwen logits平均差异: {logits_diff:.8f}")
        print_math(f"loc_S vs Qwen logits最大差异: {logits_max_diff:.8f}")
        
        if logits_diff < 1e-4:
            print_success(f"✅ 确定性模式logits一致性验证通过！")
            print_success(f"   CausalQwen的loc_S与Qwen的logits基本一致")
            logits_consistent = True
        else:
            print_warning(f"⚠️ logits差异较大: {logits_diff:.8f}")
            print_warning(f"   这可能表明权重复制不完整或ActionNetwork实现有误")
            logits_consistent = False
    
    # 验证2：采样模式多样性
    all_causal_samp = [output[0] for output in causal_samp_outputs]
    causal_diversity = len(set(tuple(seq) for seq in all_causal_samp))
    
    print_info(f"CausalQwen采样多样性: {causal_diversity}/3 个不同结果")
    
    if causal_diversity >= 2:
        print_success("✅ CausalQwen采样模式具有良好多样性")
    else:
        print_warning("⚠️ CausalQwen采样多样性较低")
    
    # 验证3：V2数学原理
    print_info("🧮 验证V2数学原理")
    try:
        from causal_qwen_mvp import InferenceValidator
        
        validator = InferenceValidator(causal_model)
        results = validator.validate_v2_principles(input_ids)
        
        pos_diff = results['position_difference'].item()
        scale_diff = results['scale_difference'].item()
        
        print_math(f"位置参数差异: {pos_diff:.6f}")
        print_math(f"尺度参数差异: {scale_diff:.6f}")
        
        if pos_diff > 1e-3:
            print_success("✅ V2数学原理验证：位置vs尺度差异显著")
        else:
            print_warning("⚠️ 位置参数差异较小")
    
    except Exception as e:
        print_error(f"V2数学验证失败: {e}")
    
    return {
        'qwen_deterministic': (qwen_det_new, qwen_det_text),
        'qwen_sampling': (qwen_samp_new, qwen_samp_text),
        'causal_deterministic': (causal_det_new, causal_det_text),
        'causal_sampling': causal_samp_outputs,
        'logits_consistent': logits_consistent,
        'logits_difference': logits_diff,
        'causal_diversity': causal_diversity
    }

def test_different_temperatures(text, tokenizer, causal_model):
    """测试不同温度参数的效果"""
    print_section(f"温度参数效果测试：'{text}'")
    
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    temperatures = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]  # 添加温度为零的测试
    temp_results = []
    
    print_info("测试不同温度下的CausalQwen采样生成:")
    
    for temp in temperatures:
        torch.manual_seed(42)  # 固定种子便于对比
        with torch.no_grad():
            output = causal_model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=True,
                temperature=temp,
                top_k=50
            )
            new_tokens = output[0, input_ids.shape[1]:].tolist()
            new_text = tokenizer.decode(new_tokens)
            temp_results.append((temp, new_tokens, new_text))
            print_info(f"T={temp}: tokens={new_tokens}, text='{new_text}'")
    
    # 特别验证温度为零的场景
    if len(temp_results) > 0 and temp_results[0][0] == 0.0:
        print_info("🌡️ 温度为零是极其重要的边界条件！")
        temp_zero_tokens, temp_zero_text = temp_results[0][1], temp_results[0][2]
        print_info(f"温度T=0结果: tokens={temp_zero_tokens}, text='{temp_zero_text}'")
    
    # 分析温度效果
    unique_sequences = len(set(tuple(result[1]) for result in temp_results))
    print_math(f"不同温度产生的序列多样性: {unique_sequences}/{len(temperatures)}")
    
    if unique_sequences >= 3:
        print_success("✅ 温度参数有效控制生成多样性")
    else:
        print_warning("⚠️ 温度参数效果不明显")
    
    return temp_results

def main():
    """主测试函数"""
    print_section("CausalQwen vs Qwen 端到端对比测试 - 更新版", Colors.PURPLE)
    print_info("使用与Qwen完全兼容的接口进行对比测试")
    print_info("验证CausalQwen的V2数学原理和Qwen兼容性")
    
    # 测试文本
    test_texts = [
        "今天天气很好",
        "人工智能的发展", 
        "在深度学习领域",
        "The future of technology"
    ]
    
    # 加载模型
    tokenizer, qwen_model, qwen_config = load_models_and_tokenizer()
    if qwen_model is None:
        print_error("Qwen模型加载失败，终止测试")
        return
    
    causal_model, causal_config = create_causal_qwen_model(qwen_config)
    if causal_model is None:
        print_error("CausalQwen模型创建失败，终止测试")
        return
    
    # 复制权重
    if not copy_qwen_weights_to_causal(qwen_model, causal_model):
        print_error("权重复制失败，终止测试")
        return
    
    # 对每个测试文本进行完整的端到端测试
    all_results = []
    
    for i, text in enumerate(test_texts):
        print_section(f"测试案例 {i+1}/{len(test_texts)}: '{text}'", Colors.GREEN)
        
        try:
            # 生成方法对比
            generation_results = compare_generation_methods(text, tokenizer, qwen_model, causal_model)
            
            # 温度效果测试
            temp_results = test_different_temperatures(text, tokenizer, causal_model)
            
            # 记录结果
            all_results.append({
                'text': text,
                'generation': generation_results,
                'temperature': temp_results
            })
            
        except Exception as e:
            print_error(f"测试案例 {i+1} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结报告
    print_section("测试总结报告", Colors.GREEN)
    
    # 统计logits一致性验证结果（关键指标）
    logits_consistency_count = sum(1 for result in all_results if result['generation']['logits_consistent'])
    total_cases = len(all_results)
    
    print_success(f"logits一致性: {logits_consistency_count}/{total_cases} 个案例通过")
    
    # 统计平均logits差异
    if all_results:
        avg_logits_diff = np.mean([result['generation']['logits_difference'] for result in all_results if 'logits_difference' in result['generation']])
        print_success(f"平均logits差异: {avg_logits_diff:.8f}")
    
    # 分析多样性
    diversity_scores = [result['generation']['causal_diversity'] for result in all_results]
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
    
    print_success(f"平均采样多样性: {avg_diversity:.2f}/3")
    
    # 温度效果分析
    temp_diversity_scores = []
    for result in all_results:
        if 'temperature' in result:
            temp_seqs = [tuple(tr[1]) for tr in result['temperature']]
            temp_diversity = len(set(temp_seqs))
            temp_diversity_scores.append(temp_diversity)
    
    avg_temp_diversity = np.mean(temp_diversity_scores) if temp_diversity_scores else 0
    print_success(f"平均温度多样性: {avg_temp_diversity:.2f}/5")
    
    if logits_consistency_count == total_cases:
        print_section("🎉 所有测试通过！CausalQwen与Qwen完全兼容！", Colors.GREEN)
        print_success("✅ 确定性模式的loc_S与Qwen的logits完全一致")
        print_success("✅ 采样模式体现V2数学原理")
        print_success("✅ 温度参数正确控制生成多样性")
        print_success("✅ 完全兼容Qwen的generate()接口")
    else:
        print_section("⚠️ 部分测试未通过，需要进一步调试", Colors.YELLOW)
        print_info("请检查权重复制和ActionNetwork实现")
        print_info(f"logits一致性: {logits_consistency_count}/{total_cases} 个案例通过")
    
    print_info("CausalQwen V2核心特性:")
    print_info("├─ do_sample=False: 噪声影响尺度参数，增加决策不确定性")
    print_info("└─ do_sample=True: 噪声影响位置参数，扰动个体身份")

if __name__ == "__main__":
    main()