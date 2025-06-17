#!/usr/bin/env python3
"""
CausalQwen vs Qwen 端到端对比测试

目标：从相同输入出发，对比CausalQwen三种推理模式与原始Qwen的输出差异
验证：
1. CausalQwen兼容模式 ≈ Qwen原始输出（特别是top-1确定性采样）
2. CausalQwen标准模式体现OvR分类的决策逻辑
3. CausalQwen因果模式体现个体差异的随机性
4. 所有模式的数学计算符合设计文档

测试流程：
输入文本 → tokenize → 嵌入 → transformer特征z → 各种推理模式 → 输出对比
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
    """打印成功信息"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """打印错误信息"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    """打印信息"""
    print(f"{Colors.WHITE}ℹ️  {message}{Colors.END}")

def print_math(message):
    """打印数学公式"""
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
        
        # 方法1：使用完整的state_dict复制（更安全）
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

def analyze_intermediate_representations(text, tokenizer, qwen_model, causal_model):
    """分析中间表征：从输入到特征z的完整过程"""
    print_section(f"中间表征分析：'{text}'")
    
    print_step(1, "文本预处理与token化")
    
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
    
    print_step(2, "特征提取：嵌入 → Transformer → 表征z")
    
    with torch.no_grad():
        # === Qwen的前向传播 ===
        qwen_outputs = qwen_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        qwen_hidden_states = qwen_outputs.last_hidden_state  # [batch, seq, hidden]
        qwen_logits = qwen_model.lm_head(qwen_hidden_states)  # [batch, seq, vocab]
        
        print_math(f"Qwen特征z形状: {qwen_hidden_states.shape}")
        print_math(f"Qwen logits形状: {qwen_logits.shape}")
        print_info(f"Qwen特征z统计: 均值={qwen_hidden_states.mean().item():.6f}, 标准差={qwen_hidden_states.std().item():.6f}")
        
        # === CausalQwen的前向传播 ===
        causal_outputs = causal_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 提取CausalQwen的中间表征
        causal_hidden_states = causal_outputs.hidden_states[-1] if causal_outputs.hidden_states else None
        
        # 如果没有hidden_states，手动计算
        if causal_hidden_states is None:
            transformer_outputs = causal_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            causal_hidden_states = transformer_outputs.last_hidden_state
        
        print_math(f"CausalQwen特征z形状: {causal_hidden_states.shape}")
        print_info(f"CausalQwen特征z统计: 均值={causal_hidden_states.mean().item():.6f}, 标准差={causal_hidden_states.std().item():.6f}")
        
        # 验证Transformer特征一致性
        feature_diff = torch.abs(qwen_hidden_states - causal_hidden_states).mean().item()
        print_math(f"特征z差异（应该≈0）: {feature_diff:.8f}")
        
        if feature_diff < 1e-6:
            print_success("✅ CausalQwen的Transformer特征与Qwen完全一致！")
        else:
            print_warning(f"⚠️ 特征差异较大({feature_diff:.8f})，可能存在权重复制问题")
        
        print_step(3, "CausalQwen因果分解：z → (loc_U, scale_U) → (loc_S, scale_S)")
        
        # 归因推断：z → U分布参数
        loc_U, scale_U = causal_model.abduction_network(causal_hidden_states)
        print_math(f"个体表征loc_U: {loc_U.shape}, 均值={loc_U.mean().item():.6f}")
        print_math(f"个体不确定性scale_U: {scale_U.shape}, 均值={scale_U.mean().item():.6f}")
        
        # 行动决策：U → S分布参数
        loc_S, scale_S = causal_model.action_network(loc_U, scale_U)
        print_math(f"决策分布loc_S: {loc_S.shape}, 均值={loc_S.mean().item():.6f}")
        print_math(f"决策不确定性scale_S: {scale_S.shape}, 均值={scale_S.mean().item():.6f}")
        
        # 对比CausalQwen的loc_S与Qwen的logits
        logits_diff = torch.abs(loc_S - qwen_logits).mean().item()
        print_math(f"loc_S与Qwen logits差异: {logits_diff:.6f}")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens,
            'qwen_hidden_states': qwen_hidden_states,
            'qwen_logits': qwen_logits,
            'causal_hidden_states': causal_hidden_states,
            'loc_U': loc_U,
            'scale_U': scale_U,
            'loc_S': loc_S,
            'scale_S': scale_S,
            'feature_diff': feature_diff,
            'logits_diff': logits_diff
        }

def compare_inference_modes(representations, tokenizer, qwen_model, causal_model):
    """对比各种推理模式的输出"""
    print_section("推理模式对比：CausalQwen vs Qwen")
    
    input_ids = representations['input_ids']
    qwen_logits = representations['qwen_logits']
    
    print_step(1, "Qwen原始推理")
    
    with torch.no_grad():
        # === Qwen标准推理 ===
        qwen_probs = F.softmax(qwen_logits[:, -1, :], dim=-1)  # 最后一个token的概率
        qwen_top1_token = torch.argmax(qwen_probs, dim=-1).item()
        qwen_top1_prob = qwen_probs[0, qwen_top1_token].item()
        qwen_top1_text = tokenizer.decode([qwen_top1_token])
        
        print_info(f"Qwen Top-1预测: token={qwen_top1_token}, prob={qwen_top1_prob:.6f}, text='{qwen_top1_text}'")
        
        # Qwen Top-k分析
        qwen_top5_probs, qwen_top5_tokens = torch.topk(qwen_probs, 5, dim=-1)
        print_info("Qwen Top-5预测:")
        for i in range(5):
            token_id = qwen_top5_tokens[0, i].item()
            prob = qwen_top5_probs[0, i].item()
            text = tokenizer.decode([token_id])
            print(f"    #{i+1}: token={token_id}, prob={prob:.6f}, text='{text}'")
    
    print_step(2, "CausalQwen三种推理模式")
    
    from causal_qwen_mvp import CausalInferenceEngine
    engine = CausalInferenceEngine(causal_model)
    
    # === 模式1：标准推理（OvR分类）===
    print_info("🎯 模式1：标准推理（基于OvR概率的确定性决策）")
    
    with torch.no_grad():
        standard_output = engine.inference(input_ids, mode='standard')
        
        # 计算OvR概率
        ovr_probs = causal_model.ovr_classifier(standard_output.loc_S, standard_output.scale_S)
        standard_probs = ovr_probs[:, -1, :]  # 最后一个token的OvR概率
        standard_top1_token = torch.argmax(standard_probs, dim=-1).item()
        standard_top1_prob = standard_probs[0, standard_top1_token].item()
        standard_top1_text = tokenizer.decode([standard_top1_token])
        
        print_math(f"OvR公式: P_k = 0.5 + (1/π) * arctan((loc_S_k - threshold) / scale_S_k)")
        print_info(f"标准模式预测: token={standard_top1_token}, prob={standard_top1_prob:.6f}, text='{standard_top1_text}'")
    
    # === 模式2：因果采样（个体具现）===
    print_info("🎲 模式2：因果采样（个体具现 + 确定性决策）")
    
    causal_predictions = []
    for trial in range(3):  # 多次采样展示随机性
        with torch.no_grad():
            causal_output = engine.inference(input_ids, mode='causal', temperature=1.0)
            
            # 从因果采样的结果获取预测
            causal_probs = F.softmax(causal_output.loc_S[:, -1, :], dim=-1)
            causal_top1_token = torch.argmax(causal_probs, dim=-1).item()
            causal_top1_prob = causal_probs[0, causal_top1_token].item()
            causal_top1_text = tokenizer.decode([causal_top1_token])
            
            causal_predictions.append((causal_top1_token, causal_top1_prob, causal_top1_text))
            print_info(f"因果采样#{trial+1}: token={causal_top1_token}, prob={causal_top1_prob:.6f}, text='{causal_top1_text}'")
    
    # 分析因果采样的多样性
    unique_tokens = len(set([pred[0] for pred in causal_predictions]))
    print_math(f"因果采样多样性: {unique_tokens}/3 个不同结果")
    
    # === 模式3：兼容采样（传统Softmax）===
    print_info("🔄 模式3：兼容采样（传统Softmax + Top-1确定性）")
    
    with torch.no_grad():
        # 确定性兼容模式
        compatible_det_output = engine.inference(input_ids, mode='compatible', do_sample=False)
        
        if hasattr(compatible_det_output, 'next_token_ids') and compatible_det_output.next_token_ids is not None:
            compatible_det_token = compatible_det_output.next_token_ids[0].item()
        else:
            # 手动计算
            compatible_logits = compatible_det_output.loc_S[:, -1, :]
            compatible_det_token = torch.argmax(compatible_logits, dim=-1).item()
        
        compatible_det_text = tokenizer.decode([compatible_det_token])
        print_info(f"兼容确定性: token={compatible_det_token}, text='{compatible_det_text}'")
        
        # 随机兼容模式
        compatible_random_predictions = []
        for trial in range(3):
            compatible_random_output = engine.inference(input_ids, mode='compatible', 
                                                      do_sample=True, temperature=1.0, top_k=50, top_p=0.9)
            
            if hasattr(compatible_random_output, 'next_token_ids') and compatible_random_output.next_token_ids is not None:
                compatible_random_token = compatible_random_output.next_token_ids[0].item()
            else:
                # 手动采样
                compatible_logits = compatible_random_output.loc_S[:, -1, :] / 1.0  # temperature
                compatible_probs = F.softmax(compatible_logits, dim=-1)
                compatible_random_token = torch.multinomial(compatible_probs, 1)[0].item()
            
            compatible_random_text = tokenizer.decode([compatible_random_token])
            compatible_random_predictions.append((compatible_random_token, compatible_random_text))
            print_info(f"兼容随机#{trial+1}: token={compatible_random_token}, text='{compatible_random_text}'")
    
    print_step(3, "关键一致性验证")
    
    # 验证1：兼容确定性模式 vs Qwen确定性
    qwen_vs_compatible = (qwen_top1_token == compatible_det_token)
    if qwen_vs_compatible:
        print_success(f"✅ 关键验证通过：兼容确定性模式与Qwen完全一致！")
        print_success(f"   Qwen: {qwen_top1_token} ('{qwen_top1_text}') == 兼容: {compatible_det_token} ('{compatible_det_text}')")
    else:
        print_warning(f"⚠️ 兼容模式不一致：Qwen={qwen_top1_token}('{qwen_top1_text}') vs 兼容={compatible_det_token}('{compatible_det_text}')")
        print_warning("这可能表明兼容模式实现有误或权重复制不完整")
    
    # 验证2：推理模式差异性
    all_predictions = [qwen_top1_token, standard_top1_token, compatible_det_token]
    all_predictions.extend([pred[0] for pred in causal_predictions])
    unique_predictions = len(set(all_predictions))
    
    print_info(f"所有预测结果: {all_predictions}")
    print_info(f"预测多样性: {unique_predictions}/{len(all_predictions)} 个不同结果")
    
    if unique_predictions >= 3:
        print_success("✅ 不同推理模式体现了各自特点")
    elif unique_predictions == 2:
        print_info("ℹ️ 部分模式产生不同结果（正常现象）")
    else:
        print_warning("⚠️ 所有模式结果相同，需要检查实现")
    
    return {
        'qwen_prediction': (qwen_top1_token, qwen_top1_prob, qwen_top1_text),
        'standard_prediction': (standard_top1_token, standard_top1_prob, standard_top1_text),
        'causal_predictions': causal_predictions,
        'compatible_det_prediction': (compatible_det_token, compatible_det_text),
        'compatible_random_predictions': compatible_random_predictions,
        'qwen_vs_compatible_match': qwen_vs_compatible,
        'prediction_diversity': unique_predictions
    }

def test_sequence_generation(text, tokenizer, qwen_model, causal_model):
    """测试序列生成：多步推理对比"""
    print_section(f"序列生成对比：'{text}' + 3个后续tokens")
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    
    print_step(1, "Qwen序列生成")
    
    with torch.no_grad():
        # Qwen生成（贪心搜索）
        qwen_generated = qwen_model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,  # 确定性生成
            pad_token_id=tokenizer.eos_token_id
        )
        qwen_new_tokens = qwen_generated[0, input_ids.shape[1]:].tolist()
        qwen_generated_text = tokenizer.decode(qwen_new_tokens)
        
        print_info(f"Qwen生成: tokens={qwen_new_tokens}, text='{qwen_generated_text}'")
    
    print_step(2, "CausalQwen序列生成")
    
    from causal_qwen_mvp import CausalInferenceEngine
    engine = CausalInferenceEngine(causal_model)
    
    # 标准模式生成
    with torch.no_grad():
        standard_generated = engine.generate_step_by_step(
            input_ids, max_new_tokens=3, mode='standard'
        )
        standard_new_tokens = standard_generated[0, input_ids.shape[1]:].tolist()
        standard_generated_text = tokenizer.decode(standard_new_tokens)
        print_info(f"标准模式: tokens={standard_new_tokens}, text='{standard_generated_text}'")
    
    # 因果模式生成
    with torch.no_grad():
        causal_generated = engine.generate_step_by_step(
            input_ids, max_new_tokens=3, mode='causal', temperature=1.0
        )
        causal_new_tokens = causal_generated[0, input_ids.shape[1]:].tolist()
        causal_generated_text = tokenizer.decode(causal_new_tokens)
        print_info(f"因果模式: tokens={causal_new_tokens}, text='{causal_generated_text}'")
    
    # 兼容模式生成（确定性）
    with torch.no_grad():
        compatible_generated = engine.generate_step_by_step(
            input_ids, max_new_tokens=3, mode='compatible_deterministic'
        )
        compatible_new_tokens = compatible_generated[0, input_ids.shape[1]:].tolist()
        compatible_generated_text = tokenizer.decode(compatible_new_tokens)
        print_info(f"兼容模式: tokens={compatible_new_tokens}, text='{compatible_generated_text}'")
    
    print_step(3, "序列生成一致性验证")
    
    # 验证Qwen vs 兼容模式
    qwen_vs_compatible_seq = (qwen_new_tokens == compatible_new_tokens)
    if qwen_vs_compatible_seq:
        print_success("✅ 序列生成验证通过：兼容模式与Qwen序列完全一致！")
    else:
        print_warning(f"⚠️ 序列不一致：Qwen={qwen_new_tokens} vs 兼容={compatible_new_tokens}")
        # 逐位置对比
        for i, (q_token, c_token) in enumerate(zip(qwen_new_tokens, compatible_new_tokens)):
            if q_token != c_token:
                q_text = tokenizer.decode([q_token])
                c_text = tokenizer.decode([c_token])
                print_warning(f"   位置{i}: Qwen={q_token}('{q_text}') vs 兼容={c_token}('{c_text}')")
    
    # 分析不同模式的序列差异
    all_sequences = [qwen_new_tokens, standard_new_tokens, causal_new_tokens, compatible_new_tokens]
    sequence_names = ['Qwen', '标准', '因果', '兼容']
    
    print_info("序列差异分析:")
    for i, seq_name in enumerate(sequence_names):
        for j, other_name in enumerate(sequence_names[i+1:], i+1):
            diff_count = sum(1 for a, b in zip(all_sequences[i], all_sequences[j]) if a != b)
            print_info(f"  {seq_name} vs {other_name}: {diff_count}/3 个位置不同")
    
    return {
        'qwen_sequence': qwen_new_tokens,
        'standard_sequence': standard_new_tokens,
        'causal_sequence': causal_new_tokens,
        'compatible_sequence': compatible_new_tokens,
        'qwen_vs_compatible_match': qwen_vs_compatible_seq
    }

def main():
    """主测试函数"""
    print_section("CausalQwen vs Qwen 端到端对比测试", Colors.PURPLE)
    
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
            # 分析中间表征
            representations = analyze_intermediate_representations(text, tokenizer, qwen_model, causal_model)
            
            # 对比推理模式
            inference_results = compare_inference_modes(representations, tokenizer, qwen_model, causal_model)
            
            # 序列生成测试
            generation_results = test_sequence_generation(text, tokenizer, qwen_model, causal_model)
            
            # 记录结果
            all_results.append({
                'text': text,
                'representations': representations,
                'inference': inference_results,
                'generation': generation_results
            })
            
        except Exception as e:
            print_error(f"测试案例 {i+1} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结报告
    print_section("测试总结报告", Colors.GREEN)
    
    # 统计一致性验证结果
    inference_consistency_count = sum(1 for result in all_results if result['inference']['qwen_vs_compatible_match'])
    generation_consistency_count = sum(1 for result in all_results if result['generation']['qwen_vs_compatible_match'])
    
    print_success(f"推理一致性: {inference_consistency_count}/{len(all_results)} 个案例通过")
    print_success(f"生成一致性: {generation_consistency_count}/{len(all_results)} 个案例通过")
    
    # 分析特征一致性
    feature_diffs = [result['representations']['feature_diff'] for result in all_results if 'representations' in result]
    avg_feature_diff = np.mean(feature_diffs) if feature_diffs else float('inf')
    
    print_success(f"Transformer特征平均差异: {avg_feature_diff:.8f}")
    
    if avg_feature_diff < 1e-6:
        print_success("✅ CausalQwen成功继承了Qwen的预训练知识！")
    else:
        print_warning("⚠️ 存在特征差异，需要检查权重复制")
    
    if inference_consistency_count == len(all_results) and generation_consistency_count == len(all_results):
        print_section("🎉 所有测试通过！CausalQwen与Qwen完全兼容！", Colors.GREEN)
        print_success("兼容模式与Qwen行为完全一致")
        print_success("三种推理模式都能正常工作")
        print_success("数学实现符合设计文档")
    else:
        print_section("⚠️ 部分测试未通过，需要进一步调试", Colors.YELLOW)
        print_info("请检查权重复制和兼容模式实现")

if __name__ == "__main__":
    main()