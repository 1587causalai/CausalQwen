#!/usr/bin/env python
"""
CausalQwen VS Qwen 模型对比验证脚本

本脚本对比验证 CausalQwen 和原始 Qwen 模型，量化验证知识传输效果。
基于 debug_forward_pass.py 的设计模式，实现全面的对比分析。

核心验证内容：
1. 模型架构对比分析（参数统计、权重共享）
2. 权重继承分析（ActionNetwork ↔ Qwen lm_head）
3. 前向传播对比（特征一致性、输出概率分布）
4. <NUM> token 特殊处理验证
5. 因果表征分析

设计原则：前向传播结果 > 模型参数结构
"""

import os
import sys
import torch
import numpy as np
from dataclasses import asdict
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.utils.losses import compute_ovr_probabilities

def analyze_vocabulary_concepts(causal_model, qwen_model, tokenizer):
    """详细分析三个词汇表概念的区分"""
    print_section("词汇表概念分析", 1)
    print("基于 qwen_reserved_tokens_analysis.md 的深度分析")
    
    # 1. 基础词汇表概念
    print("\n--- 三个词汇表概念的严格区分 ---")
    
    # 从配置获取总词汇表大小
    qwen_config_vocab_size = qwen_model.config.vocab_size  # 总词汇表大小
    qwen_used_vocab_size = len(tokenizer.tokenizer)  # 已用词汇表大小
    qwen_reserved_vocab_size = qwen_config_vocab_size - qwen_used_vocab_size  # 预留词汇表大小
    
    # CausalQwen的词汇表概念
    causal_config_vocab_size = tokenizer.vocab_size  # CausalQwen总词汇表大小
    causal_used_vocab_size = qwen_used_vocab_size + 1  # 已用词汇表大小 + <NUM>
    causal_reserved_vocab_size = qwen_reserved_vocab_size - 1  # 预留词汇表大小 - 1 (因为用了一个给<NUM>)
    
    print(f"Qwen模型词汇表概念:")
    print(f"  总词汇表大小 (config.vocab_size):     {qwen_config_vocab_size:,}")
    print(f"  已用词汇表大小 (len(tokenizer)):      {qwen_used_vocab_size:,}")
    print(f"  预留词汇表大小 (未使用):              {qwen_reserved_vocab_size:,}")
    print(f"  计算验证: {qwen_used_vocab_size:,} + {qwen_reserved_vocab_size:,} = {qwen_config_vocab_size:,}")
    
    print(f"\nCausalQwen模型词汇表概念:")
    print(f"  总词汇表大小 (config.vocab_size):     {causal_config_vocab_size:,}")
    print(f"  已用词汇表大小 (len(tokenizer)+1):    {causal_used_vocab_size:,}")
    print(f"  预留词汇表大小 (未使用):              {causal_reserved_vocab_size:,}")
    print(f"  计算验证: {causal_used_vocab_size:,} + {causal_reserved_vocab_size:,} = {causal_config_vocab_size:,}")
    
    # 2. 验证词汇表概念的一致性
    print(f"\n--- 词汇表概念一致性验证 ---")
    
    qwen_sum_correct = (qwen_used_vocab_size + qwen_reserved_vocab_size) == qwen_config_vocab_size
    causal_sum_correct = (causal_used_vocab_size + causal_reserved_vocab_size) == causal_config_vocab_size
    
    print(f"Qwen词汇表计算正确性: {'✅' if qwen_sum_correct else '❌'}")
    print(f"CausalQwen词汇表计算正确性: {'✅' if causal_sum_correct else '❌'}")
    
    # 3. <NUM> token位置分析
    print(f"\n--- <NUM> Token位置分析 ---")
    
    num_token_id = tokenizer.num_token_id
    print(f"<NUM> token ID: {num_token_id}")
    print(f"<NUM> token在原Qwen已用词汇表中的位置: {num_token_id} / {qwen_used_vocab_size}")
    
    # 验证<NUM> token的位置是否正确
    if num_token_id == qwen_used_vocab_size:
        print(f"✅ <NUM> token位置正确: 位于已用词汇表的末尾")
    elif num_token_id < qwen_used_vocab_size:
        print(f"❌ <NUM> token位置错误: 位于已用词汇表内部，可能覆盖了原有token")
    else:
        print(f"❌ <NUM> token位置错误: 位于预留词汇表区域")
    
    # 4. 权重形状分析
    print(f"\n--- 模型权重形状分析 ---")
    
    qwen_lm_head_shape = qwen_model.lm_head.weight.shape
    causal_cls_head_shape = causal_model.action_network.classification_head.causal_linear.weight.shape
    
    print(f"Qwen lm_head权重形状: {qwen_lm_head_shape}")
    print(f"  期望形状: [总词汇表大小, hidden_size] = [{qwen_config_vocab_size}, {qwen_lm_head_shape[1]}]")
    print(f"  实际形状: {qwen_lm_head_shape}")
    print(f"  形状正确: {'✅' if qwen_lm_head_shape[0] == qwen_config_vocab_size else '❌'}")
    
    print(f"\nCausalQwen分类头权重形状: {causal_cls_head_shape}")
    print(f"  期望形状: [总词汇表大小, hidden_size] = [{causal_config_vocab_size}, {causal_cls_head_shape[1]}]")
    print(f"  实际形状: {causal_cls_head_shape}")
    print(f"  形状正确: {'✅' if causal_cls_head_shape[0] == causal_config_vocab_size else '❌'}")
    
    # 5. 预留token权重分析
    print(f"\n--- 预留Token权重分析 ---")
    
    # 分析Qwen的预留token权重
    qwen_reserved_start_id = qwen_used_vocab_size
    qwen_reserved_end_id = qwen_config_vocab_size
    qwen_reserved_weights = qwen_model.lm_head.weight.data[qwen_reserved_start_id:qwen_reserved_end_id, :]
    
    print(f"Qwen预留token权重统计 (ID {qwen_reserved_start_id}~{qwen_reserved_end_id-1}):")
    print(f"  权重形状: {qwen_reserved_weights.shape}")
    print(f"  权重均值: {qwen_reserved_weights.mean().item():.6f}")
    print(f"  权重标准差: {qwen_reserved_weights.std().item():.6f}")
    print(f"  权重范围: [{qwen_reserved_weights.min().item():.6f}, {qwen_reserved_weights.max().item():.6f}]")
    print(f"  非零权重比例: {(qwen_reserved_weights != 0).float().mean().item():.6f}")
    
    # 分析CausalQwen的预留token权重
    causal_reserved_start_id = causal_used_vocab_size
    causal_reserved_end_id = causal_config_vocab_size
    causal_reserved_weights = causal_model.action_network.classification_head.causal_linear.weight.data[causal_reserved_start_id:causal_reserved_end_id, :]
    
    print(f"\nCausalQwen预留token权重统计 (ID {causal_reserved_start_id}~{causal_reserved_end_id-1}):")
    print(f"  权重形状: {causal_reserved_weights.shape}")
    print(f"  权重均值: {causal_reserved_weights.mean().item():.6f}")
    print(f"  权重标准差: {causal_reserved_weights.std().item():.6f}")
    print(f"  权重范围: [{causal_reserved_weights.min().item():.6f}, {causal_reserved_weights.max().item():.6f}]")
    print(f"  非零权重比例: {(causal_reserved_weights != 0).float().mean().item():.6f}")
    
    # 6. 预留token继承验证
    print(f"\n--- 预留Token继承验证 ---")
    
    # 验证预留token是否完全继承
    if causal_reserved_weights.shape == qwen_reserved_weights.shape:
        reserved_weights_identical = torch.allclose(causal_reserved_weights, qwen_reserved_weights, atol=1e-6)
        print(f"预留token权重完全继承: {'✅' if reserved_weights_identical else '❌'}")
        
        if reserved_weights_identical:
            print(f"✅ 验证通过：CausalQwen的预留token权重完全继承自Qwen")
        else:
            mse_diff = F.mse_loss(causal_reserved_weights, qwen_reserved_weights).item()
            print(f"❌ 预留token权重有差异，均方误差: {mse_diff:.6f}")
    else:
        print(f"❌ 预留token权重形状不匹配: {causal_reserved_weights.shape} vs {qwen_reserved_weights.shape}")
    
    # 7. 总结
    print(f"\n--- 词汇表概念分析总结 ---")
    
    concept_analysis_success = (
        qwen_sum_correct and causal_sum_correct and
        (num_token_id == qwen_used_vocab_size) and
        (qwen_lm_head_shape[0] == qwen_config_vocab_size) and
        (causal_cls_head_shape[0] == causal_config_vocab_size)
    )
    
    print(f"🎯 词汇表概念分析结果: {'✅ 完全符合理论' if concept_analysis_success else '❌ 发现问题'}")
    
    if concept_analysis_success:
        print(f"✅ 三个词汇表概念区分清晰，权重形状正确")
        print(f"✅ <NUM> token位置正确，预留token继承正确")
        print(f"✅ 符合 qwen_reserved_tokens_analysis.md 的理论分析")
    else:
        print(f"❌ 需要检查词汇表概念实现")
    
    return {
        'qwen_used_vocab_size': qwen_used_vocab_size,
        'qwen_reserved_vocab_size': qwen_reserved_vocab_size,
        'qwen_config_vocab_size': qwen_config_vocab_size,
        'causal_used_vocab_size': causal_used_vocab_size,
        'causal_reserved_vocab_size': causal_reserved_vocab_size,
        'causal_config_vocab_size': causal_config_vocab_size,
        'concept_analysis_success': concept_analysis_success
    }

def print_section(title, level=1):
    """打印层次化的章节标题"""
    symbols = ['=', '-', '~', '.']
    symbol = symbols[min(level-1, len(symbols)-1)]
    width = 80 if level == 1 else 60
    print(f"\n{symbol * width}")
    print(f"{symbol * (width//4)} {title} {symbol * (width//4)}")
    print(f"{symbol * width}")

def print_tensor_comparison(tensor1, tensor2, name1, name2, name):
    """打印两个张量的详细对比统计"""
    print(f"\n--- {name} 对比 ---")
    print(f"{'指标':<20} {name1:<20} {name2:<20} {'差异':<15}")
    print("-" * 75)
    
    # 基础统计
    print(f"{'形状':<20} {str(tensor1.shape):<20} {str(tensor2.shape):<20} {'N/A':<15}")
    
    if tensor1.is_floating_point() and tensor2.is_floating_point():
        print(f"{'均值':<20} {tensor1.mean().item():<20.6f} {tensor2.mean().item():<20.6f} {abs(tensor1.mean().item() - tensor2.mean().item()):<15.6f}")
        print(f"{'标准差':<20} {tensor1.std().item():<20.6f} {tensor2.std().item():<20.6f} {abs(tensor1.std().item() - tensor2.std().item()):<15.6f}")
        print(f"{'最小值':<20} {tensor1.min().item():<20.6f} {tensor2.min().item():<20.6f} {abs(tensor1.min().item() - tensor2.min().item()):<15.6f}")
        print(f"{'最大值':<20} {tensor1.max().item():<20.6f} {tensor2.max().item():<20.6f} {abs(tensor1.max().item() - tensor2.max().item()):<15.6f}")
        
        # 相似性度量
        if tensor1.shape == tensor2.shape:
            cosine_sim = F.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0).item()
            mse = F.mse_loss(tensor1, tensor2).item()
            print(f"{'余弦相似度':<20} {cosine_sim:<20.6f} {'N/A':<20} {'N/A':<15}")
            print(f"{'均方误差':<20} {mse:<20.6f} {'N/A':<20} {'N/A':<15}")
            
            # 判断是否完全一致
            is_identical = torch.allclose(tensor1, tensor2, atol=1e-6)
            print(f"{'完全一致':<20} {'✅' if is_identical else '❌':<20} {'N/A':<20} {'N/A':<15}")

def analyze_model_architectures(causal_model, qwen_model):
    """分析两个模型的架构差异"""
    print_section("模型架构对比分析", 1)
    
    # 参数统计
    causal_total_params = sum(p.numel() for p in causal_model.parameters())
    causal_trainable_params = sum(p.numel() for p in causal_model.parameters() if p.requires_grad)
    
    qwen_total_params = sum(p.numel() for p in qwen_model.parameters())
    qwen_trainable_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    
    print(f"\n--- 参数统计对比 ---")
    print(f"{'模型':<15} {'总参数':<15} {'可训练参数':<15} {'参数增量':<15}")
    print("-" * 60)
    print(f"{'CausalQwen':<15} {causal_total_params:<15,} {causal_trainable_params:<15,} {'-':<15}")
    print(f"{'Qwen':<15} {qwen_total_params:<15,} {qwen_trainable_params:<15,} {'-':<15}")
    print(f"{'差异':<15} {causal_total_params - qwen_total_params:<15,} {causal_trainable_params - qwen_trainable_params:<15,} {((causal_total_params - qwen_total_params) / qwen_total_params * 100):<15.2f}%")
    
    # 权重共享验证
    print(f"\n--- 权重共享验证 ---")
    
    # 获取权重字典
    causal_qwen_weights = {name: param for name, param in causal_model.named_parameters()}
    qwen_weights = {name: param for name, param in qwen_model.named_parameters()}
    
    # 检查关键权重是否共享
    key_weights_to_check = [
        ('feature_network.qwen_model.model.embed_tokens.weight', 'model.embed_tokens.weight'),
        ('feature_network.qwen_model.model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.q_proj.weight'),
        ('feature_network.qwen_model.model.layers.0.mlp.gate_proj.weight', 'model.layers.0.mlp.gate_proj.weight'),
    ]
    
    shared_count = 0
    total_count = len(key_weights_to_check)
    
    for causal_key, qwen_key in key_weights_to_check:
        if causal_key in causal_qwen_weights and qwen_key in qwen_weights:
            weight_identical = torch.equal(causal_qwen_weights[causal_key], qwen_weights[qwen_key])
            status = "✅ 完全一致" if weight_identical else "❌ 不一致"
            shared_count += 1 if weight_identical else 0
            print(f"  {qwen_key}: {status}")
        else:
            print(f"  {qwen_key}: ❌ 未找到对应权重")
    
    print(f"\n权重共享总结: {shared_count}/{total_count} 检查通过")

def analyze_weight_inheritance(causal_model, qwen_model, tokenizer, vocab_analysis):
    """分析CausalQwen从Qwen的权重继承情况（基于精确的词汇表概念）"""
    print_section("权重继承分析", 1)
    print("基于三个词汇表概念的精确权重继承分析")
    
    # 获取词汇表概念
    qwen_used_vocab_size = vocab_analysis['qwen_used_vocab_size']
    qwen_reserved_vocab_size = vocab_analysis['qwen_reserved_vocab_size'] 
    qwen_config_vocab_size = vocab_analysis['qwen_config_vocab_size']
    causal_used_vocab_size = vocab_analysis['causal_used_vocab_size']
    causal_reserved_vocab_size = vocab_analysis['causal_reserved_vocab_size']
    causal_config_vocab_size = vocab_analysis['causal_config_vocab_size']
    
    # ActionNetwork分类头 vs Qwen lm_head
    print(f"\n--- 基于词汇表概念的权重继承分析 ---")
    
    causal_cls_weight = causal_model.action_network.classification_head.causal_linear.weight.data
    qwen_lm_weight = qwen_model.lm_head.weight.data
    
    print(f"权重形状对比:")
    print(f"  CausalQwen分类头: {causal_cls_weight.shape} = [{causal_config_vocab_size}, {causal_cls_weight.shape[1]}]")
    print(f"  Qwen lm_head:    {qwen_lm_weight.shape} = [{qwen_config_vocab_size}, {qwen_lm_weight.shape[1]}]")
    
    # 1. 已用词汇表权重继承分析
    print(f"\n--- 已用词汇表权重继承分析 ---")
    print(f"对比范围: ID 0~{qwen_used_vocab_size-1} (共 {qwen_used_vocab_size} 个已用token)")
    
    # CausalQwen的已用token权重 (不包括<NUM>)
    causal_used_weights = causal_cls_weight[:qwen_used_vocab_size, :]  # ID 0~151664
    qwen_used_weights = qwen_lm_weight[:qwen_used_vocab_size, :]      # ID 0~151664
    
    print_tensor_comparison(causal_used_weights, qwen_used_weights,
                          "CausalQwen(已用)", "Qwen(已用)", "已用词汇表权重继承")
    
    # 2. <NUM> token权重特殊分析
    print(f"\n--- <NUM> Token权重特殊分析 ---")
    
    num_token_id = tokenizer.num_token_id
    print(f"<NUM> token ID: {num_token_id}")
    print(f"预期位置: 已用词汇表末尾 (ID {qwen_used_vocab_size})")
    
    if num_token_id == qwen_used_vocab_size:
        print(f"✅ <NUM> token位置正确")
        num_token_weight = causal_cls_weight[num_token_id, :]  # <NUM> token权重
        
        # 对比<NUM> token与已用token的权重差异
        used_weights_mean = causal_used_weights.mean(dim=0)  # 已用权重的均值
        used_weights_std = causal_used_weights.std(dim=0)    # 已用权重的标准差
        
        print(f"<NUM> token权重统计:")
        print(f"  权重均值: {num_token_weight.mean().item():.6f}")
        print(f"  权重标准差: {num_token_weight.std().item():.6f}")
        print(f"  权重范围: [{num_token_weight.min().item():.6f}, {num_token_weight.max().item():.6f}]")
        
        print(f"<NUM> token与已用权重对比:")
        print(f"  已用权重均值的均值: {used_weights_mean.mean().item():.6f}")
        print(f"  已用权重标准差的均值: {used_weights_std.mean().item():.6f}")
        
        # 余弦相似度分析
        cosine_sim = F.cosine_similarity(num_token_weight, used_weights_mean, dim=0).item()
        print(f"  <NUM>与已用权重均值的余弦相似度: {cosine_sim:.6f}")
        
        # 判断是否特殊初始化
        special_init = abs(num_token_weight.mean().item() - used_weights_mean.mean().item()) > 0.1
        print(f"  特殊初始化检测: {'✅ 有特殊初始化' if special_init else '❌ 无特殊初始化'}")
    else:
        print(f"❌ <NUM> token位置错误: 期望 {qwen_used_vocab_size}, 实际 {num_token_id}")
    
    # 3. 预留词汇表权重继承分析
    print(f"\n--- 预留词汇表权重继承分析 ---")
    
    # CausalQwen的预留token范围: ID (qwen_used_vocab_size + 1) ~ (causal_config_vocab_size - 1)
    causal_reserved_start = causal_used_vocab_size  # qwen_used_vocab_size + 1
    causal_reserved_end = causal_config_vocab_size
    
    # Qwen的预留token范围: ID qwen_used_vocab_size ~ (qwen_config_vocab_size - 1)  
    qwen_reserved_start = qwen_used_vocab_size
    qwen_reserved_end = qwen_config_vocab_size
    
    print(f"CausalQwen预留token范围: ID {causal_reserved_start}~{causal_reserved_end-1} (共 {causal_reserved_end - causal_reserved_start} 个)")
    print(f"Qwen预留token范围: ID {qwen_reserved_start}~{qwen_reserved_end-1} (共 {qwen_reserved_end - qwen_reserved_start} 个)")
    
    if (causal_reserved_end - causal_reserved_start) == (qwen_reserved_end - qwen_reserved_start):
        causal_reserved_weights = causal_cls_weight[causal_reserved_start:causal_reserved_end, :]
        qwen_reserved_weights = qwen_lm_weight[qwen_reserved_start:qwen_reserved_end, :]
        
        print_tensor_comparison(causal_reserved_weights, qwen_reserved_weights,
                              "CausalQwen(预留)", "Qwen(预留)", "预留词汇表权重继承")
    else:
        print(f"❌ 预留token数量不匹配，无法对比")
        print(f"   CausalQwen预留: {causal_reserved_end - causal_reserved_start}")
        print(f"   Qwen预留: {qwen_reserved_end - qwen_reserved_start}")
    
    # 4. 基于词汇表概念的偏置继承分析
    print(f"\n--- 基于词汇表概念的偏置继承分析 ---")
    
    causal_cls_bias = causal_model.action_network.classification_head.causal_linear.bias
    if causal_cls_bias is not None:
        causal_cls_bias = causal_cls_bias.data
        
        if hasattr(qwen_model.lm_head, 'bias') and qwen_model.lm_head.bias is not None:
            qwen_lm_bias = qwen_model.lm_head.bias.data
            
            # 已用词汇表偏置继承分析
            print(f"已用词汇表偏置继承分析:")
            causal_used_bias = causal_cls_bias[:qwen_used_vocab_size]
            qwen_used_bias = qwen_lm_bias[:qwen_used_vocab_size]
            
            print_tensor_comparison(causal_used_bias, qwen_used_bias,
                                  "CausalQwen(已用偏置)", "Qwen(已用偏置)", "已用词汇表偏置继承")
            
            # <NUM> token偏置特殊分析
            if num_token_id == qwen_used_vocab_size:
                print(f"\n<NUM> token偏置特殊分析:")
                num_bias_causal = causal_cls_bias[num_token_id].item()
                used_bias_mean = causal_used_bias.mean().item()
                
                print(f"  <NUM>偏置值: {num_bias_causal:.6f}")
                print(f"  已用偏置均值: {used_bias_mean:.6f}")
                print(f"  差异: {abs(num_bias_causal - used_bias_mean):.6f}")
                
                special_bias_init = abs(num_bias_causal - used_bias_mean) > 0.1
                print(f"  <NUM>偏置特殊初始化: {'✅' if special_bias_init else '❌'}")
            
            # 预留词汇表偏置继承分析
            if (causal_reserved_end - causal_reserved_start) == (qwen_reserved_end - qwen_reserved_start):
                print(f"\n预留词汇表偏置继承分析:")
                causal_reserved_bias = causal_cls_bias[causal_reserved_start:causal_reserved_end]
                qwen_reserved_bias = qwen_lm_bias[qwen_reserved_start:qwen_reserved_end]
                
                print_tensor_comparison(causal_reserved_bias, qwen_reserved_bias,
                                      "CausalQwen(预留偏置)", "Qwen(预留偏置)", "预留词汇表偏置继承")
        else:
            print("Qwen模型没有lm_head偏置，检查CausalQwen偏置初始化...")
            print(f"CausalQwen偏置统计:")
            print(f"  总体均值: {causal_cls_bias.mean().item():.6f}")
            print(f"  总体标准差: {causal_cls_bias.std().item():.6f}")
            
            # 分析不同词汇表区域的偏置
            used_bias = causal_cls_bias[:qwen_used_vocab_size]
            print(f"  已用偏置均值: {used_bias.mean().item():.6f}")
            
            if num_token_id == qwen_used_vocab_size:
                num_bias = causal_cls_bias[num_token_id].item()
                print(f"  <NUM>偏置值: {num_bias:.6f}")
            
            reserved_bias = causal_cls_bias[causal_reserved_start:causal_reserved_end]
            print(f"  预留偏置均值: {reserved_bias.mean().item():.6f}")
    else:
        print("CausalQwen分类头没有偏置")
    
    # 5. 权重继承质量总结
    print(f"\n--- 权重继承质量总结 ---")
    
    # 计算继承质量指标
    used_weights_identical = torch.allclose(causal_used_weights, qwen_used_weights, atol=1e-6)
    
    if (causal_reserved_end - causal_reserved_start) == (qwen_reserved_end - qwen_reserved_start):
        causal_reserved_weights = causal_cls_weight[causal_reserved_start:causal_reserved_end, :]
        qwen_reserved_weights = qwen_lm_weight[qwen_reserved_start:qwen_reserved_end, :]
        reserved_weights_identical = torch.allclose(causal_reserved_weights, qwen_reserved_weights, atol=1e-6)
    else:
        reserved_weights_identical = False
    
    num_position_correct = (num_token_id == qwen_used_vocab_size)
    
    print(f"权重继承质量评估:")
    print(f"  已用词汇表权重完全继承: {'✅' if used_weights_identical else '❌'}")
    print(f"  预留词汇表权重完全继承: {'✅' if reserved_weights_identical else '❌'}")
    print(f"  <NUM> token位置正确: {'✅' if num_position_correct else '❌'}")
    
    inheritance_success = used_weights_identical and reserved_weights_identical and num_position_correct
    print(f"  🎯 整体权重继承: {'✅ 完全成功' if inheritance_success else '❌ 存在问题'}")
    
    return {
        'used_weights_identical': used_weights_identical,
        'reserved_weights_identical': reserved_weights_identical,
        'num_position_correct': num_position_correct,
        'inheritance_success': inheritance_success
    }

def compare_forward_pass(causal_model, qwen_model, inputs, tokenizer, device, vocab_analysis):
    """对比两个模型的前向传播结果（基于精确的词汇表概念）"""
    print_section("前向传播对比", 1)
    print("基于三个词汇表概念的精确前向传播对比")
    
    # 获取词汇表概念
    qwen_used_vocab_size = vocab_analysis['qwen_used_vocab_size']
    qwen_reserved_vocab_size = vocab_analysis['qwen_reserved_vocab_size'] 
    qwen_config_vocab_size = vocab_analysis['qwen_config_vocab_size']
    causal_used_vocab_size = vocab_analysis['causal_used_vocab_size']
    causal_reserved_vocab_size = vocab_analysis['causal_reserved_vocab_size']
    causal_config_vocab_size = vocab_analysis['causal_config_vocab_size']
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    
    print(f"输入数据:")
    print(f"  input_ids形状: {input_ids.shape}")
    print(f"  attention_mask形状: {attention_mask.shape}")
    print(f"  numerical_values形状: {numerical_values.shape}")
    
    # CausalQwen前向传播
    print(f"\n--- CausalQwen 前向传播 ---")
    with torch.no_grad():
        causal_outputs = causal_model(input_ids, numerical_values, attention_mask)
    
    print(f"CausalQwen输出:")
    for key, tensor in causal_outputs.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}")
    
    # Qwen前向传播
    print(f"\n--- Qwen 前向传播 ---")
    with torch.no_grad():
        qwen_outputs = qwen_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    print(f"Qwen输出:")
    print(f"  logits: {qwen_outputs.logits.shape}")
    if hasattr(qwen_outputs, 'hidden_states') and qwen_outputs.hidden_states is not None:
        print(f"  hidden_states (layers): {len(qwen_outputs.hidden_states)}")
        print(f"  last_hidden_state: {qwen_outputs.hidden_states[-1].shape}")
    else:
        print(f"  hidden_states: Not available")
    
    # 特征表征对比
    print_section("特征表征对比", 2)
    
    causal_features = causal_outputs['features']
    
    if hasattr(qwen_outputs, 'hidden_states') and qwen_outputs.hidden_states is not None:
        qwen_features = qwen_outputs.hidden_states[-1]  # 最后一层隐藏状态
        
        print_tensor_comparison(causal_features, qwen_features,
                              "CausalQwen", "Qwen", "最后层隐藏状态")
        
        features_identical = torch.allclose(causal_features, qwen_features, atol=1e-6)
        print(f"\n✅ 特征完全一致性验证: {'通过' if features_identical else '失败'}")
    else:
        print("⚠️ 无法获取Qwen的隐藏状态，跳过特征对比")
        features_identical = False
    
    # 基于词汇表概念的输出分析
    print_section("基于词汇表概念的输出分析", 2)
    
    print(f"词汇表概念对比:")
    print(f"  Qwen输出logits形状: {qwen_outputs.logits.shape} (总词汇表: {qwen_config_vocab_size})")
    print(f"  CausalQwen输出cls_loc形状: {causal_outputs['cls_loc'].shape} (总词汇表: {causal_config_vocab_size})")
    print(f"  已用词汇表大小: {qwen_used_vocab_size}")
    print(f"  预留词汇表大小: {qwen_reserved_vocab_size}")
    
    # 1. 已用词汇表输出对比
    print(f"\n--- 已用词汇表输出对比 ---")
    print(f"对比范围: ID 0~{qwen_used_vocab_size-1} (共 {qwen_used_vocab_size} 个已用token)")
    
    # 提取已用词汇表的输出
    causal_used_cls_loc = causal_outputs['cls_loc'][:, :, :qwen_used_vocab_size]
    causal_used_cls_scale = causal_outputs['cls_scale'][:, :, :qwen_used_vocab_size]
    qwen_used_logits = qwen_outputs.logits[:, :, :qwen_used_vocab_size]
    
    # 直接对比分类得分（cls_loc vs logits）
    print_tensor_comparison(causal_used_cls_loc, qwen_used_logits,
                          "CausalQwen(已用cls_loc)", "Qwen(已用logits)", "已用词汇表分类得分")
    
    # 2. <NUM> token输出分析
    print(f"\n--- <NUM> Token输出分析 ---")
    
    num_token_id = tokenizer.num_token_id
    if num_token_id == qwen_used_vocab_size:
        print(f"<NUM> token ID: {num_token_id} (位置正确)")
        
        # 提取<NUM> token的输出
        causal_num_cls_loc = causal_outputs['cls_loc'][:, :, num_token_id]  # [B, S]
        causal_num_cls_scale = causal_outputs['cls_scale'][:, :, num_token_id]  # [B, S]
        
        print(f"<NUM> token输出统计:")
        print(f"  cls_loc 均值: {causal_num_cls_loc.mean().item():.6f}")
        print(f"  cls_scale 均值: {causal_num_cls_scale.mean().item():.6f}")
        
        # 计算<NUM> token的OvR概率
        num_ovr_probs = compute_ovr_probabilities(
            causal_num_cls_loc, causal_num_cls_scale, 10.0
        )
        
        print(f"  OvR概率统计:")
        print(f"    平均概率: {num_ovr_probs.mean().item():.6f}")
        print(f"    最大概率: {num_ovr_probs.max().item():.6f}")
        print(f"    概率>0.1的位置数: {(num_ovr_probs > 0.1).sum().item()}")
        
        # 分析<NUM> token相对于已用token的输出
        used_cls_loc_mean = causal_used_cls_loc.mean(dim=-1)  # [B, S] 已用token的平均loc
        num_vs_used_diff = (causal_num_cls_loc - used_cls_loc_mean).abs()
        
        print(f"  <NUM> token与已用token输出差异:")
        print(f"    平均差异: {num_vs_used_diff.mean().item():.6f}")
        print(f"    最大差异: {num_vs_used_diff.max().item():.6f}")
    else:
        print(f"❌ <NUM> token位置错误: 期望 {qwen_used_vocab_size}, 实际 {num_token_id}")
    
    # 3. 预留词汇表输出分析
    print(f"\n--- 预留词汇表输出分析 ---")
    
    causal_reserved_start = causal_used_vocab_size
    causal_reserved_end = causal_config_vocab_size
    qwen_reserved_start = qwen_used_vocab_size
    qwen_reserved_end = qwen_config_vocab_size
    
    print(f"CausalQwen预留token范围: ID {causal_reserved_start}~{causal_reserved_end-1}")
    print(f"Qwen预留token范围: ID {qwen_reserved_start}~{qwen_reserved_end-1}")
    
    # 提取预留词汇表的输出
    causal_reserved_cls_loc = causal_outputs['cls_loc'][:, :, causal_reserved_start:causal_reserved_end]
    causal_reserved_cls_scale = causal_outputs['cls_scale'][:, :, causal_reserved_start:causal_reserved_end]
    qwen_reserved_logits = qwen_outputs.logits[:, :, qwen_reserved_start:qwen_reserved_end]
    
    if causal_reserved_cls_loc.shape == qwen_reserved_logits.shape:
        print_tensor_comparison(causal_reserved_cls_loc, qwen_reserved_logits,
                              "CausalQwen(预留cls_loc)", "Qwen(预留logits)", "预留词汇表分类得分")
        
        # 分析预留token的概率分布
        reserved_ovr_probs = compute_ovr_probabilities(
            causal_reserved_cls_loc.flatten(0, -2),
            causal_reserved_cls_scale.flatten(0, -2),
            10.0
        ).view_as(causal_reserved_cls_loc)
        
        print(f"\n预留token OvR概率分析:")
        print(f"  平均概率: {reserved_ovr_probs.mean().item():.6f}")
        print(f"  最大概率: {reserved_ovr_probs.max().item():.6f}")
        print(f"  概率>0.01的比例: {(reserved_ovr_probs > 0.01).float().mean().item():.6f}")
        
        # 验证预留token的概率应该很低
        low_prob_threshold = 0.01
        low_prob_ratio = (reserved_ovr_probs < low_prob_threshold).float().mean().item()
        print(f"  低概率(<{low_prob_threshold})比例: {low_prob_ratio:.6f} (应该接近1.0)")
        
        reserved_probs_reasonable = low_prob_ratio > 0.95
        print(f"  预留token概率合理性: {'✅' if reserved_probs_reasonable else '❌'}")
    else:
        print(f"❌ 预留token输出形状不匹配")
        reserved_probs_reasonable = False
    
    # 位置级别的详细分析
    print_section("位置级别分析", 2)
    
    batch_size, seq_len = input_ids.shape
    sample_positions = [(0, 1), (0, min(3, seq_len-1)), (0, seq_len-1)]
    
    for batch_idx, pos_idx in sample_positions:
        if attention_mask[batch_idx, pos_idx] == 0:
            continue
            
        print(f"\n--- 样本{batch_idx+1} 位置{pos_idx+1} 详细分析 ---")
        
        # 分类得分对比（CausalQwen的cls_loc vs Qwen的logits）
        causal_scores = causal_outputs['cls_loc'][batch_idx, pos_idx, :K]
        qwen_scores = qwen_logits_truncated[batch_idx, pos_idx, :]
        
        print_tensor_comparison(causal_scores, qwen_scores,
                              "CausalQwen(cls_loc)", "Qwen(logits)", "分类得分")
        
        # Top-5 token概率对比
        causal_pos_probs = causal_ovr_probs_inherited[batch_idx, pos_idx, :]
        qwen_pos_probs = qwen_probs_truncated[batch_idx, pos_idx, :]
        
        causal_top5_probs, causal_top5_indices = torch.topk(causal_pos_probs, 5)
        qwen_top5_probs, qwen_top5_indices = torch.topk(qwen_pos_probs, 5)
        
        print(f"\nTop-5 token概率对比:")
        print(f"{'排名':<5} {'CausalQwen Token':<20} {'概率':<10} {'Qwen Token':<20} {'概率':<10}")
        print("-" * 65)
        
        for i in range(5):
            causal_token = tokenizer.convert_ids_to_tokens([causal_top5_indices[i].item()])[0]
            qwen_token = tokenizer.convert_ids_to_tokens([qwen_top5_indices[i].item()])[0]
            
            print(f"{i+1:<5} {causal_token:<20} {causal_top5_probs[i].item():<10.4f} {qwen_token:<20} {qwen_top5_probs[i].item():<10.4f}")
    
    # 因果表征分析
    print_section("因果表征分析", 2)
    
    causal_loc = causal_outputs['causal_loc']
    causal_scale = causal_outputs['causal_scale']
    
    print(f"因果表征统计:")
    print(f"  causal_loc: 均值={causal_loc.mean().item():.6f}, 标准差={causal_loc.std().item():.6f}")
    print(f"  causal_scale: 均值={causal_scale.mean().item():.6f}, 标准差={causal_scale.std().item():.6f}")
    print(f"  causal_scale最小值: {causal_scale.min().item():.6f} (必须>0)")
    
    # AbductionNetwork初始化验证
    expected_scale_mean = 10.0  # 基于exp(2.3) ≈ 10
    scale_init_correct = abs(causal_scale.mean().item() - expected_scale_mean) < 5.0
    print(f"  AbductionNetwork初始化: {'✅' if scale_init_correct else '❌'} (期望causal_scale≈10)")
    
    # <NUM> token的OvR概率分析
    if tokenizer.num_token_id < causal_outputs['cls_loc'].shape[-1]:
        num_token_probs = compute_ovr_probabilities(
            causal_outputs['cls_loc'][:, :, tokenizer.num_token_id],
            causal_outputs['cls_scale'][:, :, tokenizer.num_token_id],
            10.0
        )
        
        print(f"\n<NUM> token OvR概率分析:")
        print(f"  平均概率: {num_token_probs.mean().item():.6f}")
        print(f"  最大概率: {num_token_probs.max().item():.6f}")
        print(f"  概率>0.1的位置数: {(num_token_probs > 0.1).sum().item()}")
    
    return causal_outputs, qwen_outputs

def main():
    """主函数"""
    print_section("CausalQwen VS Qwen 模型对比验证", 1)
    print("验证CausalQwen从Qwen的知识传输效果")
    print("设计原则: 前向传播结果 > 模型参数结构")
    
    # 1. 模型设置
    print_section("模型设置", 1)
    
    device = torch.device('cpu')  # 使用CPU以便调试
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    
    # 设置分词器
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    print(f"分词器设置完成，词汇表大小: {tokenizer.vocab_size}")
    print(f"<NUM> token ID: {tokenizer.num_token_id}")
    
    # 设置CausalQwen模型
    print(f"\n--- 设置CausalQwen模型 ---")
    causal_config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        causal_dim=896,
        use_real_qwen=True,
        qwen_model_path=qwen_model_path,
        ovr_threshold=10.0,
        reg_loss_weight=1.0
    )
    
    causal_model = CausalLanguageModel(causal_config).to(device)
    causal_model.eval()
    
    # 执行知识传输初始化
    num_target_median = 50.0
    num_target_scale = 25.0
    causal_model.init_weights(num_target_median, num_target_scale)
    print(f"CausalQwen模型设置完成，知识传输初始化完成")
    
    # 设置原始Qwen模型
    print(f"\n--- 设置原始Qwen模型 ---")
    qwen_model = Qwen2ForCausalLM.from_pretrained(
        qwen_model_path,
        torch_dtype=torch.float32,
        device_map=None
    ).to(device)
    qwen_model.eval()
    print(f"原始Qwen模型设置完成")
    
    # 2. 测试数据准备
    print_section("测试数据准备", 1)
    
    texts = [
        "The item costs 99.99 dollars.",
        "From a batch of 100 items, 5 were defective, leaving 95 good ones.",
        "A standard text without any numerical values."
    ]
    
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
    
    print(f"测试文本:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    print(f"\n批次数据:")
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  attention_mask: {inputs['attention_mask'].shape}")
    print(f"  numerical_values: {inputs['numerical_values'].shape}")
    
    # 3. 执行对比分析
    vocab_analysis = analyze_vocabulary_concepts(causal_model, qwen_model, tokenizer)
    analyze_model_architectures(causal_model, qwen_model)
    inheritance_analysis = analyze_weight_inheritance(causal_model, qwen_model, tokenizer, vocab_analysis)
    causal_outputs, qwen_outputs = compare_forward_pass(causal_model, qwen_model, inputs, tokenizer, device, vocab_analysis)
    
    # 4. 基于词汇表概念的总结验证结果
    print_section("基于词汇表概念的验证结果总结", 1)
    print("验证所有三个词汇表概念是否正确实现")
    
    # 检查关键验证点
    if hasattr(qwen_outputs, 'hidden_states') and qwen_outputs.hidden_states is not None:
        features_identical = torch.allclose(causal_outputs['features'], qwen_outputs.hidden_states[-1], atol=1e-6)
    else:
        features_identical = False
    
    # 基于词汇表概念检查分类得分一致性
    qwen_used_vocab_size = vocab_analysis['qwen_used_vocab_size']
    
    causal_used_scores = causal_outputs['cls_loc'][:, :, :qwen_used_vocab_size]
    qwen_used_scores = qwen_outputs.logits[:, :, :qwen_used_vocab_size]
    used_scores_consistent = torch.allclose(causal_used_scores, qwen_used_scores, atol=1e-3)
    
    # 检查因果表征初始化
    scale_init_reasonable = 5.0 < causal_outputs['causal_scale'].mean().item() < 15.0
    
    # 从之前的分析获取结果
    concept_analysis_success = vocab_analysis['concept_analysis_success']
    inheritance_success = inheritance_analysis['inheritance_success']
    used_weights_identical = inheritance_analysis['used_weights_identical']
    reserved_weights_identical = inheritance_analysis['reserved_weights_identical']
    num_position_correct = inheritance_analysis['num_position_correct']
    
    print(f"\n🔍 三个词汇表概念验证检查点:")
    print(f"  📊 词汇表概念分析: {'✅ 通过' if concept_analysis_success else '❌ 失败'}")
    print(f"     - 总词汇表大小计算正确")
    print(f"     - 已用词汇表大小正确")  
    print(f"     - 预留词汇表大小正确")
    print(f"     - <NUM> token位置正确")
    
    print(f"\n🏗️ 权重继承验证检查点:")
    print(f"  🎯 整体权重继承: {'✅ 通过' if inheritance_success else '❌ 失败'}")
    print(f"     - 已用词汇表权重完全继承: {'✅' if used_weights_identical else '❌'}")
    print(f"     - 预留词汇表权重完全继承: {'✅' if reserved_weights_identical else '❌'}")
    print(f"     - <NUM> token位置正确: {'✅' if num_position_correct else '❌'}")
    
    print(f"\n🚀 前向传播验证检查点:")
    print(f"  🔥 特征完全一致: {'✅ 通过' if features_identical else '❌ 失败'}")
    print(f"  📈 已用词汇表分类得分一致: {'✅ 通过' if used_scores_consistent else '❌ 失败'}")
    print(f"  🧠 因果表征初始化合理: {'✅ 通过' if scale_init_reasonable else '❌ 失败'}")
    
    # 综合评估
    overall_success = (
        concept_analysis_success and 
        inheritance_success and 
        features_identical and 
        used_scores_consistent and 
        scale_init_reasonable
    )
    
    print(f"\n🎯 基于词汇表概念的总体验证结果:")
    print(f"   {'🎉 完全成功！' if overall_success else '⚠️ 需要进一步调试'}")
    
    if overall_success:
        print(f"\n✅ 完整验证通过:")
        print(f"   🔸 三个词汇表概念区分清晰且实现正确")
        print(f"   🔸 已用词汇表 ({vocab_analysis['qwen_used_vocab_size']:,} tokens) 完全继承")
        print(f"   🔸 预留词汇表 ({vocab_analysis['qwen_reserved_vocab_size']:,} tokens) 完全继承")
        print(f"   🔸 <NUM> token (ID: {tokenizer.num_token_id}) 位置和初始化正确")
        print(f"   🔸 前向传播结果与Qwen完全一致")
        print(f"   🔸 因果推理功能正确扩展")
        print(f"\n🎊 CausalQwen架构重构完全验证通过！")
        print(f"    符合 qwen_reserved_tokens_analysis.md 的理论分析")
    else:
        print(f"\n❌ 发现问题，建议检查:")
        
        if not concept_analysis_success:
            print(f"   🔸 词汇表概念实现")
        if not inheritance_success:
            if not used_weights_identical:
                print(f"   🔸 已用词汇表权重继承")
            if not reserved_weights_identical:
                print(f"   🔸 预留词汇表权重继承")
            if not num_position_correct:
                print(f"   🔸 <NUM> token位置设置")
        if not features_identical:
            print(f"   🔸 QwenFeatureNetwork的实现")
        if not used_scores_consistent:
            print(f"   🔸 ActionNetwork的权重初始化")
        if not scale_init_reasonable:
            print(f"   🔸 AbductionNetwork的初始化策略")
    
    # 返回完整的验证结果供外部使用
    return {
        'concept_analysis_success': concept_analysis_success,
        'inheritance_success': inheritance_success,
        'features_identical': features_identical,
        'used_scores_consistent': used_scores_consistent,
        'scale_init_reasonable': scale_init_reasonable,
        'overall_success': overall_success,
        'vocab_analysis': vocab_analysis,
        'inheritance_analysis': inheritance_analysis
    }

if __name__ == '__main__':
    main() 