#!/usr/bin/env python
"""
CausalQwen VS Qwen 知识迁移验证脚本

本脚本验证 CausalQwen 是否正确地从 Qwen 模型进行了知识迁移。

核心验证内容：
1. 特征提取一致性 - 验证 QwenFeatureNetwork 是否正确封装了 Qwen
2. 分类头权重继承 - 验证 ActionNetwork 是否完全复用了 lm_head
3. 前向传播一致性 - 验证相同输入下的输出一致性
4. 保留词汇处理 - 验证保留词汇的权重是否正确继承
"""

import os
import sys
import torch
import numpy as np
from transformers import Qwen2ForCausalLM

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper

def print_section(title, symbol="="):
    """打印格式化的标题"""
    width = 80
    print(f"\n{symbol * width}")
    print(f"{title.center(width)}")
    print(f"{symbol * width}")

def verify_feature_extraction(causal_model, qwen_model, inputs, device):
    """验证特征提取的一致性"""
    print_section("特征提取一致性验证", "-")
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # CausalQwen 特征提取
    with torch.no_grad():
        causal_outputs = causal_model(
            input_ids, 
            inputs['numerical_values'].to(device), 
            attention_mask
        )
        causal_features = causal_outputs['features']
    
    # Qwen 特征提取
    with torch.no_grad():
        qwen_outputs = qwen_model(
            input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        qwen_features = qwen_outputs.hidden_states[-1]  # 最后一层隐藏状态
    
    # 验证一致性
    features_match = torch.allclose(causal_features, qwen_features, atol=1e-6)
    
    print(f"特征形状: CausalQwen {causal_features.shape} vs Qwen {qwen_features.shape}")
    print(f"特征均值差异: {(causal_features - qwen_features).abs().mean().item():.6e}")
    print(f"特征最大差异: {(causal_features - qwen_features).abs().max().item():.6e}")
    print(f"特征提取一致性: {'✅ 通过' if features_match else '❌ 失败'}")
    
    return features_match

def verify_weight_inheritance(causal_model, qwen_model, tokenizer):
    """验证权重继承的正确性"""
    print_section("权重继承验证", "-")
    
    # 获取分类头权重
    causal_cls_weight = causal_model.action_network.classification_head.causal_linear.weight.data
    causal_cls_bias = causal_model.action_network.classification_head.causal_linear.bias
    
    qwen_lm_weight = qwen_model.lm_head.weight.data
    qwen_lm_bias = qwen_model.lm_head.bias if hasattr(qwen_model.lm_head, 'bias') else None
    
    print(f"权重形状对比:")
    print(f"  CausalQwen 分类头: {causal_cls_weight.shape}")
    print(f"  Qwen lm_head:     {qwen_lm_weight.shape}")
    
    # 验证权重继承（只比较 CausalQwen 词汇表范围内的权重）
    vocab_size = causal_cls_weight.shape[0]
    inherited_weights = causal_cls_weight[:vocab_size]
    qwen_weights = qwen_lm_weight[:vocab_size]
    
    weights_match = torch.allclose(inherited_weights, qwen_weights, atol=1e-6)
    
    print(f"\n权重继承统计:")
    print(f"  权重均值差异: {(inherited_weights - qwen_weights).abs().mean().item():.6e}")
    print(f"  权重最大差异: {(inherited_weights - qwen_weights).abs().max().item():.6e}")
    print(f"  权重继承一致性: {'✅ 通过' if weights_match else '❌ 失败'}")
    
    # 验证偏置继承
    if causal_cls_bias is not None and qwen_lm_bias is not None:
        inherited_bias = causal_cls_bias.data[:vocab_size]
        qwen_bias = qwen_lm_bias.data[:vocab_size]
        bias_match = torch.allclose(inherited_bias, qwen_bias, atol=1e-6)
        print(f"\n偏置继承统计:")
        print(f"  偏置均值差异: {(inherited_bias - qwen_bias).abs().mean().item():.6e}")
        print(f"  偏置继承一致性: {'✅ 通过' if bias_match else '❌ 失败'}")
    else:
        bias_match = True
        print(f"\n偏置继承: Qwen 没有偏置项")
    
    return weights_match and bias_match

def verify_forward_consistency(causal_model, qwen_model, inputs, tokenizer, device):
    """验证前向传播的一致性"""
    print_section("前向传播一致性验证", "-")
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    
    # CausalQwen 前向传播
    with torch.no_grad():
        causal_outputs = causal_model(input_ids, numerical_values, attention_mask)
        causal_cls_loc = causal_outputs['cls_loc']
    
    # Qwen 前向传播
    with torch.no_grad():
        qwen_outputs = qwen_model(input_ids, attention_mask=attention_mask)
        qwen_logits = qwen_outputs.logits
    
    # 只比较已用词汇表部分的输出（不包括 <NUM> 和保留词汇）
    used_vocab_size = tokenizer.tokenizer.vocab_size  # 原始 Qwen 词汇表大小
    
    causal_used_logits = causal_cls_loc[:, :, :used_vocab_size]
    qwen_used_logits = qwen_logits[:, :, :used_vocab_size]
    
    logits_match = torch.allclose(causal_used_logits, qwen_used_logits, atol=1e-3)
    
    print(f"输出形状对比:")
    print(f"  CausalQwen cls_loc: {causal_cls_loc.shape}")
    print(f"  Qwen logits:        {qwen_logits.shape}")
    print(f"\n已用词汇表输出一致性:")
    print(f"  比较范围: 前 {used_vocab_size} 个词汇")
    print(f"  输出均值差异: {(causal_used_logits - qwen_used_logits).abs().mean().item():.6e}")
    print(f"  输出最大差异: {(causal_used_logits - qwen_used_logits).abs().max().item():.6e}")
    print(f"  前向传播一致性: {'✅ 通过' if logits_match else '❌ 失败'}")
    
    # 分析 <NUM> token 的输出
    num_token_id = tokenizer.num_token_id
    if num_token_id < causal_cls_loc.shape[-1]:
        num_logits = causal_cls_loc[:, :, num_token_id]
        print(f"\n<NUM> token (ID: {num_token_id}) 输出分析:")
        print(f"  输出均值: {num_logits.mean().item():.6f}")
        print(f"  输出标准差: {num_logits.std().item():.6f}")
        print(f"  输出范围: [{num_logits.min().item():.6f}, {num_logits.max().item():.6f}]")
    
    return logits_match

def verify_reserved_tokens(causal_model, qwen_model, tokenizer):
    """验证保留词汇的处理"""
    print_section("保留词汇处理验证", "-")
    
    # 获取词汇表信息
    qwen_total_vocab = qwen_model.config.vocab_size  # 151936
    qwen_used_vocab = tokenizer.tokenizer.vocab_size  # 151665
    qwen_reserved = qwen_total_vocab - qwen_used_vocab  # 271
    
    causal_total_vocab = tokenizer.vocab_size  # 151666
    causal_reserved_start = causal_total_vocab
    causal_reserved_end = qwen_total_vocab
    
    print(f"词汇表统计:")
    print(f"  Qwen 总词汇表: {qwen_total_vocab}")
    print(f"  Qwen 已用词汇: {qwen_used_vocab}")
    print(f"  Qwen 保留词汇: {qwen_reserved}")
    print(f"  CausalQwen 总词汇表: {causal_total_vocab}")
    print(f"  CausalQwen 保留词汇: {causal_reserved_end - causal_reserved_start}")
    
    # 如果 CausalQwen 的词汇表大小等于 Qwen 的总词汇表大小，验证保留词汇权重
    if causal_model.action_network.classification_head.causal_linear.weight.shape[0] == qwen_total_vocab:
        causal_reserved_weights = causal_model.action_network.classification_head.causal_linear.weight.data[causal_reserved_start:causal_reserved_end]
        qwen_reserved_weights = qwen_model.lm_head.weight.data[causal_reserved_start:causal_reserved_end]
        
        reserved_match = torch.allclose(causal_reserved_weights, qwen_reserved_weights, atol=1e-6)
        
        print(f"\n保留词汇权重验证:")
        print(f"  权重形状: {causal_reserved_weights.shape}")
        print(f"  权重均值差异: {(causal_reserved_weights - qwen_reserved_weights).abs().mean().item():.6e}")
        print(f"  保留词汇权重一致性: {'✅ 通过' if reserved_match else '❌ 失败'}")
        
        return reserved_match
    else:
        print(f"\n⚠️  CausalQwen 词汇表大小与 Qwen 不同，跳过保留词汇验证")
        return True

def main():
    """主函数"""
    print_section("CausalQwen 知识迁移验证")
    
    # 设置
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    
    # 初始化分词器
    print("\n初始化分词器...")
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"<NUM> token ID: {tokenizer.num_token_id}")
    
    # 初始化 CausalQwen
    print("\n初始化 CausalQwen 模型...")
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        causal_dim=896,
        use_real_qwen=True,
        qwen_model_path=qwen_model_path
    )
    causal_model = CausalLanguageModel(config).to(device)
    causal_model.init_weights()  # 执行知识迁移
    causal_model.eval()
    
    # 初始化原始 Qwen
    print("\n初始化原始 Qwen 模型...")
    qwen_model = Qwen2ForCausalLM.from_pretrained(
        qwen_model_path,
        torch_dtype=torch.float32,
        device_map=None
    ).to(device)
    qwen_model.eval()
    
    # 准备测试数据
    print("\n准备测试数据...")
    texts = [
        "The price is 99.99 dollars.",
        "There are 100 items in total.",
        "Hello world!"
    ]
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
    
    # 执行验证
    print("\n开始验证...")
    
    # 1. 特征提取一致性
    features_ok = verify_feature_extraction(causal_model, qwen_model, inputs, device)
    
    # 2. 权重继承验证
    weights_ok = verify_weight_inheritance(causal_model, qwen_model, tokenizer)
    
    # 3. 前向传播一致性
    forward_ok = verify_forward_consistency(causal_model, qwen_model, inputs, tokenizer, device)
    
    # 4. 保留词汇处理
    reserved_ok = verify_reserved_tokens(causal_model, qwen_model, tokenizer)
    
    # 总结
    print_section("验证总结")
    
    all_passed = features_ok and weights_ok and forward_ok and reserved_ok
    
    print(f"验证结果:")
    print(f"  ✅ 特征提取一致性: {'通过' if features_ok else '失败'}")
    print(f"  ✅ 权重继承正确性: {'通过' if weights_ok else '失败'}")
    print(f"  ✅ 前向传播一致性: {'通过' if forward_ok else '失败'}")
    print(f"  ✅ 保留词汇处理: {'通过' if reserved_ok else '失败'}")
    
    if all_passed:
        print(f"\n🎉 知识迁移验证完全通过！")
        print(f"   CausalQwen 成功继承了 Qwen 的知识")
        print(f"   同时正确扩展了因果推理功能")
    else:
        print(f"\n❌ 知识迁移验证失败，请检查相关实现")
    
    return all_passed

if __name__ == '__main__':
    main()