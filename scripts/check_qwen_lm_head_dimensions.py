#!/usr/bin/env python3
"""
检查Qwen模型的lm_head实际输出维度

这个脚本将直接加载Qwen模型并检查其lm_head的权重形状，
以确定到底是输出151,936维还是151,665维。
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_qwen_lm_head_dimensions():
    """检查Qwen模型lm_head的实际输出维度"""
    
    print("🔍 检查Qwen模型lm_head的实际输出维度")
    print("=" * 60)
    
    # 加载模型和分词器
    model_path = "~/models/Qwen2.5-0.5B"
    expanded_path = os.path.expanduser(model_path)
    
    print(f"📁 加载模型: {expanded_path}")
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(expanded_path, trust_remote_code=True)
        print(f"✅ 分词器加载成功")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            expanded_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None
        )
        print(f"✅ 模型加载成功")
        
        # 检查模型配置
        config = model.config
        print(f"\n📊 模型配置信息:")
        print(f"   config.vocab_size: {config.vocab_size}")
        print(f"   config.hidden_size: {config.hidden_size}")
        
        # 检查分词器信息
        print(f"\n📝 分词器信息:")
        print(f"   len(tokenizer): {len(tokenizer)}")
        print(f"   tokenizer.vocab_size: {tokenizer.vocab_size}")
        
        # 检查lm_head维度
        lm_head = model.lm_head
        print(f"\n🎯 lm_head权重信息:")
        print(f"   lm_head.weight.shape: {lm_head.weight.shape}")
        print(f"   lm_head输出维度: {lm_head.weight.shape[0]}")
        print(f"   lm_head输入维度: {lm_head.weight.shape[1]}")
        
        # 检查偏置
        if hasattr(lm_head, 'bias') and lm_head.bias is not None:
            print(f"   lm_head.bias.shape: {lm_head.bias.shape}")
        else:
            print(f"   lm_head.bias: None")
        
        # 对比分析
        print(f"\n🔍 维度对比分析:")
        config_vocab_size = config.vocab_size
        lm_head_output_size = lm_head.weight.shape[0]
        tokenizer_len = len(tokenizer)
        tokenizer_vocab_size = tokenizer.vocab_size
        
        print(f"   config.vocab_size:     {config_vocab_size}")
        print(f"   lm_head输出维度:       {lm_head_output_size}")
        print(f"   len(tokenizer):        {tokenizer_len}")
        print(f"   tokenizer.vocab_size:  {tokenizer_vocab_size}")
        
        # 计算差异
        config_vs_lm_head = config_vocab_size - lm_head_output_size
        config_vs_tokenizer_len = config_vocab_size - tokenizer_len
        lm_head_vs_tokenizer_len = lm_head_output_size - tokenizer_len
        
        print(f"\n📊 差异分析:")
        print(f"   config.vocab_size - lm_head输出维度: {config_vs_lm_head}")
        print(f"   config.vocab_size - len(tokenizer): {config_vs_tokenizer_len}")
        print(f"   lm_head输出维度 - len(tokenizer): {lm_head_vs_tokenizer_len}")
        
        # 结论
        print(f"\n✅ 结论:")
        if lm_head_output_size == config_vocab_size:
            print(f"   ✅ lm_head输出维度与config.vocab_size一致 ({lm_head_output_size})")
        elif lm_head_output_size == tokenizer_len:
            print(f"   ✅ lm_head输出维度与len(tokenizer)一致 ({lm_head_output_size})")
        else:
            print(f"   ❓ lm_head输出维度与配置不一致，需要进一步分析")
            
        # 测试一个简单的前向传播
        print(f"\n🧪 测试前向传播:")
        test_input = torch.tensor([[1, 2, 3]])  # 简单的测试输入
        
        with torch.no_grad():
            outputs = model(test_input)
            logits = outputs.logits
            print(f"   模型输出logits形状: {logits.shape}")
            print(f"   logits最后一维大小: {logits.shape[-1]}")
            
        # 验证logits维度是否与lm_head一致
        if logits.shape[-1] == lm_head_output_size:
            print(f"   ✅ 模型输出维度与lm_head一致")
        else:
            print(f"   ❌ 模型输出维度与lm_head不一致")
        
        return {
            'config_vocab_size': config_vocab_size,
            'lm_head_output_size': lm_head_output_size,
            'tokenizer_len': tokenizer_len,
            'tokenizer_vocab_size': tokenizer_vocab_size,
            'logits_shape': logits.shape[-1]
        }
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

if __name__ == "__main__":
    check_qwen_lm_head_dimensions() 