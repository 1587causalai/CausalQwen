#!/usr/bin/env python
"""
深度调查Qwen2.5-0.5B词汇表大小差异

目标：
1. 找出为什么config.vocab_size=151936，但实际加载的分词器只有151665个token
2. 分析特殊token的处理
3. 检查可能的版本或配置问题
"""

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os
import json

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title.center(56)} ")
    print(f"{'='*60}")

def investigate_config_files(model_path):
    """调查模型配置文件"""
    print_separator("配置文件调查")
    
    config_path = os.path.join(model_path, 'config.json')
    tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
    
    print(f"模型路径: {model_path}")
    print(f"配置文件存在: {os.path.exists(config_path)}")
    print(f"分词器配置存在: {os.path.exists(tokenizer_config_path)}")
    
    # 读取config.json
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        print(f"\nconfig.json中的vocab_size: {config_data.get('vocab_size')}")
        print(f"model_type: {config_data.get('model_type')}")
        print(f"架构: {config_data.get('architectures')}")
    
    # 读取tokenizer_config.json
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r') as f:
            tokenizer_config = json.load(f)
        print(f"\ntokenizer_config.json信息:")
        print(f"  tokenizer_class: {tokenizer_config.get('tokenizer_class')}")
        print(f"  vocab_size: {tokenizer_config.get('vocab_size')}")
        print(f"  model_max_length: {tokenizer_config.get('model_max_length')}")

def investigate_special_tokens(tokenizer):
    """调查特殊token"""
    print_separator("特殊Token调查")
    
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': tokenizer.bos_token,
        'unk_token': tokenizer.unk_token,
        'sep_token': tokenizer.sep_token,
        'cls_token': tokenizer.cls_token,
        'mask_token': tokenizer.mask_token,
    }
    
    print(f"特殊token信息:")
    for name, token in special_tokens.items():
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {name:<12}: '{token}' (ID: {token_id})")
        else:
            print(f"  {name:<12}: None")
    
    # 检查所有特殊token
    special_tokens_dict = tokenizer.special_tokens_map
    print(f"\n所有特殊token映射:")
    for key, value in special_tokens_dict.items():
        if isinstance(value, str):
            token_id = tokenizer.convert_tokens_to_ids(value)
            print(f"  {key:<15}: '{value}' (ID: {token_id})")
        else:
            print(f"  {key:<15}: {value}")
    
    # 检查added_tokens
    added_tokens = tokenizer.added_tokens_decoder
    print(f"\n新增token (共{len(added_tokens)}个):")
    for token_id, token_obj in sorted(added_tokens.items()):
        print(f"  ID {token_id}: '{token_obj}'")

def investigate_vocab_size_calculation(tokenizer):
    """调查词汇表大小计算方式"""
    print_separator("词汇表大小计算调查")
    
    # 获取各种大小
    vocab_dict = tokenizer.get_vocab()
    base_vocab_size = tokenizer.vocab_size  # 基础词汇表大小
    total_vocab_size = len(tokenizer)        # 总大小 (包含特殊token)
    actual_vocab_size = len(vocab_dict)      # 实际词汇表大小
    
    print(f"词汇表大小分析:")
    print(f"  base vocab_size (tokenizer.vocab_size): {base_vocab_size}")
    print(f"  total size (len(tokenizer)): {total_vocab_size}")
    print(f"  actual vocab dict size: {actual_vocab_size}")
    
    # 计算差异
    base_to_total_diff = total_vocab_size - base_vocab_size
    print(f"\n大小差异分析:")
    print(f"  总大小 - 基础大小 = {total_vocab_size} - {base_vocab_size} = {base_to_total_diff}")
    print(f"  这{base_to_total_diff}个token应该是特殊token")
    
    # 验证added_tokens数量
    added_tokens_count = len(tokenizer.added_tokens_decoder)
    print(f"  实际added_tokens数量: {added_tokens_count}")
    print(f"  数量匹配: {'✅' if base_to_total_diff == added_tokens_count else '❌'}")
    
    return base_vocab_size, total_vocab_size, actual_vocab_size

def investigate_model_weights(model_path):
    """调查模型权重"""
    print_separator("模型权重调查")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype='auto',
            device_map=None
        )
        
        lm_head_shape = model.lm_head.weight.shape
        embedding_shape = model.model.embed_tokens.weight.shape
        
        print(f"模型权重形状:")
        print(f"  lm_head.weight: {lm_head_shape}")
        print(f"  embed_tokens.weight: {embedding_shape}")
        
        print(f"\n权重一致性检查:")
        weights_consistent = (lm_head_shape[0] == embedding_shape[0])
        print(f"  lm_head和embed_tokens词汇表大小一致: {'✅' if weights_consistent else '❌'}")
        print(f"  权重中的词汇表大小: {lm_head_shape[0]}")
        
        return lm_head_shape[0]
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def compare_with_official_specs():
    """对比官方规格"""
    print_separator("官方规格对比")
    
    print(f"根据你的记忆:")
    print(f"  Qwen2.5-0.5B应该有151936个token")
    
    print(f"\n可能的解释:")
    print(f"  1. 模型文件版本不同")
    print(f"  2. 加载时进行了某种过滤")
    print(f"  3. 特殊token处理方式不同")
    print(f"  4. transformers库版本差异")

def main():
    """主函数"""
    print_separator("Qwen2.5-0.5B词汇表大小差异深度调查")
    
    model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    
    # 步骤1：调查配置文件
    investigate_config_files(model_path)
    
    # 步骤2：加载分词器和配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print_separator("基础信息总结")
    print(f"config.vocab_size: {config.vocab_size}")
    print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
    print(f"len(tokenizer): {len(tokenizer)}")
    print(f"len(tokenizer.get_vocab()): {len(tokenizer.get_vocab())}")
    
    # 步骤3：调查特殊token
    investigate_special_tokens(tokenizer)
    
    # 步骤4：调查词汇表大小计算
    base_size, total_size, actual_size = investigate_vocab_size_calculation(tokenizer)
    
    # 步骤5：调查模型权重
    weight_vocab_size = investigate_model_weights(model_path)
    
    # 步骤6：对比官方规格
    compare_with_official_specs()
    
    # 最终分析
    print_separator("最终分析")
    
    print(f"发现的所有词汇表大小:")
    sizes = {
        'config.vocab_size': config.vocab_size,
        'tokenizer.vocab_size': base_size,
        'len(tokenizer)': total_size,
        'len(get_vocab())': actual_size,
        'model weights': weight_vocab_size,
        '你的记忆': 151936
    }
    
    for name, size in sizes.items():
        print(f"  {name:<20}: {size}")
    
    # 寻找规律
    print(f"\n规律分析:")
    config_to_actual_diff = config.vocab_size - actual_size
    config_to_weight_diff = config.vocab_size - (weight_vocab_size or 0)
    
    print(f"  配置大小 - 实际大小 = {config.vocab_size} - {actual_size} = {config_to_actual_diff}")
    if weight_vocab_size:
        print(f"  配置大小 - 权重大小 = {config.vocab_size} - {weight_vocab_size} = {config_to_weight_diff}")
    
    # 可能的解释
    print(f"\n可能的解释:")
    if config_to_actual_diff == 271:  # 151936 - 151665
        print(f"  ✓ 差异为{config_to_actual_diff}，可能是某些token被过滤或未加载")
    
    print(f"\n结论:")
    print(f"  需要使用config.vocab_size ({config.vocab_size}) 作为理论值")
    print(f"  这是最接近你记忆中151936的值")
    print(f"  CausalQwen应该是 {config.vocab_size + 1} = {config.vocab_size + 1}")

if __name__ == '__main__':
    main() 