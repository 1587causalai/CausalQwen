#!/usr/bin/env python
"""快速检查Qwen2.5-0.5B词汇表大小"""

from transformers import AutoTokenizer, AutoConfig
import os

def main():
    model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    print(f'模型路径: {model_path}')
    
    # 检查配置
    print(f'\n=== 模型配置检查 ===')
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f'config.vocab_size: {config.vocab_size}')
    
    # 检查分词器
    print(f'\n=== 分词器检查 ===')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f'len(tokenizer): {len(tokenizer)}')
    print(f'tokenizer.vocab_size: {tokenizer.vocab_size}')
    print(f'实际词汇表大小: {len(tokenizer.get_vocab())}')
    
    # 检查是否有差异
    methods = {
        'len(tokenizer)': len(tokenizer),
        'tokenizer.vocab_size': tokenizer.vocab_size,
        'len(get_vocab())': len(tokenizer.get_vocab()),
        'config.vocab_size': config.vocab_size
    }
    
    print(f'\n=== 各种方法对比 ===')
    for method, size in methods.items():
        print(f'{method:<20}: {size}')
    
    # 检查是否一致
    unique_sizes = set(methods.values())
    print(f'\n=== 一致性检查 ===')
    print(f'唯一值数量: {len(unique_sizes)}')
    print(f'唯一值: {sorted(unique_sizes)}')
    
    if len(unique_sizes) == 1:
        print('✅ 所有方法结果一致')
        print(f'Qwen2.5-0.5B词汇表大小: {list(unique_sizes)[0]}')
    else:
        print('❌ 不同方法给出不同结果')
        print('需要进一步调查原因')

if __name__ == '__main__':
    main() 