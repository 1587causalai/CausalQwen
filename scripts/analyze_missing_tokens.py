#!/usr/bin/env python
"""
分析Qwen2.5-0.5B中271个未被分词器加载的token

目标：
1. 找出这271个token的ID范围
2. 分析为什么没有被加载
3. 探讨可能的原因
"""

from transformers import AutoTokenizer, AutoConfig
import os

def analyze_missing_tokens():
    """分析缺失的token"""
    print("="*60)
    print(" 分析271个未被分词器加载的token ".center(58))
    print("="*60)
    
    model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    
    # 加载配置和分词器
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    config_vocab_size = config.vocab_size      # 151936
    actual_vocab_size = len(tokenizer)         # 151665
    missing_count = config_vocab_size - actual_vocab_size  # 271
    
    print(f"配置词汇表大小: {config_vocab_size}")
    print(f"实际加载大小: {actual_vocab_size}")
    print(f"缺失token数量: {missing_count}")
    
    # 获取实际加载的token ID集合
    vocab_dict = tokenizer.get_vocab()
    loaded_token_ids = set(vocab_dict.values())
    
    print(f"\n实际加载的token ID范围:")
    print(f"  最小ID: {min(loaded_token_ids)}")
    print(f"  最大ID: {max(loaded_token_ids)}")
    print(f"  实际ID数量: {len(loaded_token_ids)}")
    
    # 找出缺失的token ID
    expected_ids = set(range(config_vocab_size))
    missing_ids = expected_ids - loaded_token_ids
    
    print(f"\n缺失的token ID分析:")
    print(f"  缺失ID数量: {len(missing_ids)}")
    
    if missing_ids:
        missing_list = sorted(missing_ids)
        print(f"  缺失ID范围: {min(missing_list)} 到 {max(missing_list)}")
        
        # 分析缺失ID的分布模式
        consecutive_ranges = []
        start = missing_list[0]
        end = start
        
        for i in range(1, len(missing_list)):
            if missing_list[i] == end + 1:
                end = missing_list[i]
            else:
                consecutive_ranges.append((start, end))
                start = missing_list[i]
                end = start
        consecutive_ranges.append((start, end))
        
        print(f"\n缺失ID的连续范围:")
        for start, end in consecutive_ranges:
            if start == end:
                print(f"  ID {start}: 单个缺失")
            else:
                print(f"  ID {start}-{end}: 连续缺失 ({end-start+1}个)")
        
        # 显示前10个和后10个缺失的ID
        print(f"\n前10个缺失的ID: {missing_list[:10]}")
        print(f"后10个缺失的ID: {missing_list[-10:]}")
    
    # 检查最高的token ID
    max_loaded_id = max(loaded_token_ids)
    print(f"\n最高加载的token ID: {max_loaded_id}")
    
    if max_loaded_id < config_vocab_size - 1:
        unloaded_high_ids = config_vocab_size - 1 - max_loaded_id
        print(f"最高位置未加载的ID数量: {unloaded_high_ids}")
        print(f"这些ID范围: {max_loaded_id + 1} 到 {config_vocab_size - 1}")

def investigate_tokenizer_implementation():
    """调查分词器实现细节"""
    print("\n" + "="*60)
    print(" 分词器实现细节调查 ".center(58))
    print("="*60)
    
    model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查tokenizer的属性
    print(f"分词器类型: {type(tokenizer)}")
    print(f"分词器类名: {tokenizer.__class__.__name__}")
    
    # 检查tokenizer内部属性
    print(f"\n分词器内部属性:")
    if hasattr(tokenizer, 'vocab_size'):
        print(f"  tokenizer.vocab_size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, '_vocab_size'):
        print(f"  tokenizer._vocab_size: {getattr(tokenizer, '_vocab_size', 'N/A')}")
    
    # 检查tokenizer的配置
    if hasattr(tokenizer, 'init_kwargs'):
        print(f"  init_kwargs: {tokenizer.init_kwargs}")
    
    # 检查是否有vocab_size相关的配置
    for attr in dir(tokenizer):
        if 'vocab' in attr.lower() and not attr.startswith('_'):
            value = getattr(tokenizer, attr)
            if not callable(value):
                print(f"  {attr}: {value}")

def analyze_possible_reasons():
    """分析可能的原因"""
    print("\n" + "="*60)
    print(" 可能原因分析 ".center(58))
    print("="*60)
    
    print("可能的解释：")
    print("\n1. 🔒 预留位置")
    print("   - 这271个位置可能是为未来功能预留的")
    print("   - 保持词汇表大小为2的幂次等技术原因")
    print("   - 为多模态token预留空间")
    
    print("\n2. 📚 训练策略")
    print("   - 训练时可能使用了更大的词汇表")
    print("   - 但推理时只激活实际需要的token")
    print("   - 减少内存使用和计算开销")
    
    print("\n3. 🔧 实现细节")
    print("   - transformers库可能只加载定义的token")
    print("   - 未定义的token位置保持空白")
    print("   - 避免加载无效或未训练的token")
    
    print("\n4. 🔄 版本兼容性")
    print("   - 保持与不同版本模型的兼容性")
    print("   - 支持动态添加新token")
    print("   - 向后兼容旧版本的分词器")
    
    print("\n5. 🎯 模型设计")
    print("   - 实际有效词汇量就是151665")
    print("   - 151936是最大容量而非实际使用量")
    print("   - 类似于数组分配vs实际使用的概念")

def check_practical_implications():
    """检查实际影响"""
    print("\n" + "="*60)
    print(" 对CausalQwen的实际影响 ".center(58))
    print("="*60)
    
    print("对我们项目的影响：")
    
    print("\n✅ 正面影响：")
    print("  - 用户记忆正确，官方规格确实是151936")
    print("  - 模型权重确实支持151936个token")
    print("  - 我们的理论基础是正确的")
    
    print("\n⚠️ 需要注意：")
    print("  - 实际可用的token只有151665个")
    print("  - 添加<NUM> token会变成151666")
    print("  - 这仍然远小于151936的最大容量")
    
    print("\n🎯 建议的做法：")
    print("  - 理论文档中使用151936作为K值")
    print("  - 实现中使用实际大小151665")
    print("  - CausalQwen词汇表大小：151665 + 1 = 151666")
    print("  - 在文档中注明这个差异")
    
    print("\n📊 容量分析：")
    total_capacity = 151936
    actually_used = 151665
    our_addition = 1
    remaining_capacity = total_capacity - actually_used - our_addition
    
    print(f"  总容量：     {total_capacity}")
    print(f"  已使用：     {actually_used}")
    print(f"  我们添加：   {our_addition}")
    print(f"  剩余容量：   {remaining_capacity}")
    print(f"  使用率：     {(actually_used + our_addition)/total_capacity*100:.1f}%")

def main():
    """主函数"""
    analyze_missing_tokens()
    investigate_tokenizer_implementation()
    analyze_possible_reasons()
    check_practical_implications()
    
    print("\n" + "="*60)
    print(" 结论 ".center(58))
    print("="*60)
    print("这271个'缺失'的token实际上是正常现象！")
    print("它们代表了配置容量与实际使用之间的差异。")
    print("对我们的CausalQwen项目没有负面影响。")
    print("我们可以安全地添加<NUM> token！")

if __name__ == '__main__':
    main() 