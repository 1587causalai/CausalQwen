#!/usr/bin/env python
"""
诊断数值提取功能
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tokenizer import QwenTokenizerWrapper

def main():
    print("=== 数值提取诊断 ===\n")
    
    # 初始化分词器
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    
    # 更全面的测试文本
    test_texts = [
        # 基础数值
        "The price is 99.99 dollars.",
        "There are 100 items.",
        "Temperature is -5.2 degrees.",
        "Growth rate: 15.7%",
        
        # 多个数值
        "I bought 3 apples for 2.50 each, total is 7.50.",
        "From 100 items, 25 were sold, 75 remain.",
        
        # 数值运算场景
        "Calculate: 10 + 20 = 30",
        "The discount is 20%, so 100 becomes 80.",
        "Average of 10, 20, 30 is 20.",
        
        # 特殊格式
        "Pi is approximately 3.14159",
        "The year 2024 has 366 days.",
        "My phone number is 123-456-7890.",
        
        # 科学记数法
        "The distance is 1.5e6 meters.",
        "Avogadro's number is 6.022e23.",
        
        # 混合文本
        "In Q1 2024, revenue was $1.2M, up 15% from $1.04M in Q4 2023.",
        
        # 纯文本对照
        "Hello world!",
        "This is a test without numbers."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: '{text}'")
        print(f"{'='*60}")
        
        # 编码
        encoded = tokenizer(text, return_tensors='pt')
        
        # 获取 tokens 和数值
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist())
        token_ids = encoded['input_ids'][0].tolist()
        numerical_values = encoded['numerical_values'][0].tolist()
        
        # 打印详细的 token 分析
        print("\nToken 详细分析:")
        print(f"{'位置':<6} {'Token':<20} {'ID':<8} {'数值':<10}")
        print("-" * 50)
        
        num_token_count = 0
        for j, (token, token_id, value) in enumerate(zip(tokens, token_ids, numerical_values)):
            # 高亮显示 <NUM> token
            if token_id == tokenizer.num_token_id:
                print(f"{j:<6} {'<NUM>':<20} {token_id:<8} {value:<10.4f} ⭐")
                num_token_count += 1
            else:
                value_str = f"{value:.4f}" if value != 0.0 else "-"
                print(f"{j:<6} {token:<20} {token_id:<8} {value_str:<10}")
        
        # 统计信息
        non_zero_count = sum(1 for v in numerical_values if v != 0.0)
        print(f"\n统计:")
        print(f"  - Token 总数: {len(tokens)}")
        print(f"  - <NUM> token 数量: {num_token_count}")
        print(f"  - 非零数值数量: {non_zero_count}")
        
        # 检查是否正确提取了数字
        import re
        numbers_in_text = re.findall(r'-?\d+\.?\d*%?', text)
        if numbers_in_text:
            print(f"  - 文本中的数字: {numbers_in_text}")
            if num_token_count == 0:
                print(f"  ⚠️  警告: 文本中有 {len(numbers_in_text)} 个数字但没有 <NUM> token!")
            elif num_token_count != len(numbers_in_text):
                print(f"  ⚠️  警告: 数字数量 ({len(numbers_in_text)}) 与 <NUM> token 数量 ({num_token_count}) 不匹配!")
            else:
                print(f"  ✅ 成功: 所有数字都被正确替换为 <NUM> token")
                
                # 验证数值是否正确
                extracted_values = [v for v, tid in zip(numerical_values, token_ids) if tid == tokenizer.num_token_id]
                print(f"  - 提取的数值: {extracted_values}")
        else:
            if num_token_count > 0:
                print(f"  ⚠️  警告: 没有数字但有 {num_token_count} 个 <NUM> token!")
            else:
                print(f"  ✅ 正确: 文本中没有数字")
    
    # 最后做一个综合测试
    print(f"\n{'='*60}")
    print("综合功能测试")
    print(f"{'='*60}")
    
    # 测试数值提取的正确性
    test_cases = [
        ("100", 100.0),
        ("-5.2", -5.2),
        ("15.7%", 0.157),  # 百分号应该转换为小数
        ("3.14159", 3.14159),
        ("1.5e6", 1.5e6),
    ]
    
    print("\n数值提取准确性测试:")
    all_correct = True
    for text, expected in test_cases:
        encoded = tokenizer(f"The value is {text}.", return_tensors='pt')
        token_ids = encoded['input_ids'][0].tolist()
        numerical_values = encoded['numerical_values'][0].tolist()
        
        # 找到 <NUM> token 的位置
        num_positions = [i for i, tid in enumerate(token_ids) if tid == tokenizer.num_token_id]
        
        if num_positions:
            extracted = numerical_values[num_positions[0]]
            is_correct = abs(extracted - expected) < 1e-6
            all_correct = all_correct and is_correct
            status = "✅" if is_correct else "❌"
            print(f"  {text} -> {extracted:.6f} (期望: {expected:.6f}) {status}")
        else:
            print(f"  {text} -> 未找到 <NUM> token ❌")
            all_correct = False
    
    if all_correct:
        print("\n✅ 所有数值提取测试通过！")
    else:
        print("\n❌ 数值提取存在问题，需要检查分词器实现")
    
    # 诊断建议
    print(f"\n{'='*60}")
    print("诊断建议:")
    print(f"{'='*60}")
    print("如果数值提取不工作，请检查:")
    print("1. QwenTokenizerWrapper._extract_and_replace_numbers() 方法")
    print("2. encode_plus() 方法是否正确调用了 _extract_and_replace_numbers")
    print("3. <NUM> token 是否正确添加到词汇表")
    print("4. 正则表达式是否能匹配所有数字格式")

if __name__ == '__main__':
    main()
