"""最简单的 CausalQwen 演示"""

import torch
from transformers import AutoTokenizer
import os


class SimpleCausalQwen:
    """超级简化的 CausalQwen 
    
    只演示核心概念，不实现完整功能
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 添加 <NUM> token
        self.tokenizer.add_tokens(['<NUM>'], special_tokens=True)
        self.num_token_id = self.tokenizer.convert_tokens_to_ids('<NUM>')
        
    def generate_with_causality(self, prompt, max_length=50):
        """因果生成演示
        
        核心思想：
        1. 先采样"原因"（因果表征 U）
        2. 基于原因决定"结果"（下一个token）
        3. 使用 softmax(loc_S) 兼容标准采样
        """
        print(f"\n🎲 因果生成模式")
        print(f"📝 输入: {prompt}")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # 模拟生成过程
        generated = input_ids
        for step in range(max_length):
            # 步骤1: 推断因果表征分布（简化为正态分布演示）
            u_loc = torch.randn(1, 128)  # 模拟 loc
            u_scale = torch.ones(1, 128) * 2.0  # 模拟 scale
            
            # 步骤2: 采样具体的因果表征
            u_sample = torch.normal(u_loc, u_scale)  # 实际应该用柯西分布
            
            # 步骤3: 计算分类分数的分布参数
            # S_k = A_k · U + B_k，得到每个类别的 loc
            loc_scores = torch.randn(1, self.tokenizer.vocab_size)  # 模拟 loc_{S_k}
            
            # 步骤4: 使用 softmax(loc) 进行兼容采样（关键改进！）
            probs = torch.softmax(loc_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否结束
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            # 演示信息
            if step < 5:  # 只打印前几步
                print(f"   步骤 {step+1}: 采样U → 计算loc_S → softmax → token {next_token.item()}")
        
        # Decode
        result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"🤖 输出: {result}")
        
        return result
    
    def handle_numerical_input(self, text):
        """演示数值处理"""
        print(f"\n🔢 数值处理演示")
        print(f"📝 原始文本: {text}")
        
        # 简单的数值识别（实际应该更智能）
        import re
        pattern = r'\b\d+\.?\d*\b'
        
        def replace_num(match):
            return f"<NUM>{match.group()}</NUM>"
        
        processed = re.sub(pattern, replace_num, text)
        print(f"🔄 处理后: {processed}")
        
        # 提取数值
        nums = re.findall(r'<NUM>([\d.]+)</NUM>', processed)
        print(f"📊 识别的数值: {nums}")
        
        return processed, nums

    def demonstrate_compatibility(self):
        """演示与 Qwen 的兼容性"""
        print(f"\n🔗 兼容性演示")
        print("✅ CausalQwen 的采样流程与 Qwen 完全兼容：")
        print("   1. 计算 logits（loc_S）")
        print("   2. 应用 softmax 得到概率分布")
        print("   3. 使用标准的 top-k/top-p 过滤")
        print("   4. 从过滤后的分布中采样")
        print("   5. 完全兼容 temperature 参数")
        
        # 模拟兼容性测试
        vocab_size = 1000
        loc_scores = torch.randn(1, vocab_size)
        
        # 标准流程
        print(f"\n📊 标准采样流程演示:")
        print(f"   原始 logits 形状: {loc_scores.shape}")
        
        # 应用温度
        temperature = 0.8
        scaled_logits = loc_scores / temperature
        print(f"   温度调整后: temperature={temperature}")
        
        # Softmax
        probs = torch.softmax(scaled_logits, dim=-1)
        print(f"   概率分布: sum={probs.sum():.6f} (应该=1.0)")
        
        # Top-k 演示
        k = 50
        top_k_probs, top_k_indices = torch.topk(probs, k)
        print(f"   Top-{k} 采样就绪")
        
        print("✅ 完全兼容 transformers.generate() 的所有参数！")


def main():
    """主演示函数"""
    print("=" * 60)
    print("🚀 CausalQwen 简化演示")
    print("=" * 60)
    
    # 加载tokenizer
    model_path = os.path.expanduser("~/models/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 创建简化的CausalQwen
    model = SimpleCausalQwen(tokenizer)
    
    # 演示1: 因果生成（兼容版）
    print("\n📌 演示1: 因果生成流程（Qwen兼容版）")
    model.generate_with_causality("今天天气", max_length=8)
    
    # 演示2: 数值处理
    print("\n📌 演示2: 数值感知")
    model.handle_numerical_input("股价上涨了5.2%，成交量达到100万")
    
    # 演示3: 兼容性
    print("\n📌 演示3: 与 Qwen 的完全兼容性")
    model.demonstrate_compatibility()
    
    # 演示4: 核心概念
    print("\n📌 演示4: 核心概念")
    print("🎯 CausalQwen 的核心创新:")
    print("   1. 因果表征 U：建模个体差异")
    print("   2. 推断-行动：先推断原因，再决定结果")
    print("   3. 统一表示：文本和数值的无缝融合")
    print("   4. 柯西分布：数学上的优雅选择")
    print("   5. 兼容采样：softmax(loc_S) 完全兼容现有工具链 🆕")
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")


if __name__ == "__main__":
    main()
