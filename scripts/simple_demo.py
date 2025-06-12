"""CausalQwen 简单演示"""

import torch
from transformers import AutoTokenizer
import os


class SimpleMockCausalQwen:
    """简化的CausalQwen演示版本"""
    
    def __init__(self):
        # 加载tokenizer - 使用绝对路径
        model_path = os.path.expanduser("~/models/Qwen2.5-0.5B")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 30,
        sampling_mode: str = "causal"
    ) -> str:
        """简化的文本生成（演示用）"""
        
        # 1. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']
        
        print(f"📝 输入tokens: {input_ids.shape}")
        print(f"🔧 采样模式: {sampling_mode}")
        
        # 2. 模拟生成过程
        generated_ids = input_ids.clone()
        
        print("🔄 开始生成...")
        for step in range(max_new_tokens):
            if sampling_mode == "causal":
                # 模拟因果采样：采样U -> 计算分数 -> 选择token
                print(f"  步骤 {step+1}: 因果采样 -> 选择原因 -> 观察结果")
                next_token = torch.randint(1000, 30000, (1, 1))  # 随机模拟
            else:
                # 模拟传统采样
                print(f"  步骤 {step+1}: 传统采样 -> 概率分布 -> 随机选择")
                next_token = torch.randint(1000, 30000, (1, 1))  # 随机模拟
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # 模拟EOS停止
            if torch.rand(1) < 0.1:  # 10%概率停止
                print(f"  🛑 在步骤 {step+1} 遇到EOS，停止生成")
                break
        
        # 3. Decode
        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print("✅ 生成完成！")
        return result


def demo():
    """演示函数"""
    print("🚀 CausalQwen 简单演示")
    print("=" * 60)
    
    # 创建模型
    model = SimpleMockCausalQwen()
    
    # 测试用例
    test_cases = [
        "你好呀！",
        "今天天气怎么样？",
        "请告诉我一个笑话",
        "Hello, how are you?"
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}: {prompt}")
        print("-" * 40)
        
        # 因果采样
        print("🎲 因果采样模式:")
        result1 = model.generate_text(prompt, max_new_tokens=10, sampling_mode="causal")
        print(f"结果: {result1}\n")
        
        # 传统采样
        print("🎯 传统采样模式:")
        result2 = model.generate_text(prompt, max_new_tokens=10, sampling_mode="traditional") 
        print(f"结果: {result2}")
        
        print("=" * 60)


if __name__ == "__main__":
    demo()
