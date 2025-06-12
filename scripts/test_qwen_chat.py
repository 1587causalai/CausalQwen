"""测试 Qwen 模型的对话功能"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_qwen_basic():
    """测试 Qwen 的基本对话功能"""
    print("=" * 60)
    print("🧪 Qwen 基础对话测试")
    print("=" * 60)
    
    # 加载模型
    model_path = os.path.expanduser("~/models/Qwen2.5-0.5B")
    print(f"\n📂 模型路径: {model_path}")
    
    # 加载 tokenizer 和模型
    print("🔄 加载 Qwen 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    model.eval()
    
    print("✅ 模型加载成功！")
    print(f"   - 模型类型: {type(model).__name__}")
    print(f"   - 词汇表大小: {tokenizer.vocab_size}")
    print(f"   - 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 测试简单生成
    print("\n🎯 测试文本生成...")
    test_prompts = [
        "Hello, how are you?",
        "今天天气真好",
        "1 + 1 = ",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 输入: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 输出: {response}")
    
    print("\n" + "=" * 60)
    print("✅ Qwen 测试完成！")


if __name__ == "__main__":
    test_qwen_basic()
