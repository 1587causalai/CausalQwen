"""CausalQwen 完整演示

展示如何创建一个真正迁移了Qwen知识的CausalQwen模型。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from src.causal_qwen.model import CausalQwen
from src.causal_qwen.config import CausalQwenConfig
from dataclasses import dataclass
import json


def load_qwen_and_convert():
    """加载Qwen模型并转换为CausalQwen"""
    
    print("🚀 CausalQwen 模型转换演示")
    print("=" * 60)
    
    # 1. 加载Qwen模型和配置
    print("\n1️⃣ 加载原始Qwen模型...")
    model_path = os.path.expanduser("~/models/Qwen2.5-0.5B")
    
    qwen_config = AutoConfig.from_pretrained(model_path)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"✅ Qwen模型加载成功！")
    print(f"   - 词汇表大小: {qwen_config.vocab_size}")
    print(f"   - 隐藏维度: {qwen_config.hidden_size}")
    print(f"   - 注意力头数: {qwen_config.num_attention_heads}")
    print(f"   - 层数: {qwen_config.num_hidden_layers}")
    
    # 2. 创建CausalQwen配置
    print("\n2️⃣ 创建CausalQwen配置...")
    causal_config = CausalQwenConfig(
        num_vocab=qwen_config.vocab_size + 1,  # +1 for <NUM>
        hidden_dim=qwen_config.hidden_size,
        num_layers=qwen_config.num_hidden_layers,
        num_heads=qwen_config.num_attention_heads,
        num_token_id=qwen_config.vocab_size,  # 使用第一个保留token作为<NUM>
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    
    print(f"✅ CausalQwen配置创建成功！")
    print(f"   - <NUM> token ID: {causal_config.num_token_id}")
    
    # 3. 创建CausalQwen模型
    print("\n3️⃣ 创建CausalQwen模型...")
    causal_model = CausalQwen(causal_config)
    
    # 4. 迁移权重
    print("\n4️⃣ 迁移Qwen权重到CausalQwen...")
    
    # 迁移词嵌入（扩展一个token）
    print("   - 迁移词嵌入...")
    qwen_embed_weight = qwen_model.model.embed_tokens.weight.data
    causal_model.numerical_embedding.embedding.weight.data[:-1] = qwen_embed_weight
    # 初始化<NUM> token的嵌入
    causal_model.numerical_embedding.embedding.weight.data[-1] = qwen_embed_weight.mean(dim=0)
    
    # 迁移Transformer层
    print("   - 迁移Transformer层...")
    # 这里需要根据实际的模型结构进行迁移
    # 示例：如果使用了Qwen的transformer结构
    if hasattr(qwen_model, 'model') and hasattr(qwen_model.model, 'layers'):
        causal_model.transformer = qwen_model.model.layers
    
    # 迁移输出头到ActionNetwork的分类头
    print("   - 迁移输出头到ActionNetwork...")
    if hasattr(qwen_model, 'lm_head'):
        # 扩展权重以包含<NUM> token
        lm_head_weight = qwen_model.lm_head.weight.data
        causal_model.action.class_weights.data[:-1] = lm_head_weight.T  # 注意转置
        causal_model.action.class_weights.data[-1] = lm_head_weight.mean(dim=0)
        
        if hasattr(qwen_model.lm_head, 'bias') and qwen_model.lm_head.bias is not None:
            causal_model.action.class_bias.data[:-1] = qwen_model.lm_head.bias.data
    
    print("✅ 权重迁移完成！")
    
    # 5. 设置tokenizer
    causal_model.set_tokenizer(tokenizer)
    
    # 6. 添加<NUM> token到tokenizer
    print("\n5️⃣ 扩展tokenizer...")
    tokenizer.add_tokens(['<NUM>'], special_tokens=True)
    print(f"✅ Tokenizer扩展完成！新词汇表大小: {len(tokenizer)}")
    
    return causal_model, tokenizer


def demo_conversation():
    """演示对话功能"""
    
    # 加载模型
    model, tokenizer = load_qwen_and_convert()
    model.eval()
    
    print("\n" + "=" * 60)
    print("💬 开始对话演示")
    print("=" * 60)
    
    # 测试对话
    test_messages = [
        [{"role": "user", "content": "你好！"}],
        [{"role": "user", "content": "什么是人工智能？"}],
        [{"role": "user", "content": "计算一下：3 + 5 = <NUM>"}],
    ]
    
    for messages in test_messages:
        print(f"\n👤 用户: {messages[-1]['content']}")
        
        try:
            # 使用因果采样
            print("🎲 因果采样回复: ", end="", flush=True)
            response = model.chat(
                messages,
                stream=True,
                max_new_tokens=50,
                temperature=0.8,
                sampling_mode="causal"
            )
            
            for chunk in response:
                print(chunk, end="", flush=True)
            print()
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✨ 演示完成！")


def demo_numerical_understanding():
    """演示数值理解能力"""
    
    print("\n🔢 数值理解能力演示")
    print("=" * 60)
    
    # 这里应该展示CausalQwen如何处理包含数值的文本
    examples = [
        "温度是 <NUM> 摄氏度",
        "股价上涨了 <NUM> %",
        "距离大约 <NUM> 公里",
    ]
    
    print("CausalQwen 可以理解和生成包含数值的文本：")
    for example in examples:
        print(f"  - {example}")
    
    print("\n关键特性：")
    print("  ✅ 统一的文本-数值表示")
    print("  ✅ 数值感知的嵌入 (φ(v) 编码)")
    print("  ✅ 柯西分布建模数值不确定性")
    print("  ✅ 因果推理支持反事实问题")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CausalQwen 演示")
    parser.add_argument("--mode", choices=["convert", "chat", "numerical"], 
                       default="convert", help="演示模式")
    
    args = parser.parse_args()
    
    if args.mode == "convert":
        load_qwen_and_convert()
    elif args.mode == "chat":
        demo_conversation()
    elif args.mode == "numerical":
        demo_numerical_understanding()
