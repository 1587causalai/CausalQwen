#!/usr/bin/env python3
"""
CausalQwen与Qwen简单集成测试 - 修复版
测试目标：输入"你好"，验证compatible模式下的数值一致性
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    print("🚀 开始CausalQwen简单集成测试")
    print("目标：验证'你好'输入的数值一致性")
    print("=" * 50)
    
    try:
        # 导入模块
        print("📦 导入模块...")
        from transformers import Qwen2ForCausalLM, AutoTokenizer
        from causal_qwen_mvp.models import CausalQwenMVPForCausalLM, CausalQwen2Config
        print("✅ 模块导入成功")
        
        # 加载Qwen模型
        print("📥 加载原始Qwen2.5-0.5B...")
        qwen_model_path = Path("~/models/Qwen2.5-0.5B").expanduser()
        
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        qwen_model = Qwen2ForCausalLM.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print(f"✅ Qwen模型加载成功，参数量: {sum(p.numel() for p in qwen_model.parameters()):,}")
        
        # 创建CausalQwen模型
        print("🔧 创建CausalQwen模型...")
        causal_config = CausalQwen2Config(
            vocab_size=qwen_model.config.vocab_size,
            hidden_size=qwen_model.config.hidden_size,
            intermediate_size=qwen_model.config.intermediate_size,
            num_hidden_layers=qwen_model.config.num_hidden_layers,
            num_attention_heads=qwen_model.config.num_attention_heads,
            num_key_value_heads=getattr(qwen_model.config, 'num_key_value_heads', qwen_model.config.num_attention_heads),
            max_position_embeddings=qwen_model.config.max_position_embeddings,
            individual_dim=128,
            num_individuals=16
        )
        
        causal_qwen_model = CausalQwenMVPForCausalLM(causal_config)
        print(f"✅ CausalQwen模型创建成功，参数量: {sum(p.numel() for p in causal_qwen_model.parameters()):,}")
        
        # 关键权重映射
        print("🔄 执行关键权重映射...")
        with torch.no_grad():
            # 映射词嵌入
            causal_qwen_model.qwen_model.embed_tokens.weight.copy_(
                qwen_model.model.embed_tokens.weight
            )
            
            # 映射lm_head到action_network.linear（关键！）
            causal_qwen_model.action_network.linear.weight.copy_(
                qwen_model.lm_head.weight
            )
            
            # 映射norm层
            causal_qwen_model.qwen_model.norm.weight.copy_(
                qwen_model.model.norm.weight
            )
        print("✅ 关键权重映射完成")
        
        # 执行测试
        print("\n🧪 执行'你好'一致性测试...")
        test_input = "你好"
        
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        print(f"输入tokens: {input_ids}")
        
        with torch.no_grad():
            # Qwen输出
            qwen_outputs = qwen_model(input_ids)
            qwen_logits = qwen_outputs.logits[:, -1, :]
            
            # CausalQwen输出
            causal_outputs = causal_qwen_model(input_ids)
            causal_logits = causal_outputs.loc_S[:, -1, :]
            
            # 计算差异
            max_diff = torch.max(torch.abs(qwen_logits - causal_logits)).item()
            mean_diff = torch.mean(torch.abs(qwen_logits - causal_logits)).item()
            
            print(f"\n📊 数值对比结果:")
            print(f"  最大差异: {max_diff:.8f}")
            print(f"  平均差异: {mean_diff:.8f}")
            
            print(f"\n📋 前5个logits对比:")
            print(f"  Qwen:      {qwen_logits[0, :5].numpy()}")
            print(f"  CausalQwen: {causal_logits[0, :5].numpy()}")
            
            # 判断成功
            success = max_diff < 1e-4
            print(f"\n🎯 测试结果:")
            if success:
                print(f"  ✅ 成功！数值差异很小 ({max_diff:.8f})")
            else:
                print(f"  ❌ 失败！数值差异较大 ({max_diff:.8f})")
        
        print(f"\n🏆 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 