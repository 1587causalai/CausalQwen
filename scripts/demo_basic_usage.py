#!/usr/bin/env python3
"""
CausalQwen 简单演示 - 完全兼容Qwen接口

演示CausalQwen的核心功能：
1. 与Qwen完全相同的使用方式
2. do_sample参数控制V2核心行为
3. 完整的生成能力展示
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🚀 CausalQwen 简单演示 - 与Qwen完全兼容")
    print("="*50)
    
    # 1. 创建模型（与Qwen相同方式）
    print("\n📦 创建CausalQwen模型...")
    try:
        from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config
        
        # 小型演示配置
        config = CausalQwen2Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            causal_size=64  # CausalQwen特有参数
        )
        
        model = CausalQwenMVPForCausalLM(config)
        print("✅ 模型创建成功")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 2. 准备输入
    print("\n🔤 准备测试输入...")
    input_ids = torch.randint(0, 1000, (1, 8))
    print(f"输入序列: {input_ids.tolist()}")
    
    # 3. 确定性生成（与Qwen相同）
    print("\n🎯 确定性生成 (do_sample=False)")
    print("   V2原理: 噪声影响尺度参数，增加决策不确定性")
    
    try:
        det_output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,  # 关键参数
            temperature=1.0   # 确定性模式下无效
        )
        
        new_tokens = det_output[0, input_ids.shape[1]:].tolist()
        print(f"   生成结果: {new_tokens}")
        print("✅ 确定性生成成功")
        
    except Exception as e:
        print(f"❌ 确定性生成失败: {e}")
    
    # 4. 采样生成（与Qwen相同）
    print("\n🎲 采样生成 (do_sample=True)")
    print("   V2原理: 噪声影响位置参数，扰动个体身份")
    
    try:
        samp_output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,      # 关键参数
            temperature=0.8,     # 控制随机性
            top_k=50,           # Top-K采样
            top_p=0.9           # Nucleus采样
        )
        
        new_tokens = samp_output[0, input_ids.shape[1]:].tolist()
        print(f"   生成结果: {new_tokens}")
        print("✅ 采样生成成功")
        
    except Exception as e:
        print(f"❌ 采样生成失败: {e}")
    
    # 5. 对比不同温度效果
    print("\n🌡️  温度效果对比")
    temperatures = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]  # 添加温度为零的测试
    
    for temp in temperatures:
        try:
            torch.manual_seed(42)  # 固定随机种子便于对比
            temp_output = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=True,
                temperature=temp,
                top_k=100
            )
            
            new_tokens = temp_output[0, input_ids.shape[1]:].tolist()
            print(f"   T={temp}: {new_tokens}")
            
            # 特别说明温度为零的重要性
            if temp == 0.0:
                pass
                # print("   🌡️ 温度为零是极其重要的边界条件！")
            
        except Exception as e:
            print(f"   T={temp}: 失败 - {e}")
    
    # 6. 验证V2数学原理
    print("\n🧮 V2数学原理验证")
    try:
        from causal_qwen_mvp import InferenceValidator
        
        validator = InferenceValidator(model)
        results = validator.validate_v2_principles(input_ids)
        
        pos_diff = results['position_difference'].item()
        scale_diff = results['scale_difference'].item()
        
        print(f"   位置参数差异: {pos_diff:.6f}")
        print(f"   尺度参数差异: {scale_diff:.6f}")
        
        if pos_diff > 1e-3:
            print("✅ V2数学原理验证通过")
        else:
            print("⚠️  位置参数差异较小")
            
    except Exception as e:
        print(f"❌ V2验证失败: {e}")
    
    # 7. 批量生成演示
    print("\n📦 批量生成演示")
    try:
        batch_input = torch.randint(0, 1000, (3, 6))  # 3个序列
        
        batch_output = model.generate(
            batch_input,
            max_new_tokens=4,
            do_sample=True,
            temperature=1.0,
            top_k=50
        )
        
        print("   批量输入:")
        for i, seq in enumerate(batch_input):
            print(f"     序列{i}: {seq.tolist()}")
        
        print("   批量输出:")
        for i, seq in enumerate(batch_output):
            new_part = seq[batch_input.shape[1]:].tolist()
            print(f"     序列{i}: {new_part}")
        
        print("✅ 批量生成成功")
        
    except Exception as e:
        print(f"❌ 批量生成失败: {e}")
    
    # 8. 总结
    print("\n🎉 演示完成！")
    print("="*50)
    print("CausalQwen核心特性:")
    print("├─ 完全兼容Qwen接口：generate(), do_sample, temperature等")
    print("├─ V2核心创新：位置vs尺度的精妙差异")
    print("├─ do_sample=False: 噪声影响尺度参数（确定性+不确定性）")
    print("├─ do_sample=True: 噪声影响位置参数（扰动个体身份）")
    print("└─ 完整的柯西分布数学基础")
    
    print("\n使用方法（与Qwen完全相同）:")
    print("```python")
    print("from causal_qwen_mvp import CausalQwenMVPForCausalLM")
    print("model = CausalQwenMVPForCausalLM.from_pretrained('path')")
    print("output = model.generate(input_ids, do_sample=True, temperature=0.8)")
    print("```")

if __name__ == "__main__":
    main()