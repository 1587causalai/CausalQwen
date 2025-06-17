#!/usr/bin/env python3
"""
CausalQwen与Qwen简单集成测试
最小可运行的端到端验证

测试目标：输入"你好"，验证compatible模式下的数值一致性

Author: CausalQwen Team
Date: 2024-01-16
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 确保src目录在Python路径中
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_simple_integration():
    """简单集成测试"""
    
    print("🚀 开始CausalQwen简单集成测试")
    print("目标：验证'你好'输入的数值一致性")
    print("=" * 50)
    
    try:
        # 1. 导入模块
        print("📦 导入模块...")
        from transformers import Qwen2ForCausalLM, AutoTokenizer
        from causal_qwen_mvp.models import CausalQwenMVPForCausalLM
        from causal_qwen_mvp.models import CausalQwen2Config
        
        # 2. 加载Qwen模型
        print("📥 加载原始Qwen2.5-0.5B...")
        qwen_model_path = "~/models/Qwen2.5-0.5B"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
            qwen_model = Qwen2ForCausalLM.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            print(f"✅ Qwen模型加载成功，参数量: {sum(p.numel() for p in qwen_model.parameters()):,}")
        except Exception as e:
            print(f"❌ Qwen模型加载失败: {e}")
            print("💡 请确保 ~/models/Qwen2.5-0.5B 路径存在")
            return
        
        # 3. 创建CausalQwen模型
        print("🔧 创建CausalQwen模型...")
        try:
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
        except Exception as e:
            print(f"❌ CausalQwen模型创建失败: {e}")
            return
        
        # 4. 关键权重映射（简化版）
        print("🔄 执行关键权重映射...")
        try:
            with torch.no_grad():
                # 映射词嵌入
                causal_qwen_model.qwen_model.embed_tokens.weight.copy_(
                    qwen_model.model.embed_tokens.weight
                )
                
                # 映射lm_head到action_network.linear（关键映射！）
                causal_qwen_model.action_network.linear.weight.copy_(
                    qwen_model.lm_head.weight
                )
                
                # 映射norm层
                causal_qwen_model.qwen_model.norm.weight.copy_(
                    qwen_model.model.norm.weight
                )
                
                # 映射Transformer层（简化：只复制能匹配的）
                for i, (qwen_layer, causal_layer) in enumerate(
                    zip(qwen_model.model.layers, causal_qwen_model.qwen_model.layers)
                ):
                    try:
                        # 尝试直接复制state_dict
                        causal_layer.load_state_dict(qwen_layer.state_dict(), strict=False)
                    except Exception as layer_e:
                        print(f"  ⚠️ 第{i}层映射失败，跳过: {layer_e}")
                        continue
                
            print("✅ 关键权重映射完成")
        except Exception as e:
            print(f"❌ 权重映射失败: {e}")
            return
        
        # 5. 执行简单测试
        print("\n🧪 执行'你好'一致性测试...")
        test_input = "你好"
        
        # 准备输入
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        print(f"输入tokens: {input_ids}")
        
        with torch.no_grad():
            # Qwen输出
            qwen_outputs = qwen_model(input_ids)
            qwen_logits = qwen_outputs.logits[:, -1, :]  # 最后一个token的logits
            
            # CausalQwen输出
            causal_outputs = causal_qwen_model(input_ids)
            causal_logits = causal_outputs.loc_S[:, -1, :]  # 在compatible模式下应该等价
            
            # 计算差异
            max_diff = torch.max(torch.abs(qwen_logits - causal_logits)).item()
            mean_diff = torch.mean(torch.abs(qwen_logits - causal_logits)).item()
            relative_error = torch.mean(
                torch.abs(qwen_logits - causal_logits) / (torch.abs(qwen_logits) + 1e-8)
            ).item()
            
            print(f"\n📊 数值对比结果:")
            print(f"  最大差异: {max_diff:.8f}")
            print(f"  平均差异: {mean_diff:.8f}")
            print(f"  相对误差: {relative_error:.8f}")
            
            # 显示前5个logits作为样本
            print(f"\n📋 前5个logits对比:")
            print(f"  Qwen:      {qwen_logits[0, :5].numpy()}")
            print(f"  CausalQwen: {causal_logits[0, :5].numpy()}")
            
            # 判断成功标准
            success_threshold = 1e-4
            is_success = max_diff < success_threshold
            
            print(f"\n🎯 测试结果:")
            if is_success:
                print(f"  ✅ 成功！数值差异 ({max_diff:.8f}) < 阈值 ({success_threshold})")
                print("  🎉 CausalQwen的compatible模式基本与Qwen一致！")
            else:
                print(f"  ❌ 失败！数值差异 ({max_diff:.8f}) >= 阈值 ({success_threshold})")
                print("  🔧 需要检查权重映射或模型结构")
        
        # 6. 简单的生成测试
        print(f"\n🗣️ 简单生成对比测试...")
        try:
            # Qwen生成
            qwen_generated = qwen_model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            qwen_text = tokenizer.decode(qwen_generated[0], skip_special_tokens=True)
            
            # CausalQwen compatible生成
            causal_generated = causal_qwen_model.generate(
                input_ids,
                max_new_tokens=3,
                causal_mode='compatible',
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            causal_text = tokenizer.decode(causal_generated[0], skip_special_tokens=True)
            
            print(f"  Qwen输出:      '{qwen_text}'")
            print(f"  CausalQwen输出: '{causal_text}'")
            
            if qwen_text == causal_text:
                print("  ✅ 生成输出完全一致！")
            else:
                print("  ⚠️ 生成输出有差异，但这可能正常（因为权重映射不完整）")
                
        except Exception as e:
            print(f"  ⚠️ 生成测试失败: {e}")
        
        print(f"\n" + "=" * 50)
        print("🏆 简单集成测试完成！")
        if is_success:
            print("📋 下一步建议：完善权重映射，测试更多样例")
        else:
            print("📋 下一步建议：调试权重映射逻辑，检查模型结构")
            
    except Exception as e:
        print(f"❌ 测试过程失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_integration() 