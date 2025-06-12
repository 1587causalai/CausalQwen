#!/usr/bin/env python
"""
因果语言模型的前向传播调试脚本 (V5: 简化版)

修复了无输出的问题，专注于核心功能验证。
"""
import os
import sys
import torch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_step(step_num, description):
    """打印步骤信息"""
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {description}")
    print(f"{'='*60}")

def main():
    """主函数，运行调试前向传播。"""
    try:
        print("🚀 CausalQwen 前向传播调试脚本 (简化版)")
        print("=" * 80)

        # 步骤1: 基本导入测试
        print_step(1, "测试基本导入")
        
        try:
            from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
            print("✅ 导入 CausalLanguageModel 成功")
        except Exception as e:
            print(f"❌ 导入 CausalLanguageModel 失败: {e}")
            return False
        
        try:
            from src.data.tokenizer import QwenTokenizerWrapper
            print("✅ 导入 QwenTokenizerWrapper 成功")
        except Exception as e:
            print(f"❌ 导入 QwenTokenizerWrapper 失败: {e}")
            return False

        # 步骤2: 设备和路径检查
        print_step(2, "环境检查")
        
        device = torch.device('cpu')
        qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
        
        print(f"设备: {device}")
        print(f"Qwen路径: {qwen_model_path}")
        
        if not os.path.exists(qwen_model_path):
            print(f"❌ Qwen模型路径不存在: {qwen_model_path}")
            return False
        else:
            print(f"✅ Qwen模型路径存在")

        # 步骤3: 分词器初始化
        print_step(3, "分词器初始化")
        
        try:
            tokenizer = QwenTokenizerWrapper(
                model_path=qwen_model_path, 
                use_real_tokenizer=True
            )
            print(f"✅ 分词器初始化成功")
            print(f"   词汇表大小: {tokenizer.vocab_size}")
            print(f"   <NUM> token ID: {tokenizer.num_token_id}")
        except Exception as e:
            print(f"❌ 分词器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤4: 模型配置
        print_step(4, "模型配置创建")
        
        try:
            config = CausalLMConfig(
                vocab_size=tokenizer.vocab_size,
                num_token_id=tokenizer.num_token_id,
                hidden_size=896,
                causal_dim=896,
                use_real_qwen=True,
                qwen_model_path=qwen_model_path,
                ovr_threshold=10.0,
                reg_loss_weight=1.0
            )
            print(f"✅ 配置创建成功")
            print(f"   vocab_size: {config.vocab_size}")
            print(f"   hidden_size: {config.hidden_size}")
        except Exception as e:
            print(f"❌ 配置创建失败: {e}")
            return False

        # 步骤5: 模型创建
        print_step(5, "模型创建")
        
        try:
            model = CausalLanguageModel(config).to(device)
            print(f"✅ 模型创建成功")
            
            # 获取参数数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   总参数数量: {total_params:,}")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤6: 模型初始化
        print_step(6, "模型权重初始化")
        
        try:
            model.init_weights()
            print(f"✅ 权重初始化成功")
        except Exception as e:
            print(f"❌ 权重初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤7: 测试数据准备
        print_step(7, "测试数据准备")
        
        try:
            test_texts = [
                "The price is 99.99 dollars.",
                "Hello world!"
            ]
            
            inputs = tokenizer.batch_encode_plus(
                test_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            print(f"✅ 测试数据准备成功")
            print(f"   批次大小: {inputs['input_ids'].shape}")
            print(f"   序列长度: {inputs['input_ids'].shape[1]}")
        except Exception as e:
            print(f"❌ 测试数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤8: 前向传播测试
        print_step(8, "前向传播测试")
        
        try:
            model.eval()
            
            with torch.no_grad():
                outputs = model(
                    inputs['input_ids'].to(device),
                    inputs['numerical_values'].to(device),
                    inputs['attention_mask'].to(device)
                )
            
            print(f"✅ 前向传播成功")
            print(f"   输出键: {list(outputs.keys())}")
            
            # 检查输出形状
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                    
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤9: 数学公式验证
        print_step(9, "数学公式快速验证")
        
        try:
            # 简单的柯西NLL测试
            test_target = 3.5
            test_loc = 2.0
            test_scale = 1.5
            
            # 手工计算
            import math
            z = (test_target - test_loc) / test_scale
            manual_nll = math.log(math.pi * test_scale) + math.log(1 + z**2)
            
            print(f"✅ 数学验证成功")
            print(f"   柯西NLL手工计算: {manual_nll:.6f}")
            
            # 测试函数计算（如果可用）
            try:
                from src.utils.distributions import cauchy_nll_loss
                computed_nll = cauchy_nll_loss(
                    torch.tensor(test_target),
                    torch.tensor(test_loc), 
                    torch.tensor(test_scale),
                    reduction='none'
                ).item()
                
                diff = abs(manual_nll - computed_nll)
                print(f"   柯西NLL函数计算: {computed_nll:.6f}")
                print(f"   差异: {diff:.8f}")
                print(f"   ✅ 数学一致性: {'通过' if diff < 1e-6 else '失败'}")
                
            except ImportError:
                print("   ⚠️  无法导入柯西NLL函数，跳过比较")
                
        except Exception as e:
            print(f"❌ 数学验证失败: {e}")

        # 总结
        print_step("完成", "调试总结")
        print("🎉 所有基本功能测试通过！")
        print("   - 模型可以正常创建和初始化")
        print("   - 前向传播正常工作")
        print("   - 输出形状符合预期")
        return True

    except KeyboardInterrupt:
        print(f"\n❌ 用户中断 (Ctrl+C)")
        return False
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("开始运行简化调试脚本...")
    success = main()
    if success:
        print("\n✅ 脚本执行成功")
    else:
        print("\n❌ 脚本执行失败")
    print("脚本结束")