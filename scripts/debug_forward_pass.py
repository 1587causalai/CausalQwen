#!/usr/bin/env python
"""
因果语言模型的前向传播调试脚本 (V6: 深度分析版)

专注于验证前向传播中每个核心组件的数学逻辑和数据流。
"""
import os
import sys
import torch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_step(step_num, description):
    """打印步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  步骤 {step_num}: {description}")
    print(f"{'='*70}")

def print_tensor_stats(name, tensor):
    """打印张量的详细统计信息，用于调试。"""
    if not isinstance(tensor, torch.Tensor):
        print(f"   - {name}: Not a tensor")
        return
    
    # 统一转换到CPU和float32进行分析，避免设备和类型问题
    tensor = tensor.detach().cpu().to(torch.float32)
    
    print(f"   - {name}:")
    print(f"     - Shape: {tensor.shape}")
    print(f"     - Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"     - Has Inf: {torch.isinf(tensor).any().item()}")
    print(f"     - Mean: {tensor.mean().item():.6f}")
    print(f"     - Std:  {tensor.std().item():.6f}")
    print(f"     - Min:  {tensor.min().item():.6f}")
    print(f"     - Max:  {tensor.max().item():.6f}")

def main():
    """主函数，运行调试前向传播。"""
    try:
        print("🚀 CausalQwen 前向传播调试脚本 (深度分析版)")
        print("=" * 80)

        # 步骤1: 基本导入测试
        print_step(1, "测试基本导入")
            from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
            from src.data.tokenizer import QwenTokenizerWrapper
        print("✅ 导入成功")

        # 步骤2: 环境和路径检查
        print_step(2, "环境和路径检查")
        device = torch.device('cpu')
        qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
        print(f"设备: {device}")
        print(f"Qwen路径: {qwen_model_path}")
        assert os.path.exists(qwen_model_path), "Qwen模型路径不存在"
            print(f"✅ Qwen模型路径存在")

        # 步骤3: 分词器初始化
        print_step(3, "分词器初始化")
        tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
            print(f"✅ 分词器初始化成功")
        vocab_info = tokenizer.vocab_size_info()
        print(f"   - CausalQwen 词汇表大小: {vocab_info['causalqwen_vocab']}")
        print(f"   - <NUM> token ID: {tokenizer.num_token_id}")

        # 步骤4: 模型配置
        print_step(4, "模型配置创建")
            config = CausalLMConfig(
            vocab_size=vocab_info['causalqwen_vocab'],
                num_token_id=tokenizer.num_token_id,
                hidden_size=896,
                causal_dim=896,
                use_real_qwen=True,
            qwen_model_path=qwen_model_path
            )
            print(f"✅ 配置创建成功")

        # 步骤5: 模型创建与初始化
        print_step(5, "模型创建与初始化")
            model = CausalLanguageModel(config).to(device)
        model.init_weights()
        print(f"✅ 模型创建与初始化成功")
            total_params = sum(p.numel() for p in model.parameters())
        print(f"   - 总参数数量: {total_params:,}")

        # 步骤6: 测试数据准备
        print_step(6, "准备测试数据")
            test_texts = [
            "The price of the book is 99.99 dollars and the temperature is -10.5 degrees.",
            "Hello world! This is a test without numbers."
            ]
            inputs = tokenizer.batch_encode_plus(
            test_texts, padding=True, truncation=True, return_tensors='pt'
            )
        input_ids = inputs['input_ids'].to(device)
        numerical_values = inputs['numerical_values'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        print(f"✅ 测试数据准备成功 (Batch Size: {input_ids.shape[0]}, Seq Len: {input_ids.shape[1]})")

        # 步骤7: 分步前向传播与验证
        print_step(7, "分步前向传播与验证")
        model.eval()
            with torch.no_grad():
            
            # 7.1 特征提取网络
            print("\n--- 7.1. 特征提取 (Feature Extraction) ---")
            print("输入: input_ids, numerical_values")
            print("输出: 上下文特征 z")
            print("理论: z = FeatureNetwork(NumericalEmbedding(input))")
            
            # 注意：在我们的模型中，特征提取网络包含了数值嵌入的逻辑
            z = model.feature_network(
                input_ids=input_ids,
                numerical_values=numerical_values,
                attention_mask=attention_mask
            )
            print_tensor_stats("上下文特征 z", z)
            assert z.shape == (input_ids.shape[0], input_ids.shape[1], config.hidden_size), "z 形状错误"

            # 7.2 归因推断网络
            print("\n--- 7.2. 归因推断 (Abduction) ---")
            print("输入: 上下文特征 z")
            print("输出: loc_U, scale_U")
            print("理论: loc_U 应该约等于 z (因初始化为恒等映射), scale_U 应该为较大的正数 (约10)")
            
            loc_U, scale_U = model.abduction_network(z)
            print_tensor_stats("因果表征位置 loc_U", loc_U)
            print_tensor_stats("因果表征尺度 scale_U", scale_U)
            assert torch.allclose(loc_U, z, atol=1e-5), "loc_U 与 z 不一致"
            assert scale_U.mean() > 5, "scale_U 的均值过小"

            # 7.3 行动决策网络
            print("\n--- 7.3. 行动决策 (Action) ---")
            print("输入: loc_U, scale_U")
            print("输出: loc_S, scale_S (分类), loc_Y, scale_Y (回归)")
            print("理论: scale_S 和 scale_Y 应该也是较大的正数，因为它们是 scale_U 的线性变换")

            output_dict = model.action_network(loc_U, scale_U)
            
            print("\n   --- 分类输出 ---")
            print_tensor_stats("分类 logits (loc_S)", output_dict.get('loc_S'))
            print_tensor_stats("分类尺度 (scale_S)", output_dict.get('scale_S'))

            print("\n   --- 回归输出 ---")
            print_tensor_stats("回归预测 (loc_Y)", output_dict.get('loc_Y'))
            print_tensor_stats("回归不确定性 (scale_Y)", output_dict.get('scale_Y'))
            
            # 关键验证
            final_scale_Y = output_dict.get('scale_Y')
            assert final_scale_Y is not None, "模型未输出 scale_Y"
            if final_scale_Y.mean().item() < 1.0:
                 print("\n🚨🚨🚨 警告: 回归不确定性 scale_Y 的均值非常小! 这可能是导致 PICP=0 的直接原因。")
            else:
                 print("\n✅ 回归不确定性 scale_Y 的均值看起来合理。")

        # 步骤8: 总结
        print_step("完成", "调试总结")
        print("🎉 脚本执行完毕。请仔细检查上面每个步骤的输出，特别是 `scale_U` 和 `scale_Y` 的值。")
        return True

    except KeyboardInterrupt:
        print(f"\n❌ 用户中断 (Ctrl+C)")
        return False
    except Exception as e:
        print(f"\n❌ 脚本执行中出现意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)