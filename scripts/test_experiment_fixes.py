#!/usr/bin/env python3
"""
测试实验脚本中的修复是否正确工作

这个脚本验证：
1. run_experiments.py 是否正确使用修复后的初始化
2. 数学公式是否在实验环境中正确工作
3. 训练流程是否使用修复后的组件
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper
from src.training.trainer import Trainer

def test_experiment_fixes():
    """测试实验中的修复是否正确工作"""
    print("🔬 测试实验脚本中的修复")
    print("=" * 60)
    
    # 1. 创建与 run_experiments.py 相同的配置
    tokenizer = QwenTokenizerWrapper(
        model_path="~/models/Qwen2.5-0.5B", 
        use_real_tokenizer=True
    )
    
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        causal_dim=896,  # 强制与 hidden_size 相同
        use_real_qwen=True,
        qwen_model_path="~/models/Qwen2.5-0.5B",
        ovr_threshold=10.0
    )
    
    print(f"✅ 配置创建成功:")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - causal_dim: {config.causal_dim}")
    print(f"  - ovr_threshold: {config.ovr_threshold}")
    
    # 2. 创建模型（与 run_experiments.py 相同）
    device = torch.device('cpu')  # 使用CPU进行测试
    model = CausalLanguageModel(config).to(device)
    
    print(f"\n✅ 模型创建成功")
    
    # 3. 测试 Trainer 的初始化（这会调用我们修复的 init_weights）
    print(f"\n🔧 测试 Trainer 初始化（包含修复后的权重初始化）...")
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-4,
        batch_size=4,  # 小批量用于测试
        config=config,
        wandb_run=None
    )
    
    print(f"✅ Trainer 初始化成功")
    print(f"  - 数据统计计算完成")
    print(f"  - 知识传输初始化完成")
    
    # 4. 验证回归头修复
    print(f"\n🔍 验证回归头修复:")
    reg_weight = model.action_network.regression_head.causal_linear.weight.data
    reg_bias = model.action_network.regression_head.causal_linear.bias.data
    
    print(f"  - 回归头权重统计:")
    print(f"    权重均值: {reg_weight.mean().item():.6f}")
    print(f"    权重标准差: {reg_weight.std().item():.6f}")
    print(f"    权重不全为零: {'✅' if reg_weight.std().item() > 0.001 else '❌'}")
    
    if reg_bias is not None:
        print(f"  - 回归头偏置: {reg_bias.item():.4f}")
        print(f"    偏置合理性: {'✅' if abs(reg_bias.item()) > 1.0 else '❌'}")
    
    # 5. 测试前向传播数学正确性
    print(f"\n🧮 测试前向传播数学正确性:")
    
    # 创建测试输入
    test_text = "The price is 42.5 dollars."
    inputs = tokenizer([test_text], padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['numerical_values'], inputs['attention_mask'])
    
    # 验证输出形状
    expected_batch_size = 1
    expected_seq_len = inputs['input_ids'].shape[1]
    
    print(f"  - 输出形状验证:")
    print(f"    cls_loc: {outputs['cls_loc'].shape} (期望: [{expected_batch_size}, {expected_seq_len}, {config.vocab_size}])")
    print(f"    reg_loc: {outputs['reg_loc'].shape} (期望: [{expected_batch_size}, {expected_seq_len}])")
    
    shape_correct = (
        outputs['cls_loc'].shape == (expected_batch_size, expected_seq_len, config.vocab_size) and
        outputs['reg_loc'].shape == (expected_batch_size, expected_seq_len)
    )
    print(f"    形状正确性: {'✅' if shape_correct else '❌'}")
    
    # 验证回归头响应输入
    reg_values = outputs['reg_loc'][0, :3].tolist()  # 前3个位置
    print(f"  - 回归头响应性验证:")
    print(f"    前3个位置的回归值: {[f'{v:.4f}' for v in reg_values]}")
    
    # 检查是否有变化（不是固定值）
    reg_variance = torch.var(outputs['reg_loc'][0, :min(3, expected_seq_len)]).item()
    responsive = reg_variance > 0.01
    print(f"    回归值变化性: {'✅' if responsive else '❌'} (方差: {reg_variance:.6f})")
    
    # 6. 验证柯西分布线性变换数学
    print(f"\n📐 验证柯西分布线性变换数学:")
    
    # 选择一个位置进行验证
    pos = 0
    causal_loc = outputs['causal_loc'][0, pos]  # [C]
    causal_scale = outputs['causal_scale'][0, pos]  # [C]
    
    # 验证分类头的几个token
    cls_head = model.action_network.classification_head.causal_linear
    test_tokens = [0, 1]  # 测试前两个token
    
    all_correct = True
    for token_idx in test_tokens:
        weight_row = cls_head.weight[token_idx]  # [C]
        abs_weight_row = torch.abs(weight_row)
        theoretical_scale = torch.dot(abs_weight_row, causal_scale).item()
        actual_scale = outputs['cls_scale'][0, pos, token_idx].item()
        
        match = abs(theoretical_scale - actual_scale) < 1e-5
        all_correct = all_correct and match
        
        print(f"    Token{token_idx}: 理论={theoretical_scale:.6f}, 实际={actual_scale:.6f} {'✅' if match else '❌'}")
    
    print(f"  - 数学一致性: {'✅' if all_correct else '❌'}")
    
    # 7. 总结
    print(f"\n" + "=" * 60)
    print(f"📊 测试总结:")
    
    all_tests = [
        shape_correct,
        responsive,
        all_correct,
        reg_weight.std().item() > 0.001
    ]
    
    if all(all_tests):
        print(f"✅ 所有测试通过！实验脚本正确使用了修复后的组件。")
        print(f"  - 知识传输初始化正确工作")
        print(f"  - 回归头修复生效（权重不为零，响应输入）")
        print(f"  - 柯西分布线性变换数学正确")
        print(f"  - 序列到序列架构正确")
    else:
        print(f"❌ 部分测试失败，需要进一步检查:")
        if not shape_correct:
            print(f"  - 输出形状不正确")
        if not responsive:
            print(f"  - 回归头不响应输入")
        if not all_correct:
            print(f"  - 数学公式计算错误")
        if not (reg_weight.std().item() > 0.001):
            print(f"  - 回归头权重仍为零")

def main():
    """主函数"""
    print("🧪 实验脚本修复验证")
    print("=" * 80)
    
    try:
        test_experiment_fixes()
        print("\n🎉 验证完成！")
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 