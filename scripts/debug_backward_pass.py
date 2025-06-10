#!/usr/bin/env python
"""
因果语言模型反向传播与梯度流调试脚本

本脚本旨在验证模型的反向传播过程是否健康，检查关键组件的梯度是否存在、
是否为NaN/inf，以及其数值范围是否合理。这是在进行完整训练前一个
至关重要的"预检"步骤。
"""
import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.utils.losses import CausalLMLoss

def print_grad_stats(model, network_name):
    """辅助函数，打印指定网络模块的梯度统计信息。"""
    print(f"\n--- {network_name} 的梯度统计 ---")
    has_grads = False
    for name, param in model.named_parameters():
        if network_name in name and param.grad is not None:
            has_grads = True
            grad = param.grad
            is_nan = torch.isnan(grad).any()
            is_inf = torch.isinf(grad).any()
            
            print(f"  - {name}:")
            print(f"    - 形状: {grad.shape}")
            if is_nan or is_inf:
                print(f"    - 状态: {'存在NaN' if is_nan else ''}{'存在Inf' if is_inf else ''}  <-- 严重问题！")
            else:
                print(f"    - 状态: 健康")
                print(f"    - 均值: {grad.mean():.6f}, 标准差: {grad.std():.6f}")
                print(f"    - 范围: [{grad.min():.6f}, {grad.max():.6f}]")
    
    if not has_grads:
        print("  - 警告: 未找到任何梯度！该网络可能没有被正确地连接到损失函数。")
    print("-" * 40)


def main():
    """主函数，运行调试反向传播。"""
    print("=" * 80)
    print("=      因果语言模型反向传播与梯度流调试脚本      =")
    print("=" * 80)

    # --- 1. 设置 (与前向传播脚本一致) ---
    print("\n[步骤 1. 设置模型、分词器和配置...]")
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    
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
    
    model = CausalLanguageModel(config).to(device)
    # 确保模型处于训练模式以计算梯度
    model.train()
    
    # 正确初始化
    print("Initializing AbductionNetwork with proper scale...")
    model.abduction_network.init_weights()
    print("AbductionNetwork initialization completed.")
    print("设置完成。")

    # --- 2. 准备单批次数据 ---
    print("\n[步骤 2. 构建单批次数据...]")
    texts = ["The item costs 99.99 dollars."]
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[:, :-1] = labels[:, 1:].clone()
    labels[:, -1] = -100

    target_values = torch.full_like(numerical_values, float('nan'))
    shifted_numerical_values = numerical_values.clone()
    shifted_numerical_values[:, :-1] = numerical_values[:, 1:].clone()
    shifted_numerical_values[:, -1] = 0.0
    
    num_mask = (labels == tokenizer.num_token_id)
    target_values[num_mask] = shifted_numerical_values[num_mask]
    
    # --- 3. 前向传播 ---
    print("\n[步骤 3. 执行单次前向传播...]")
    outputs = model(input_ids, numerical_values, attention_mask)
    print("前向传播完成。")

    # --- 4. 损失计算 ---
    print("\n[步骤 4. 计算损失...]")
    loss_fn = CausalLMLoss(
        num_classes=config.vocab_size,
        num_token_id=config.num_token_id,
        regression_weight=config.reg_loss_weight,
        ovr_threshold=config.ovr_threshold
    )
    loss_dict = loss_fn(
        outputs["cls_loc"], outputs["cls_scale"],
        outputs["reg_loc"], outputs["reg_scale"],
        labels, target_values
    )
    loss = loss_dict['loss']
    print(f"计算得到的总损失: {loss.item():.4f}")

    # --- 5. 反向传播 ---
    print("\n[步骤 5. 执行单次反向传播...]")
    # 清除旧的梯度
    model.zero_grad()
    # 计算梯度
    loss.backward()
    print("反向传播完成。")
    
    # --- 6. 梯度检查 ---
    print("\n[步骤 6. 检查关键组件的梯度...]")
    print_grad_stats(model, "abduction_network")
    print_grad_stats(model, "action_network")
    
    print("\n" + "="*80)
    print("=      梯度诊断脚本执行完毕。")
    print("=      请检查以上输出，确保梯度健康。")
    print("="*80)

if __name__ == '__main__':
    main() 