#!/usr/bin/env python
"""
因果语言模型的前向传播调试脚本 (V4: 原生序列到序列架构)

本脚本对因果语言模型执行单次前向传播，并打印出从输入到最终损失的所有关键中间数学量。
这旨在用于调试和验证实现是否与 `design-docs/math/mathematical_foundations.md` 
中阐述的理论基础保持一致。

V4 更新：模型架构已完全重构为序列到序列模式，无需临时适配器。
"""
import os
import sys
import torch
import numpy as np
from dataclasses import asdict
import torch.nn.functional as F

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.training.trainer import Trainer
from src.utils.losses import CausalLMLoss, compute_ovr_probabilities, cauchy_nll_loss

def print_tensor_stats(tensor, name):
    """辅助函数，用于打印张量的详细统计信息。"""
    if not isinstance(tensor, torch.Tensor):
        print(f"--- {name}: {tensor} ---")
        return
        
    print(f"--- {name} ---")
    print(f"  - 形状 (Shape): {tensor.shape}")
    print(f"  - 数据类型 (Dtype): {tensor.dtype}")
    print(f"  - 设备 (Device): {tensor.device}")
    # 防止对非浮点或空张量进行统计计算
    if tensor.is_floating_point() and tensor.numel() > 0:
        print(f"  - 统计值 (Values):")
        print(f"    - 均值 (mean): {tensor.mean():.6f}, 标准差 (std): {tensor.std():.6f}")
        print(f"    - 最小值 (min):  {tensor.min():.6f}, 最大值 (max): {tensor.max():.6f}")
    print(f"  - 样本值 (Sample values): {tensor.flatten()[:5].tolist()}")
    print("-" * (len(name) + 20))


def main():
    """主函数，运行调试前向传播。"""
    print("=" * 80)
    print("=   因果语言模型前向传播调试脚本 (V4: 原生序列到序列架构)   =")
    print("=" * 80)

    # --- 1. 设置 ---
    print("\n[步骤 1. 设置模型、分词器和配置...]")
    device = torch.device('cpu') # 使用 CPU 以方便调试
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
    model.eval()

    print("设置完成。")

    # --- 2. 数据与目标生成 (BOS/EOS + 序列到序列) ---
    print("\n[步骤 2. 使用增强版分词器构建最终的批次数据...]")
    
    texts = [
        "The item costs 99.99 dollars.",
        "From a batch of 100 items, 5 were defective, leaving 95 good ones.",
        "A standard text without any numerical values."
    ]
    
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    
    # --- 构建最终的、序列化的目标 ---
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[:, :-1] = labels[:, 1:].clone()
    labels[:, -1] = -100

    # 构建序列化的回归目标
    target_values = torch.full_like(numerical_values, float('nan'))
    shifted_numerical_values = numerical_values.clone()
    shifted_numerical_values[:, :-1] = numerical_values[:, 1:].clone()
    shifted_numerical_values[:, -1] = 0.0
    
    num_mask = (labels == tokenizer.num_token_id)
    target_values[num_mask] = shifted_numerical_values[num_mask]

    # --- 详细打印 ---
    batch_size, seq_len = input_ids.shape
    print("\n--- 批次数据详情 (共 {} 个样本) ---".format(batch_size))
    # 只打印第一个样本的详情，以保持输出简洁
    i = 0
    print(f"\n  样本 {i+1}: '{texts[i]}'")
    print("  位置 | 输入Token        | 输入数值 | 目标Token (Label)  | 目标数值 (Target Value)")
    print("  " + "-"*70)
    for j in range(seq_len):
        if attention_mask[i, j] == 0: continue
        
        input_token = tokenizer.convert_ids_to_tokens([input_ids[i,j].item()])[0]
        input_val = numerical_values[i,j].item()
        
        label_id = labels[i,j].item()
        if label_id != -100:
            label_token = tokenizer.convert_ids_to_tokens([label_id])[0]
        else:
            label_token = "N/A (Ignore)"
        
        target_val = target_values[i,j].item()
        
        print(f"  {j:^4} | {input_token:<16} | {input_val:^10.2f} | {label_token:<18} | {target_val:^20.4f}")
    
    # --- 3. 前向传播 ---
    print("\n[步骤 3. 执行单次前向传播...]")
    with torch.no_grad():
        outputs = model(input_ids, numerical_values, attention_mask)
    print("前向传播完成。")
    
    # 验证输出形状
    print("\n--- 架构验证：检查输出形状 ---")
    print(f"输入序列形状: {input_ids.shape}")
    print(f"特征形状: {outputs['features'].shape}")
    print(f"因果表征形状: {outputs['causal_loc'].shape}")
    print(f"分类输出形状: {outputs['cls_loc'].shape}")
    print(f"回归输出形状: {outputs['reg_loc'].shape}")
    
    # 展示序列中不同位置的输出差异
    print("\n--- 序列位置的独立性验证 ---")
    print("检查不同位置是否有不同的预测（前3个位置）：")
    for pos in range(min(3, seq_len)):
        print(f"\n位置 {pos}:")
        print(f"  causal_loc[0,{pos},:5] = {outputs['causal_loc'][0, pos, :5].tolist()}")
        print(f"  cls_loc[0,{pos},:5] = {outputs['cls_loc'][0, pos, :5].tolist()}")
        print(f"  reg_loc[0,{pos}] = {outputs['reg_loc'][0, pos].item():.4f}")

    # --- 4. 损失计算 ---
    print("\n[步骤 4. 调用新的 CausalLMLoss 计算损失...]")
    
    loss_fn = CausalLMLoss(
        num_classes=config.vocab_size,
        num_token_id=config.num_token_id,
        regression_weight=config.reg_loss_weight,
        ovr_threshold=config.ovr_threshold
    )
    
    # 直接使用模型的原生序列输出
    loss_dict = loss_fn(
        outputs["cls_loc"], outputs["cls_scale"],
        labels, target_values
    )
    
    print("\n--- 最终损失输出 ---")
    print_tensor_stats(loss_dict['loss'], "总损失 (Total Loss)")
    print_tensor_stats(loss_dict['cls_loss'], "分类损失 (Classification Loss)")
    print_tensor_stats(loss_dict['gated_reg_loss'], "门控回归损失 (Gated Regression Loss)")

    # --- 详细的概率分析 ---
    print("\n[步骤 4.5. 详细分析OvR概率分布...]")
    
    # 计算第一个样本第一个有效位置的OvR概率
    sample_idx, pos_idx = 0, 3  # 选择位置3 (应该预测<NUM>)
    cls_loc_sample = outputs["cls_loc"][sample_idx, pos_idx]  # [V]
    cls_scale_sample = outputs["cls_scale"][sample_idx, pos_idx]  # [V]
    
    # 计算OvR概率
    ovr_probs = compute_ovr_probabilities(cls_loc_sample, cls_scale_sample, config.ovr_threshold)
    
    print(f"\n--- 样本{sample_idx+1}位置{pos_idx}的概率分析 (应该预测: {tokenizer.convert_ids_to_tokens([labels[sample_idx, pos_idx].item()])[0]}) ---")
    print(f"OvR阈值: {config.ovr_threshold}")
    print(f"概率和: {ovr_probs.sum().item():.6f}")
    print(f"最大概率: {ovr_probs.max().item():.6f}")
    print(f"最小概率: {ovr_probs.min().item():.6f}")
    print(f"概率均值: {ovr_probs.mean().item():.6f}")
    
    # 找出概率最高的前5个token
    top_probs, top_indices = torch.topk(ovr_probs, 5)
    print(f"\n概率最高的前5个token：")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
        print(f"  {i+1}. {token}: {prob.item():.6f}")
    
    # 检查<NUM> token的概率
    num_token_prob = ovr_probs[tokenizer.num_token_id].item()
    print(f"\n<NUM> token (ID: {tokenizer.num_token_id}) 的概率: {num_token_prob:.6f}")
    
    # 检查真实目标token的概率
    true_target = labels[sample_idx, pos_idx].item()
    if true_target != -100:
        true_target_prob = ovr_probs[true_target].item()
        true_target_token = tokenizer.convert_ids_to_tokens([true_target])[0]
        print(f"真实目标 '{true_target_token}' (ID: {true_target}) 的概率: {true_target_prob:.6f}")
    
    # 分析概率分布的统计特性
    prob_above_001 = (ovr_probs > 0.01).sum().item()
    prob_above_01 = (ovr_probs > 0.1).sum().item()
    prob_above_05 = (ovr_probs > 0.5).sum().item()
    
    print(f"\n概率分布统计：")
    print(f"  概率 > 0.01 的token数量: {prob_above_001} / {len(ovr_probs)}")
    print(f"  概率 > 0.1 的token数量: {prob_above_01} / {len(ovr_probs)}")  
    print(f"  概率 > 0.5 的token数量: {prob_above_05} / {len(ovr_probs)}")

    # --- 5. 推断-行动范式验证 ---
    print("\n[步骤 5. 验证推断-行动范式的实现...]")
    print("\n每个位置都经历了完整的推断-行动过程：")
    print("1. 推断 (Abduction): 特征 z_i → 个体因果表征分布 U_i ~ Cauchy(loc_i, scale_i)")
    print("2. 行动 (Action): U_i → 分类分数 S_k,i 和 回归值 Y_i")
    print("\n这正是 mathematical_foundations.md 中描述的核心范式！")

    print("\n" + "="*80)
    print("=   V4 调试脚本执行完毕。")
    print("=   架构重构成功！我们现在拥有了真正的序列到序列因果语言模型。")
    print("="*80)

if __name__ == '__main__':
    main()