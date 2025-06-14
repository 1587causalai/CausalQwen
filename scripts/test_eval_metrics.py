#!/usr/bin/env python
"""
评估指标计算验证脚本

本脚本旨在白盒验证 `evaluator.py` 中各项评估指标计算的正确性，
特别关注数值词元预测（Precision, Recall, F1）和回归误差（MAE, MdAE）的计算逻辑。
"""
import os
import sys
import torch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tokenizer import QwenTokenizerWrapper

def print_step(step_name, description):
    """打印流程图步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  {step_name}: {description}")
    print(f"{'-'*70}")

def compute_eval_metrics_mock(
    pred_tokens, true_tokens, 
    true_values, pred_values, 
    num_token_id
):
    """
    一个模拟 Evaluator 核心逻辑的函数，用于计算各项指标。
    """
    metrics = {}
    
    # --- 1. 数值词元预测性能 (Precision, Recall, F1) ---
    is_true_num = (true_tokens == num_token_id)
    is_pred_num = (pred_tokens == num_token_id)
    
    tp = np.sum(is_true_num & is_pred_num)
    fp = np.sum(~is_true_num & is_pred_num)
    fn = np.sum(is_true_num & ~is_pred_num)
    
    # 计算 Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['num_precision'] = precision
    metrics['num_recall'] = recall
    metrics['num_f1'] = f1
    metrics['num_tp'] = tp
    metrics['num_fp'] = fp
    metrics['num_fn'] = fn

    # --- 2. 回归性能 (MAE, MdAE) ---
    # 关键逻辑: 回归指标只应在模型正确识别出<NUM>词元(即True Positives)的位置上计算。
    # 对于FN(False Negative), 模型没有预测为<NUM>，因此不存在回归预测值，不应计算误差。
    # 对于FP(False Positive), 没有真实的数值可供比较，也不计算误差。
    tp_mask = is_true_num & is_pred_num
    
    reg_true_values = true_values[tp_mask]
    reg_pred_values = pred_values[tp_mask]

    # 过滤掉 NaN (如果存在)
    valid_mask = ~np.isnan(reg_true_values)
    reg_true_values = reg_true_values[valid_mask]
    reg_pred_values = reg_pred_values[valid_mask]

    if len(reg_true_values) > 0:
        metrics['reg_mae'] = np.mean(np.abs(reg_true_values - reg_pred_values))
        metrics['reg_mdae'] = np.median(np.abs(reg_true_values - reg_pred_values))
    else:
        metrics['reg_mae'] = 0.0
        metrics['reg_mdae'] = 0.0
        
    return metrics

def main():
    print("🚀 CausalQwen - 评估指标计算逻辑验证")

    # --- 步骤 1: 初始化与设置 ---
    print_step("步骤 1", "初始化分词器并定义黄金测试数据")
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    NUM_ID = tokenizer.num_token_id
    OTHER_ID = NUM_ID + 1  # 代表任何非<NUM>的词元

    # 定义"黄金"测试数据
    # 场景:
    # - Pos 0: TP (True Positive)  -  正确预测 <NUM>
    # - Pos 1: FN (False Negative) -  漏报 <NUM>
    # - Pos 2: TN (True Negative)  -  正确预测非 <NUM>
    # - Pos 3: FP (False Positive) -  误报 <NUM>
    # - Pos 4: TP (True Positive)  -  正确预测 <NUM>
    true_tokens = np.array([NUM_ID, NUM_ID, OTHER_ID, OTHER_ID, NUM_ID])
    pred_tokens = np.array([NUM_ID, OTHER_ID, OTHER_ID, NUM_ID, NUM_ID])
    
    true_values = np.array([10.0, 20.0, 0.0, 0.0, 30.0])
    pred_values = np.array([12.0, 99.0, 0.0, 88.0, 25.0]) # 预测值
    
    print("   - <NUM> Token ID:", NUM_ID)
    print("   - 真实 Token:", true_tokens)
    print("   - 预测 Token:", pred_tokens)
    print("   - 真实数值:", true_values)
    print("   - 预测数值:", pred_values)


    # --- 步骤 2: 手动计算期望结果 ---
    print_step("步骤 2", "手动计算期望的评估指标 (Ground Truth)")
    # TP = 2 (pos 0, 4)
    # FN = 1 (pos 1)
    # FP = 1 (pos 3)
    expected_precision = 2 / (2 + 1)  # TP / (TP + FP)
    expected_recall = 2 / (2 + 1)   # TP / (TP + FN)
    expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
    
    # 关键修正：回归误差只在模型正确预测为 <NUM> 的位置 (True Positives) 计算
    # 对应位置: 0, 4
    # 在这些位置上的真实值 vs 预测值: (10.0 vs 12.0), (30.0 vs 25.0)
    errors = np.abs([10.0 - 12.0, 30.0 - 25.0]) # [2.0, 5.0]
    expected_mae = np.mean(errors) # (2 + 5) / 2 = 3.5
    expected_mdae = np.median(errors) # median(2, 5) = 3.5

    print(f"\n   --- 手动计算的期望值 (修正后) ---")
    print(f"     - 回归误差只在 TP 位置计算")
    print(f"     - TP=2, FP=1, FN=1")
    print(f"     - 期望 Precision: {expected_precision:.4f}")
    print(f"     - 期望 Recall: {expected_recall:.4f}")
    print(f"     - 期望 F1 Score: {expected_f1:.4f}")
    print(f"     - 期望 MAE: {expected_mae:.4f}")
    print(f"     - 期望 MdAE: {expected_mdae:.4f}")

    # --- 步骤 3: 使用模拟函数计算 ---
    print_step("步骤 3", "使用 `compute_eval_metrics_mock` 函数计算指标")
    calculated_metrics = compute_eval_metrics_mock(
        pred_tokens, true_tokens, true_values, pred_values, NUM_ID
    )
    print(f"\n   --- 函数计算结果 ---")
    for key, value in calculated_metrics.items():
        print(f"     - {key}: {value:.4f}" if isinstance(value, float) else f"     - {key}: {value}")

    # --- 步骤 4: 验证结果 ---
    print_step("步骤 4", "验证计算结果是否与期望一致")
    all_passed = True
    
    tests = {
        "Precision": (calculated_metrics['num_precision'], expected_precision),
        "Recall": (calculated_metrics['num_recall'], expected_recall),
        "F1 Score": (calculated_metrics['num_f1'], expected_f1),
        "MAE": (calculated_metrics['reg_mae'], expected_mae),
        "MdAE": (calculated_metrics['reg_mdae'], expected_mdae),
    }

    for name, (calc, exp) in tests.items():
        is_close = np.isclose(calc, exp)
        status = "✅ 通过" if is_close else "❌ 失败"
        print(f"   - {name} 验证: {status}")
        if not is_close:
            all_passed = False
            print(f"     - 计算值: {calc:.4f}, 期望值: {exp:.4f}")

    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 全部验证成功！指标计算逻辑正确。")
    else:
        print("❌ 部分或全部验证失败！请检查 `compute_eval_metrics_mock` 的实现。")

if __name__ == '__main__':
    main() 