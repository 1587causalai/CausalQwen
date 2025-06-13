#!/usr/bin/env python
"""
总损失计算及 OvR 阈值影响实验脚本

本脚本旨在完成两项任务：
1. 白盒验证 `compute_total_loss` 函数的实现是否正确，特别是对分类和
   回归损失使用不同分母进行平均的核心逻辑。
2. 通过实验对比不同的 OvR 决策阈值 (`ovr_threshold`) 对初始损失值的影响，
   以验证其数学行为。
"""
import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.utils.model_utils import get_qwen_model_info
from src.losses.loss_functions import ovr_classification_loss, gated_regression_loss
from src.utils.losses import compute_ovr_probabilities

def print_step(step_name, description):
    """打印流程图步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  {step_name}: {description}")
    print(f"{'-'*70}")

def main():
    print("🚀 CausalQwen - 总损失计算阈值实验")

    # --- 步骤 1: 通用初始化 ---
    print_step("步骤 1", "初始化分词器和共享测试数据")
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    model_info = get_qwen_model_info(qwen_model_path)
    if not model_info:
        sys.exit(1)
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)

    test_samples = [
        "This is a sentence without any numbers.",
        "The item costs 50.5 and the tax is 4.5.",
    ]
    inputs = tokenizer.batch_encode_plus(test_samples, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    cls_labels = torch.full_like(input_ids, -100)
    cls_labels[:, :-1] = input_ids[:, 1:]

    print("   ✅ 通用初始化完成。")

    thresholds_to_test = [-10.0, 0.0, 10.0, 10000.0]
    all_tests_passed = True
    results = []

    for threshold in thresholds_to_test:
        print(f"\n\n{'#'*80}")
        print(f"🔬 开始测试 OvR 阈值 (Threshold): {threshold}")
        print(f"{'#'*80}")

        # --- 步骤 2: 针对当前阈值初始化模型 ---
        print_step("步骤 2", f"使用阈值 {threshold} 初始化模型配置")
        config = CausalLMConfig(
            vocab_size=model_info['vocab_size'],
            num_token_id=tokenizer.num_token_id,
            hidden_size=model_info['hidden_size'],
            use_real_qwen=False,
            reg_loss_gating_alpha=1.0,
            reg_loss_weight=1.0,
            ovr_threshold=threshold  # <--- 核心改动
        )
        model = CausalLanguageModel(config).to(device)
        print(f"   ✅ 模型已使用 ovr_threshold={threshold} 进行配置。")


        with torch.no_grad():
            # --- 步骤 3: 生成模拟输出 ---
            print_step("步骤 3", "生成模拟输出分布")
            # 为保证跨阈值测试的可比性，我们为每个循环使用相同的随机种子
            torch.manual_seed(42)
            outputs = {
                'cls_loc': torch.randn(len(test_samples), input_ids.shape[1], config.vocab_size),
                'cls_scale': torch.rand(len(test_samples), input_ids.shape[1], config.vocab_size) * 5 + 1,
                'reg_loc': torch.randn(len(test_samples), input_ids.shape[1]),
                'reg_scale': torch.rand(len(test_samples), input_ids.shape[1]) * 5 + 1,
            }
            print("   - 已生成固定的随机模拟模型输出。")

            # --- 步骤 4: 手动计算总损失 (Ground Truth) ---
            print_step("步骤 4", "手动计算总损失 (Ground Truth)")
            print("   - 遵循 `mathematical_foundations.md` 中的分离平均公式")

            # a. 提取所需张量
            cls_loc, cls_scale = outputs['cls_loc'], outputs['cls_scale']
            reg_loc, reg_scale = outputs['reg_loc'], outputs['reg_scale']
            
            # b. 计算逐词元损失
            # 注意：这里我们使用了 model.config.ovr_threshold 来确保与模型行为一致
            cls_probs = compute_ovr_probabilities(cls_loc, cls_scale, model.config.ovr_threshold)
            L_cls_per_token = ovr_classification_loss(cls_probs, cls_labels, reduction='none')

            num_probs = cls_probs[:, :, tokenizer.num_token_id]
            num_mask = (cls_labels == tokenizer.num_token_id).float() * attention_mask
            L_reg_per_token = gated_regression_loss(
                reg_loc, reg_scale, numerical_values,
                gate_prob=num_probs, mask=num_mask, 
                alpha=model.config.reg_loss_gating_alpha, reduction='none'
            )

            # c. 分别计算平均损失
            active_cls_tokens = attention_mask.sum()
            # 需要处理潜在的 inf 值，将其替换为 0 以便求和
            L_cls_per_token_safe = torch.nan_to_num(L_cls_per_token, nan=0.0, posinf=0.0, neginf=0.0)
            L_cls_mean = (L_cls_per_token_safe * attention_mask).sum() / active_cls_tokens
            
            active_reg_tokens = num_mask.sum()
            L_reg_eff = L_reg_per_token.sum() / (active_reg_tokens + 1e-8)

            # d. 加权合并
            lambda_weight = model.config.reg_loss_weight
            manual_total_loss = L_cls_mean + lambda_weight * L_reg_eff
            
            print(f"\n   --- 手动计算结果 ---")
            print(f"     - 有效分类词元数 (attention_mask.sum()): {active_cls_tokens.item()}")
            print(f"     - 有效回归词元数 (num_mask.sum()): {active_reg_tokens.item()}")
            print(f"     - 平均分类损失 (L_cls_mean): {L_cls_mean.item():.4f}")
            print(f"     - 有效回归损失 (L_reg_eff): {L_reg_eff.item():.4f}")
            print(f"     - 回归权重 (λ): {lambda_weight}")
            print(f"     - 手动计算总损失 (Manual Total Loss): {manual_total_loss.item():.4f}")

            # --- 步骤 5: 使用模型内置方法计算损失 ---
            print_step("步骤 5", "使用 `model.compute_loss` 计算总损失")
            loss_dict = model.compute_loss(
                outputs, targets=cls_labels, 
                numerical_values=numerical_values, attention_mask=attention_mask
            )
            model_total_loss = loss_dict['total']
            print(f"\n   --- 模型计算结果 ---")
            print(f"     - 平均分类损失 (cls_loss): {loss_dict['cls']:.4f}")
            print(f"     - 有效回归损失 (effective_reg_loss): {loss_dict['effective_reg_loss']:.4f}")
            print(f"     - 模型计算总损失 (Model Total Loss): {model_total_loss.item():.4f}")

            # --- 步骤 6: 核心数学逻辑验证 ---
            print_step("步骤 6", "核心数学逻辑验证")
            # 由于可能存在 NaN，我们在比较前也对模型输出进行处理
            model_total_loss_safe = torch.nan_to_num(model_total_loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss_match = torch.allclose(manual_total_loss, model_total_loss_safe, atol=1e-5)
            
            if not loss_match:
                all_tests_passed = False

            print(f"\n   --- 验证: 手动计算 vs. 模型计算 ---")
            print(f"     - 结论: {'✅ 通过' if loss_match else '❌ 失败'}")
            if not loss_match:
                diff = torch.abs(manual_total_loss - model_total_loss_safe)
                print(f"     - 绝对差异: {diff.item():.8f}")
            
            results.append({
                "threshold": threshold,
                "cls_loss": loss_dict['cls'],
                "reg_loss": loss_dict['effective_reg_loss'],
                "total_loss": model_total_loss,
                "passed": loss_match
            })

    # --- 总结报告 ---
    print(f"\n\n{'='*80}")
    print("📊 实验总结报告")
    print(f"{'='*80}")
    print(f"{'阈值':>10} | {'分类损失':>15} | {'有效回归损失':>15} | {'总损失':>15} | {'验证通过':>10}")
    print(f"{'-'*11}+{'-'*17}+{'-'*17}+{'-'*17}+{'-'*12}")
    for res in results:
        status = "✅" if res['passed'] else "❌"
        # 处理可能的 inf/-inf/nan
        cls_loss_str = f"{torch.nan_to_num(res['cls_loss']).item():>15.4f}"
        reg_loss_str = f"{torch.nan_to_num(res['reg_loss']).item():>15.4f}"
        total_loss_str = f"{torch.nan_to_num(res['total_loss']).item():>15.4f}"
        
        print(f"{res['threshold']:>10.1f} | {cls_loss_str} | {reg_loss_str} | {total_loss_str} | {status:>10}")

    print(f"\n{'='*80}")
    if all_tests_passed:
        print("🎉 全部验证成功！`compute_total_loss` 在所有测试阈值下均表现正确。")
    else:
        print("❌ 部分或全部验证失败！请检查 `compute_total_loss` 的实现或阈值逻辑。")


if __name__ == '__main__':
    main() 