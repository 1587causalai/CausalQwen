#!/usr/bin/env python
"""
总损失计算白盒验证脚本

本脚本严格遵循 `mathematical_foundations.md` 中关于总损失合并的精确定义，
旨在白盒验证 `compute_total_loss` 函数的实现是否正确，特别是对分类和
回归损失使用不同分母进行平均的核心逻辑。
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
    print("🚀 CausalQwen - 总损失 (Total Loss) 计算逻辑深度验证")
    
    # --- 步骤 1: 初始化 ---
    print_step("步骤 1", "初始化真实模型、分词器及测试数据")
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    model_info = get_qwen_model_info(qwen_model_path)
    if not model_info:
        sys.exit(1)
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    
    config = CausalLMConfig(
        vocab_size=model_info['vocab_size'],
        num_token_id=tokenizer.num_token_id,
        hidden_size=model_info['hidden_size'],
        use_real_qwen=False, # 损失计算与模型权重无关，设为False以加速
        reg_loss_gating_alpha=0.0,
        reg_loss_weight=0.5 # 使用一个非1的权重来验证乘法
    )
    # 损失计算是静态方法，不需要加载真实模型或初始化权重
    model = CausalLanguageModel(config).to(device)

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
    
    print("   ✅ 初始化完成。")

    with torch.no_grad():
        # --- 步骤 2: 生成模拟输出 ---
        print_step("步骤 2", "生成模拟输出分布")
        # 由于我们只测试损失函数，可以直接创建随机的输出分布
        # 而无需通过真实的前向传播，这样测试更纯粹、更快速
        outputs = {
            'cls_loc': torch.randn(len(test_samples), input_ids.shape[1], config.vocab_size),
            'cls_scale': torch.rand(len(test_samples), input_ids.shape[1], config.vocab_size) * 5 + 1,
            'reg_loc': torch.randn(len(test_samples), input_ids.shape[1]),
            'reg_scale': torch.rand(len(test_samples), input_ids.shape[1]) * 5 + 1,
        }
        print("   - 已生成随机的模拟模型输出。")

        # --- 步骤 3: 手动计算总损失 (Ground Truth) ---
        print_step("步骤 3", "手动计算总损失 (Ground Truth)")
        print("   - 遵循 `mathematical_foundations.md` 中的分离平均公式")

        # a. 提取所需张量
        cls_loc, cls_scale = outputs['cls_loc'], outputs['cls_scale']
        reg_loc, reg_scale = outputs['reg_loc'], outputs['reg_scale']
        
        # b. 计算逐词元损失
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
        L_cls_mean = (L_cls_per_token * attention_mask).sum() / active_cls_tokens
        
        active_reg_tokens = num_mask.sum()
        # L_reg_per_token 已经包含了门控和掩码，直接求和即可
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

        # --- 步骤 4: 使用模型内置方法计算损失 ---
        print_step("步骤 4", "使用 `model.compute_loss` 计算总损失")
        loss_dict = model.compute_loss(
            outputs, targets=cls_labels, 
            numerical_values=numerical_values, attention_mask=attention_mask
        )
        model_total_loss = loss_dict['total']
        print(f"\n   --- 模型计算结果 ---")
        print(f"     - 平均分类损失 (cls_loss): {loss_dict['cls_loss']:.4f}")
        print(f"     - 有效回归损失 (effective_reg_loss): {loss_dict['effective_reg_loss']:.4f}")
        print(f"     - 模型计算总损失 (Model Total Loss): {model_total_loss.item():.4f}")

        # --- 步骤 5: 核心数学逻辑验证 ---
        print_step("步骤 5", "核心数学逻辑验证")
        loss_match = torch.allclose(manual_total_loss, model_total_loss, atol=1e-5)
        
        print(f"\n   --- 验证: 手动计算 vs. 模型计算 ---")
        print(f"     - 结论: {'✅ 通过' if loss_match else '❌ 失败'}")
        if not loss_match:
            diff = torch.abs(manual_total_loss - model_total_loss)
            print(f"     - 绝对差异: {diff.item():.8f}")
            
    print(f"\n\n{'='*80}")
    if loss_match:
        print("🎉 验证成功！`compute_total_loss` 的实现完全符合数学设计，正确地对损失进行了分离平均。")
    else:
        print("❌ 验证失败！请检查 `compute_total_loss` 的实现逻辑。")

if __name__ == '__main__':
    main() 