#!/usr/bin/env python
"""
端到端回归流程验证脚本

本脚本旨在完整地展示从原始文本输入到最终门控回归损失计算的全过程。
它使用 `test_numerical_aware_embedding.py` 中的4个真实语料样本，
通过完整的 CausalQwen 模型，验证核心的回归任务流程和门控机制。
"""
import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.utils.model_utils import get_qwen_model_info

def print_step(step_name, description):
    """打印流程图步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  {step_name}: {description}")
    print(f"{'-'*70}")

def main():
    print("🚀 CausalQwen - 端到端回归流程与门控机制验证")
    
    # --- 步骤 1: 初始化真实模型和分词器 ---
    print_step("步骤 1", "初始化真实模型和分词器")
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
        causal_dim=model_info['hidden_size'],
        use_real_qwen=True,
        qwen_model_path=qwen_model_path,
        use_numerical_features=True,
        ovr_threshold=100.0,
        reg_loss_gating_alpha=0.0 # 使用完全门控，方便验证
    )
    model = CausalLanguageModel(config).to(device)
    model.init_weights()
    model.eval()
    print("   ✅ 组件初始化完成")

    # --- 步骤 2: 准备输入数据 ---
    print_step("步骤 2", "准备输入数据 (来自 test_numerical_aware_embedding.py)")
    test_samples = [
        "This is a sentence without any numbers.",
        "The item costs 50.5 and the tax is 4.5.",
        "The dimensions are 10 by 20 by 30 cm.",
        "What is 2 plus 3? 5.",
    ]
    inputs = tokenizer.batch_encode_plus(test_samples, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    print(f"   - 使用 {len(test_samples)} 个测试样本进行批量处理。")

    # --- 步骤 3: 准备回归任务的标签 ---
    print_step("步骤 3", "准备回归任务的标签 (Labels)")
    # 对于回归任务，"标签" 就是 `numerical_values` 张量本身。
    # 我们也需要一个分类任务的标签，但在此测试中我们主要关注回归损失。
    cls_labels = torch.full_like(input_ids, -100)
    cls_labels[:, :-1] = input_ids[:, 1:]
    
    print("   - 回归标签: `numerical_values` 张量")
    print("   - 分类标签: `input_ids` 左移一位 (常规语言模型目标)")
    
    print("\n   --- 查看样本 2 的输入和回归目标 ---")
    sample_idx = 1
    num_token_id = tokenizer.num_token_id
    
    print(f"     输入 IDs: {input_ids[sample_idx].tolist()}")
    print(f"     回归目标: {numerical_values[sample_idx].tolist()}")
    
    # 找到<NUM> token的位置
    num_positions = (input_ids[sample_idx] == num_token_id).nonzero(as_tuple=True)[0]
    target_values = numerical_values[sample_idx][num_positions]
    
    print(f"     - 在位置 {num_positions.tolist()} 发现了 <NUM> 词元。")
    print(f"     - 它们对应的回归目标值是: {target_values.tolist()}")
    
    print("\n   --- 查看样本 1 (无数值) 的回归目标 ---")
    print(f"     回归目标: {numerical_values[0].tolist()}")
    print("     - 结论: 对于没有数值的句子，回归目标应全为0。")

    with torch.no_grad():
        # --- 步骤 4: 执行模型完整前向传播 ---
        print_step("步骤 4", "执行模型完整前向传播")
        outputs = model(
            input_ids=input_ids,
            numerical_values=numerical_values,
            attention_mask=attention_mask
        )
        print("   - 模型前向传播完成。")
        print(f"   - 输出 reg_loc 形状: {outputs['reg_loc'].shape}")

        # --- 步骤 5: 调用 `compute_loss` 计算损失 ---
        print_step("步骤 5", "调用 `compute_loss` 计算损失")
        loss_dict = model.compute_loss(
            outputs,
            targets=cls_labels,          # 分类目标
            numerical_values=numerical_values, # 回归目标
            attention_mask=attention_mask
        )
        
        reg_loss = loss_dict.get('reg_loss')
        
        print(f"\n   - 计算得到的总损失 (Total Loss): {loss_dict['total']:.4f}")
        print(f"   - 计算得到的分类损失 (Classification Loss): {loss_dict['cls_loss']:.4f}")
        print(f"   - 计算得到的回归损失 (Regression Loss): {reg_loss:.4f}")
        print(f"   - 门控权重均值: {loss_dict.get('gate_weights_mean', -1):.4f}")
        print(f"   - (理论: 回归损失应该只由包含数值的样本贡献)")

    print(f"\n\n{'='*80}")
    print("🎉 端到端回归流程验证完成！")
    print("   脚本清晰地展示了门控回归损失是如何基于对齐的数值目标计算的。")

if __name__ == '__main__':
    main() 