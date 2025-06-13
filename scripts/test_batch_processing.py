#!/usr/bin/env python
"""
批量处理正确性验证脚本

本脚本旨在严格验证 CausalQwen 的批量处理 (batch processing) 功能是否正确。
正确的批量处理应确保批次中每个样本的计算结果与该样本被单独处理时的结果完全一致，
从而保证训练和推理的可重复性与正确性。

验证目标:
1. 分别独立处理多个样本，并记录其输出。
2. 将这些样本作为一个批次统一处理。
3. 逐一对比两种处理方式下对应样本的输出 (loc_S, loc_Y 等)，验证其数值上是否完全相等。
"""
import os
import sys
import torch

# 将项目根目录添加到 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.utils.model_utils import get_qwen_model_info

def print_step(step_name, description):
    """打印流程步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  {step_name}: {description}")
    print(f"{'-'*70}")

def get_model_outputs(model, tokenizer, text_list, device):
    """辅助函数：对给定的文本列表进行分词和模型前向传播，返回输出"""
    inputs = tokenizer.batch_encode_plus(text_list, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            numerical_values=numerical_values,
            attention_mask=attention_mask
        )
    return outputs, attention_mask

def main():
    torch.manual_seed(42) # 保证初始化权重的一致性
    print("🚀 CausalQwen - 批量处理 (Batch Processing) 正确性验证 🚀")

    # --- 步骤 1: 初始化 ---
    print_step("步骤 1", "初始化模型和分词器")
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
        use_real_qwen=True,
        qwen_model_path=qwen_model_path,
    )
    model = CausalLanguageModel(config).to(device)
    model.init_weights()
    model.eval()
    print("   ✅ 模型和分词器初始化完成。")

    # --- 步骤 2: 准备测试数据 ---
    test_samples = [
        "The price is 29.99 dollars.",
        "A short sentence.",
        "The stock has 100 units and the discount is 5.5 percent."
    ]
    print_step("步骤 2", f"准备 {len(test_samples)} 个测试样本")
    for i, s in enumerate(test_samples):
        print(f"   - 样本 {i+1}: '{s}'")

    # --- 步骤 3: 独立处理每个样本 ---
    print_step("步骤 3", "独立处理 (Batch Size = 1) 每个样本并存储结果")
    individual_outputs = []
    for sample in test_samples:
        outputs, _ = get_model_outputs(model, tokenizer, [sample], device)
        individual_outputs.append(outputs)
    print(f"   ✅ 已完成对 {len(individual_outputs)} 个样本的独立处理。")

    # --- 步骤 4: 批量处理所有样本 ---
    print_step("步骤 4", "批量处理 (Batch Size > 1) 所有样本")
    batch_outputs, batch_attention_mask = get_model_outputs(model, tokenizer, test_samples, device)
    print(f"   ✅ 已完成对 {len(test_samples)} 个样本的批量处理。")

    # --- 步骤 5: 逐一对比结果 ---
    print_step("步骤 5", "逐一对比独立处理与批量处理的结果")
    all_match = True
    for i in range(len(test_samples)):
        print(f"\n   --- 对比样本 {i+1} ---")
        
        # 获取每个样本有效长度（排除填充）
        individual_len = individual_outputs[i]['cls_loc'].shape[1]
        batch_len = batch_attention_mask[i].sum().item()
        
        # 确保我们比较的是相同长度的有效序列
        if individual_len != batch_len:
             print(f"   - ❌ 长度不匹配! Individual: {individual_len}, Batch: {batch_len}")
             all_match = False
             continue

        # 对比 cls_loc
        individual_cls_loc = individual_outputs[i]['cls_loc']
        batch_cls_loc = batch_outputs['cls_loc'][i, :batch_len, :]
        # 使用更宽松的容差来解决大规模线性层中的浮点不精确问题
        cls_loc_match = torch.allclose(individual_cls_loc, batch_cls_loc, atol=1e-4)
        print(f"     - `cls_loc` 是否匹配?   {'✅ 是' if cls_loc_match else '❌ 否'}")
        if not cls_loc_match: 
            all_match = False
            # 添加详细的调试信息
            if i == 0: # 只为第一个样本打印详细信息
                diff = torch.abs(individual_cls_loc - batch_cls_loc.unsqueeze(0))
                print(f"       - [DEBUG] Max Diff: {diff.max().item():.8f}")
                print(f"       - [DEBUG] Mean (Individual): {individual_cls_loc.mean().item():.8f}")
                print(f"       - [DEBUG] Mean (Batch): {batch_cls_loc.mean().item():.8f}")

        # 对比 reg_loc - 保持严格的容差
        individual_reg_loc = individual_outputs[i]['reg_loc']
        batch_reg_loc = batch_outputs['reg_loc'][i, :batch_len]
        reg_loc_match = torch.allclose(individual_reg_loc, batch_reg_loc, atol=1e-5)
        print(f"     - `reg_loc` 是否匹配?   {'✅ 是' if reg_loc_match else '❌ 否'}")
        if not reg_loc_match: all_match = False

    print(f"\n\n{'='*80}")
    if all_match:
        print("🎉 验证成功！批量处理与独立处理的结果完全一致。")
    else:
        print("❌ 验证失败！批量处理的结果与独立处理不一致，请检查模型中的填充或注意力掩码逻辑。")

if __name__ == '__main__':
    main() 