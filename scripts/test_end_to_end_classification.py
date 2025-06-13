#!/usr/bin/env python
"""
端到端分类流程验证脚本

本脚本旨在完整地展示从原始文本输入到最终分类损失计算的全过程。
它使用 `test_numerical_aware_embedding.py` 中的4个真实语料样本，
通过完整的 CausalQwen 模型，验证核心的分类任务流程。
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
    print("🚀 CausalQwen - 端到端分类流程验证")
    
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
        ovr_threshold=100.0 # 与之前的测试保持一致
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
    print(f"   - 输入形状 (input_ids): {input_ids.shape}")

    # --- 步骤 3: 准备用于损失计算的标签 (Labels) ---
    print_step("步骤 3", "准备用于损失计算的标签 (Labels)")
    # 语言模型的目标是预测下一个词，所以标签是输入向左移动一位。
    # 我们创建一个与输入形状相同，但填充为-100的标签张量。
    # 然后，我们将 input_ids 的 [1:] 部分复制到 labels 的 [:-1] 部分。
    # 这样，模型在位置 i 的输出，将与位置 i+1 的输入进行比较。
    # 最后一个词元的标签将保持-100，损失函数会自动忽略它。
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]
    
    print("   - 标签已生成 (input_ids 左移一位，最后一个词元和 padding 位置将被忽略)")
    
    print("\n   --- 查看样本 2 的输入和标签 ---")
    sample_idx = 1
    # Decode for human readability - skip special tokens for input, but not for labels
    original_tokens = tokenizer.decode(input_ids[sample_idx, :attention_mask[sample_idx].sum()])
    
    print(f"     原始句子: '{original_tokens}'")
    print(f"     输入 IDs: {input_ids[sample_idx].tolist()}")
    print(f"     标签 IDs: {labels[sample_idx].tolist()}")
    print("     (注: -100 对应位置不计算损失)")


    with torch.no_grad():
        # --- 步骤 4: 执行模型完整前向传播 ---
        print_step("步骤 4", "执行模型完整前向传播")
        outputs = model(
            input_ids=input_ids,
            numerical_values=numerical_values,
            attention_mask=attention_mask
        )
        print("   - 模型前向传播完成。")
        print(f"   - 输出 cls_loc 形状: {outputs['cls_loc'].shape}")


        # --- 步骤 5: 调用 `compute_loss` 计算损失 ---
        print_step("步骤 5", "调用 `compute_loss` 计算损失")
        loss_dict = model.compute_loss(
            outputs,
            targets=labels,
            numerical_values=numerical_values, # 回归目标，这里我们不关心
            attention_mask=attention_mask
        )
        
        cls_loss = loss_dict.get('cls_loss')
        
        print(f"\n   - 计算得到的总损失 (Total Loss): {loss_dict['total']:.4f}")
        print(f"   - 计算得到的分类损失 (Classification Loss): {cls_loss:.4f}")
        print(f"   - 计算得到的回归损失 (Regression Loss): {loss_dict['reg_loss']:.4f}")

    print(f"\n\n{'='*80}")
    print("🎉 端到端分类流程验证完成！")
    print("   脚本清晰地展示了从原始文本 -> Tokenization -> Labels -> Forward Pass -> Loss 的完整过程。")

if __name__ == '__main__':
    main() 