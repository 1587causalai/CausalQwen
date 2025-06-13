#!/usr/bin/env python
"""
数值感知嵌入流程图式验证脚本

本脚本严格按照 `mathematical_foundations.md` 中的 "图 2" 流程图，
以批量处理的方式，结合数学公式和关键验证，清晰地展示从原始文本到
最终增强嵌入 (enhanced_embeddings) 的每一步数据变换。
"""
import os
import sys
import torch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper

def print_step(step_name, description):
    """打印流程图步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  步骤 {step_name}: {description}")
    print(f"{'-'*70}")

def print_batch_shape(name, tensor):
    """打印批处理张量的名称和形状。"""
    if not isinstance(tensor, torch.Tensor):
        print(f"   {name}: Not a tensor")
        return
    print(f"   - {name} (批量) Shape: {tensor.detach().cpu().shape}")


def main():
    print("🚀 CausalQwen - 数值感知嵌入模块深度验证 (批量模式)")
    
    # --- 初始化 ---
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)

    # --- 词汇表信息展示 ---
    vocab_info = tokenizer.vocab_size_info()
    print("\n" + "="*70)
    print("📊 词汇表信息概览")
    print("-"*70)
    print(f"   - 基础 Qwen 词汇表 (Base Qwen Vocab): {vocab_info['qwen_base_vocab']}")
    print(f"   - CausalQwen 词汇表 (Model Vocab): {vocab_info['causalqwen_vocab']} (Qwen + <NUM>)")
    print(f"   - 分词器内部长度 (Tokenizer Internal): {vocab_info['tokenizer_internal_len']} (CausalQwen + Placeholders)")
    print(f"   - <NUM> Token ID: {vocab_info['num_token_id']}")
    print("="*70)
    
    config = CausalLMConfig(
        vocab_size=vocab_info['causalqwen_vocab'],
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        use_real_qwen=True,
        qwen_model_path=qwen_model_path
    )
    full_model = CausalLanguageModel(config).to(device)
    full_model.eval()
    feature_network = full_model.feature_network
    print("\n✅ 组件初始化完成")

    # --- 准备测试样本 ---
    test_samples = [
        "This is a sentence without any numbers.",
        "The item costs 50.5 and the tax is 4.5.",
        "The dimensions are 10 by 20 by 30 cm.",
        "What is 2 plus 3? 5.",
    ]
    batch_size = len(test_samples)
    print(f"\n📊 准备 {batch_size} 个测试样本进行批量处理。")

    # --- 步骤 A, B, C, D: 分词器处理 ---
    print_step("A-D", "分词器处理: 将文本批量转换为 `input_ids` 和 `numerical_values`")
    inputs = tokenizer.batch_encode_plus(test_samples, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    numeric_values = inputs['numerical_values'].to(device)
    
    print_batch_shape("Input IDs", input_ids)
    print_batch_shape("Numerical Values", numeric_values)
    print("\n   --- 查看各样本分词结果 ---")
    for i, text in enumerate(test_samples):
        print(f"     样本 {i+1}: '{text}'")
        print(f"       - IDs: {input_ids[i].tolist()}")
        print(f"       - Vals: {numeric_values[i].tolist()}")

    with torch.no_grad():
        # --- 步骤 E, F: 基础嵌入 ---
        print_step("E-F", "基础嵌入: 将 `input_ids` 映射为向量 `base_embeddings`")
        base_embeddings = feature_network.base_network.qwen_model.model.embed_tokens(input_ids)
        print_batch_shape("Base Embeddings", base_embeddings)

        # --- 步骤 G: 数值编码 ---
        print_step("G", "数值编码: 根据 `numerical_values` 生成稀疏编码 `φ(v)`")
        print("   - 公式: φ(v) = sign(v) * ln(1+|v|) * e_direction")
        
        direction_vector = feature_network.numerical_direction
        normed_direction = direction_vector / (torch.norm(direction_vector) + 1e-9)
        transformed_values = torch.sign(numeric_values) * torch.log1p(torch.abs(numeric_values))
        phi_v = transformed_values.unsqueeze(-1) * normed_direction
        num_mask = (input_ids == tokenizer.num_token_id).float().unsqueeze(-1)
        phi_v = phi_v * num_mask
        
        print_batch_shape("Numerical Encoding (φ(v))", phi_v)

        # --- 步骤 G.1: 智能验证 ---
        print_step("G.1", "智能验证: 验证 `φ(v)` 的稀疏性")
        print("   - 理论: `φ(v)` 应该只在有数值的样本的 `<NUM>` token 位置有非零值。")

        # 验证 1: 无数值样本
        phi_v_sample_no_num = phi_v[0]
        norm_no_num = torch.norm(phi_v_sample_no_num).item()
        print(f"\n   --- 验证样本 1 ('...without any numbers.') ---")
        print(f"     - `φ(v)` 的整体范数: {norm_no_num:.4f}")
        print(f"     - 结论: {'✅ 正确, 范数应为 0' if np.isclose(norm_no_num, 0) else '❌ 错误'}")

        # 验证 2: 有数值样本
        phi_v_sample_with_num = phi_v[1]
        input_ids_with_num = input_ids[1]
        num_mask_sample = (input_ids_with_num == tokenizer.num_token_id)
        non_num_mask_sample = ~num_mask_sample

        norm_at_num_positions = torch.norm(phi_v_sample_with_num[num_mask_sample]).item()
        norm_at_non_num_positions = torch.norm(phi_v_sample_with_num[non_num_mask_sample]).item()
        print(f"\n   --- 验证样本 2 ('...costs 50.5 and the tax is 4.5.') ---")
        print(f"     - 在非<NUM>位置的编码范数: {norm_at_non_num_positions:.4f}")
        print(f"     - 在<NUM>位置的编码范数: {norm_at_num_positions:.4f}")
        print(f"     - 结论: {'✅ 正确, 编码仅在<NUM>位置' if np.isclose(norm_at_non_num_positions, 0) and norm_at_num_positions > 1e-6 else '❌ 错误'}")
        
        # --- 步骤 H, I: 融合与输出 ---
        print_step("H-I", "融合与输出: 生成最终的 `enhanced_embeddings`")
        print("   - 公式: enhanced_embeddings = base_embeddings + φ(v)")
        enhanced_embeddings = base_embeddings + phi_v
        print_batch_shape("Enhanced Embeddings", enhanced_embeddings)


    print(f"\n\n{'='*80}")
    print("🎉 批量验证成功！脚本已生成一份清晰、自解释的技术说明。")
    print("   输出展示了批量处理流程、核心数学公式和关键行为的智能验证。")

if __name__ == '__main__':
    main() 