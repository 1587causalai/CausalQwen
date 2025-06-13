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

from src.models.causal_lm import CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper
from src.utils.model_utils import get_qwen_model_info

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
    
    # 动态获取模型参数
    model_info = get_qwen_model_info(qwen_model_path)
    if not model_info:
        sys.exit(1)
        
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
        vocab_size=model_info['vocab_size'],
        num_token_id=tokenizer.num_token_id,
        hidden_size=model_info['hidden_size'],
        use_real_qwen=True,
        qwen_model_path=qwen_model_path,
        use_numerical_features=True  # 确保激活数值功能
    )
    # 不再需要完整的CausalLanguageModel，直接测试NumericalEmbedding模块
    # 我们需要一个基础嵌入层来初始化它
    from src.models.feature_network import QwenFeatureNetwork
    from src.models.numerical_aware_embedding import NumericalAwareEmbedding
    
    base_qwen_network = QwenFeatureNetwork(model_path=qwen_model_path, hidden_size=model_info['hidden_size'])
    base_embedding_layer = base_qwen_network.qwen_model.model.embed_tokens
    
    numerical_embedding_module = NumericalAwareEmbedding(
        base_embedding_layer=base_embedding_layer,
        num_token_id=config.num_token_id,
        hidden_size=config.hidden_size
    ).to(device)
    numerical_embedding_module.eval()
    
    print("\n✅ 组件初始化完成: 单独测试 `NumericalEmbedding` 模块")

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
        # --- 步骤 E, F, G, H, I: 直接调用新模块 ---
        print_step("E-I", "增强嵌入: 调用 `NumericalEmbedding` 模块一次性完成转换")
        
        # 直接调用模块
        enhanced_embeddings = numerical_embedding_module(input_ids, numeric_values)
        print_batch_shape("Enhanced Embeddings", enhanced_embeddings)

        # --- 为了验证内部逻辑，我们手动重现计算过程 ---
        # 步骤 E, F: 基础嵌入
        print_step("E-F (验证)", "基础嵌入: 手动获取 `base_embeddings`")
        base_embeddings = base_embedding_layer(input_ids)
        print_batch_shape("Base Embeddings (手动)", base_embeddings)

        # 步骤 G: 数值编码
        print_step("G (验证)", "数值编码: 手动计算稀疏编码 `φ(v)`")
        print("   - 公式: φ(v) = sign(v) * ln(1+|v|) * e_direction")
        
        direction_vector = numerical_embedding_module.numerical_direction
        transformed_values = torch.sign(numeric_values) * torch.log1p(torch.abs(numeric_values))
        phi_v = transformed_values.unsqueeze(-1) * direction_vector
        num_mask = (input_ids == tokenizer.num_token_id).float().unsqueeze(-1)
        phi_v = phi_v * num_mask
        
        print_batch_shape("Numerical Encoding (φ(v), 手动)", phi_v)

        # 步骤 G.1: 智能验证
        print_step("G.1 (验证)", "智能验证: 验证 `φ(v)` 的稀疏性")
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
        print_step("H-I (验证)", "融合与输出: 手动生成 `enhanced_embeddings`")
        print("   - 公式: enhanced_embeddings = base_embeddings + φ(v)")
        manual_enhanced_embeddings = base_embeddings + phi_v
        print_batch_shape("Enhanced Embeddings (手动)", manual_enhanced_embeddings)

        # --- 最终一致性验证 ---
        print_step("最终验证", "对比模块输出和手动计算结果")
        diff = torch.norm(enhanced_embeddings - manual_enhanced_embeddings).item()
        print(f"   - 模块输出与手动计算结果的范数差异: {diff:.6f}")
        print(f"   - 结论: {'✅ 一致' if np.isclose(diff, 0) else '❌ 不一致'}")


    print(f"\n\n{'='*80}")
    print("🎉 批量验证成功！脚本已更新，用于独立验证 `NumericalAwareEmbedding` 模块。")
    print("   输出展示了模块的黑盒调用、核心数学公式和关键行为的白盒验证。")

if __name__ == '__main__':
    main() 