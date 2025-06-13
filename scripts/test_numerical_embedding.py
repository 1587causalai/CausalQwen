#!/usr/bin/env python
"""
数值感知嵌入模块 (Numerical-aware Embedding) 专项测试脚本

本脚本旨在清晰、独立地验证数值感知嵌入模块的数学正确性，
确保其行为与 `mathematical_foundations.md` 中定义的理论完全一致。
"""
import os
import sys
import torch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig
from src.models.feature_network import NumAwareFeatureNetwork
from src.data.tokenizer import QwenTokenizerWrapper
from src.models.causal_lm import CausalLanguageModel

def print_step(step_num, description):
    """打印步骤信息"""
    print(f"\n{'='*70}")
    print(f"➡️  步骤 {step_num}: {description}")
    print(f"{'='*70}")

def print_tensor_stats(name, tensor):
    """打印张量的简要统计信息。"""
    if not isinstance(tensor, torch.Tensor):
        print(f"   - {name}: Not a tensor")
        return
    tensor = tensor.detach().cpu().to(torch.float32)
    print(f"   - {name} | Shape: {tensor.shape} | Mean: {tensor.mean():.4f} | Std: {tensor.std():.4f}")

def main():
    print("🚀 CausalQwen - 数值感知嵌入模块专项测试")
    
    # --- 1. 初始化 ---
    print_step(1, "初始化分词器和模型组件")
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    assert os.path.exists(qwen_model_path), "Qwen模型路径不存在"
    
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size_info()['causalqwen_vocab'],
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        use_real_qwen=True, # 必须加载真实模型以获取嵌入层
        qwen_model_path=qwen_model_path
    )
    
    # 必须创建完整的CausalLanguageModel以正确初始化所有组件
    full_model = CausalLanguageModel(config).to(device)
    full_model.eval()
    
    # 从完整模型中提取我们想要测试的组件
    feature_network = full_model.feature_network
    
    print("✅ 组件初始化完成")
    print(f"   <NUM> token ID: {config.num_token_id}")

    # --- 2. 准备测试数据 ---
    print_step(2, "准备两组测试数据 (有/无数值)")
    text_with_num = "The price is 99.9 dollars"
    text_without_num = "Hello world, this is a test"
    
    inputs_with_num = tokenizer.batch_encode_plus([text_with_num], return_tensors='pt')
    inputs_without_num = tokenizer.batch_encode_plus([text_without_num], return_tensors='pt')

    print(f"   - 有数值文本: '{text_with_num}'")
    print(f"   - 无数值文本: '{text_without_num}'")

    # --- 3. 核心验证：模块的整体行为 ---
    print_step(3, "验证模块的整体行为")
    print("理论：对于有数值和无数值的输入，模块输出的差异应该只体现在数值编码上。")

    with torch.no_grad():
        # 获取"无数值"情况下的输出特征
        # 注意：这里我们调用完整的 forward，而不是试图拆解它
        # `numerical_values` 仍然需要传入，即使它全是零
        features_without_num = feature_network(
            input_ids=inputs_without_num['input_ids'],
            numerical_values=inputs_without_num['numerical_values']
        )
        
        # 获取"有数值"情况下的输出特征
        features_with_num = feature_network(
            input_ids=inputs_with_num['input_ids'],
            numerical_values=inputs_with_num['numerical_values']
        )

    print_tensor_stats("输出 (无数值)", features_without_num)
    print_tensor_stats("输出 (有数值)", features_with_num)
    
    # --- 4. 验证数值编码的独立性 ---
    print_step(4, "验证数值编码的独立性 (e' = e_base + φ(v))")
    print("理论：模块的 `forward` 方法输出的特征，可以被看作是 `e_base + φ(v)`。")
    
    with torch.no_grad():
        ids = inputs_with_num['input_ids']
        num_values = inputs_with_num['numerical_values']
        
        # 找到<NUM>词元的位置
        num_token_pos = (ids == config.num_token_id).nonzero(as_tuple=True)
        pos_idx = num_token_pos[1][0]

        # 1. 计算基础特征 (base_features)
        # 这是模块在没有数值信息时应该输出的结果。
        # 我们通过传入一个全零的 numerical_values 来模拟这种情况。
        zeros_for_nums = torch.zeros_like(num_values)
        base_features_equivalent = feature_network(
            input_ids=ids,
            numerical_values=zeros_for_nums
        )

        # 2. 计算实际输出 (features_with_num)
        # 这个我们已经在上一步计算过了
        
        # 3. 计算差值，它应该等于数值编码 φ(v)
        phi_v = features_with_num - base_features_equivalent

    print_tensor_stats("模拟的基础特征 (e_base)", base_features_equivalent)
    print_tensor_stats("实际的输出 (e')", features_with_num)
    print_tensor_stats("两者的差值 (φ(v))", phi_v)

    # 验证非<NUM>位置
    print("\n   --- 验证非<NUM>位置 ---")
    is_zero_elsewhere = (phi_v[:, :pos_idx].abs().max() < 1e-8) and \
                        (phi_v[:, pos_idx+1:].abs().max() < 1e-8)
    print(f"   - 非<NUM>位置的差值为零: {'✅ 通过' if is_zero_elsewhere else '❌ 失败'}")
    assert is_zero_elsewhere, "非<NUM>位置的嵌入不应被改变"

    # 验证<NUM>位置
    print("\n   --- 验证<NUM>位置 ---")
    phi_v_at_pos = phi_v[0, pos_idx]
    is_nonzero_at_pos = phi_v_at_pos.abs().max() > 1e-6
    print(f"   - <NUM>位置的差值非零: {'✅ 通过' if is_nonzero_at_pos else '❌ 失败'}")
    assert is_nonzero_at_pos, "<NUM>位置的嵌入应该被改变"

    # 根据公式验证 φ(v) 的范数
    v = num_values[num_token_pos].item()
    expected_norm = np.abs(np.log(1 + np.abs(v)))
    actual_norm = torch.norm(phi_v_at_pos).item()
    norm_match = np.isclose(expected_norm, actual_norm, rtol=1e-6)
    print(f"   - 验证 φ({v:.4f}) 的范数:")
    print(f"     - 理论范数 |ln(1+|v|)|: {expected_norm:.6f}")
    print(f"     - 实际范数 ||φ(v)||: {actual_norm:.6f}")
    print(f"     - 范数是否匹配: {'✅ 通过' if norm_match else '❌ 失败'}")
    assert norm_match, "φ(v) 的范数与理论不符"
    
    print("\n   结论：数值编码模块的行为完全符合数学定义，验证通过！")

if __name__ == '__main__':
    main() 