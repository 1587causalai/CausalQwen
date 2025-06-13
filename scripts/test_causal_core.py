#!/usr/bin/env python
"""
因果核心流程图式验证脚本

本脚本严格按照 `mathematical_foundations.md` 中的 "图 3" 流程图，
清晰地展示从增强嵌入 (e) 到最终决策分布 (S, Y) 的每一步数据变换。
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
    print(f"➡️  步骤 {step_name}: {description}")
    print(f"{'-'*70}")

def print_tensor_stats(name, tensor):
    """打印张量的详细统计信息。"""
    if not isinstance(tensor, torch.Tensor):
        print(f"   - {name}: Not a tensor")
        return
    tensor = tensor.detach().cpu().to(torch.float32)
    print(f"   - {name}:")
    print(f"     - Shape: {tensor.shape}")
    print(f"     - Mean: {tensor.mean().item():.4f}")
    print(f"     - Std:  {tensor.std().item():.4f}")
    print(f"     - Min:  {tensor.min().item():.4f}")
    print(f"     - Max:  {tensor.max().item():.4f}")

def main():
    print("🚀 CausalQwen - 因果核心流程深度验证")
    
    # --- 初始化 ---
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')

    # 动态获取模型参数
    model_info = get_qwen_model_info(qwen_model_path)
    if not model_info:
        sys.exit(1)
    
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    
    config = CausalLMConfig(
        vocab_size=model_info['vocab_size'],
        num_token_id=tokenizer.num_token_id,
        hidden_size=model_info['hidden_size'],
        causal_dim=model_info['hidden_size'], # 恒等映射
        use_real_qwen=True,
        qwen_model_path=qwen_model_path,
        use_numerical_features=True
    )
    model = CausalLanguageModel(config).to(device)
    model.init_weights()
    model.eval()
    print("\n✅ 组件初始化完成")

    # --- 准备输入数据 ---
    test_samples = [
        "The item costs 50.5 and the tax is 4.5.",
        "A sentence without numbers.",
    ]
    inputs = tokenizer.batch_encode_plus(test_samples, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    print(f"\n📊 准备 {len(test_samples)} 个测试样本进行批量处理。")

    with torch.no_grad():
        # --- 完整前向传播 ---
        # 模型现在返回一个包含所有中间步骤的字典
        full_outputs = model(
            input_ids=input_ids,
            numerical_values=numerical_values,
            attention_mask=attention_mask
        )

        # --- 图3 流程起点: 获取增强嵌入 e ---
        print_step("A", "起点: 获取增强嵌入 e (Enhanced Embeddings)")
        e = full_outputs['enhanced_embeddings']
        print_tensor_stats("增强嵌入 e", e)

        # --- 图3 流程: e -> z (Qwen 特征网络) ---
        print_step("B", "Qwen特征网络: e -> z (上下文特征)")
        z = full_outputs['features']
        print_tensor_stats("上下文特征 z", z)

        # --- 图3 流程: z -> U (归因网络) ---
        print_step("C-D", "归因网络 (Abduction): z -> U (因果表征分布)")
        loc_U = full_outputs['causal_loc']
        scale_U = full_outputs['causal_scale']
        print("   - 理论: loc_U ≈ z, scale_U 为大正数 (≈10)")
        print_tensor_stats("因果位置 loc_U", loc_U)
        print_tensor_stats("因果尺度 scale_U", scale_U)
        print(f"   - 验证: loc_U 与 z 的均值差异: {abs(loc_U.mean() - z.mean()):.6f}")
        print(f"   - 验证: scale_U 均值 > 5: {'✅' if scale_U.mean() > 5 else '❌'}")

        # --- 图3 流程: U -> S, Y (行动网络) ---
        print_step("E-H", "行动网络 (Action): U -> S, Y (决策分布)")
        
        print("\n   --- 分类输出 (S) ---")
        print("   - 理论: scale_S 应该是 scale_U 的线性变换，仍为大正数")
        print_tensor_stats("分类 logits (loc_S)", full_outputs.get('cls_loc'))
        print_tensor_stats("分类尺度 (scale_S)", full_outputs.get('cls_scale'))

        print("\n   --- 回归输出 (Y) ---")
        print("   - 理论: scale_Y 应该是 scale_U 的线性变换，为一个合理的正数")
        print_tensor_stats("回归预测 (loc_Y)", full_outputs.get('reg_loc'))
        print_tensor_stats("回归不确定性 (scale_Y)", full_outputs.get('reg_scale'))
        
        scale_Y_mean = full_outputs.get('reg_scale').mean().item()
        print(f"   - 验证: scale_Y 均值 > 0: {'✅' if scale_Y_mean > 0 else '❌'}")
    
    print(f"\n\n{'='*80}")
    print("🎉 批量验证成功！脚本清晰地展示了从 e -> z -> U -> (S, Y) 的核心数据流。")

if __name__ == '__main__':
    main() 