#!/usr/bin/env python
"""
因果采样模式验证脚本

本脚本严格按照 `docs/mathematical_foundations.md` 中 "3.2 因果采样" 章节的定义，
对 CausalQwen 独有的因果采样推理模式进行白盒验证。

验证目标:
1. "采样原因": 验证模型能从 `Cauchy(loc_U, scale_U)` 分布中正确采样出具体的因果表征 `u`。
2. "观察结果": 验证将采样的 `u` 传入行动网络后，能得到确定性的分类和回归输出。
3. 随机性验证: 验证多次运行因果采样会产生不同的结果，证明随机性来源。
"""
import os
import sys
import torch
import torch.nn.functional as F

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

def print_tensor_stats(name, tensor):
    """打印张量的详细统计信息，使用对柯西分布鲁棒的指标。"""
    if not isinstance(tensor, torch.Tensor):
        print(f"   - {name}: Not a tensor")
        return
    tensor = tensor.detach().cpu().to(torch.float32)
    median = torch.median(tensor).item()
    q1 = torch.quantile(tensor, 0.25).item()
    q3 = torch.quantile(tensor, 0.75).item()
    iqr = q3 - q1
    print(f"   - {name}:")
    print(f"     - Shape:  {tensor.shape}")
    print(f"     - Median: {median:.4f} (中位数)")
    print(f"     - IQR:    {iqr:.4f} (四分位距)")

def main():
    print("🚀 CausalQwen - 因果采样 (Causal Sampling) 模式验证 🚀")

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
        causal_dim=model_info['hidden_size'],
        use_real_qwen=True,
        qwen_model_path=qwen_model_path,
        use_numerical_features=True,
    )
    model = CausalLanguageModel(config).to(device)
    model.init_weights()
    model.eval()
    print("   ✅ 模型和分词器初始化完成。")

    # --- 步骤 2: 准备输入数据 ---
    print_step("步骤 2", "准备单个输入样本")
    text = "The price is 99.9."
    inputs = tokenizer.batch_encode_plus([text], padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    numerical_values = inputs['numerical_values'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    print(f"   - 输入文本: '{text}'")

    # --- 步骤 3: 获取因果分布参数 ---
    print_step("步骤 3", "前向传播获取因果表征 U 的分布参数")
    with torch.no_grad():
        # 执行完整的模型前向传播
        outputs = model(
            input_ids=input_ids,
            numerical_values=numerical_values,
            attention_mask=attention_mask,
        )
        loc_U = outputs.get('causal_loc')
        scale_U = outputs.get('causal_scale')

    # 分析最后一个词元的分布
    last_token_idx = -1
    loc_U_last = loc_U[:, last_token_idx, :]
    scale_U_last = scale_U[:, last_token_idx, :]
    print("   - 已获取最后一个词元的因果表征分布参数:")
    print_tensor_stats("loc_U (last token)", loc_U_last)
    print_tensor_stats("scale_U (last token)", scale_U_last)

    # --- 步骤 4: 执行因果采样 ---
    print_step("步骤 4", "执行两次因果采样以验证随机性")

    previous_u_sample = None
    for i in range(2):
        print(f"\n   --- 采样运行 #{i+1} ---")
        
        # 1. 采样"原因"
        torch.manual_seed(42 + i) # 使用不同的种子以确保结果不同
        cauchy_dist = torch.distributions.Cauchy(loc_U_last, scale_U_last)
        u_sample = cauchy_dist.sample()
        
        print("   - 步骤 4.1: 采样'原因' (Sample the Cause)")
        print_tensor_stats("采样得到的 u_sample", u_sample)

        # 2. 观察'结果' - 手动执行确定性线性变换
        with torch.no_grad():
            # 分类: logit = u * W_cls^T
            cls_logits = F.linear(
                u_sample,
                weight=model.action_network.classification_head.causal_linear.weight
            )
            # 回归: value = u * W_reg^T + b_reg
            reg_output = F.linear(
                u_sample,
                weight=model.action_network.regression_head.causal_linear.weight,
                bias=model.action_network.regression_head.causal_linear.bias
            ).squeeze(-1) # 挤压掉最后的维度
        
        predicted_token_id = torch.argmax(cls_logits, dim=-1).item()
        predicted_token = tokenizer.decode([predicted_token_id])
        predicted_value = reg_output.item()

        print("\n   - 步骤 4.2: 观察'结果' (Observe the Effect)")
        print(f"     - 预测的词元 (Argmax of Logits): '{predicted_token}' (ID: {predicted_token_id})")
        print(f"     - 预测的数值: {predicted_value:.4f}")

        # 3. 验证与上一次采样的结果是否不同
        if previous_u_sample is not None:
            u_is_different = not torch.allclose(u_sample, previous_u_sample)
            print("\n   - 步骤 4.3: 随机性验证")
            print(f"     - 本次采样的 `u` 与上次是否不同? {'✅ 是' if u_is_different else '❌ 否'}")
            assert u_is_different, "两次采样的 u 相同，随机性可能存在问题！"
        
        previous_u_sample = u_sample

    print(f"\n\n{'='*80}")
    print("🎉 验证成功！因果采样流程符合预期。")
    print("   - 成功从 `Cauchy(loc_U, scale_U)` 分布中采样得到了 `u`。")
    print("   - 成功将 `u` 传入行动网络获得了确定性输出。")
    print("   - 连续两次采样产生了不同的 `u`，验证了随机性。")

if __name__ == '__main__':
    main() 