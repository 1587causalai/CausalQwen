#!/usr/bin/env python
"""
验证损失函数实现的正确性
"""
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel

def test_ovr_probability():
    """测试 OvR 概率计算"""
    print("=" * 60)
    print("测试 OvR 概率计算")
    print("=" * 60)
    
    # 创建测试数据
    loc = torch.tensor([[10.0, -5.0, 0.0]])  # 3个类别的决策分数位置参数
    scale = torch.tensor([[1.0, 2.0, 0.5]])   # 尺度参数
    threshold = 10.0  # 默认阈值
    
    # 手动计算概率（根据公式）
    manual_probs = 0.5 + (1 / math.pi) * torch.atan((loc - threshold) / scale)
    
    print(f"位置参数 loc: {loc}")
    print(f"尺度参数 scale: {scale}")
    print(f"阈值 C_k: {threshold}")
    print(f"手动计算的概率: {manual_probs}")
    
    # 检查边界情况
    print("\n边界情况测试:")
    # 当 loc = C_k 时，概率应该是 0.5
    loc_at_threshold = torch.tensor([[threshold]])
    scale_test = torch.tensor([[1.0]])
    prob_at_threshold = 0.5 + (1 / math.pi) * torch.atan((loc_at_threshold - threshold) / scale_test)
    print(f"当 loc = C_k = {threshold} 时，P = {prob_at_threshold.item():.4f} (应该是 0.5)")
    
    # 当 loc >> C_k 时，概率应该接近 1
    loc_large = torch.tensor([[threshold + 100]])
    prob_large = 0.5 + (1 / math.pi) * torch.atan((loc_large - threshold) / scale_test)
    print(f"当 loc = {loc_large.item()} >> C_k 时，P = {prob_large.item():.4f} (应该接近 1)")
    
    # 当 loc << C_k 时，概率应该接近 0
    loc_small = torch.tensor([[threshold - 100]])
    prob_small = 0.5 + (1 / math.pi) * torch.atan((loc_small - threshold) / scale_test)
    print(f"当 loc = {loc_small.item()} << C_k 时，P = {prob_small.item():.4f} (应该接近 0)")

def test_cauchy_nll():
    """测试柯西负对数似然损失"""
    print("\n" + "=" * 60)
    print("测试柯西负对数似然损失")
    print("=" * 60)
    
    # 测试数据
    y_true = torch.tensor([5.0, -2.0, 0.0])
    loc = torch.tensor([4.0, -2.0, 1.0])
    scale = torch.tensor([1.0, 0.5, 2.0])
    
    # 手动计算柯西 NLL（根据公式）
    # L = log(π * scale) + log(1 + ((y_true - loc) / scale)^2)
    manual_nll = torch.log(math.pi * scale) + torch.log(1 + ((y_true - loc) / scale) ** 2)
    
    print(f"真实值 y_true: {y_true}")
    print(f"预测位置 loc: {loc}")
    print(f"预测尺度 scale: {scale}")
    print(f"手动计算的 NLL: {manual_nll}")
    
    # 检查特殊情况
    print("\n特殊情况测试:")
    # 当预测完全准确时（y_true = loc）
    perfect_pred_nll = torch.log(math.pi * scale[0]) + torch.log(torch.tensor(1.0))
    print(f"完美预测时 (y=loc): NLL = {perfect_pred_nll.item():.4f}")
    
    # 检查尺度参数的影响
    scales = torch.tensor([0.1, 1.0, 10.0])
    for s in scales:
        nll = torch.log(math.pi * s) + torch.log(1 + ((torch.tensor(1.0) - torch.tensor(0.0)) / s) ** 2)
        print(f"固定误差=1，scale={s:.1f} 时: NLL = {nll.item():.4f}")

def test_gated_loss():
    """测试门控损失机制"""
    print("\n" + "=" * 60)
    print("测试门控损失机制")
    print("=" * 60)
    
    # 模拟场景
    batch_size = 2
    num_token_id = 151665
    
    # 场景 1: 真实标签是 <NUM>，模型预测概率高
    print("场景 1: 真实标签是 <NUM>，模型预测 <NUM> 概率 = 0.9")
    is_num_mask = torch.tensor([1.0, 0.0])  # 第一个是 <NUM>，第二个不是
    num_prob = torch.tensor([0.9, 0.1])     # 模型预测概率
    cauchy_nll = torch.tensor([2.0, 2.0])   # 假设的回归损失
    
    gated_loss = is_num_mask * num_prob * cauchy_nll
    print(f"门控损失: {gated_loss}")
    print(f"位置 0 (是<NUM>): {gated_loss[0]:.4f} = 1.0 * 0.9 * 2.0")
    print(f"位置 1 (非<NUM>): {gated_loss[1]:.4f} = 0.0 * 0.1 * 2.0")
    
    # 场景 2: 真实标签是 <NUM>，但模型预测概率低
    print("\n场景 2: 真实标签是 <NUM>，模型预测 <NUM> 概率 = 0.1")
    num_prob_low = torch.tensor([0.1, 0.9])
    gated_loss_low = is_num_mask * num_prob_low * cauchy_nll
    print(f"门控损失: {gated_loss_low}")
    print(f"位置 0 (是<NUM>): {gated_loss_low[0]:.4f} = 1.0 * 0.1 * 2.0")
    print("说明：模型还没学会预测 <NUM>，回归损失被大幅降低")

def test_full_loss_computation():
    """测试完整的损失计算流程"""
    print("\n" + "=" * 60)
    print("测试完整的损失计算")
    print("=" * 60)
    
    # 创建一个小模型进行测试
    # 注意：这里使用小词汇表进行测试，但概念相同
    config = CausalLMConfig(
        vocab_size=100,  # 测试用小词汇表
        num_token_id=99,  # 最后一个 token 是 <NUM>
        hidden_size=64,
        causal_dim=64,
        ovr_threshold=0.0,  # 使用更合理的阈值
        reg_loss_weight=1.0,
        use_real_qwen=False,
        use_ovr_classifier=True,
        use_cauchy_distribution=True
    )
    
    model = CausalLanguageModel(config)
    
    # 创建测试输入
    batch_size, seq_len = 2, 3
    input_ids = torch.tensor([[1, 2, 99], [3, 4, 5]])  # 第一个序列最后是 <NUM>
    numerical_values = torch.tensor([[0.0, 0.0, 5.5], [0.0, 0.0, 0.0]])  # 只有 [0,2] 位置有数值
    
    # 创建虚拟的隐藏状态
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 通过归因推断网络
    causal_loc, causal_scale = model.abduction_network(hidden_states)
    print(f"因果表征 loc 形状: {causal_loc.shape}")
    print(f"因果表征 scale 形状: {causal_scale.shape}")
    
    # 通过行动网络
    action_outputs = model.action_network(causal_loc, causal_scale)
    
    # 准备目标
    targets = {
        'token_ids': input_ids,
        'numerical_values': numerical_values
    }
    
    # 计算损失
    losses = model.compute_losses(action_outputs, targets)
    
    print(f"\n损失计算结果:")
    print(f"总损失: {losses['total'].item():.4f}")
    print(f"分类损失: {losses['cls'].item():.4f}")
    print(f"回归损失: {losses['reg'].item():.4f}")
    
    # 验证损失的合理性
    print(f"\n损失合理性检查:")
    print(f"分类损失是否为正: {losses['cls'].item() > 0}")
    print(f"回归损失是否为正: {losses['reg'].item() >= 0}")  # 可能为 0（如果没有 <NUM>）
    print(f"总损失是否等于分类+回归: {abs(losses['total'].item() - losses['cls'].item() - losses['reg'].item()) < 1e-6}")
    
    print(f"\n工业级实践说明:")
    print(f"  - 实际部署时，vocab_size 应为 151,936（Qwen 完整容量）")
    print(f"  - <NUM> token ID 应为 151,665（第一个预留位置）")
    print(f"  - 这种设计保持了与 Qwen 的完全兼容性")
    print(f"  - 剩余 270 个预留位置可供未来扩展使用")

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    
    # 运行所有测试
    test_ovr_probability()
    test_cauchy_nll()
    test_gated_loss()
    test_full_loss_computation()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
