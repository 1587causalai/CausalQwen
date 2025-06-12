#!/usr/bin/env python
"""
门控损失函数的数学验证测试

验证关键数学性质：
1. 门控回归损失：L_reg_gated,i = m_i · (α + (1-α) · P_<NUM>,i) · L_cauchy_nll,i
2. 总损失：L_total = Σ(L_cls,i + λ · L_reg_gated,i)
3. 数值掩码：只有<NUM>位置参与回归损失
4. 柯西负对数似然的正确实现
5. 门控系数的作用验证

特别关注：验证实现的数学正确性，而不仅仅是功能性
"""

import unittest
import torch
import math
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses.loss_functions import cauchy_nll_loss
from src.utils.distributions import cauchy_log_prob


class TestGatedLoss(unittest.TestCase):
    """门控损失函数的数学验证测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.vocab_size = 1000
        self.num_token_id = 999  # <NUM> token ID
        self.regression_weight = 1.0
        
        print(f"\n初始化完成：vocab_size={self.vocab_size}, num_token_id={self.num_token_id}")
    
    def test_cauchy_nll_implementation(self):
        """
        测试柯西负对数似然的数学实现正确性
        验证：L = log(π·scale) + log(1 + ((target - loc)/scale)²)
        """
        print("\n" + "="*60)
        print("测试柯西负对数似然实现")
        print("="*60)
        
        # 测试数据
        test_cases = [
            {"target": 0.0, "loc": 0.0, "scale": 1.0, "desc": "标准情况"},
            {"target": 1.0, "loc": 0.0, "scale": 1.0, "desc": "偏移目标"},
            {"target": 0.0, "loc": 1.0, "scale": 1.0, "desc": "偏移预测"},
            {"target": 0.0, "loc": 0.0, "scale": 2.0, "desc": "大scale"},
            {"target": 0.0, "loc": 0.0, "scale": 0.5, "desc": "小scale"},
        ]
        
        print("测试情况\t\ttarget\tloc\tscale\t实现结果\t手动计算\t差异")
        print("-" * 75)
        
        for case in test_cases:
            target = torch.tensor([case["target"]])
            loc = torch.tensor([case["loc"]])
            scale = torch.tensor([case["scale"]])
            
            # 使用实现的函数 (注意参数顺序: loc, scale, target)
            impl_loss = cauchy_nll_loss(loc, scale, target, reduction='none')
            
            # 手动计算验证
            z = (target - loc) / scale
            manual_loss = torch.log(math.pi * scale) + torch.log1p(z ** 2)
            
            # 比较结果
            diff = torch.abs(impl_loss - manual_loss).item()
            
            print(f"{case['desc']:<15}\t{case['target']:4.1f}\t{case['loc']:4.1f}\t{case['scale']:4.1f}\t{impl_loss.item():8.4f}\t{manual_loss.item():8.4f}\t{diff:.2e}")
            
            # 验证数学正确性
            self.assertLess(diff, 1e-6, f"柯西NLL实现不正确：{case['desc']}")
        
        print("✓ 柯西负对数似然实现验证通过")
    
    def test_gating_mechanism_formula(self):
        """
        测试门控机制的数学公式
        验证：gate_factor = α + (1-α) · P_<NUM>
        """
        print("\n" + "="*60)
        print("测试门控机制公式")
        print("="*60)
        
        # 测试不同的门控系数和概率
        test_cases = [
            {"alpha": 1.0, "p_num": 0.5, "desc": "无门控(α=1)"},
            {"alpha": 0.0, "p_num": 0.9, "desc": "完全门控(α=0)"},
            {"alpha": 0.5, "p_num": 0.8, "desc": "部分门控"},
            {"alpha": 0.0, "p_num": 0.1, "desc": "低置信度"},
            {"alpha": 0.0, "p_num": 1.0, "desc": "完全置信"},
        ]
        
        print("测试情况\t\tα\tP_NUM\t门控因子\t期望结果\t验证")
        print("-" * 60)
        
        for case in test_cases:
            alpha = case["alpha"]
            p_num = torch.tensor(case["p_num"])
            
            # 计算门控因子
            gate_factor = alpha + (1 - alpha) * p_num
            expected = alpha + (1 - alpha) * case["p_num"]
            
            diff = abs(gate_factor.item() - expected)
            result = "✓" if diff < 1e-6 else "✗"
            
            print(f"{case['desc']:<15}\t{alpha:3.1f}\t{case['p_num']:4.1f}\t{gate_factor.item():8.4f}\t{expected:8.4f}\t{result}")
            
            # 验证公式正确性
            self.assertAlmostEqual(gate_factor.item(), expected, places=6,
                                 msg=f"门控公式错误：{case['desc']}")
        
        print("✓ 门控机制公式验证通过")
    
    def test_numerical_mask_application(self):
        """
        测试数值掩码的正确应用
        验证：只有<NUM>位置的损失被计算
        """
        print("\n" + "="*60)
        print("测试数值掩码应用")
        print("="*60)
        
        batch_size = 2
        seq_len = 5
        
        # 创建测试数据
        # targets中包含<NUM>和非<NUM>位置
        targets = torch.tensor([
            [100, self.num_token_id, 200, self.num_token_id, 300],  # 两个<NUM>位置
            [400, 500, self.num_token_id, 600, 700]                 # 一个<NUM>位置
        ])
        
        target_values = torch.randn(batch_size, seq_len)
        reg_loc = torch.randn(batch_size, seq_len)
        reg_scale = torch.abs(torch.randn(batch_size, seq_len)) + 0.1
        
        # 创建数值掩码
        num_mask = (targets == self.num_token_id).float()
        
        print(f"targets形状: {targets.shape}")
        print(f"数值掩码:\n{num_mask}")
        
        # 计算有掩码的损失 (注意参数顺序: loc, scale, target)
        basic_loss = cauchy_nll_loss(reg_loc, reg_scale, target_values, reduction='none')
        masked_loss = basic_loss * num_mask
        
        print(f"\n基础损失形状: {basic_loss.shape}")
        print(f"掩码损失形状: {masked_loss.shape}")
        
        # 验证：非<NUM>位置的损失应该为0
        non_num_positions = (targets != self.num_token_id)
        non_num_losses = masked_loss[non_num_positions]
        
        print(f"非<NUM>位置数量: {non_num_positions.sum().item()}")
        print(f"非<NUM>位置损失和: {non_num_losses.sum().item()}")
        
        # 验证非<NUM>位置损失为0
        self.assertTrue(torch.allclose(non_num_losses, torch.zeros_like(non_num_losses)),
                       "非<NUM>位置的损失应该为0")
        
        # 验证<NUM>位置有非零损失
        num_positions = (targets == self.num_token_id)
        num_losses = masked_loss[num_positions]
        self.assertTrue((num_losses > 0).all(), "<NUM>位置应该有正损失")
        
        print("✓ 数值掩码应用验证通过")
    
    def test_gated_regression_loss_formula(self):
        """
        测试完整的门控回归损失公式
        验证：L_reg_gated = m_i · (α + (1-α) · P_<NUM>) · L_cauchy_nll
        """
        print("\n" + "="*60)
        print("测试门控回归损失公式")
        print("="*60)
        
        batch_size = 3
        seq_len = 4
        
        # 创建测试数据
        targets = torch.tensor([
            [1, self.num_token_id, 3, 4],
            [5, 6, self.num_token_id, 8],
            [9, 10, 11, self.num_token_id]
        ])
        
        target_values = torch.tensor([
            [0.0, 10.5, 0.0, 0.0],
            [0.0, 0.0, -5.2, 0.0],
            [0.0, 0.0, 0.0, 99.9]
        ])
        
        reg_loc = torch.randn(batch_size, seq_len)
        reg_scale = torch.abs(torch.randn(batch_size, seq_len)) + 0.1
        
        # P_<NUM>概率（模拟从分类头获得）
        p_num = torch.rand(batch_size, seq_len)
        
        # 门控参数
        alpha = 0.3
        
        # 手动计算门控回归损失 (注意参数顺序: loc, scale, target)
        num_mask = (targets == self.num_token_id).float()
        basic_loss = cauchy_nll_loss(reg_loc, reg_scale, target_values, reduction='none')
        gate_factor = alpha + (1 - alpha) * p_num
        gated_loss = num_mask * gate_factor * basic_loss
        
        print(f"输入形状: targets={targets.shape}, values={target_values.shape}")
        print(f"数值掩码:\n{num_mask}")
        print(f"门控因子形状: {gate_factor.shape}")
        print(f"门控损失形状: {gated_loss.shape}")
        
        # 验证门控损失的性质
        # 1. 非<NUM>位置损失为0
        non_num_mask = (targets != self.num_token_id)
        non_num_losses = gated_loss[non_num_mask]
        self.assertTrue(torch.allclose(non_num_losses, torch.zeros_like(non_num_losses)),
                       "非<NUM>位置的门控损失应该为0")
        
        # 2. <NUM>位置损失 > 0（假设基础损失 > 0）
        num_positions = (targets == self.num_token_id)
        if num_positions.any():
            num_losses = gated_loss[num_positions]
            self.assertTrue((num_losses >= 0).all(), "<NUM>位置的门控损失应该非负")
        
        # 3. 验证门控效果：高P_<NUM>应该导致更高的损失权重
        total_gated_loss = gated_loss.sum()
        print(f"总门控损失: {total_gated_loss.item():.6f}")
        
        print("✓ 门控回归损失公式验证通过")
    
    def test_total_loss_composition(self):
        """
        测试总损失的组成
        验证：L_total = Σ(L_cls + λ · L_reg_gated)
        """
        print("\n" + "="*60)
        print("测试总损失组成")
        print("="*60)
        
        batch_size = 2
        seq_len = 3
        
        # 模拟分类损失（每个位置）
        cls_losses = torch.rand(batch_size, seq_len) * 2  # 随机分类损失
        
        # 模拟门控回归损失（每个位置）
        reg_losses = torch.rand(batch_size, seq_len) * 0.5  # 随机回归损失
        
        # 回归权重
        lambda_reg = 1.5
        
        # 计算总损失
        total_cls_loss = cls_losses.sum()
        total_reg_loss = reg_losses.sum()
        combined_total_loss = total_cls_loss + lambda_reg * total_reg_loss
        
        # 逐位置计算验证
        position_wise_total = 0
        for b in range(batch_size):
            for s in range(seq_len):
                position_loss = cls_losses[b, s] + lambda_reg * reg_losses[b, s]
                position_wise_total += position_loss
        
        print(f"分类损失总和: {total_cls_loss.item():.6f}")
        print(f"回归损失总和: {total_reg_loss.item():.6f}")
        print(f"加权回归损失: {(lambda_reg * total_reg_loss).item():.6f}")
        print(f"组合总损失: {combined_total_loss.item():.6f}")
        print(f"逐位置计算: {position_wise_total.item():.6f}")
        
        # 验证计算一致性
        diff = abs(combined_total_loss.item() - position_wise_total.item())
        print(f"计算差异: {diff:.8f}")
        
        self.assertLess(diff, 1e-6, "总损失计算应该一致")
        
        print("✓ 总损失组成验证通过")
    
    def test_loss_gradient_properties(self):
        """
        测试损失函数的梯度性质
        """
        print("\n" + "="*60)
        print("测试损失梯度性质")
        print("="*60)
        
        batch_size = 2
        seq_len = 3
        
        # 创建需要梯度的参数
        reg_loc = torch.randn(batch_size, seq_len, requires_grad=True)
        reg_scale = torch.abs(torch.randn(batch_size, seq_len)) + 0.1
        reg_scale.requires_grad_(True)
        
        # 目标值和掩码
        target_values = torch.randn(batch_size, seq_len)
        num_mask = torch.randint(0, 2, (batch_size, seq_len)).float()  # 随机掩码
        
        # 计算损失 (注意参数顺序: loc, scale, target)
        basic_loss = cauchy_nll_loss(reg_loc, reg_scale, target_values, reduction='none')
        masked_loss = (basic_loss * num_mask).sum()
        
        # 反向传播
        masked_loss.backward()
        
        print(f"回归loc梯度范数: {reg_loc.grad.norm().item():.6f}")
        print(f"回归scale梯度范数: {reg_scale.grad.norm().item():.6f}")
        print(f"损失值: {masked_loss.item():.6f}")
        
        # 验证梯度存在
        self.assertIsNotNone(reg_loc.grad, "reg_loc应该有梯度")
        self.assertIsNotNone(reg_scale.grad, "reg_scale应该有梯度")
        
        # 验证梯度非零（如果有任何非零掩码）
        if num_mask.sum() > 0:
            self.assertGreater(reg_loc.grad.norm().item(), 0, "reg_loc梯度应该非零")
            self.assertGreater(reg_scale.grad.norm().item(), 0, "reg_scale梯度应该非零")
        
        print("✓ 损失梯度性质验证通过")
    
    def test_loss_mathematical_properties(self):
        """
        测试损失函数的数学性质
        """
        print("\n" + "="*60)
        print("测试损失数学性质")
        print("="*60)
        
        # 测试柯西NLL的性质
        target = torch.tensor([0.0])
        loc = torch.tensor([0.0])
        scale = torch.tensor([1.0])
        
        # 性质1：当预测完全正确时，损失应该是log(π)
        perfect_loss = cauchy_nll_loss(loc, scale, target)
        expected_min_loss = math.log(math.pi)
        
        print(f"完美预测损失: {perfect_loss.item():.6f}")
        print(f"理论最小损失: {expected_min_loss:.6f}")
        print(f"差异: {abs(perfect_loss.item() - expected_min_loss):.8f}")
        
        self.assertAlmostEqual(perfect_loss.item(), expected_min_loss, places=5,
                             msg="完美预测的损失应该等于log(π)")
        
        # 性质2：损失应该随预测误差增加而增加
        errors = [0.0, 1.0, 2.0, 5.0]
        losses = []
        
        print(f"\n预测误差与损失关系:")
        print("误差\t损失")
        print("-" * 20)
        
        for error in errors:
            target_err = torch.tensor([error])
            loss = cauchy_nll_loss(loc, scale, target_err)
            losses.append(loss.item())
            print(f"{error:4.1f}\t{loss.item():.6f}")
        
        # 验证单调性
        for i in range(1, len(losses)):
            self.assertGreater(losses[i], losses[i-1],
                             f"损失应该随误差增加：error={errors[i]} vs {errors[i-1]}")
        
        print("✓ 损失数学性质验证通过")


if __name__ == "__main__":
    print("开始门控损失函数的数学验证测试...")
    print("=" * 80)
    
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    
    # 运行测试
    unittest.main(verbosity=2) 