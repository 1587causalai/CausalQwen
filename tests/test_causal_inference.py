#!/usr/bin/env python
"""
因果推断阶段的数学验证测试

验证关键数学性质：
1. U_i | z_i ~ Cauchy(loc(z_i), scale(z_i))
2. 条件独立性：P(U_i | z) = P(U_i | z_i)
3. 柯西分布线性稳定性：Y = aU + b ~ Cauchy(aμ + b, |a|γ)
4. scale > 0 始终成立
5. 并行处理性质
"""

import unittest
import torch
import math
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.abduction_network import AbductionNetwork
from src.utils.distributions import CauchyLinear


class TestCausalInference(unittest.TestCase):
    """因果推断阶段的数学验证测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.hidden_size = 128
        self.causal_dim = 64
        self.abduction_network = AbductionNetwork(
            hidden_size=self.hidden_size,
            causal_dim=self.causal_dim
        )
        
        print(f"\n初始化完成：hidden_size={self.hidden_size}, causal_dim={self.causal_dim}")
    
    def test_cauchy_distribution_parameters(self):
        """
        测试柯西分布参数的生成
        验证：U_i | z_i ~ Cauchy(loc(z_i), scale(z_i))
        """
        print("\n" + "="*60)
        print("测试柯西分布参数生成")
        print("="*60)
        
        batch_sizes = [1, 2, 5]
        seq_lengths = [1, 4, 10]
        
        print("批次大小\t序列长度\tloc形状\t\t\tscale形状\t\tscale>0")
        print("-" * 70)
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # 创建随机特征
                features = torch.randn(batch_size, seq_len, self.hidden_size)
                
                with torch.no_grad():
                    loc, scale = self.abduction_network(features)
                
                expected_shape = (batch_size, seq_len, self.causal_dim)
                
                # 验证形状
                self.assertEqual(loc.shape, expected_shape,
                               f"loc形状不匹配：期望 {expected_shape}，实际 {loc.shape}")
                self.assertEqual(scale.shape, expected_shape,
                               f"scale形状不匹配：期望 {expected_shape}，实际 {scale.shape}")
                
                # 验证scale > 0
                scale_positive = (scale > 0).all().item()
                
                print(f"{batch_size}\t\t{seq_len}\t\t{loc.shape}\t{scale.shape}\t{'✓' if scale_positive else '✗'}")
                
                # 验证数学性质
                self.assertTrue(scale_positive, "所有scale值必须大于0")
                self.assertTrue(torch.isfinite(loc).all(), "所有loc值必须是有限的")
                self.assertTrue(torch.isfinite(scale).all(), "所有scale值必须是有限的")
        
        print("✓ 柯西分布参数生成验证通过")
    
    def test_conditional_independence(self):
        """
        测试条件独立性：P(U_i | z) = P(U_i | z_i)
        验证每个位置的推断只依赖于该位置的特征
        """
        print("\n" + "="*60)
        print("测试条件独立性")
        print("="*60)
        
        batch_size = 2
        seq_len = 4
        
        # 创建基础特征序列
        base_features = torch.randn(batch_size, seq_len, self.hidden_size)
        
        # 创建修改版本：只改变位置1的特征
        modified_features = base_features.clone()
        modified_features[:, 1, :] = torch.randn(batch_size, self.hidden_size)
        
        with torch.no_grad():
            base_loc, base_scale = self.abduction_network(base_features)
            modified_loc, modified_scale = self.abduction_network(modified_features)
            
            print(f"特征形状: {base_features.shape}")
            print(f"输出形状: loc={base_loc.shape}, scale={base_scale.shape}")
            
            # 分析每个位置的变化
            print("\n位置独立性分析:")
            print("位置\tloc差异范数\tscale差异范数\t是否受影响")
            print("-" * 50)
            
            affected_positions = 0
            for pos in range(seq_len):
                loc_diff = torch.norm(modified_loc[:, pos, :] - base_loc[:, pos, :]).item()
                scale_diff = torch.norm(modified_scale[:, pos, :] - base_scale[:, pos, :]).item()
                
                is_affected = loc_diff > 1e-6 or scale_diff > 1e-6
                if is_affected:
                    affected_positions += 1
                
                print(f"{pos}\t{loc_diff:.6f}\t\t{scale_diff:.6f}\t\t{'✓' if is_affected else '✗'}")
            
            print(f"\n总共 {affected_positions} 个位置受到影响")
            
            # 验证条件独立性：只有位置1应该受影响
            self.assertEqual(affected_positions, 1, 
                           "只有修改的位置应该受到影响（条件独立性）")
        
        print("✓ 条件独立性验证通过")
    
    def test_cauchy_linear_stability(self):
        """
        测试柯西分布的线性稳定性
        验证：Y = aU + b ~ Cauchy(aμ + b, |a|γ)
        """
        print("\n" + "="*60)
        print("测试柯西分布线性稳定性")
        print("="*60)
        
        # 创建 CauchyLinear 层
        input_dim = 64
        output_dim = 32
        cauchy_linear = CauchyLinear(input_dim, output_dim)
        
        # 测试数据
        batch_size = 3
        loc_input = torch.randn(batch_size, input_dim)
        scale_input = torch.abs(torch.randn(batch_size, input_dim)) + 0.1  # 确保 > 0
        
        with torch.no_grad():
            loc_output, scale_output = cauchy_linear(loc_input, scale_input)
            
            print(f"输入形状: loc={loc_input.shape}, scale={scale_input.shape}")
            print(f"输出形状: loc={loc_output.shape}, scale={scale_output.shape}")
            
            # 验证输出形状
            expected_output_shape = (batch_size, output_dim)
            self.assertEqual(loc_output.shape, expected_output_shape,
                           f"loc输出形状不匹配")
            self.assertEqual(scale_output.shape, expected_output_shape,
                           f"scale输出形状不匹配")
            
            # 验证线性变换性质
            print("\n线性变换验证:")
            print("  - loc变换: loc_out = W * loc_in + b")
            print("  - scale变换: scale_out = |W| * scale_in")
            
            # 手动计算期望结果
            weight = cauchy_linear.weight
            bias = cauchy_linear.bias
            
            expected_loc = torch.nn.functional.linear(loc_input, weight, bias)
            expected_scale = torch.nn.functional.linear(scale_input, weight.abs(), None)
            
            # 验证计算正确性
            loc_diff = torch.norm(loc_output - expected_loc).item()
            scale_diff = torch.norm(scale_output - expected_scale).item()
            
            print(f"  - loc计算差异: {loc_diff:.8f}")
            print(f"  - scale计算差异: {scale_diff:.8f}")
            
            self.assertLess(loc_diff, 1e-6, "loc线性变换计算不正确")
            self.assertLess(scale_diff, 1e-6, "scale线性变换计算不正确")
            
            # 验证 scale > 0
            self.assertTrue((scale_output > 0).all(), "输出scale必须大于0")
        
        print("✓ 柯西分布线性稳定性验证通过")
    
    def test_parallel_computation_consistency(self):
        """
        测试并行计算的一致性
        验证批处理和逐个处理的结果一致
        """
        print("\n" + "="*60)
        print("测试并行计算一致性")
        print("="*60)
        
        batch_size = 3
        seq_len = 5
        
        # 创建测试特征
        features = torch.randn(batch_size, seq_len, self.hidden_size)
        
        with torch.no_grad():
            # 批处理计算
            batch_loc, batch_scale = self.abduction_network(features)
            
            # 逐个位置计算
            individual_locs = []
            individual_scales = []
            
            for b in range(batch_size):
                for s in range(seq_len):
                    # 单个位置的特征 [1, 1, hidden_size]
                    single_feature = features[b:b+1, s:s+1, :]
                    single_loc, single_scale = self.abduction_network(single_feature)
                    individual_locs.append(single_loc[0, 0, :])
                    individual_scales.append(single_scale[0, 0, :])
            
            # 重构为批处理形状
            individual_loc_batch = torch.stack(individual_locs).view(batch_size, seq_len, self.causal_dim)
            individual_scale_batch = torch.stack(individual_scales).view(batch_size, seq_len, self.causal_dim)
            
            # 比较结果
            loc_diff = torch.norm(batch_loc - individual_loc_batch).item()
            scale_diff = torch.norm(batch_scale - individual_scale_batch).item()
            
            print(f"特征形状: {features.shape}")
            print(f"批处理结果形状: {batch_loc.shape}")
            print(f"逐个计算结果形状: {individual_loc_batch.shape}")
            print(f"loc差异范数: {loc_diff:.8f}")
            print(f"scale差异范数: {scale_diff:.8f}")
            
            # 验证一致性
            self.assertLess(loc_diff, 1e-5, "批处理和逐个计算的loc结果应该一致")
            self.assertLess(scale_diff, 1e-5, "批处理和逐个计算的scale结果应该一致")
        
        print("✓ 并行计算一致性验证通过")
    
    def test_distribution_properties(self):
        """
        测试分布参数的数学性质
        """
        print("\n" + "="*60)
        print("测试分布参数性质")
        print("="*60)
        
        batch_size = 50  # 更大的批次用于统计分析
        seq_len = 10
        
        # 创建多样化的特征
        features = torch.randn(batch_size, seq_len, self.hidden_size)
        
        with torch.no_grad():
            loc, scale = self.abduction_network(features)
            
            # 统计分析
            loc_mean = loc.mean().item()
            loc_std = loc.std().item()
            scale_mean = scale.mean().item()
            scale_std = scale.std().item()
            scale_min = scale.min().item()
            scale_max = scale.max().item()
            
            print(f"分布参数统计:")
            print(f"  loc  - 均值: {loc_mean:.6f}, 标准差: {loc_std:.6f}")
            print(f"  scale - 均值: {scale_mean:.6f}, 标准差: {scale_std:.6f}")
            print(f"  scale - 范围: [{scale_min:.6f}, {scale_max:.6f}]")
            
            # 验证性质
            self.assertTrue(scale_min > 0, "scale最小值必须大于0")
            self.assertGreater(scale_std, 0, "scale应该有变化（不是常数）")
            self.assertTrue(torch.isfinite(loc).all(), "所有loc值必须有限")
            self.assertTrue(torch.isfinite(scale).all(), "所有scale值必须有限")
            
            # 检查合理的数值范围
            self.assertLess(abs(loc_mean), 5.0, "loc均值应该在合理范围内")
            self.assertLess(scale_mean, 10.0, "scale均值应该在合理范围内")
        
        print("✓ 分布参数性质验证通过")


if __name__ == "__main__":
    print("开始因果推断阶段的数学验证测试...")
    print("=" * 80)
    
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    
    # 运行测试
    unittest.main(verbosity=2) 