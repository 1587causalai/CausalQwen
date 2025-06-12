#!/usr/bin/env python
"""
数值编码函数的数学验证测试

验证关键数学性质：
1. φ(0) = 0 (零值退化)
2. 符号保持：sign(φ(v)) = sign(v)
3. 数值稳定性：大数值不会导致溢出
4. 单调性：φ'(v) > 0 对于 v > 0
"""

import unittest
import torch
import math
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.feature_network import NumAwareFeatureNetwork, MockFeatureNetwork


class TestNumericalEncoding(unittest.TestCase):
    """数值编码函数的数学验证测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.vocab_size = 50000
        self.hidden_size = 64
        self.num_token_id = 12345
        
        # 创建基础网络和数值感知网络
        self.base_network = MockFeatureNetwork(
            vocab_size=self.vocab_size, 
            hidden_size=self.hidden_size
        )
        self.num_aware_network = NumAwareFeatureNetwork(
            base_network=self.base_network,
            num_token_id=self.num_token_id,
            hidden_size=self.hidden_size
        )
        
        print(f"\n初始化完成：vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, num_token_id={self.num_token_id}")
    
    def test_phi_zero_property(self):
        """
        测试关键数学性质：φ(0) = 0
        这是整个数值感知机制的基础性质
        """
        print("\n" + "="*60)
        print("测试 φ(0) = 0 性质")
        print("="*60)
        
        # 测试零值
        zero_value = torch.tensor(0.0)
        
        # 手动计算 φ(0)
        sign_v = torch.sign(zero_value)
        log_v = torch.log1p(torch.abs(zero_value))
        phi_result = sign_v * log_v
        
        print(f"输入值: {zero_value.item()}")
        print(f"sign(0): {sign_v.item()}")
        print(f"ln(1 + |0|): {log_v.item()}")
        print(f"φ(0) = sign(0) * ln(1 + |0|): {phi_result.item()}")
        
        # 验证性质
        self.assertAlmostEqual(phi_result.item(), 0.0, places=10, 
                              msg="φ(0) 应该等于 0")
        
        # 测试实际网络中的实现
        batch_size = 1
        seq_len = 3
        input_ids = torch.tensor([[1, self.num_token_id, 2]])  # 中间是 <NUM>
        numerical_values = torch.tensor([[0.0, 0.0, 0.0]])  # 全零数值
        
        # 获取基础特征
        with torch.no_grad():
            base_features = self.base_network(input_ids, numerical_values)
            num_aware_features = self.num_aware_network(input_ids, numerical_values)
            
            # 对于零数值，数值感知特征应该等于基础特征
            feature_diff = torch.norm(num_aware_features - base_features)
            print(f"基础特征 vs 数值感知特征的差异范数: {feature_diff.item():.10f}")
            
            self.assertLess(feature_diff.item(), 1e-8, 
                          msg="当数值为0时，数值感知特征应该等于基础特征")
        
        print("✓ φ(0) = 0 性质验证通过")
    
    def test_sign_preservation(self):
        """
        测试符号保持性质：sign(φ(v)) = sign(v)
        """
        print("\n" + "="*60)
        print("测试符号保持性质")
        print("="*60)
        
        test_values = [
            -100.0, -10.0, -1.0, -0.1, 
            0.1, 1.0, 10.0, 100.0
        ]
        
        print("测试值\t\t原始符号\tφ(v)符号\t验证结果")
        print("-" * 50)
        
        for value in test_values:
            v_tensor = torch.tensor(value)
            
            # 计算 φ(v)
            sign_v = torch.sign(v_tensor)
            log_v = torch.log1p(torch.abs(v_tensor))
            phi_result = sign_v * log_v
            
            original_sign = torch.sign(v_tensor).item()
            phi_sign = torch.sign(phi_result).item()
            
            print(f"{value:8.1f}\t\t{original_sign:4.0f}\t\t{phi_sign:4.0f}\t\t{'✓' if original_sign == phi_sign else '✗'}")
            
            # 验证符号保持（零值除外）
            if value != 0:
                self.assertEqual(original_sign, phi_sign, 
                               f"值 {value} 的符号应该保持不变")
        
        print("✓ 符号保持性质验证通过")
    
    def test_numerical_stability(self):
        """
        测试数值稳定性：大数值不会导致溢出
        """
        print("\n" + "="*60)
        print("测试数值稳定性")
        print("="*60)
        
        # 测试极大值
        large_values = [1e3, 1e6, 1e9, 1e12]
        
        print("测试值\t\tφ(v)结果\t\t是否有限")
        print("-" * 40)
        
        for value in large_values:
            v_tensor = torch.tensor(value)
            
            # 计算 φ(v)
            sign_v = torch.sign(v_tensor)
            log_v = torch.log1p(torch.abs(v_tensor))
            phi_result = sign_v * log_v
            
            is_finite = torch.isfinite(phi_result).item()
            
            print(f"{value:10.0e}\t\t{phi_result.item():8.4f}\t\t{'✓' if is_finite else '✗'}")
            
            # 验证结果是有限的
            self.assertTrue(torch.isfinite(phi_result).item(), 
                          f"φ({value}) 应该是有限值")
            
            # 验证结果是合理的
            expected_result = math.log1p(value)  # ln(1 + |v|)
            self.assertAlmostEqual(phi_result.item(), expected_result, places=4,
                                 msg=f"φ({value}) 的计算结果不正确")
        
        print("✓ 数值稳定性验证通过")
    
    def test_monotonicity(self):
        """
        测试单调性：对于 v > 0，φ(v) 应该单调递增
        """
        print("\n" + "="*60)
        print("测试单调性性质")
        print("="*60)
        
        # 测试正数的单调性
        positive_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        print("测试值\t\tφ(v)结果\t\t单调性检查")
        print("-" * 45)
        
        previous_result = None
        for i, value in enumerate(positive_values):
            v_tensor = torch.tensor(value)
            
            # 计算 φ(v)
            log_v = torch.log1p(torch.abs(v_tensor))
            phi_result = log_v.item()  # 正数的符号是1
            
            monotonic = "✓" if previous_result is None or phi_result > previous_result else "✗"
            
            print(f"{value:8.2f}\t\t{phi_result:8.4f}\t\t{monotonic}")
            
            # 验证单调性
            if previous_result is not None:
                self.assertGreater(phi_result, previous_result, 
                                 f"φ({value}) 应该大于 φ({positive_values[i-1]})")
            
            previous_result = phi_result
        
        print("✓ 单调性验证通过")
    
    def test_integration_with_network(self):
        """
        测试数值编码在完整网络中的集成效果
        """
        print("\n" + "="*60)
        print("测试网络集成效果")
        print("="*60)
        
        batch_size = 2
        seq_len = 4
        
        # 创建测试数据：包含 <NUM> 和非 <NUM> 位置
        input_ids = torch.tensor([
            [1, self.num_token_id, 3, self.num_token_id],  # 两个数值位置
            [4, 5, self.num_token_id, 6]  # 一个数值位置
        ])
        
        numerical_values = torch.tensor([
            [0.0, 10.5, 0.0, -2.3],  # 对应 <NUM> 位置的数值
            [0.0, 0.0, 99.9, 0.0]
        ])
        
        with torch.no_grad():
            # 获取基础特征和数值感知特征
            base_features = self.base_network(input_ids, numerical_values)
            num_aware_features = self.num_aware_network(input_ids, numerical_values)
            
            print(f"输入形状: {input_ids.shape}")
            print(f"数值形状: {numerical_values.shape}")
            print(f"基础特征形状: {base_features.shape}")
            print(f"数值感知特征形状: {num_aware_features.shape}")
            
            # 分析 <NUM> 位置的特征变化
            print("\n<NUM> 位置的特征分析:")
            print("位置\t数值\t\t特征差异范数")
            print("-" * 35)
            
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    if input_ids[batch_idx, seq_idx] == self.num_token_id:
                        value = numerical_values[batch_idx, seq_idx].item()
                        
                        # 计算特征差异
                        base_feat = base_features[batch_idx, seq_idx]
                        aware_feat = num_aware_features[batch_idx, seq_idx]
                        diff_norm = torch.norm(aware_feat - base_feat).item()
                        
                        print(f"{batch_idx},{seq_idx}\t{value:8.1f}\t\t{diff_norm:.6f}")
                        
                        # 验证：数值不为0时，特征应该有差异
                        if abs(value) > 1e-6:
                            self.assertGreater(diff_norm, 1e-6, 
                                             f"数值 {value} 应该影响特征")
        
        print("✓ 网络集成效果验证通过")


if __name__ == "__main__":
    print("开始数值编码函数的数学验证测试...")
    print("=" * 80)
    
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    
    # 运行测试
    unittest.main(verbosity=2) 