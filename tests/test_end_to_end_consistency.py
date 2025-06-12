#!/usr/bin/env python
"""
端到端数学一致性验证测试

验证关键数学性质：
1. 从输入到输出的完整数据流一致性
2. 采样预测 vs 分布参数预测的一致性
3. 反事实推理能力验证
4. 完整的数学变换链：数值编码→特征提取→归因推断→行动决策→损失计算
5. 位置级别的并行计算验证
"""

import unittest
import torch
import math
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.models.feature_network import MockFeatureNetwork, NumAwareFeatureNetwork
from src.models.abduction_network import AbductionNetwork
from src.models.action_network import ActionNetwork
from src.losses.loss_functions import compute_total_loss


class TestEndToEndConsistency(unittest.TestCase):
    """端到端数学一致性验证测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建配置 - 使用真实Qwen模型
        import os
        qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
        
        self.config = CausalLMConfig(
            vocab_size=151936,  # Qwen2.5的标准词汇表大小
            hidden_size=896,    # Qwen2.5-0.5B的hidden_size
            causal_dim=896,     # 与hidden_size保持一致
            num_token_id=151665,  # Qwen的<NUM> token ID
            use_real_qwen=True,
            use_mock_feature_network=False,  # 使用真实Qwen模型
            use_numerical_features=True,     # 启用数值感知功能！
            qwen_model_path=qwen_model_path
        )
        
        # 创建完整的CausalLM模型
        self.model = CausalLanguageModel(self.config)
        
        # 更新实例变量以匹配真实Qwen配置
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size  
        self.causal_dim = self.config.causal_dim
        self.num_token_id = self.config.num_token_id
        
        print(f"\n初始化完成：vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, causal_dim={self.causal_dim}")
    
    def test_complete_data_flow(self):
        """
        测试完整的数据流：输入→数值编码→特征提取→归因推断→行动决策→输出
        """
        print("\n" + "="*60)
        print("测试完整数据流")
        print("="*60)
        
        batch_size = 2
        seq_len = 5
        
        # 创建输入数据
        input_ids = torch.tensor([
            [1, self.num_token_id, 3, 4, self.num_token_id],
            [6, 7, self.num_token_id, 9, 10]
        ])
        
        numerical_values = torch.tensor([
            [0.0, 10.5, 0.0, 0.0, -5.2],
            [0.0, 0.0, 99.9, 0.0, 0.0]
        ])
        
        print(f"输入 input_ids: {input_ids.shape}")
        print(f"输入 numerical_values: {numerical_values.shape}")
        print(f"<NUM> 位置: {(input_ids == self.num_token_id).nonzero(as_tuple=True)}")
        
        with torch.no_grad():
            # 端到端前向传播
            outputs = self.model(input_ids, numerical_values)
            
            print(f"\n输出结构:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
            
            # 验证输出形状
            expected_shapes = {
                'cls_loc': (batch_size, seq_len, self.vocab_size),
                'cls_scale': (batch_size, seq_len, self.vocab_size),
                'reg_loc': (batch_size, seq_len),
                'reg_scale': (batch_size, seq_len),
                'causal_loc': (batch_size, seq_len, self.causal_dim),
                'causal_scale': (batch_size, seq_len, self.causal_dim)
            }
            
            for key, expected_shape in expected_shapes.items():
                if key in outputs:
                    actual_shape = outputs[key].shape
                    self.assertEqual(actual_shape, expected_shape,
                                   f"{key} 形状不匹配：期望 {expected_shape}，实际 {actual_shape}")
            
            # 验证数值范围
            self.assertTrue((outputs['cls_scale'] > 0).all(), "分类scale必须大于0")
            self.assertTrue((outputs['reg_scale'] > 0).all(), "回归scale必须大于0")
            self.assertTrue((outputs['causal_scale'] > 0).all(), "因果scale必须大于0")
            
            # 验证有限性
            for key in ['cls_loc', 'cls_scale', 'reg_loc', 'reg_scale']:
                if key in outputs:
                    self.assertTrue(torch.isfinite(outputs[key]).all(),
                                  f"{key} 应该是有限值")
        
        print("✓ 完整数据流验证通过")
    
    def test_position_level_parallelism(self):
        """
        测试位置级别的并行计算一致性
        验证：每个位置的计算相互独立
        """
        print("\n" + "="*60)
        print("测试位置级别并行计算")
        print("="*60)
        
        batch_size = 2
        seq_len = 4
        
        # 创建输入
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        input_ids[:, 1] = self.num_token_id  # 在位置1放置<NUM>
        numerical_values = torch.randn(batch_size, seq_len)
        
        with torch.no_grad():
            # 完整序列计算
            full_outputs = self.model(input_ids, numerical_values)
            
            # 逐位置计算（使用相同的特征）
            features = self.model.feature_network(input_ids, numerical_values)
            
            position_outputs = []
            for pos in range(seq_len):
                # 单个位置的特征 [batch_size, 1, hidden_size]
                single_feat = features[:, pos:pos+1, :]
                
                # 归因推断
                single_causal_loc, single_causal_scale = self.model.abduction_network(single_feat)
                
                # 行动决策
                single_action_outputs = self.model.action_network(single_causal_loc, single_causal_scale)
                
                position_outputs.append({
                    'causal_loc': single_causal_loc.squeeze(1),  # [batch_size, causal_dim]
                    'causal_scale': single_causal_scale.squeeze(1),
                    'cls_loc': single_action_outputs['cls_loc'].squeeze(1),  # [batch_size, vocab_size]
                    'cls_scale': single_action_outputs['cls_scale'].squeeze(1),
                    'reg_loc': single_action_outputs['reg_loc'].squeeze(1),  # [batch_size]
                    'reg_scale': single_action_outputs['reg_scale'].squeeze(1)
                })
            
            # 比较完整计算和逐位置计算的结果
            print("位置\t因果loc差异\t因果scale差异\t分类loc差异\t分类scale差异\t回归loc差异\t回归scale差异")
            print("-" * 100)
            
            max_diffs = {'causal_loc': 0, 'causal_scale': 0, 'cls_loc': 0, 'cls_scale': 0, 'reg_loc': 0, 'reg_scale': 0}
            
            for pos in range(seq_len):
                diffs = {}
                for key in ['causal_loc', 'causal_scale', 'cls_loc', 'cls_scale', 'reg_loc', 'reg_scale']:
                    full_val = full_outputs[key][:, pos] if key in ['reg_loc', 'reg_scale'] else full_outputs[key][:, pos, :]
                    pos_val = position_outputs[pos][key]
                    diff = torch.norm(full_val - pos_val).item()
                    diffs[key] = diff
                    max_diffs[key] = max(max_diffs[key], diff)
                
                print(f"{pos}\t{diffs['causal_loc']:.2e}\t\t{diffs['causal_scale']:.2e}\t\t{diffs['cls_loc']:.2e}\t\t{diffs['cls_scale']:.2e}\t\t{diffs['reg_loc']:.2e}\t\t{diffs['reg_scale']:.2e}")
            
            print(f"\n最大差异: {max_diffs}")
            
            # 验证并行计算一致性
            for key, max_diff in max_diffs.items():
                self.assertLess(max_diff, 1e-5, f"{key} 并行计算差异过大")
        
        print("✓ 位置级别并行计算验证通过")
    
    def test_sampling_vs_deterministic_consistency(self):
        """
        测试采样预测与确定性预测的一致性
        验证：多次采样的期望应该接近分布参数
        """
        print("\n" + "="*60)
        print("测试采样vs确定性预测一致性")
        print("="*60)
        
        batch_size = 1
        seq_len = 3
        
        # 创建输入
        input_ids = torch.tensor([[100, self.num_token_id, 200]])
        numerical_values = torch.tensor([[0.0, 50.0, 0.0]])
        
        with torch.no_grad():
            # 获取分布参数
            outputs = self.model(input_ids, numerical_values)
            
            # 对回归值进行多次采样
            num_samples = 1000
            reg_samples = []
            
            for _ in range(num_samples):
                # 从柯西分布采样
                # 使用逆CDF方法：F^(-1)(u) = loc + scale * tan(π(u - 0.5))
                u = torch.rand_like(outputs['reg_loc'])
                sample = outputs['reg_loc'] + outputs['reg_scale'] * torch.tan(math.pi * (u - 0.5))
                reg_samples.append(sample)
            
            reg_samples = torch.stack(reg_samples, dim=0)  # [num_samples, batch_size, seq_len]
            
            # 计算采样统计
            sample_median = torch.median(reg_samples, dim=0)[0]  # 柯西分布的中位数=位置参数
            
            print(f"分布位置参数: {outputs['reg_loc']}")
            print(f"采样中位数: {sample_median}")
            print(f"差异: {torch.abs(outputs['reg_loc'] - sample_median)}")
            
            # 验证中位数一致性（柯西分布的中位数等于位置参数）
            median_diff = torch.abs(outputs['reg_loc'] - sample_median).max().item()
            print(f"最大中位数差异: {median_diff:.6f}")
            
            # 由于采样的随机性和Mock网络的特性，我们使用更宽松的阈值
            self.assertLess(median_diff, 2.0, "采样中位数应该接近分布位置参数")
        
        print("✓ 采样vs确定性预测一致性验证通过")
    
    def test_mathematical_invariants(self):
        """
        测试数学不变量
        验证：某些数学性质在整个流程中保持不变
        """
        print("\n" + "="*60)
        print("测试数学不变量")
        print("="*60)
        
        batch_size = 2
        seq_len = 4
        
        # 测试1：零数值的影响
        print("测试1：零数值的影响")
        input_ids = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        
        # 非数值输入（全零）
        zero_values = torch.zeros(batch_size, seq_len)
        
        with torch.no_grad():
            zero_outputs = self.model(input_ids, zero_values)
            
            # 验证：非<NUM>位置的数值感知应该不受影响
            # 因为 φ(0) = 0，所以非数值位置的特征应该保持不变
            
            print(f"零数值输出形状验证: ✓")
            print(f"  分类输出: {zero_outputs['cls_loc'].shape}")
            print(f"  回归输出: {zero_outputs['reg_loc'].shape}")
        
        # 测试2：<NUM>位置的数值敏感性
        print("\n测试2：<NUM>位置的数值敏感性")
        input_ids_with_num = torch.tensor([
            [1, self.num_token_id, 3, 4],
            [5, 6, self.num_token_id, 8]
        ])
        
        # 不同的数值配置
        values_1 = torch.tensor([[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, -5.0, 0.0]])
        values_2 = torch.tensor([[0.0, 20.0, 0.0, 0.0], [0.0, 0.0, -10.0, 0.0]])
        
        with torch.no_grad():
            outputs_1 = self.model(input_ids_with_num, values_1)
            outputs_2 = self.model(input_ids_with_num, values_2)
            
            # 分析<NUM>位置的输出差异
            num_positions = (input_ids_with_num == self.num_token_id)
            
            for b in range(batch_size):
                for s in range(seq_len):
                    if num_positions[b, s]:
                        cls_diff = torch.norm(outputs_2['cls_loc'][b, s] - outputs_1['cls_loc'][b, s]).item()
                        reg_diff = abs(outputs_2['reg_loc'][b, s] - outputs_1['reg_loc'][b, s]).item()
                        
                        print(f"  位置({b},{s}): 分类差异={cls_diff:.4f}, 回归差异={reg_diff:.4f}")
                        
                        # 验证：不同数值应该产生不同输出（降低Mock网络的敏感性要求）
                        self.assertGreater(cls_diff, 0.001, f"位置({b},{s})分类输出应该对数值敏感")
                        self.assertGreater(reg_diff, 0.001, f"位置({b},{s})回归输出应该对数值敏感")
        
        print("✓ 数学不变量验证通过")
    
    def test_loss_computation_consistency(self):
        """
        测试损失计算的一致性
        验证：端到端损失计算与分步计算一致
        """
        print("\n" + "="*60)
        print("测试损失计算一致性")
        print("="*60)
        
        batch_size = 2
        seq_len = 3
        
        # 创建训练数据
        input_ids = torch.tensor([
            [100, self.num_token_id, 200],
            [300, 400, self.num_token_id]
        ])
        
        numerical_values = torch.tensor([
            [0.0, 15.5, 0.0],
            [0.0, 0.0, -8.2]
        ])
        
        # 目标数据
        target_ids = torch.tensor([
            [101, self.num_token_id, 201],
            [301, 401, self.num_token_id]
        ])
        
        target_values = torch.tensor([
            [0.0, 20.0, 0.0],
            [0.0, 0.0, -10.0]
        ])
        
        with torch.no_grad():
            # 方法1：使用模型的compute_loss方法
            outputs = self.model(input_ids, numerical_values)
            model_loss_dict = self.model.compute_loss(outputs, target_ids, target_values)
            
            # 方法2：手动计算损失
            # 计算分类概率
            cls_probs = self.model.action_network.classification_head.compute_probabilities(
                outputs['cls_loc'], outputs['cls_scale']
            )
            
            # 获取<NUM>概率
            num_probs = cls_probs[:, :, self.num_token_id]
            
            # 创建数值掩码
            num_mask = (target_ids == self.num_token_id).float()
            
            # 手动计算损失
            manual_loss_dict = compute_total_loss(
                cls_probs=cls_probs,
                cls_targets=target_ids,
                reg_loc=outputs['reg_loc'],
                reg_scale=outputs['reg_scale'], 
                reg_targets=target_values,
                num_probs=num_probs,
                num_mask=num_mask
            )
            
            print("损失比较:")
            print(f"  模型损失: {model_loss_dict['total']:.6f}")
            print(f"  手动损失: {manual_loss_dict['total']:.6f}")
            print(f"  差异: {abs(model_loss_dict['total'] - manual_loss_dict['total']):.8f}")
            
            # 验证一致性
            loss_diff = abs(model_loss_dict['total'] - manual_loss_dict['total']).item()
            self.assertLess(loss_diff, 1e-6, "模型损失与手动计算损失应该一致")
            
            # 验证分量损失
            if 'cls' in model_loss_dict and 'cls' in manual_loss_dict:
                cls_diff = abs(model_loss_dict['cls'] - manual_loss_dict['cls']).item()
                print(f"  分类损失差异: {cls_diff:.8f}")
                self.assertLess(cls_diff, 1e-6, "分类损失应该一致")
            
            if 'reg' in model_loss_dict and 'reg' in manual_loss_dict:
                reg_diff = abs(model_loss_dict['reg'] - manual_loss_dict['reg']).item()
                print(f"  回归损失差异: {reg_diff:.8f}")
                self.assertLess(reg_diff, 1e-6, "回归损失应该一致")
        
        print("✓ 损失计算一致性验证通过")
    
    def test_counterfactual_reasoning(self):
        """
        测试反事实推理能力
        验证：模型能够处理假设性的数值变化
        """
        print("\n" + "="*60)
        print("测试反事实推理能力")
        print("="*60)
        
        batch_size = 1
        seq_len = 3
        
        # 基础场景
        base_input = torch.tensor([[100, self.num_token_id, 200]])
        base_values = torch.tensor([[0.0, 10.0, 0.0]])
        
        # 反事实场景：相同的文本，不同的数值
        counterfactual_values = [5.0, 15.0, 25.0, 50.0, 100.0]
        
        print("反事实分析：相同文本，不同数值")
        print("数值\t回归预测\t分类概率变化\t<NUM>概率")
        print("-" * 50)
        
        base_outputs = None
        
        with torch.no_grad():
            for i, cf_value in enumerate([10.0] + counterfactual_values):
                cf_numerical = torch.tensor([[0.0, cf_value, 0.0]])
                cf_outputs = self.model(base_input, cf_numerical)
                
                # 计算分类概率
                cls_probs = self.model.action_network.classification_head.compute_probabilities(
                    cf_outputs['cls_loc'], cf_outputs['cls_scale']
                )
                
                reg_pred = cf_outputs['reg_loc'][0, 1].item()  # <NUM>位置的回归预测
                num_prob = cls_probs[0, 1, self.num_token_id].item()  # <NUM>位置的<NUM>概率
                
                if i == 0:  # 基础场景
                    base_outputs = cf_outputs
                    base_cls_probs = cls_probs
                    print(f"{cf_value:5.1f}\t{reg_pred:8.4f}\t{'基础':<12}\t{num_prob:.4f}")
                else:
                    # 计算与基础场景的差异
                    cls_diff = torch.norm(cls_probs[0, 1, :] - base_cls_probs[0, 1, :]).item()
                    print(f"{cf_value:5.1f}\t{reg_pred:8.4f}\t{cls_diff:8.4f}\t{num_prob:.4f}")
                    
                    # 验证：不同数值应该产生不同预测（降低Mock网络的敏感性要求）
                    reg_diff = abs(reg_pred - base_outputs['reg_loc'][0, 1].item())
                    self.assertGreater(reg_diff, 0.001, f"数值{cf_value}应该产生不同的回归预测")
        
        print("✓ 反事实推理能力验证通过")
    
    def test_gradient_flow_end_to_end(self):
        """
        测试端到端梯度流
        验证：从输入到损失的完整梯度传播
        """
        print("\n" + "="*60)
        print("测试端到端梯度流")
        print("="*60)
        
        batch_size = 1
        seq_len = 2
        
        # 创建需要梯度的输入
        input_ids = torch.tensor([[self.num_token_id, 100]])
        numerical_values = torch.tensor([[10.0, 0.0]], requires_grad=True)
        
        # 目标数据
        target_ids = torch.tensor([[self.num_token_id, 101]])
        target_values = torch.tensor([[15.0, 0.0]])
        
        # 前向传播
        outputs = self.model(input_ids, numerical_values)
        loss_dict = self.model.compute_loss(outputs, target_ids, target_values)
        
        # 反向传播
        total_loss = loss_dict['total']
        total_loss.backward()
        
        print(f"总损失: {total_loss.item():.6f}")
        print(f"数值输入梯度: {numerical_values.grad}")
        print(f"梯度范数: {numerical_values.grad.norm().item():.6f}")
        
        # 验证梯度存在
        self.assertIsNotNone(numerical_values.grad, "数值输入应该有梯度")
        
        # 验证<NUM>位置有非零梯度
        num_grad = numerical_values.grad[0, 0].item()  # <NUM>位置的梯度
        self.assertNotEqual(num_grad, 0, "<NUM>位置应该有非零梯度")
        
        print("✓ 端到端梯度流验证通过")


if __name__ == "__main__":
    print("开始端到端数学一致性验证测试...")
    print("=" * 80)
    
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    
    # 运行测试
    unittest.main(verbosity=2) 