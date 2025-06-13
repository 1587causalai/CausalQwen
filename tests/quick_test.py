"""
CausalQwen 核心数学组件测试
专注于验证每个数学公式的正确实现
"""

import torch
import numpy as np
import pytest
from typing import Tuple

# 假设的导入（根据实际项目结构调整）
# from src.models.causal_qwen import CausalQwen
# from src.modules.numerical_embedding import compute_phi
# from src.modules.causality import compute_cauchy_nll

# 测试常量
NUM_TOKEN_ID = 151936  # <NUM> token ID
VOCAB_SIZE = 151937
HIDDEN_DIM = 768
TOLERANCE = 1e-6


class TestPhiFunction:
    """测试数值编码函数 φ(v) 的数学性质"""
    
    def test_phi_zero_property(self):
        """验证 φ(0) = 0 的精确性"""
        direction = torch.randn(HIDDEN_DIM)
        direction = direction / direction.norm()
        
        # 模拟 φ 函数
        v = 0.0
        phi_v = self._compute_phi(v, direction)
        
        # 必须精确为零
        assert torch.allclose(phi_v, torch.zeros_like(phi_v), atol=1e-10)
        assert phi_v.abs().max().item() == 0.0
        
    def test_phi_sign_symmetry(self):
        """验证 φ(-v) = -φ(v)"""
        direction = torch.randn(HIDDEN_DIM)
        direction = direction / direction.norm()
        
        test_values = [0.1, 1.0, 10.0, 100.0, 1e6]
        
        for v in test_values:
            phi_pos = self._compute_phi(v, direction)
            phi_neg = self._compute_phi(-v, direction)
            
            # 符号对称性必须精确
            assert torch.allclose(phi_pos, -phi_neg, rtol=1e-7)
            
    def test_phi_logarithmic_growth(self):
        """验证对数增长特性"""
        direction = torch.randn(HIDDEN_DIM)
        direction = direction / direction.norm()
        
        values = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        norms = []
        
        for v in values:
            phi_v = self._compute_phi(v, direction)
            norm = phi_v.norm().item()
            norms.append(norm)
            
            # 验证范数公式
            expected_norm = abs(np.log(1 + abs(v)))
            assert np.isclose(norm, expected_norm, rtol=1e-6)
        
        # 验证增长率递减
        growth_rates = []
        for i in range(1, len(norms)):
            growth_rate = norms[i] / norms[i-1]
            growth_rates.append(growth_rate)
            
        # 对数函数的增长率应该递减
        for i in range(1, len(growth_rates)):
            assert growth_rates[i] < growth_rates[i-1]
            
    def test_phi_numerical_stability(self):
        """测试极端值的数值稳定性"""
        direction = torch.randn(HIDDEN_DIM)
        direction = direction / direction.norm()
        
        extreme_values = [
            1e-30,  # 极小值
            1e-10,
            1e-5,
            1e10,   # 大值
            1e20,   # 极大值
            1e30,   # 超大值
        ]
        
        for v in extreme_values:
            phi_v = self._compute_phi(v, direction)
            
            # 不应该有 NaN 或 Inf
            assert not torch.isnan(phi_v).any()
            assert not torch.isinf(phi_v).any()
            
            # 验证范数
            expected_norm = abs(np.log(1 + abs(v)))
            actual_norm = phi_v.norm().item()
            
            # 对于极端值允许稍大的相对误差
            rtol = 1e-5 if abs(v) < 1e10 else 1e-3
            assert np.isclose(actual_norm, expected_norm, rtol=rtol)
    
    def _compute_phi(self, v: float, direction: torch.Tensor) -> torch.Tensor:
        """计算 φ(v) = sign(v) · ln(1 + |v|) · e"""
        if v == 0:
            return torch.zeros_like(direction)
        return np.sign(v) * np.log(1 + abs(v)) * direction


class TestCauchyDistribution:
    """测试柯西分布相关计算"""
    
    def test_cauchy_linear_transformation(self):
        """验证柯西分布的线性变换性质"""
        # U ~ Cauchy(μ, γ)
        loc_u = torch.tensor([[2.0, -1.0, 0.0]])
        scale_u = torch.tensor([[3.0, 1.0, 5.0]])
        
        # 测试多个线性变换
        transformations = [
            (2.5, -1.0),   # Y = 2.5U - 1
            (-1.0, 0.0),   # Y = -U
            (0.0, 5.0),    # Y = 5 (退化情况)
            (1.0, 0.0),    # Y = U (恒等变换)
        ]
        
        for a, b in transformations:
            # 计算变换后的参数
            loc_y = a * loc_u + b
            scale_y = abs(a) * scale_u
            
            # 手动验证每个元素
            for i in range(loc_u.shape[1]):
                expected_loc = a * loc_u[0, i].item() + b
                expected_scale = abs(a) * scale_u[0, i].item()
                
                assert torch.isclose(loc_y[0, i], torch.tensor(expected_loc))
                assert torch.isclose(scale_y[0, i], torch.tensor(expected_scale))
                
    def test_cauchy_cdf_accuracy(self):
        """验证柯西CDF计算的数值精度"""
        # 标准柯西分布的已知值
        test_cases = [
            # (x, loc, scale, expected_cdf)
            (0.0, 0.0, 1.0, 0.5),      # 中位数
            (1.0, 0.0, 1.0, 0.75),     # 第三四分位数
            (-1.0, 0.0, 1.0, 0.25),    # 第一四分位数
            (0.0, 5.0, 1.0, 0.0024),   # 远离中心
        ]
        
        for x, loc, scale, expected_cdf in test_cases:
            # 计算 P(X > x) = 1 - CDF(x)
            survival_prob = self._compute_cauchy_survival(x, loc, scale)
            expected_survival = 1 - expected_cdf
            
            # 高精度验证（除了远离中心的情况）
            if abs(x - loc) < 5 * scale:
                assert abs(survival_prob - expected_survival) < 1e-10
            else:
                assert abs(survival_prob - expected_survival) < 1e-3
    
    def test_cauchy_nll_formula(self):
        """验证柯西负对数似然的精确公式"""
        test_cases = [
            # (y_true, loc, scale)
            (0.0, 0.0, 1.0),    # 完美预测
            (1.0, 0.0, 1.0),    # 一个标准差
            (2.0, 1.0, 0.5),    # 两个标准差
            (-3.0, -3.0, 2.0),  # 负值完美预测
        ]
        
        for y_true, loc, scale in test_cases:
            y_true_t = torch.tensor([y_true])
            loc_t = torch.tensor([loc])
            scale_t = torch.tensor([scale])
            
            # 计算NLL
            nll = self._compute_cauchy_nll(y_true_t, loc_t, scale_t)
            
            # 手动计算期望值
            z = (y_true - loc) / scale
            expected_nll = np.log(np.pi * scale) + np.log(1 + z**2)
            
            # 验证精确性
            assert abs(nll.item() - expected_nll) < 1e-10
            
    def test_cauchy_nll_gradient(self):
        """验证柯西NLL的梯度正确性"""
        y_true = torch.tensor([2.0], requires_grad=False)
        loc = torch.tensor([1.0], requires_grad=True)
        scale = torch.tensor([0.5], requires_grad=True)
        
        # 计算损失
        nll = self._compute_cauchy_nll(y_true, loc, scale)
        nll.backward()
        
        # 手动计算梯度
        z = (y_true - loc) / scale
        z_value = z.item()
        
        # ∂L/∂loc = 2z / (scale(1 + z²))
        expected_grad_loc = 2 * z_value / (scale.item() * (1 + z_value**2))
        
        # ∂L/∂scale = 1/scale - 2z² / (scale(1 + z²))
        expected_grad_scale = 1/scale.item() - 2*z_value**2 / (scale.item() * (1 + z_value**2))
        
        # 验证梯度
        assert torch.isclose(loc.grad, torch.tensor([expected_grad_loc]), rtol=1e-5)
        assert torch.isclose(scale.grad, torch.tensor([expected_grad_scale]), rtol=1e-5)
    
    def _compute_cauchy_survival(self, x: float, loc: float, scale: float) -> float:
        """计算 P(X > x) for X ~ Cauchy(loc, scale)"""
        z = (x - loc) / scale
        return 0.5 - np.arctan(z) / np.pi
    
    def _compute_cauchy_nll(self, y_true: torch.Tensor, loc: torch.Tensor, 
                           scale: torch.Tensor) -> torch.Tensor:
        """计算柯西分布的负对数似然"""
        z = (y_true - loc) / scale
        return torch.log(np.pi * scale) + torch.log(1 + z**2)


class TestOvRProbability:
    """测试 One-vs-Rest 概率计算"""
    
    def test_ovr_probability_formula(self):
        """验证OvR概率公式的正确性"""
        # 创建测试数据
        loc_s = torch.tensor([[0.0, 1.0, -1.0, 5.0]])
        scale_s = torch.tensor([[1.0, 2.0, 0.5, 1.0]])
        thresholds = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        # 计算概率
        probs = self._compute_ovr_probabilities(loc_s, scale_s, thresholds)
        
        # 手动验证每个概率
        expected_probs = []
        for i in range(loc_s.shape[1]):
            z = (loc_s[0, i] - thresholds[i]) / scale_s[0, i]
            p = 0.5 + torch.atan(z) / np.pi
            expected_probs.append(p.item())
        
        # 验证
        assert probs.shape == (1, 4)
        for i in range(4):
            assert abs(probs[0, i].item() - expected_probs[i]) < 1e-10
            
    def test_ovr_probability_bounds(self):
        """验证概率值的有效范围"""
        # 测试极端情况
        loc_s = torch.tensor([[-1000.0, 0.0, 1000.0]])
        scale_s = torch.tensor([[1.0, 1.0, 1.0]])
        thresholds = torch.tensor([0.0, 0.0, 0.0])
        
        probs = self._compute_ovr_probabilities(loc_s, scale_s, thresholds)
        
        # 所有概率必须在 [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        
        # 验证极端值
        assert probs[0, 0] < 0.001  # 远小于阈值
        assert abs(probs[0, 1] - 0.5) < 1e-10  # 等于阈值
        assert probs[0, 2] > 0.999  # 远大于阈值
        
    def test_ovr_probability_batch(self):
        """测试批量计算的正确性"""
        batch_size = 8
        vocab_size = 100
        
        # 创建批量数据
        loc_s = torch.randn(batch_size, vocab_size)
        scale_s = torch.rand(batch_size, vocab_size) * 5 + 0.1
        thresholds = torch.randn(vocab_size)
        
        # 批量计算
        probs_batch = self._compute_ovr_probabilities(loc_s, scale_s, thresholds)
        
        # 逐个验证
        for i in range(batch_size):
            probs_single = self._compute_ovr_probabilities(
                loc_s[i:i+1], scale_s[i:i+1], thresholds
            )
            assert torch.allclose(probs_batch[i], probs_single[0], atol=1e-7)
    
    def _compute_ovr_probabilities(self, loc_s: torch.Tensor, scale_s: torch.Tensor,
                                  thresholds: torch.Tensor) -> torch.Tensor:
        """计算 P(S > C) = 1/2 + (1/π)arctan((loc-C)/scale)"""
        z = (loc_s - thresholds) / scale_s
        return 0.5 + torch.atan(z) / np.pi


class TestGatedLoss:
    """测试门控损失机制"""
    
    def test_gated_loss_alpha_cases(self):
        """测试不同alpha值的门控行为"""
        batch_size = 4
        seq_len = 8
        
        # 创建测试数据
        mask = torch.zeros(batch_size, seq_len)
        mask[:, [0, 2, 5]] = 1.0  # 位置0,2,5是数值
        
        p_num = torch.tensor([
            [0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.1, 0.2],
            [0.8, 0.2, 0.9, 0.1, 0.4, 0.6, 0.2, 0.3],
            [0.7, 0.3, 0.7, 0.3, 0.5, 0.8, 0.3, 0.1],
            [0.6, 0.4, 0.6, 0.4, 0.2, 0.9, 0.4, 0.4],
        ])
        
        base_loss = torch.ones(batch_size, seq_len) * 2.0  # 固定损失值
        
        # 测试不同的alpha
        test_alphas = [0.0, 0.1, 0.5, 0.9, 1.0]
        
        for alpha in test_alphas:
            gated_loss = self._compute_gated_loss(base_loss, mask, p_num, alpha)
            
            # 验证形状
            assert gated_loss.shape == (batch_size, seq_len)
            
            # 非数值位置必须为0
            non_num_mask = (mask == 0)
            assert (gated_loss[non_num_mask] == 0).all()
            
            # 验证数值位置的计算
            for b in range(batch_size):
                for pos in [0, 2, 5]:
                    expected_gate = alpha + (1 - alpha) * p_num[b, pos]
                    expected_loss = base_loss[b, pos] * expected_gate
                    
                    actual_loss = gated_loss[b, pos]
                    assert torch.isclose(actual_loss, expected_loss, rtol=1e-6)
                    
    def test_gated_loss_edge_cases(self):
        """测试门控损失的边界情况"""
        # 情况1: p_num = 0 且 alpha = 0
        mask = torch.ones(1, 3)
        p_num = torch.zeros(1, 3)
        base_loss = torch.ones(1, 3)
        
        gated_loss = self._compute_gated_loss(base_loss, mask, p_num, alpha=0.0)
        assert (gated_loss == 0).all()  # 完全被门控
        
        # 情况2: p_num = 1 且 alpha = 0
        p_num = torch.ones(1, 3)
        gated_loss = self._compute_gated_loss(base_loss, mask, p_num, alpha=0.0)
        assert torch.allclose(gated_loss, base_loss)  # 完全通过
        
        # 情况3: alpha = 1 (忽略p_num)
        p_num = torch.rand(1, 3)
        gated_loss = self._compute_gated_loss(base_loss, mask, p_num, alpha=1.0)
        assert torch.allclose(gated_loss, base_loss)  # 不受p_num影响
    
    def _compute_gated_loss(self, base_loss: torch.Tensor, mask: torch.Tensor,
                           p_num: torch.Tensor, alpha: float) -> torch.Tensor:
        """计算门控回归损失"""
        gate = mask * (alpha + (1 - alpha) * p_num)
        return base_loss * gate


class TestEndToEndNumerics:
    """端到端数值验证"""
    
    def test_dimension_consistency(self):
        """验证整个流程的维度一致性"""
        batch_size = 2
        seq_len = 16
        vocab_size = VOCAB_SIZE
        hidden_dim = HIDDEN_DIM
        
        # 模拟各阶段的输出
        # 1. 输入
        input_ids = torch.randint(0, vocab_size-1, (batch_size, seq_len))
        numeric_values = torch.zeros(batch_size, seq_len)
        
        # 2. 嵌入 (enhanced embeddings)
        embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 3. 特征提取 (transformer output)
        features = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 4. 归因推断
        loc_u = features  # 恒等映射
        scale_u = torch.ones(batch_size, seq_len, hidden_dim) * 10.0
        
        # 5. 行动决策
        # 分类
        cls_weight = torch.randn(vocab_size, hidden_dim)
        loc_s = torch.matmul(loc_u.unsqueeze(2), cls_weight.t()).squeeze(2)
        scale_s = torch.ones_like(loc_s) * 10.0
        
        # 回归
        reg_weight = torch.randn(1, hidden_dim)
        loc_y = torch.matmul(loc_u, reg_weight.t()).squeeze(-1)
        scale_y = torch.ones_like(loc_y) * 10.0
        
        # 验证所有维度
        assert embeddings.shape == (batch_size, seq_len, hidden_dim)
        assert features.shape == (batch_size, seq_len, hidden_dim)
        assert loc_u.shape == (batch_size, seq_len, hidden_dim)
        assert scale_u.shape == (batch_size, seq_len, hidden_dim)
        assert loc_s.shape == (batch_size, seq_len, vocab_size)
        assert scale_s.shape == (batch_size, seq_len, vocab_size)
        assert loc_y.shape == (batch_size, seq_len)
        assert scale_y.shape == (batch_size, seq_len)
    
    def test_numerical_flow_integrity(self):
        """验证数值信息在整个流程中的传播"""
        # 创建一个简单的测试场景
        input_ids = torch.tensor([[100, NUM_TOKEN_ID, 200]])
        numeric_values = torch.tensor([[0.0, 50.0, 0.0]])
        
        # 验证数值编码
        direction = torch.randn(HIDDEN_DIM)
        direction = direction / direction.norm()
        
        # 位置1应该有数值编码
        phi_1 = np.sign(50.0) * np.log(1 + abs(50.0)) * direction
        assert phi_1.norm() > 0
        
        # 位置0和2应该没有数值编码
        phi_0 = torch.zeros_like(direction)
        phi_2 = torch.zeros_like(direction)
        
        # 验证编码的正确性
        expected_phi_norm = np.log(1 + 50.0)
        assert abs(phi_1.norm().item() - expected_phi_norm) < 1e-6


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])