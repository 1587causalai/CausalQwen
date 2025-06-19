"""
测试CausalQwen的核心数学框架

测试内容：
1. 柯西分布的线性稳定性
2. ActionNetwork的双模式（位置vs尺度）
3. 温度参数的选择性作用
4. AbductionNetwork的个体推断
"""

import pytest
import torch
import torch.nn.functional as F


def assert_cauchy_linear_stability(loc_input, scale_input, weight, bias, 
                                  loc_output, scale_output, tolerance):
    """验证柯西分布的线性稳定性"""
    # 验证位置参数：loc_output = loc_input @ weight.T + bias
    expected_loc = loc_input @ weight.T + bias
    torch.testing.assert_close(loc_output, expected_loc, 
                             atol=tolerance.get('atol', 1e-6),
                             rtol=tolerance.get('rtol', 1e-5))
    
    # 验证尺度参数：scale_output = scale_input @ |weight|.T
    expected_scale = scale_input @ torch.abs(weight).T
    torch.testing.assert_close(scale_output, expected_scale,
                             atol=tolerance.get('atol', 1e-6),
                             rtol=tolerance.get('rtol', 1e-5))


def assert_v2_mode_difference(loc_S_det, scale_S_det, loc_S_samp, scale_S_samp):
    """验证双模式的差异"""
    # 位置参数应该有差异（采样模式扰动了位置）
    loc_diff = torch.abs(loc_S_samp - loc_S_det).mean().item()
    assert loc_diff > 1e-6, f"位置参数差异过小: {loc_diff}"
    
    # 尺度参数的差异取决于具体实现
    scale_diff = torch.abs(scale_S_samp - scale_S_det).mean().item()
    return loc_diff, scale_diff


class TestCauchyMath:
    """测试柯西分布数学工具类"""
    
    def test_cauchy_linear_stability_location(self, tolerance):
        """测试柯西分布位置参数的线性稳定性"""
        from causal_qwen_mvp.components import CauchyMath
        
        # 准备测试数据
        batch_size, input_dim, output_dim = 4, 128, 256
        loc_input = torch.randn(batch_size, input_dim)
        weight = torch.randn(output_dim, input_dim)
        bias = torch.randn(output_dim)
        
        # 计算输出
        loc_output = CauchyMath.cauchy_linear_stable_loc(loc_input, weight, bias)
        
        # 验证形状
        assert loc_output.shape == (batch_size, output_dim)
        
        # 验证数学公式：output = input @ weight.T + bias
        expected_output = loc_input @ weight.T + bias
        torch.testing.assert_close(loc_output, expected_output, 
                                 atol=tolerance.get('atol', 1e-6), 
                                 rtol=tolerance.get('rtol', 1e-5))
    
    def test_cauchy_linear_stability_scale(self, tolerance):
        """测试柯西分布尺度参数的线性稳定性"""
        from causal_qwen_mvp.components import CauchyMath
        
        # 准备测试数据
        batch_size, input_dim, output_dim = 4, 128, 256
        scale_input = torch.abs(torch.randn(batch_size, input_dim)) + 0.1
        weight = torch.randn(output_dim, input_dim)
        
        # 计算输出
        scale_output = CauchyMath.cauchy_linear_stable_scale(scale_input, weight)
        
        # 验证形状
        assert scale_output.shape == (batch_size, output_dim)
        
        # 验证数学公式：output = input @ |weight|.T
        expected_output = scale_input @ torch.abs(weight).T
        torch.testing.assert_close(scale_output, expected_output,
                                 atol=tolerance.get('atol', 1e-6),
                                 rtol=tolerance.get('rtol', 1e-5))
    
    def test_combined_linear_transformation(self):
        """测试完整的线性变换（位置和尺度同时）"""
        from causal_qwen_mvp.components import CauchyMath
        
        # 准备数据
        batch_size, input_dim, output_dim = 2, 64, 100
        loc_input = torch.randn(batch_size, input_dim)
        scale_input = torch.abs(torch.randn(batch_size, input_dim)) + 0.5
        weight = torch.randn(output_dim, input_dim)
        bias = torch.randn(output_dim)
        
        # 计算
        loc_output = CauchyMath.cauchy_linear_stable_loc(loc_input, weight, bias)
        scale_output = CauchyMath.cauchy_linear_stable_scale(scale_input, weight)
        
        # 验证线性稳定性
        assert_cauchy_linear_stability(
            loc_input, scale_input, weight, bias,
            loc_output, scale_output, {'atol': 1e-6, 'rtol': 1e-5}
        )


class TestActionNetworkModes:
    """测试ActionNetwork的双模式"""
    
    def test_action_network_creation(self, action_network):
        """测试ActionNetwork的创建"""
        assert action_network is not None
        assert hasattr(action_network, 'lm_head')
        assert hasattr(action_network, 'b_noise')
        
        # 验证参数形状
        assert action_network.b_noise.shape == (action_network.config.causal_size,)
    
    def test_v2_non_sampling_mode(self, action_network, sample_loc_U, sample_scale_U, tolerance):
        """测试非采样模式：噪声影响尺度参数"""
        with torch.no_grad():
            loc_S_det, scale_S_det = action_network(
                sample_loc_U, sample_scale_U, do_sample=False
            )
        
        # 验证形状
        expected_shape = (sample_loc_U.shape[0], sample_loc_U.shape[1], 
                         action_network.config.vocab_size)
        assert loc_S_det.shape == expected_shape
        assert scale_S_det.shape == expected_shape
        
        # 验证数学实现
        expected_scale_U_noisy = sample_scale_U + torch.abs(action_network.b_noise)
        expected_loc_S = action_network.lm_head(sample_loc_U)
        expected_scale_S = expected_scale_U_noisy @ torch.abs(action_network.lm_head.weight).T
        
        torch.testing.assert_close(loc_S_det, expected_loc_S,
                                 atol=tolerance.get('atol', 1e-6),
                                 rtol=tolerance.get('rtol', 1e-5))
        torch.testing.assert_close(scale_S_det, expected_scale_S,
                                 atol=tolerance.get('atol', 1e-6),
                                 rtol=tolerance.get('rtol', 1e-5))
    
    def test_v2_sampling_mode(self, action_network, sample_loc_U, sample_scale_U, 
                            set_random_seed, tolerance):
        """测试采样模式：噪声影响位置参数"""
        set_random_seed(42)
        
        with torch.no_grad():
            loc_S_samp, scale_S_samp = action_network(
                sample_loc_U, sample_scale_U, do_sample=True, temperature=1.0
            )
        
        # 验证形状
        expected_shape = (sample_loc_U.shape[0], sample_loc_U.shape[1], 
                         action_network.config.vocab_size)
        assert loc_S_samp.shape == expected_shape
        assert scale_S_samp.shape == expected_shape
        
        # 验证尺度参数计算（采样模式下尺度不受噪声影响）
        expected_scale_S = sample_scale_U @ torch.abs(action_network.lm_head.weight).T
        torch.testing.assert_close(scale_S_samp, expected_scale_S,
                                 atol=tolerance.get('atol', 1e-6),
                                 rtol=tolerance.get('rtol', 1e-5))
    
    def test_mode_differences(self, action_network, sample_loc_U, sample_scale_U, set_random_seed):
        """测试采样与非采样模式的差异"""
        set_random_seed(42)
        
        # 非采样模式
        with torch.no_grad():
            loc_S_det, scale_S_det = action_network(
                sample_loc_U, sample_scale_U, do_sample=False
            )
        
        # 采样模式
        set_random_seed(42)
        with torch.no_grad():
            loc_S_samp, scale_S_samp = action_network(
                sample_loc_U, sample_scale_U, do_sample=True, temperature=1.0
            )
        
        # 验证差异
        loc_diff, scale_diff = assert_v2_mode_difference(
            loc_S_det, scale_S_det, loc_S_samp, scale_S_samp
        )
        
        # 位置参数应该有显著差异
        assert loc_diff > 1e-3, f"位置参数差异过小: {loc_diff}"
    
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_temperature_effect_in_sampling_mode(self, action_network, sample_loc_U, 
                                               sample_scale_U, temperature, set_random_seed):
        """测试温度参数在采样模式下的效果"""
        set_random_seed(42)
        
        # 基准（temperature=1.0）
        with torch.no_grad():
            loc_S_base, _ = action_network(
                sample_loc_U, sample_scale_U, do_sample=True, temperature=1.0
            )
        
        # 不同温度
        set_random_seed(42)
        with torch.no_grad():
            loc_S_temp, _ = action_network(
                sample_loc_U, sample_scale_U, do_sample=True, temperature=temperature
            )
        
        # 计算差异
        diff = torch.abs(loc_S_temp - loc_S_base).mean().item()
        
        if temperature == 1.0:
            assert diff < 1e-8, "温度为1.0时应该与基准相同"
        else:
            # 温度不同时应该有差异，且差异大小与温度相关
            assert diff > 1e-6, f"温度{temperature}未产生预期差异"
    
    def test_temperature_effect_in_standard_mode(self, action_network, 
                                                       sample_loc_U, sample_scale_U):
        """测试温度参数在标准模式下的效果"""
        with torch.no_grad():
            # temperature=0 (纯因果模式)
            loc_S_0, scale_S_0 = action_network(
                sample_loc_U, sample_scale_U, do_sample=False, temperature=0.0
            )
            
            # temperature=1.0 (标准模式)
            loc_S_1, scale_S_1 = action_network(
                sample_loc_U, sample_scale_U, do_sample=False, temperature=1.0
            )
            
            # temperature=2.0 (标准模式)
            loc_S_2, scale_S_2 = action_network(
                sample_loc_U, sample_scale_U, do_sample=False, temperature=2.0
            )
        
        # 位置参数应该相同（标准模式下位置参数不受温度影响）
        torch.testing.assert_close(loc_S_0, loc_S_1, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(loc_S_1, loc_S_2, atol=1e-8, rtol=1e-8)
        
        # 尺度参数应该随温度增加而增加
        scale_mean_0 = scale_S_0.mean().item()
        scale_mean_1 = scale_S_1.mean().item()
        scale_mean_2 = scale_S_2.mean().item()
        
        assert scale_mean_0 < scale_mean_1, f"温度1.0应该比温度0产生更大的尺度参数: {scale_mean_0} vs {scale_mean_1}"
        assert scale_mean_1 < scale_mean_2, f"温度2.0应该比温度1.0产生更大的尺度参数: {scale_mean_1} vs {scale_mean_2}"


class TestAbductionNetwork:
    """测试归因推断网络"""
    
    def test_abduction_network_creation(self, abduction_network):
        """测试AbductionNetwork的创建"""
        assert abduction_network is not None
        assert hasattr(abduction_network, 'loc_net')
        assert hasattr(abduction_network, 'scale_net')
    
    def test_abduction_forward(self, abduction_network, sample_hidden_states):
        """测试归因推断的前向传播"""
        loc_U, scale_U = abduction_network(sample_hidden_states)
        
        # 验证形状
        expected_shape = (sample_hidden_states.shape[0], 
                         sample_hidden_states.shape[1],
                         abduction_network.config.causal_size)
        assert loc_U.shape == expected_shape
        assert scale_U.shape == expected_shape
        
        # 验证尺度参数为正
        assert torch.all(scale_U > 0), "尺度参数必须为正"
    
    def test_abduction_initialization(self, abduction_network, sample_hidden_states):
        """测试归因网络的初始化策略"""
        # 初始化后应该接近恒等映射
        loc_U, scale_U = abduction_network(sample_hidden_states)
        
        # 如果是恒等初始化，loc_U应该接近输入
        if abduction_network.config.hidden_size == abduction_network.config.causal_size:
            # 检查是否接近恒等映射（允许一定偏差）
            diff = torch.abs(loc_U - sample_hidden_states).mean().item()
            assert diff < 1.0, f"初始化偏离恒等映射过多: {diff}"
        
        # 尺度参数应该有合理的初始值
        scale_mean = scale_U.mean().item()
        assert 0.1 < scale_mean < 10.0, f"尺度参数初始值不合理: {scale_mean}"


class TestIntegration:
    """集成测试：多个组件协同工作"""
    
    def test_abduction_to_action_flow(self, abduction_network, action_network, 
                                     sample_hidden_states, tolerance):
        """测试从归因到行动的完整流程"""
        # Step 1: 归因推断
        loc_U, scale_U = abduction_network(sample_hidden_states)
        
        # Step 2: 行动决策（两种模式）
        with torch.no_grad():
            # 非采样模式
            loc_S_det, scale_S_det = action_network(loc_U, scale_U, do_sample=False)
            
            # 采样模式
            loc_S_samp, scale_S_samp = action_network(loc_U, scale_U, do_sample=True)
        
        # 验证输出形状
        batch_size, seq_len = sample_hidden_states.shape[:2]
        vocab_size = action_network.config.vocab_size
        expected_shape = (batch_size, seq_len, vocab_size)
        
        assert loc_S_det.shape == expected_shape
        assert scale_S_det.shape == expected_shape
        assert loc_S_samp.shape == expected_shape
        assert scale_S_samp.shape == expected_shape
        
        # 验证模式差异
        loc_diff, _ = assert_v2_mode_difference(
            loc_S_det, scale_S_det, loc_S_samp, scale_S_samp
        )
        assert loc_diff > 1e-6, "两种模式应该产生不同的位置参数" 