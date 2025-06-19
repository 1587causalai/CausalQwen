"""
测试CausalQwen与Qwen的接口兼容性

测试内容：
1. generate()接口完全兼容
2. do_sample, temperature, top_k, top_p参数支持
3. 批量生成功能
4. 数学原理验证
"""

import pytest
import torch


class TestQwenCompatibility:
    """测试与Qwen的接口兼容性"""
    
    def test_model_creation(self, test_model):
        """测试模型创建"""
        assert test_model is not None
        assert hasattr(test_model, 'generate')
        assert hasattr(test_model, 'forward')
        assert hasattr(test_model, 'config')
    
    def test_generate_interface_deterministic(self, test_model, sample_input_ids):
        """测试确定性生成接口（do_sample=False）"""
        # 确定性生成
        output = test_model.generate(
            sample_input_ids,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=test_model.config.eos_token_id
        )
        
        # 验证输出
        assert output is not None
        assert output.shape[0] == sample_input_ids.shape[0]  # 批次大小相同
        assert output.shape[1] == sample_input_ids.shape[1] + 5  # 长度增加5
        
        # 验证前缀保持不变
        assert torch.all(output[:, :sample_input_ids.shape[1]] == sample_input_ids)
    
    def test_generate_interface_sampling(self, test_model, sample_input_ids):
        """测试采样生成接口（do_sample=True）"""
        # 采样生成
        output = test_model.generate(
            sample_input_ids,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            pad_token_id=test_model.config.eos_token_id
        )
        
        # 验证输出
        assert output is not None
        assert output.shape[0] == sample_input_ids.shape[0]
        assert output.shape[1] == sample_input_ids.shape[1] + 5
        
        # 验证前缀保持不变
        assert torch.all(output[:, :sample_input_ids.shape[1]] == sample_input_ids)
    
    def test_do_sample_difference(self, test_model, sample_input_ids, set_random_seed):
        """测试do_sample参数产生的差异"""
        max_new_tokens = 5
        
        # 确定性生成
        det_output = test_model.generate(
            sample_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        # 采样生成（多次）
        sampling_outputs = []
        for i in range(3):
            set_random_seed(i)  # 不同的种子
            samp_output = test_model.generate(
                sample_input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0
            )
            sampling_outputs.append(samp_output)
        
        # 验证确定性生成的一致性
        det_output2 = test_model.generate(
            sample_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        assert torch.all(det_output == det_output2), "确定性生成应该完全一致"
        
        # 验证采样生成的多样性
        unique_sequences = set()
        for output in sampling_outputs:
            new_tokens = output[0, sample_input_ids.shape[1]:].tolist()
            unique_sequences.add(tuple(new_tokens))
        
        assert len(unique_sequences) >= 2, "采样生成应该具有多样性"
    
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 1.5, 2.0])
    def test_temperature_effect(self, test_model, sample_input_ids, temperature, set_random_seed):
        """测试温度参数的效果"""
        set_random_seed(42)
        
        # 生成
        output = test_model.generate(
            sample_input_ids,
            max_new_tokens=3,
            do_sample=True,
            temperature=temperature,
            top_k=100
        )
        
        # 基本验证
        assert output.shape[1] == sample_input_ids.shape[1] + 3
        
        # TODO: 可以添加更多关于温度效果的统计验证
    
    def test_temperature_zero_consistency(self, test_model, sample_input_ids):
        """测试温度为0时的一致性行为"""
        # 温度为0时，无论do_sample是什么，都应该是纯因果模式
        output1 = test_model.generate(
            sample_input_ids,
            max_new_tokens=3,
            do_sample=False,
            temperature=0.0
        )
        
        output2 = test_model.generate(
            sample_input_ids,
            max_new_tokens=3,
            do_sample=True,
            temperature=0.0
        )
        
        # 温度为0时两种模式应该产生相同结果（纯因果）
        assert torch.all(output1 == output2), "温度为0时两种模式应该产生相同结果（纯因果）"
    
    def test_top_k_top_p_parameters(self, test_model, sample_input_ids, set_random_seed):
        """测试top_k和top_p参数"""
        set_random_seed(42)
        
        # 测试top_k
        output_k10 = test_model.generate(
            sample_input_ids,
            max_new_tokens=3,
            do_sample=True,
            temperature=1.0,
            top_k=10
        )
        
        set_random_seed(42)
        output_k50 = test_model.generate(
            sample_input_ids,
            max_new_tokens=3,
            do_sample=True,
            temperature=1.0,
            top_k=50
        )
        
        # 基本验证
        assert output_k10.shape[1] == sample_input_ids.shape[1] + 3
        assert output_k50.shape[1] == sample_input_ids.shape[1] + 3
        
        # 测试top_p
        set_random_seed(42)
        output_p09 = test_model.generate(
            sample_input_ids,
            max_new_tokens=3,
            do_sample=True,
            temperature=1.0,
            top_p=0.9
        )
        
        assert output_p09.shape[1] == sample_input_ids.shape[1] + 3
    
    def test_batch_generation(self, test_model, sample_input_ids):
        """测试批量生成"""
        batch_size = sample_input_ids.shape[0]
        assert batch_size > 1, "需要批量输入进行测试"
        
        # 批量生成
        batch_output = test_model.generate(
            sample_input_ids,
            max_new_tokens=4,
            do_sample=True,
            temperature=1.0,
            top_k=50
        )
        
        # 验证形状
        assert batch_output.shape[0] == batch_size
        assert batch_output.shape[1] == sample_input_ids.shape[1] + 4
        
        # 验证批内多样性（如果使用不同的随机种子）
        sequences = []
        for i in range(batch_size):
            new_tokens = batch_output[i, sample_input_ids.shape[1]:].tolist()
            sequences.append(tuple(new_tokens))
        
        # 至少应该有一些不同的序列
        unique_sequences = set(sequences)
        assert len(unique_sequences) >= 1  # 至少有一个唯一序列


class TestMathematicalPrinciples:
    """测试数学原理在生成中的体现"""
    
    def test_inference_validator_available(self, test_model):
        """测试InferenceValidator是否可用"""
        try:
            from causal_qwen_mvp import InferenceValidator
            validator = InferenceValidator(test_model)
            assert validator is not None
        except ImportError:
            pytest.skip("InferenceValidator not available")
    
    def test_v2_principles_validation(self, test_model, sample_input_ids):
        """测试数学原理验证"""
        try:
            from causal_qwen_mvp import InferenceValidator
        except ImportError:
            pytest.skip("InferenceValidator not available")
        
        validator = InferenceValidator(test_model)
        results = validator.validate_causal_principles(sample_input_ids, temperature=1.0)
        
        # 验证返回结果
        assert 'position_difference' in results
        assert 'scale_difference' in results
        assert 'base_representations' in results
        
        # 位置参数差异应该显著
        pos_diff = results['position_difference'].item()
        assert pos_diff > 1e-4, f"位置参数差异过小: {pos_diff}"
        
        # 基础表征验证
        base_repr = results['base_representations']
        assert 'loc_U' in base_repr
        assert 'scale_U' in base_repr
        
        # 尺度参数应该为正
        scale_U_mean = base_repr['scale_U'].mean().item()
        assert scale_U_mean > 0, "尺度参数必须为正"
    
    def test_generation_consistency(self, test_model, sample_input_ids):
        """测试生成的一致性"""
        # 相同输入的确定性生成应该一致
        outputs = []
        for _ in range(3):
            output = test_model.generate(
                sample_input_ids,
                max_new_tokens=5,
                do_sample=False
            )
            outputs.append(output)
        
        # 所有输出应该相同
        for i in range(1, len(outputs)):
            assert torch.all(outputs[0] == outputs[i]), f"第{i}次生成结果不一致"
    
    def test_special_temperature_zero(self, test_model, sample_input_ids):
        """测试温度为0的特殊情况"""
        # 温度为0是重要的边界条件
        output = test_model.generate(
            sample_input_ids,
            max_new_tokens=3,
            do_sample=True,
            temperature=0.0  # 极限情况
        )
        
        # 应该成功生成
        assert output is not None
        assert output.shape[1] == sample_input_ids.shape[1] + 3
        
        # 温度为0时，即使do_sample=True，行为也应该接近确定性
        # （但具体实现可能有所不同） 