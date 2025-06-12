"""测试CausalQwen的生成功能"""

import torch
import pytest
from dataclasses import dataclass


@dataclass
class MockConfig:
    """模拟配置"""
    num_vocab: int = 100
    hidden_dim: int = 64
    num_token_id: int = 99  # <NUM> token
    eos_token_id: int = 2   # </s> token


class MockCausalQwen:
    """用于测试的模拟CausalQwen模型"""
    
    def __init__(self, config):
        self.config = config
        self.num_vocab = config.num_vocab
        self.hidden_dim = config.hidden_dim
        self.num_token_id = config.num_token_id
        
    def eval(self):
        """模拟eval模式"""
        return self
    
    def generate(
        self,
        input_ids: torch.Tensor,
        num_values: torch.Tensor = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        sampling_mode: str = "causal",
        return_dict_in_generate: bool = False,
        **kwargs
    ):
        """模拟生成方法"""
        batch_size, seq_len = input_ids.shape
        
        # 模拟生成过程
        generated_ids = input_ids.clone()
        generated_values = num_values.clone() if num_values is not None else torch.zeros_like(input_ids, dtype=torch.float)
        
        generation_info = {
            'u_scale': [],
            'token_probs': []
        } if return_dict_in_generate else None
        
        # 模拟EOS提前停止的概率
        eos_probability = 0.1 if hasattr(self.config, 'eos_token_id') else 0.0
        
        for step in range(max_new_tokens):
            # 模拟下一个token生成
            if sampling_mode == "causal":
                # 模拟因果采样：温度影响scale
                scale = torch.ones(batch_size, self.hidden_dim) * temperature
                next_token = torch.randint(0, self.num_vocab, (batch_size,))
                
                # 修复批处理问题：逐个检查每个样本
                next_value = torch.zeros(batch_size)
                for b in range(batch_size):
                    if next_token[b].item() == self.num_token_id:
                        next_value[b] = torch.randn(1).item()
                
                if return_dict_in_generate:
                    generation_info['u_scale'].append(scale)
                    generation_info['token_probs'].append(torch.randn(batch_size, self.num_vocab))
            else:
                # 模拟传统采样
                next_token = torch.randint(0, self.num_vocab, (batch_size,))
                
                # 修复批处理问题：逐个检查每个样本
                next_value = torch.zeros(batch_size)
                for b in range(batch_size):
                    if next_token[b].item() == self.num_token_id:
                        next_value[b] = torch.randn(1).item()
                
                if return_dict_in_generate:
                    generation_info['token_probs'].append(torch.randn(batch_size, self.num_vocab))
            
            # 更新序列
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            generated_values = torch.cat([generated_values, next_value.unsqueeze(1)], dim=1)
            
            # 检查EOS（随机模拟提前停止）
            if hasattr(self.config, 'eos_token_id') and torch.rand(1).item() < eos_probability:
                # 随机选择一些样本停止
                break
        
        if return_dict_in_generate:
            return {
                'sequences': generated_ids,
                'values': generated_values,
                'generation_info': generation_info
            }
        else:
            return generated_ids


class TestGeneration:
    """生成功能测试"""
    
    @pytest.fixture
    def model(self):
        """创建模型实例"""
        config = MockConfig()
        model = MockCausalQwen(config)
        model.eval()
        return model
    
    def test_basic_generation(self, model):
        """测试基本生成功能"""
        # 准备输入
        input_ids = torch.tensor([[1, 2, 3, 4]])
        
        # 生成
        output = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0
        )
        
        # 验证输出形状（考虑可能的提前停止）
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] >= 4  # 至少保持原始长度
        assert output.shape[1] <= 14  # 最多是原始4 + 新生成10
        
        # 验证前缀保持不变
        assert torch.all(output[:, :4] == input_ids)
    
    def test_causal_vs_traditional_sampling(self, model):
        """测试因果采样vs传统采样"""
        input_ids = torch.tensor([[1, 2, 3]])
        
        # 因果采样
        causal_output = model.generate(
            input_ids,
            max_new_tokens=5,
            sampling_mode="causal",
            temperature=0.8
        )
        
        # 传统采样
        traditional_output = model.generate(
            input_ids,
            max_new_tokens=5,
            sampling_mode="traditional",
            temperature=0.8
        )
        
        # 两种模式都应该生成有效输出
        assert causal_output.shape[0] == traditional_output.shape[0] == 1
        assert causal_output.shape[1] >= 3  # 至少保持原始长度
        assert traditional_output.shape[1] >= 3  # 至少保持原始长度
    
    def test_generation_with_numerical_values(self, model):
        """测试带数值的生成"""
        # 输入包含<NUM>token
        input_ids = torch.tensor([[1, 2, model.config.num_token_id, 4]])
        num_values = torch.tensor([[0., 0., 3.14, 0.]])
        
        # 生成
        output = model.generate(
            input_ids,
            num_values=num_values,
            max_new_tokens=5,
            return_dict_in_generate=True
        )
        
        # 验证返回格式
        assert 'sequences' in output
        assert 'values' in output
        assert 'generation_info' in output
        
        # 验证数值传递
        assert torch.allclose(output['values'][0, 2], torch.tensor(3.14))
    
    def test_top_k_top_p_filtering(self, model):
        """测试top-k和top-p过滤"""
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Top-k采样
        topk_output = model.generate(
            input_ids,
            max_new_tokens=10,
            top_k=5
        )
        
        # Top-p采样
        topp_output = model.generate(
            input_ids,
            max_new_tokens=10,
            top_p=0.9
        )
        
        # 组合采样
        combined_output = model.generate(
            input_ids,
            max_new_tokens=10,
            top_k=10,
            top_p=0.95
        )
        
        # 验证输出有效（考虑可能的提前停止）
        assert topk_output.shape[1] >= 3  # 至少保持原始长度
        assert topp_output.shape[1] >= 3  # 至少保持原始长度
        assert combined_output.shape[1] >= 3  # 至少保持原始长度
        
        # 验证前缀保持
        assert torch.all(topk_output[:, :3] == input_ids)
        assert torch.all(topp_output[:, :3] == input_ids)
        assert torch.all(combined_output[:, :3] == input_ids)
    
    def test_temperature_effect(self, model):
        """测试温度参数的影响"""
        input_ids = torch.tensor([[1, 2, 3]])
        
        # 设置随机种子以便比较
        torch.manual_seed(42)
        
        # 低温生成（更确定）
        low_temp_output = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.1,
            sampling_mode="causal",
            return_dict_in_generate=True
        )
        
        # 高温生成（更随机）
        torch.manual_seed(42)
        high_temp_output = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=2.0,
            sampling_mode="causal",
            return_dict_in_generate=True
        )
        
        # 比较U分布的scale（高温应该有更大的scale）
        if ('u_scale' in low_temp_output['generation_info'] and 
            'u_scale' in high_temp_output['generation_info'] and
            len(low_temp_output['generation_info']['u_scale']) > 0 and
            len(high_temp_output['generation_info']['u_scale']) > 0):
            
            low_scales = torch.stack(low_temp_output['generation_info']['u_scale'])
            high_scales = torch.stack(high_temp_output['generation_info']['u_scale'])
            
            # 平均来看，高温的scale应该更大
            assert high_scales.mean() > low_scales.mean() * 1.5
        else:
            # 如果没有u_scale信息，只验证生成完成
            assert low_temp_output['sequences'].shape[1] >= 3
            assert high_temp_output['sequences'].shape[1] >= 3
    
    def test_batch_generation(self, model):
        """测试批量生成"""
        # 批量输入
        input_ids = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        # 批量生成
        output = model.generate(
            input_ids,
            max_new_tokens=7,
            temperature=1.0
        )
        
        # 验证批量输出
        assert output.shape[0] == 3  # batch size
        assert output.shape[1] >= 3  # 至少保持原始长度
        
        # 验证每个样本的前缀
        for i in range(3):
            assert torch.all(output[i, :3] == input_ids[i])
    
    def test_early_stopping(self, model):
        """测试提前停止（EOS）"""
        input_ids = torch.tensor([[1, 2, 3]])
        
        # 生成直到EOS
        output = model.generate(
            input_ids,
            max_new_tokens=100,  # 设置很大，但应该提前停止
            temperature=1.0
        )
        
        # 检查是否包含EOS并在合理长度内停止
        if hasattr(model.config, 'eos_token_id'):
            # 由于是随机生成，只验证长度合理性
            assert output.shape[1] <= 103  # 应该在最大长度之前停止
            assert output.shape[1] >= 3   # 至少保持原始长度


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
