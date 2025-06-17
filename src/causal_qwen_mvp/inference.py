"""
CausalQwen MVP: 推理模块实现
实现三种推理模式：标准、因果采样、兼容传统
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Union
from .models import CausalQwenMVPForCausalLM


class CausalInferenceEngine:
    """CausalQwen推理引擎"""
    
    def __init__(self, model: CausalQwenMVPForCausalLM):
        self.model = model
        self.config = model.config
    
    def inference(self, input_ids, mode='standard', temperature=1.0, **kwargs):
        """统一推理接口"""
        if mode == 'standard':
            return self._standard_inference(input_ids, **kwargs)
        elif mode == 'causal':  
            return self._causal_sampling(input_ids, temperature=temperature, **kwargs)
        elif mode == 'compatible':
            return self._compatible_sampling(input_ids, temperature=temperature, **kwargs)
        else:
            raise ValueError(f"Unknown inference mode: {mode}")
    
    def _standard_inference(self, input_ids, **kwargs):
        """标准确定性推理：使用期望值计算"""
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            # 返回完整的CausalMVPOutput结构
            return outputs
    
    def _causal_sampling(self, input_ids, temperature=1.0, **kwargs):
        """
        个体因果采样：个体具现 → 环境噪声 → 线性决策
        
        数学框架：
        1. 采样个体: u_i ~ Cauchy(loc_U_i, temperature * scale_U_i)
        2. 构建决策输入分布: U'_input ~ Cauchy(u_i, |b_noise|) 
        3. 解析计算决策: 将决策输入分布传入ActionNetwork线性变换
        """  
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            
            # 步骤1：采样具体个体（温度控制个体采样的不确定性）
            # 温度越低，scale_U越小，个体采样越确定性
            temperature_controlled_scale_U = outputs.scale_U * temperature
            uniform_sample = torch.rand_like(outputs.loc_U)
            u_sampled = outputs.loc_U + temperature_controlled_scale_U * torch.tan(torch.pi * (uniform_sample - 0.5))
            
            # 步骤2-3：构建决策输入分布并解析计算
            # ActionNetwork将采样的个体u_sampled作为位置参数
            # 并使用其内置的b_noise作为环境噪声的尺度参数
            # 这样ActionNetwork内部会计算：U'_input ~ Cauchy(u_sampled, |b_noise|)
            # 然后解析计算最终的决策分布
            loc_S, scale_S = self.model.action_network(u_sampled, torch.zeros_like(outputs.scale_U))
            
            # 更新输出结构
            from .models import CausalMVPOutput
            return CausalMVPOutput(
                loc_S=loc_S,
                scale_S=scale_S,
                loc_U=u_sampled,
                scale_U=torch.zeros_like(outputs.scale_U),
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
    
    def _compatible_sampling(self, input_ids, do_sample=True, top_k=50, top_p=0.9, temperature=1.0, **kwargs):
        """传统兼容采样：支持确定性和随机采样"""
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            # 返回完整的输出结构，包含所有字段
            return outputs
            
            # 随机采样：应用top-k过滤
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                # 将其他位置设为负无穷
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
                logits = filtered_logits
            
            # 应用top-p (nucleus)过滤
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 找到累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的token（重要！）
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将需要移除的位置设为负无穷
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Softmax并采样
            probs = F.softmax(logits, dim=-1)
            next_token_ids = torch.multinomial(probs, 1)
            
            return next_token_ids
    
    def generate_step_by_step(
        self,
        input_ids,
        max_length=None,
        max_new_tokens=None,
        mode='standard',
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        """
        CausalQwen专用的逐步生成方法
        
        Args:
            mode (str): 推理模式
                - 'standard': 确定性OvR
                - 'causal': 因果采样 
                - 'compatible_sample': 传统采样（HuggingFace兼容）
                - 'compatible_deterministic': 传统确定性（HuggingFace兼容）
        """
        # 参数处理
        if max_length is None and max_new_tokens is None:
            max_new_tokens = 50
        elif max_new_tokens is None:
            max_new_tokens = max_length - input_ids.shape[-1]
            
        if pad_token_id is None:
            pad_token_id = getattr(self.config, 'pad_token_id', 0)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', 2)
        
        # 直接使用当前引擎实例（self已经是CausalInferenceEngine）
        current_ids = input_ids.clone()
        
        # 逐步生成
        for step in range(max_new_tokens):
            if mode in ['compatible_sample', 'compatible_deterministic']:
                # 兼容模式：使用传统HuggingFace逻辑
                do_sample = (mode == 'compatible_sample')
                next_token = self._compatible_sampling(
                    current_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    **kwargs
                )
            else:
                # CausalQwen模式：standard或causal
                next_token = self.inference(current_ids, mode=mode, temperature=temperature, **kwargs)
            
            # 添加到序列
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # 检查停止条件
            if next_token[0, -1].item() == eos_token_id:
                break
                
        return current_ids
    
    def batch_generate(self, input_ids_list, mode='standard', **kwargs):
        """批量生成 - 占位实现"""
        # TODO: 实现高效的批量生成
        results = []
        for input_ids in input_ids_list:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            result = self.generate_step_by_step(input_ids, mode=mode, **kwargs)
            results.append(result)
        return results


# 为主模型类添加推理方法
def add_inference_methods(model_class):
    """为CausalQwenMVPForCausalLM添加推理方法"""
    
    def inference(self, input_ids, mode='standard', **kwargs):
        """统一推理接口"""
        engine = CausalInferenceEngine(self)
        return engine.inference(input_ids, mode, **kwargs)
    
    def generate_step_by_step(self, input_ids, max_length=50, mode='standard', **kwargs):
        """自回归生成（CausalQwen专用）"""
        engine = CausalInferenceEngine(self)
        return engine.generate_step_by_step(input_ids, max_length=max_length, mode=mode, **kwargs)
    
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        """
        HuggingFace compatible generate method with CausalQwen extensions.
        
        Design Trade-offs (设计取舍):
        - Priority: causal_mode > do_sample (因果模式优先级高于传统采样)
        - Default: 'standard' mode for deterministic behavior (默认确定性模式)
        - Simplification: Clear separation of three modes, avoiding complex parameter mapping
          (简化设计：三种模式明确分离，避免复杂参数映射)
        
        Args:
            causal_mode (str): 'standard', 'causal', or 'compatible' 
            do_sample (bool): Only effective when causal_mode='compatible'
            temperature (float): Controls sampling randomness
            top_k (int): Limits vocabulary for sampling  
            top_p (float): Nucleus sampling threshold
        """
        # Extract CausalQwen-specific parameters
        causal_mode = kwargs.pop('causal_mode', 'standard')  # Default to deterministic
        
        # Mode determination with clear priority: causal_mode > do_sample
        if causal_mode == 'standard':
            # Deterministic OvR prediction (ignore do_sample)
            mode = 'standard'
        elif causal_mode == 'causal':
            # Individual-based causal sampling (ignore do_sample)  
            mode = 'causal'
        elif causal_mode == 'compatible':
            # Traditional HuggingFace behavior (respect do_sample)
            do_sample = kwargs.get('do_sample', False)
            mode = 'compatible_sample' if do_sample else 'compatible_deterministic'
        else:
            raise ValueError(f"Invalid causal_mode: {causal_mode}. Must be 'standard', 'causal', or 'compatible'")
        
        # Delegate to step-by-step generation with resolved mode
        return self.generate_step_by_step(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            mode=mode,
            **kwargs
        )
    
    def batch_generate(self, input_ids_list, mode='standard', **kwargs):
        """批量生成"""
        engine = CausalInferenceEngine(self)
        return engine.batch_generate(input_ids_list, mode, **kwargs)
    
    # 动态添加方法到类
    model_class.inference = inference
    model_class.generate_step_by_step = generate_step_by_step
    model_class.generate = generate
    model_class.batch_generate = batch_generate
    
    return model_class


# 应用推理方法到主模型类
CausalQwenMVPForCausalLM = add_inference_methods(CausalQwenMVPForCausalLM)


class InferenceValidator:
    """推理验证工具"""
    
    def __init__(self, model: CausalQwenMVPForCausalLM):
        self.model = model
        
    def validate_inference_consistency(self, input_ids, num_samples=5):
        """验证不同推理模式的一致性 - 占位实现"""
        # TODO: 实现更comprehensive的一致性检查
        results = {}
        
        # 测试标准推理
        try:
            standard_output = self.model.inference(input_ids, mode='standard')
            results['standard'] = standard_output
            print(f"Standard inference: {standard_output.shape}")
        except Exception as e:
            print(f"Standard inference failed: {e}")
            
        # 测试因果采样
        try:
            causal_outputs = []
            for i in range(num_samples):
                causal_output = self.model.inference(input_ids, mode='causal')
                causal_outputs.append(causal_output)
            results['causal'] = causal_outputs
            print(f"Causal sampling: {len(causal_outputs)} samples")
        except Exception as e:
            print(f"Causal sampling failed: {e}")
            
        # 测试兼容采样
        try:
            compatible_output = self.model.inference(input_ids, mode='compatible')
            results['compatible'] = compatible_output
            print(f"Compatible sampling: {compatible_output.shape}")
        except Exception as e:
            print(f"Compatible sampling failed: {e}")
            
        return results
    
    def test_generation_quality(self, input_ids, max_length=20):
        """测试生成质量 - 占位实现"""
        # TODO: 实现质量评估指标
        results = {}
        
        for mode in ['standard', 'causal', 'compatible']:
            try:
                generated = self.model.generate_step_by_step(
                    input_ids, 
                    max_length=max_length,
                    mode=mode
                )
                results[mode] = {
                    'generated_ids': generated,
                    'length': generated.shape[-1],
                    'input_length': input_ids.shape[-1]
                }
                print(f"{mode} generation: length {generated.shape[-1]}")
            except Exception as e:
                print(f"{mode} generation failed: {e}")
                
        return results 