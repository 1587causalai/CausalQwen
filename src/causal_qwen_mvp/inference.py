"""
CausalQwen 推理引擎 - 与Qwen完全兼容

V2核心创新：位置vs尺度的精妙差异
├─ do_sample=True：噪声影响位置参数，扰动个体身份  
└─ do_sample=False：噪声影响尺度参数，增加决策不确定性

完全兼容Qwen接口：do_sample, temperature, top_k, top_p, max_new_tokens
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .models import CausalQwenMVPForCausalLM, CausalMVPOutput


class CausalInferenceEngine:
    """CausalQwen推理引擎 - 与Qwen完全兼容"""
    
    def __init__(self, model: CausalQwenMVPForCausalLM):
        self.model = model
        self.config = model.config
    
    def __call__(self, input_ids, **kwargs) -> CausalMVPOutput:
        """直接调用推理，与Qwen接口兼容"""
        return self.model(input_ids, **kwargs)
    
    def generate_next_token(self, input_ids, do_sample=False, temperature=1.0, 
                          top_k=50, top_p=0.9, **kwargs) -> torch.LongTensor:
        """生成下一个token - 与Qwen生成接口兼容
        
        Args:
            input_ids: 输入序列
            do_sample: 是否采样（V2核心参数）
            temperature: 温度参数
            top_k: top-k采样
            top_p: nucleus采样
            
        Returns:
            next_token: 下一个token [batch_size, 1]
        """
        output = self.model(input_ids, do_sample=do_sample, temperature=temperature, **kwargs)
        
        # 计算OvR概率
        ovr_probs = self.model.ovr_classifier(output.loc_S, output.scale_S)
        logits = ovr_probs[:, -1, :]  # 最后一个位置的概率
        
        if do_sample:
            # 采样生成
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k过滤
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 多项分布采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            # 贪心生成
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token
    
    def generate(self, input_ids, max_new_tokens=20, do_sample=True, temperature=1.0,
                top_k=50, top_p=0.9, pad_token_id=None, eos_token_id=None, **kwargs):
        """序列生成 - 完全兼容Qwen.generate()接口
        
        Args:
            input_ids: 初始序列 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
            do_sample: 是否采样
            temperature: 温度参数  
            top_k: top-k采样
            top_p: nucleus采样
            pad_token_id: padding token id
            eos_token_id: 结束token id
            
        Returns:
            generated_ids: 完整序列 [batch_size, seq_len + new_tokens]
        """
        batch_size = input_ids.shape[0]
        current_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                next_token = self.generate_next_token(
                    current_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    **kwargs
                )
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # 检查是否遇到结束token
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
        
        return current_ids


class InferenceValidator:
    """推理验证器 - 验证V2数学原理"""
    
    def __init__(self, model: CausalQwenMVPForCausalLM):
        self.model = model
        self.engine = CausalInferenceEngine(model)
    
    def validate_v2_principles(self, input_ids, temperature=1.0):
        """验证V2数学原理：位置vs尺度差异"""
        results = {}
        
        with torch.no_grad():
            # 基础表征
            transformer_output = self.model.model(input_ids)
            hidden_states = transformer_output[0]
            loc_U, scale_U = self.model.abduction_network(hidden_states)
            
            # 确定性模式：噪声影响尺度参数
            det_output = self.model(input_ids, do_sample=False)
            
            # 采样模式：噪声影响位置参数
            samp_output = self.model(input_ids, do_sample=True, temperature=temperature)
            
            results = {
                'base_representations': {
                    'loc_U': loc_U,
                    'scale_U': scale_U
                },
                'deterministic_mode': {
                    'loc_S': det_output.loc_S,
                    'scale_S': det_output.scale_S
                },
                'sampling_mode': {
                    'loc_S': samp_output.loc_S,
                    'scale_S': samp_output.scale_S
                },
                'position_difference': torch.abs(det_output.loc_S - samp_output.loc_S).mean(),
                'scale_difference': torch.abs(det_output.scale_S - samp_output.scale_S).mean()
            }
        
        return results
    
    def compare_with_qwen(self, input_ids, qwen_model=None):
        """与原始Qwen模型对比，验证兼容性"""
        if qwen_model is None:
            return None
            
        results = {}
        
        with torch.no_grad():
            # CausalQwen确定性输出
            causal_output = self.model(input_ids, do_sample=False)
            causal_logits = causal_output.loc_S  # 位置参数作为logits
            
            # Qwen确定性输出
            qwen_output = qwen_model(input_ids)
            qwen_logits = qwen_output.logits
            
            # 对比分析
            logits_diff = torch.abs(causal_logits - qwen_logits).mean()
            
            # 预测一致性
            causal_pred = torch.argmax(causal_logits[:, -1, :], dim=-1)
            qwen_pred = torch.argmax(qwen_logits[:, -1, :], dim=-1)
            pred_match = (causal_pred == qwen_pred).float().mean()
            
            results = {
                'logits_difference': logits_diff.item(),
                'prediction_match_rate': pred_match.item(),
                'causal_prediction': causal_pred.tolist(),
                'qwen_prediction': qwen_pred.tolist()
            }
        
        return results