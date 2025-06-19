"""
CausalQwen 推理引擎 - 与Qwen完全兼容

核心创新：温度参数统一控制噪声强度，四种推理模式
├─ Causal模式 (temperature=0): 纯因果生成，无外生噪声
├─ Standard模式 (do_sample=False, temperature>0): 噪声增加决策不确定性
├─ Sampling模式 (do_sample=True, temperature>0): 噪声扰动个体身份
└─ Compatible模式: 传统Softmax，与原始Qwen兼容

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
                          **kwargs) -> torch.LongTensor:
        """生成下一个token - CausalQwen专用推理
        
        CausalQwen的"采样"是在ActionNetwork内部通过噪声注入实现的，
        不是传统的多项分布采样！
        
        Args:
            input_ids: 输入序列
            do_sample: 是否采样（控制ActionNetwork内部噪声注入方式）
            temperature: 温度参数（控制噪声强度，在ActionNetwork内部生效）
            
        Returns:
            next_token: 下一个token [batch_size, 1]
        """
        # CausalQwen推理：ActionNetwork内部已经完成了"采样"
        output = self.model(input_ids, do_sample=do_sample, temperature=temperature, **kwargs)
        
        # 计算OvR概率：P_k = 1/2 + (1/π) × arctan((loc_S_k - C_k) / scale_S_k)
        ovr_probs = self.model.ovr_classifier(output.loc_S, output.scale_S)
        
        # CausalQwen最终决策：直接argmax，无需传统采样
        next_token = torch.argmax(ovr_probs[:, -1, :], dim=-1, keepdim=True)
        
        return next_token
    
    def generate(self, input_ids, max_new_tokens=20, do_sample=True, temperature=1.0,
                top_k=None, top_p=None, pad_token_id=None, eos_token_id=None, **kwargs):
        """序列生成 - CausalQwen专用推理
        
        CausalQwen不使用传统的top_k/top_p采样！
        "采样"是在ActionNetwork内部通过噪声注入实现的。
        
        Args:
            input_ids: 初始序列 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
            do_sample: 是否采样（控制ActionNetwork内部噪声注入方式）
            temperature: 温度参数（控制噪声强度）
            top_k: 兼容性参数，CausalQwen中无效
            top_p: 兼容性参数，CausalQwen中无效
            pad_token_id: padding token id
            eos_token_id: 结束token id
            
        Returns:
            generated_ids: 完整序列 [batch_size, seq_len + new_tokens]
        """
        # 警告：top_k/top_p在CausalQwen中无效
        if top_k is not None or top_p is not None:
            pass 
            # print("⚠️ Warning: top_k/top_p参数在CausalQwen中无效，采样由ActionNetwork内部噪声控制")
        
        batch_size = input_ids.shape[0]
        current_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                next_token = self.generate_next_token(
                    current_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                    **kwargs
                )
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # 检查是否遇到结束token
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
        
        return current_ids


class InferenceValidator:
    """推理验证器 - 验证数学原理"""
    
    def __init__(self, model: CausalQwenMVPForCausalLM):
        self.model = model
        self.engine = CausalInferenceEngine(model)
    
    def validate_causal_principles(self, input_ids, temperature=1.0):
        """验证数学原理：温度统一控制噪声强度"""
        results = {}
        
        with torch.no_grad():
            # 基础表征
            transformer_output = self.model.model(input_ids)
            hidden_states = transformer_output[0]
            
            # 使用 CausalEngine 获取个体表征
            engine_output = self.model.causal_engine(hidden_states, do_sample=False, temperature=1.0)
            loc_U = engine_output['loc_U']
            scale_U = engine_output['scale_U']
            
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