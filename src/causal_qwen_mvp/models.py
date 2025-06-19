"""
CausalQwen MVP: 主模型实现
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import Qwen2ForCausalLM

from .config import CausalQwen2Config, CausalMVPOutput
from .components import AbductionNetwork, ActionNetwork, OvRClassifier


class CausalQwenMVPForCausalLM(Qwen2ForCausalLM):
    """CausalQwen MVP主模型类"""
    
    config_class = CausalQwen2Config
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__(config)
        
        # 添加因果模块
        self.abduction_network = AbductionNetwork(config)
        self.action_network = ActionNetwork(config)  
        self.ovr_classifier = OvRClassifier(config)
        
        # 初始化因果权重
        self._init_causal_weights()
    
    def _init_causal_weights(self):
        """初始化因果模块权重"""
        # 因果模块已在各自的__init__中完成初始化
        pass
    
    def copy_pretrained_weights(self, qwen_model_path_or_model):
        """从预训练Qwen2模型复制权重"""
        if isinstance(qwen_model_path_or_model, str):
            from transformers import Qwen2ForCausalLM
            print(f"正在加载预训练模型: {qwen_model_path_or_model}")
            qwen_model = Qwen2ForCausalLM.from_pretrained(qwen_model_path_or_model)
        else:
            qwen_model = qwen_model_path_or_model
            
        # 复制ActionNetwork的lm_head权重
        self.action_network.copy_weights_from_qwen(qwen_model)
        
        # 验证vocab_size一致性（包含预留词汇）
        if hasattr(qwen_model, 'config') and hasattr(qwen_model.config, 'vocab_size'):
            expected_vocab_size = qwen_model.config.vocab_size
            actual_vocab_size = self.config.vocab_size
            if expected_vocab_size != actual_vocab_size:
                print(f"⚠️  词汇表大小不匹配: 期望 {expected_vocab_size}, 实际 {actual_vocab_size}")
                print("请确保配置中的vocab_size包含了所有预留词汇")
            else:
                print(f"✅ 词汇表大小一致: {actual_vocab_size} (包含预留词汇)")
        
        print("权重复制完成！")
    
    def generate(self, input_ids, max_new_tokens=20, do_sample=True, temperature=1.0,
                top_k=None, top_p=None, pad_token_id=None, eos_token_id=None, **kwargs):
        """序列生成 - CausalQwen专用推理（不使用传统采样）"""
        from .inference import CausalInferenceEngine
        engine = CausalInferenceEngine(self)
        return engine.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        **kwargs
    ) -> Union[Tuple, CausalMVPOutput]:
        """前向传播 - 双模式框架实现
        
        核心特性：
        - do_sample=False: 非采样模式，噪声影响尺度参数
        - do_sample=True: 采样模式，噪声影响位置参数
        """
        
        # 1. 获取Transformer特征
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )
        hidden_states = transformer_outputs[0]
        
        # 2. 因果推理链路
        loc_U, scale_U = self.abduction_network(hidden_states)  # 个体推断
        loc_S, scale_S = self.action_network(
            loc_U, scale_U, 
            do_sample=do_sample, 
            temperature=temperature
        )  # 决策推断
        
        # 3. 损失计算
        loss = None
        if labels is not None:
            probs = self.ovr_classifier(loc_S, scale_S)
            loss = self._compute_ovr_loss(probs, labels)
        
        return CausalMVPOutput(
            loss=loss,
            loc_S=loc_S,
            scale_S=scale_S,
            loc_U=loc_U,
            scale_U=scale_U,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
        )
    
    def _compute_ovr_loss(self, probs, labels):
        """计算OvR损失 - 占位实现"""
        # TODO: 实现更sophisticated的损失函数
        # 简化实现：二元交叉熵损失
        targets = F.one_hot(labels, num_classes=self.config.vocab_size).float()
        loss = F.binary_cross_entropy(probs, targets, reduction='mean')
        return loss
    
    def debug_forward_pass(self, input_ids):
        """调试前向传播的每个步骤"""
        with torch.no_grad():
            # 1. Transformer输出
            transformer_outputs = self.model(input_ids)
            print(f"Hidden states shape: {transformer_outputs[0].shape}")
            
            # 2. 归因网络输出
            loc_U, scale_U = self.abduction_network(transformer_outputs[0])
            print(f"U distribution - loc: {loc_U.mean():.4f}, scale: {scale_U.mean():.4f}")
            
            # 3. 行动网络输出
            loc_S, scale_S = self.action_network(loc_U, scale_U)
            print(f"S distribution - loc: {loc_S.mean():.4f}, scale: {scale_S.mean():.4f}")
            
            return loc_S, scale_S 