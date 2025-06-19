"""
CausalQwen MVP - 基于 CausalEngine 的第一个应用实例

这个模块展示了如何将独立的 CausalEngine 应用到具体的语言模型（Qwen）上。
CausalQwen 只是 CausalEngine 的一个客户，一个应用案例。

主从关系：
- CausalEngine 是主：定义因果推理的核心算法
- CausalQwen 是从：调用引擎，将其应用到 Qwen 模型上
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import Qwen2ForCausalLM

# 关键：从顶层导入独立的 CausalEngine
from causal_engine import CausalEngine

from .config import CausalQwen2Config, CausalMVPOutput
from .components import OvRClassifier


class CausalQwenMVPForCausalLM(Qwen2ForCausalLM):
    """
    CausalQwen - CausalEngine 在 Qwen 上的应用实例
    
    这个类展示了如何将革命性的 CausalEngine 集成到现有的语言模型中。
    它继承自 Qwen2ForCausalLM，但用 CausalEngine 替换了原始的生成机制。
    
    这只是一个开始 - CausalEngine 可以被应用到任何 Transformer 模型上。
    """
    
    config_class = CausalQwen2Config
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__(config)
        
        # 核心创新：实例化独立的 CausalEngine
        # 注意：我们从 config 中提取原子参数，而不是传递整个 config
        self.causal_engine = CausalEngine(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            causal_size=config.causal_size,
            b_noise_init=config.b_noise_init,
            gamma_init=config.gamma_init
        )
        
        # OvR 分类器（可选组件）
        self.ovr_classifier = OvRClassifier(config)
        
        # 打印引擎信息
        print(f"✨ CausalEngine 已加载: {self.causal_engine}")
    
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
        """
        前向传播 - 展示 CausalEngine 的应用
        
        工作流程：
        1. 使用 Qwen 的 Transformer 提取特征
        2. 将特征传递给 CausalEngine 进行因果推理
        3. 计算损失（如果有标签）
        """
        
        # Step 1: 使用 Qwen 的 Transformer 获取特征
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
        
        # Step 2: 调用独立的 CausalEngine 进行因果推理
        causal_outputs = self.causal_engine(
            hidden_states=hidden_states,
            do_sample=do_sample,
            temperature=temperature,
            return_dict=True
        )
        
        # Step 3: 计算损失（如果需要）
        loss = None
        if labels is not None:
            # 使用 OvR 分类器计算概率
            probs = self.ovr_classifier(
                causal_outputs['loc_S'], 
                causal_outputs['scale_S']
            )
            loss = self._compute_ovr_loss(probs, labels)
        
        # 返回结果
        return CausalMVPOutput(
            loss=loss,
            loc_S=causal_outputs['loc_S'],
            scale_S=causal_outputs['scale_S'],
            loc_U=causal_outputs['loc_U'],
            scale_U=causal_outputs['scale_U'],
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
        )
    
    def _compute_ovr_loss(self, probs, labels):
        """计算 OvR (One-vs-Rest) 损失"""
        # 将标签转换为 one-hot
        targets = F.one_hot(labels, num_classes=self.config.vocab_size).float()
        # 计算二元交叉熵
        loss = F.binary_cross_entropy(probs, targets, reduction='mean')
        return loss
    
    def generate(self, input_ids, max_new_tokens=20, do_sample=True, temperature=1.0,
                top_k=None, top_p=None, pad_token_id=None, eos_token_id=None, **kwargs):
        """生成方法 - 使用 CausalEngine 进行自回归生成"""
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
    
    def get_causal_engine_config(self):
        """获取内部 CausalEngine 的配置"""
        return self.causal_engine.get_config()
    
    @classmethod
    def from_pretrained_with_engine(cls, pretrained_model_name_or_path, **kwargs):
        """
        便捷方法：加载预训练 Qwen 并自动配置 CausalEngine
        
        这展示了如何将任何预训练模型"升级"为因果模型
        """
        # 加载配置
        config = CausalQwen2Config.from_pretrained(pretrained_model_name_or_path)
        
        # 更新配置参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 创建模型
        model = cls(config)
        
        # 加载预训练权重（仅 Transformer 部分）
        # 注意：CausalEngine 使用自己的初始化
        pretrained_model = Qwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path)
        model.model.load_state_dict(pretrained_model.model.state_dict())
        model.embed_tokens = pretrained_model.embed_tokens
        
        print(f"✅ 成功将 {pretrained_model_name_or_path} 升级为因果模型")
        print(f"   使用的引擎: {model.causal_engine}")
        
        return model 