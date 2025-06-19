"""
CausalQwen MVP: 配置和输出数据结构
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import Qwen2Config
from transformers.modeling_outputs import ModelOutput


@dataclass
class CausalMVPOutput(ModelOutput):
    """CausalQwen MVP输出结构"""
    loss: Optional[torch.FloatTensor] = None
    loc_S: torch.FloatTensor = None
    scale_S: torch.FloatTensor = None
    loc_U: torch.FloatTensor = None
    scale_U: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    next_token_ids: Optional[torch.LongTensor] = None  # 用于兼容模式的采样结果


class CausalQwen2Config(Qwen2Config):
    """扩展Qwen2Config以支持因果模型参数"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 因果模型特有参数
        self.causal_size = kwargs.get('causal_size', self.hidden_size)
        self.abduction_init_strategy = kwargs.get('abduction_init_strategy', 'identity')
        self.b_noise_init = kwargs.get('b_noise_init', 0.1)
        self.ovr_threshold_init = kwargs.get('ovr_threshold_init', 0.0)
        self.gamma_init = kwargs.get('gamma_init', 10.0)  # AbductionNetwork尺度初始化
        self.inference_mode = kwargs.get('inference_mode', 'standard') 