"""
CausalQwen MVP: 核心模型实现
使用占位式逻辑快速搭建框架，后续逐步完善具体实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers import Qwen2ForCausalLM, Qwen2Config, PretrainedConfig
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


class CauchyMath:
    """Cauchy分布数学工具类，实现严格的线性稳定性"""
    
    @staticmethod
    def cauchy_linear_stable_loc(loc_input, weight, bias=None):
        """Cauchy分布位置参数的线性变换"""
        # 位置参数变换：直接矩阵乘法
        result = loc_input @ weight.T
        if bias is not None:
            result = result + bias
        return result
    
    @staticmethod  
    def cauchy_linear_stable_scale(scale_input, weight):
        """Cauchy分布尺度参数的线性变换"""
        # 尺度参数变换：直接矩阵乘法
        return scale_input @ torch.abs(weight).T


class AbductionNetwork(nn.Module):
    """归因网络：从隐藏状态推断个体表征分布"""
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        
        # 修正：添加bias项，符合设计文档要求
        self.loc_net = nn.Linear(config.hidden_size, config.causal_size, bias=True)
        self.scale_net = nn.Linear(config.hidden_size, config.causal_size, bias=True)
        
        self._init_identity_mapping()
    
    def _init_identity_mapping(self):
        """初始化为恒等映射，符合设计文档"""
        with torch.no_grad():
            if self.config.hidden_size == self.config.causal_size:
                # loc_net恒等映射初始化
                self.loc_net.weight.copy_(torch.eye(self.config.causal_size))
                self.loc_net.bias.zero_()
                
                # scale_net初始化：weight=0, bias=γ_init 产生宽分布
                self.scale_net.weight.zero_()
                self.scale_net.bias.fill_(self.config.gamma_init)
            else:
                # 如果维度不匹配，使用Xavier初始化
                nn.init.xavier_uniform_(self.loc_net.weight)
                nn.init.zeros_(self.loc_net.bias)
                nn.init.xavier_uniform_(self.scale_net.weight)
                self.scale_net.weight.data *= 0.1
                self.scale_net.bias.fill_(self.config.gamma_init)
    
    def forward(self, hidden_states):
        """前向传播，符合设计文档的数学要求"""
        # 位置参数：标准线性变换
        loc_U = self.loc_net(hidden_states)
        
        # 尺度参数：使用softplus确保正性，符合设计文档
        scale_U = F.softplus(self.scale_net(hidden_states))
        
        return loc_U, scale_U


class ActionNetwork(nn.Module):
    """行动网络：从个体表征到决策分布"""
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        
        # 修正：添加bias项，符合设计文档要求
        self.lm_head = nn.Linear(config.causal_size, config.vocab_size, bias=True)
        
        # 修正：b_noise维度应该是causal_size，用于外生噪声融合
        self.b_noise = nn.Parameter(torch.zeros(config.causal_size))
        
        self._init_from_original_lm_head()
    
    def _init_from_original_lm_head(self):
        """从原始lm_head复制权重，符合知识继承原则"""
        # 外生噪声应有合理的初始值，而非假设无噪声
        nn.init.constant_(self.b_noise, self.config.b_noise_init)
        
        # TODO: 当有预训练模型可用时，应从其复制权重
        # 目前使用标准初始化作为备选
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)
        
    def copy_weights_from_qwen(self, qwen_model):
        """从预训练Qwen2模型复制lm_head权重"""
        if hasattr(qwen_model, 'lm_head'):
            print("正在复制Qwen2预训练权重...")
            with torch.no_grad():
                # 确保vocab_size一致（包含预留词汇）
                if qwen_model.lm_head.weight.shape == self.lm_head.weight.shape:
                    self.lm_head.weight.copy_(qwen_model.lm_head.weight)
                    if hasattr(qwen_model.lm_head, 'bias') and qwen_model.lm_head.bias is not None:
                        self.lm_head.bias.copy_(qwen_model.lm_head.bias)
                    print(f"✅ 成功复制权重，词汇表大小: {qwen_model.lm_head.weight.shape[0]}")
                else:
                    print(f"❌ 权重形状不匹配: Qwen({qwen_model.lm_head.weight.shape}) vs CausalQwen({self.lm_head.weight.shape})")
                    print("使用标准初始化...")
        else:
            print("❌ 源模型没有lm_head，使用标准初始化...")
        
    def forward(self, loc_U, scale_U=None):
        """前向传播，严格实现柯西分布线性稳定性"""
        # Step 1: 外生噪声融合（添加到尺度参数）
        if scale_U is None:
            scale_U = torch.zeros_like(loc_U)  # Cauchy 分布尺度参数为0 时，scale_U=0
        scale_U_noisy = scale_U + torch.abs(self.b_noise)
        
        # Step 2: 位置参数的线性变换
        loc_S = self.lm_head(loc_U)
        
        # Step 3: 尺度参数的线性稳定性变换
        scale_S = scale_U_noisy @ torch.abs(self.lm_head.weight).T
        
        return loc_S, scale_S


class OvRClassifier(nn.Module):
    """One-vs-Rest分类器"""
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        self.thresholds = nn.Parameter(torch.full((config.vocab_size,), config.ovr_threshold_init))
    
    def forward(self, loc_S, scale_S):
        """计算OvR概率 - 占位实现"""
        # TODO: 实现严格的Cauchy分布CDF计算
        # 占位：使用简化的概率计算
        # P(y_k=1) = P(Cauchy(loc_S_k, scale_S_k) > threshold_k)
        normalized_diff = (loc_S - self.thresholds.unsqueeze(0).unsqueeze(0)) / scale_S
        # 使用atan近似Cauchy CDF: P = 0.5 + (1/π) * atan(x)
        probs = 0.5 + (1/torch.pi) * torch.atan(normalized_diff)
        return probs


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
    
    def inference(self, input_ids, mode='standard', **kwargs):
        """推理接口 - 调用推理引擎"""
        from .inference import CausalInferenceEngine
        engine = CausalInferenceEngine(self)
        return engine.inference(input_ids, mode=mode, **kwargs)
    
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
        **kwargs
    ) -> Union[Tuple, CausalMVPOutput]:
        """前向传播 - 框架实现"""
        
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
        loc_S, scale_S = self.action_network(loc_U, scale_U)    # 决策推断
        
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