"""
Feature Network module.

This module implements the feature extraction network for the causal language model.
In the initial implementation, we use a mock feature network that generates random features.
Later, this can be replaced with a real Qwen-0.5B backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FeatureNetworkBase(nn.Module):
    """
    Base class for feature networks.
    """
    
    def __init__(self):
        """Initialize the base feature network."""
        super().__init__()
    
    def forward(self, input_ids: torch.Tensor, 
                numerical_values: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features from input tokens.
        
        Args:
            input_ids: Input token IDs
            numerical_values: Numerical values (for compatibility)
            attention_mask: Attention mask
            
        Returns:
            features: Extracted features
        """
        raise NotImplementedError("Subclasses must implement forward method")


class QwenFeatureNetwork(FeatureNetworkBase):
    """
    Qwen特征网络，基于预训练的Qwen模型提取文本特征。
    """
    
    def __init__(self, model_path: str, hidden_size: int = 896):
        super().__init__()  # 调用基类的无参数初始化
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.use_real_model = True  # 添加缺失的属性
        
        try:
            from transformers import Qwen2ForCausalLM
            self.qwen_model = Qwen2ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=None  # 让调用者控制设备
            )
            print(f"Successfully loaded Qwen model from {model_path}")
            print(f"Successfully loaded Qwen model with hidden size {self.qwen_model.config.hidden_size}")
            
            # 验证hidden_size
            actual_hidden_size = self.qwen_model.config.hidden_size
            if actual_hidden_size != hidden_size:
                print(f"⚠️  Warning: Expected hidden_size {hidden_size}, got {actual_hidden_size}")
                self.hidden_size = actual_hidden_size
                
        except Exception as e:
            print(f"❌ Failed to load Qwen model: {e}")
            raise e
    
    def get_lm_head(self):
        """
        获取Qwen模型的语言模型头。
        
        Returns:
            nn.Linear: lm_head层
        """
        if not self.use_real_model:
            raise ValueError("Cannot get lm_head from mock model")
        
        if hasattr(self.qwen_model, 'lm_head'):
            return self.qwen_model.lm_head
        else:
            raise AttributeError("Qwen model does not have 'lm_head' attribute")
    
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，提取序列特征。
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            inputs_embeds: 增强嵌入 [batch_size, seq_len, hidden_size]
            
        Returns:
            features: 序列特征 [batch_size, seq_len, hidden_size]
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        # 使用Qwen模型提取特征
        # 如果需要梯度（训练时），则不使用no_grad
        if self.qwen_model.training:
            outputs = self.qwen_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        else:
            with torch.no_grad():
                outputs = self.qwen_model(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
        
        # 获取最后一层隐藏状态
        features = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        return features


class MockFeatureNetwork(FeatureNetworkBase):
    """
    Mock特征网络，用于测试和调试。
    生成基于输入的特征向量，支持数值感知，避免加载大型模型的开销。
    """
    
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()  # 调用基类的无参数初始化
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_real_model = False  # 添加属性以保持一致性
        
        # 创建token嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 数值方向向量（让数值变化产生不同特征）
        self.numerical_direction = nn.Parameter(
            torch.randn(hidden_size) / math.sqrt(hidden_size)
        )
        
        # 初始化嵌入权重
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids: torch.Tensor, 
                numerical_values: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成支持数值感知的mock特征向量。
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            numerical_values: 数值信息 [batch_size, seq_len]
            attention_mask: 注意力掩码（暂时忽略）
            
        Returns:
            features: mock特征 [batch_size, seq_len, hidden_size]
        """
        # 基础token特征
        base_features = self.embedding(input_ids)
        
        # 如果提供了数值信息，则融合数值特征
        if numerical_values is not None:
            # 数值变换：φ(v) = sign(v) * ln(1 + |v|)
            # 对于v=0，结果为0
            transformed_values = torch.sign(numerical_values) * torch.log1p(torch.abs(numerical_values))
            
            # 扩展到特征维度 [batch_size, seq_len, 1]
            transformed_values = transformed_values.unsqueeze(-1)
            
            # 计算数值嵌入 [batch_size, seq_len, hidden_size]
            numerical_embeddings = transformed_values * self.numerical_direction
            
            # 融合特征：基础特征 + 数值特征
            features = base_features + numerical_embeddings
        else:
            features = base_features
            
        return features
    
    def get_lm_head(self):
        """
        Mock模型没有真实的lm_head。
        
        Returns:
            None: Mock模型不支持lm_head
        """
        return None

