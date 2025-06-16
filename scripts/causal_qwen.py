import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import re
import random
from typing import Dict, Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader

class NumericalAwareTokenizer:
    """数值感知的分词器 - 处理混合文本/数值输入"""
    
    def __init__(self, base_tokenizer, num_token_id: int):
        self.base_tokenizer = base_tokenizer
        self.num_token_id = num_token_id
        self.number_pattern = re.compile(r'(-?\d+\.?\d*)')
        # 添加 pad_token_id 属性
        self.pad_token_id = self.base_tokenizer.pad_token_id
        
    def tokenize_with_numbers(self, text: str) -> Dict[str, torch.Tensor]:
        """
        将文本转换为 token IDs 和对齐的数值序列
        
        Args:
            text: 输入文本，如 "价格是99.9元"
            
        Returns:
            dict: 包含 'input_ids' 和 'numeric_values' 的字典
        """
        # 提取数值并替换为占位符
        numeric_values = []
        
        def replace_num(match):
            numeric_values.append(float(match.group(1)))
            return ' <NUM> '
        
        processed_text = self.number_pattern.sub(replace_num, text)
        
        # 使用基础分词器
        tokens = self.base_tokenizer(processed_text, return_tensors='pt')
        input_ids = tokens['input_ids'][0]  # 去掉 batch 维度
        
        # 对齐数值到正确位置
        aligned_values = []
        num_idx = 0
        
        for i, token_id in enumerate(input_ids):
            # 检查当前 token 是否对应 <NUM>
            # 注意：这里简化处理，实际可能需要更复杂的逻辑
            token_text = self.base_tokenizer.decode([token_id], skip_special_tokens=False).strip()
            if token_text == '<NUM>' or token_id == self.num_token_id:
                if num_idx < len(numeric_values):
                    aligned_values.append(numeric_values[num_idx])
                    num_idx += 1
                    # 替换为正确的 NUM_TOKEN_ID
                    input_ids[i] = self.num_token_id
                else:
                    aligned_values.append(0.0)
            else:
                aligned_values.append(0.0)
        
        return {
            'input_ids': input_ids,
            'numeric_values': torch.tensor(aligned_values, dtype=torch.float)
        }

    def batch_tokenize(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """批量处理文本"""
        batch_input_ids = []
        batch_numeric_values = []
        
        for text in texts:
            result = self.tokenize_with_numbers(text)
            batch_input_ids.append(result['input_ids'])
            batch_numeric_values.append(result['numeric_values'])
        
        # Padding
        max_len = min(max_length, max(len(ids) for ids in batch_input_ids))
        
        padded_input_ids = []
        padded_numeric_values = []
        attention_mask = []
        
        for input_ids, numeric_values in zip(batch_input_ids, batch_numeric_values):
            # 截断
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                numeric_values = numeric_values[:max_len]
            
            # 填充
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                padded_input_ids.append(
                    torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id)])
                )
                padded_numeric_values.append(
                    torch.cat([numeric_values, torch.zeros(pad_len)])
                )
                attention_mask.append(
                    torch.cat([torch.ones(len(input_ids)), torch.zeros(pad_len)])
                )
            else:
                padded_input_ids.append(input_ids)
                padded_numeric_values.append(numeric_values)
                attention_mask.append(torch.ones(len(input_ids)))
        
        return {
            'input_ids': torch.stack(padded_input_ids).long(),
            'numeric_values': torch.stack(padded_numeric_values).float(),
            'attention_mask': torch.stack(attention_mask).long()
        }

class NumericalEmbedding(nn.Module):
    """数值编码模块 - 实现 φ(v) = sign(v) * ln(1 + |v|) * w_num"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # 数值编码的方向向量 w_num
        self.direction_vector = nn.Parameter(torch.randn(hidden_size))
        self._init_weights()
    
    def _init_weights(self):
        # 初始化为单位向量
        with torch.no_grad():
            self.direction_vector.data = F.normalize(self.direction_vector.data, dim=0)
    
    def forward(self, numeric_values: torch.Tensor) -> torch.Tensor:
        """
        计算数值编码 φ(v) = sign(v) * ln(1 + |v|) * w_num
        
        Args:
            numeric_values: [B, S] - 数值序列
        
        Returns:
            encoding: [B, S, H] - 数值编码
        """
        # 扩展维度以便广播
        values = numeric_values.unsqueeze(-1)  # [B, S, 1]
        
        # 计算 sign(v) * ln(1 + |v|)
        magnitude = torch.sign(values) * torch.log1p(torch.abs(values))  # [B, S, 1]
        
        # 与方向向量相乘
        encoding = magnitude * self.direction_vector.unsqueeze(0).unsqueeze(0)  # [B, S, H]
        
        return encoding

class NumericalAwareEmbedding(nn.Module):
    """数值感知嵌入模块 - 融合词元嵌入和数值编码"""
    
    def __init__(self, token_embedding: nn.Embedding, hidden_size: int):
        super().__init__()
        self.token_embedding = token_embedding
        self.numerical_embedding = NumericalEmbedding(hidden_size)
    
    def forward(self, input_ids: torch.Tensor, numeric_values: torch.Tensor) -> torch.Tensor:
        """
        计算增强嵌入 e_i = base_embed_i + φ(v_i)
        
        Args:
            input_ids: [B, S] - 词元ID序列
            numeric_values: [B, S] - 对应的数值
        
        Returns:
            enhanced_embeddings: [B, S, H] - 增强后的嵌入
        """
        # 获取基础词元嵌入
        base_embeddings = self.token_embedding(input_ids)  # [B, S, H]
        
        # 计算数值编码
        numeric_encoding = self.numerical_embedding(numeric_values)  # [B, S, H]
        
        # 融合
        enhanced_embeddings = base_embeddings + numeric_encoding  # [B, S, H]
        
        return enhanced_embeddings

class QwenTransformer(nn.Module):
    """Qwen Transformer 包装器 - 特征提取网络"""
    
    def __init__(self, qwen_model):
        super().__init__()
        self.transformer = qwen_model.model  # Qwen 的核心 transformer
        
    def forward(self, 
                embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        使用 Qwen Transformer 提取特征
        
        Args:
            embeddings: [B, S, H] - 输入嵌入
            attention_mask: [B, S] - 注意力掩码
        
        Returns:
            features: [B, S, H] - 上下文特征
        """
        # 准备位置编码
        batch_size, seq_length = embeddings.shape[:2]
        position_ids = torch.arange(seq_length, device=embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 调用 Qwen transformer
        outputs = self.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )
        
        return outputs.last_hidden_state  # [B, S, H]

class AbductionNetwork(nn.Module):
    """归因推断网络 - 推断个体因果表征分布"""
    
    def __init__(self, hidden_size: int, causal_hidden_size: int, gamma_init: float = 10.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.causal_hidden_size = causal_hidden_size
        self.gamma_init = gamma_init
        
        self.loc_net = nn.Linear(hidden_size, causal_hidden_size)
        self.scale_net = nn.Linear(hidden_size, causal_hidden_size)
        self._init_weights()
    
    def _init_weights(self):
        # 位置网络初始化为恒等映射
        nn.init.eye_(self.loc_net.weight)
        nn.init.zeros_(self.loc_net.bias)
        
        # 尺度网络初始化 - b_scale = log(gamma_init)
        nn.init.zeros_(self.scale_net.weight)
        nn.init.constant_(self.scale_net.bias, math.log(self.gamma_init))
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从特征推断个体因果表征分布
        
        Args:
            features: [B, S, H] - 上下文特征
        
        Returns:
            loc_U: [B, S, C] - 位置参数
            scale_U: [B, S, C] - 尺度参数（正值）
        """
        # 计算位置参数
        loc_U = self.loc_net(features)  # [B, S, C]
        
        # 计算尺度参数（使用 softplus 保证正值）
        scale_U = F.softplus(self.scale_net(features))  # [B, S, C]
        
        return loc_U, scale_U

class ActionNetwork(nn.Module):
    """行动网络 - 基于个体表征进行决策"""
    
    def __init__(self, causal_hidden_size: int, vocab_size: int):
        super().__init__()
        # 分类头
        self.classification_head = nn.Linear(causal_hidden_size, vocab_size)
        # 回归头
        self.regression_head = nn.Linear(causal_hidden_size, 1)
        # 噪声参数
        self.b_noise = nn.Parameter(torch.zeros(causal_hidden_size))
        
    def forward(self, loc_U: torch.Tensor, scale_U: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        基于个体表征分布进行决策
        
        两步流程：
        1. 噪声注入：U' ~ Cauchy(loc_U, scale_U + |b_noise|)
        2. 并行决策：分类和回归
        
        Args:
            loc_U: [B, S, C] - 个体表征位置参数
            scale_U: [B, S, C] - 个体表征尺度参数
        
        Returns:
            loc_S: [B, S, V] - 分类决策位置参数
            scale_S: [B, S, V] - 分类决策尺度参数
            loc_Y: [B, S] - 回归决策位置参数
            scale_Y: [B, S] - 回归决策尺度参数
        """
        # Step 1: 噪声融合
        noise_scale = torch.abs(self.b_noise).unsqueeze(0).unsqueeze(0)  # [1, 1, C]
        scale_U_fused = scale_U + noise_scale  # [B, S, C]
        
        # Step 2: 分类决策
        loc_S = self.classification_head(loc_U)  # [B, S, V]
        
        # 计算 scale_S: |W_cls| · scale_U_fused
        W_cls_abs = torch.abs(self.classification_head.weight)  # [V, C]
        scale_S = torch.matmul(scale_U_fused, W_cls_abs.T)  # [B, S, V]
        
        # Step 3: 回归决策
        loc_Y = self.regression_head(loc_U).squeeze(-1)  # [B, S]
        
        # 计算 scale_Y: |W_reg| · scale_U_fused
        W_reg_abs = torch.abs(self.regression_head.weight)  # [1, C]
        scale_Y = torch.matmul(scale_U_fused, W_reg_abs.T).squeeze(-1)  # [B, S]
        
        return loc_S, scale_S, loc_Y, scale_Y

class OvrClassificationLoss(nn.Module):
    """OvR (One-vs-Rest) 分类损失"""
    
    def __init__(self, vocab_size: int, C_ovr: float = 100.0, learnable_threshold: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        
        if learnable_threshold:
            # 可学习的阈值参数
            self.C_ovr = nn.Parameter(torch.full((vocab_size,), C_ovr))
        else:
            # 固定阈值
            self.register_buffer('C_ovr', torch.tensor(C_ovr))
    
    def forward(self, 
                loc_S: torch.Tensor, 
                scale_S: torch.Tensor, 
                labels: torch.Tensor,
                ignore_index: int = -100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 OvR 分类损失
        
        Args:
            loc_S: [B, S, V] - 分类位置参数
            scale_S: [B, S, V] - 分类尺度参数
            labels: [B, S] - 真实标签
            ignore_index: 忽略的标签值
        
        Returns:
            loss: [B, S] - 每个位置的损失
            valid_mask: [B, S] - 有效位置掩码
        """
        B, S, V = loc_S.shape
        
        # 创建有效位置掩码
        valid_mask = (labels != ignore_index).float()  # [B, S]
        
        # 计算 OvR 概率: P_k = 1/2 + (1/π)arctan((loc_S_k - C_k)/scale_S_k)
        if len(self.C_ovr.shape) == 0:  # 标量
            C_ovr = self.C_ovr.unsqueeze(0).unsqueeze(0)  # [1, 1, 1]
        else:  # 向量
            C_ovr = self.C_ovr.unsqueeze(0).unsqueeze(0)  # [1, 1, V]
            
        z = (loc_S - C_ovr) / scale_S  # [B, S, V]
        P = 0.5 + torch.atan(z) / math.pi  # [B, S, V]
        
        # 创建 one-hot 标签
        y_onehot = torch.zeros_like(loc_S)  # [B, S, V]
        valid_labels = labels.clone()
        valid_labels[labels == ignore_index] = 0  # 避免索引错误
        y_onehot.scatter_(2, valid_labels.unsqueeze(-1), 1)
        y_onehot = y_onehot * valid_mask.unsqueeze(-1)  # 应用掩码
        
        # 计算二元交叉熵
        eps = 1e-7
        bce = -(y_onehot * torch.log(P + eps) + 
                (1 - y_onehot) * torch.log(1 - P + eps))  # [B, S, V]
        
        # 对词汇表维度求和
        loss = bce.sum(dim=-1)  # [B, S]
        
        # 应用有效位置掩码
        loss = loss * valid_mask  # [B, S]
        
        return loss, valid_mask

class GatedRegressionLoss(nn.Module):
    """门控回归损失 - 柯西分布的负对数似然"""
    
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self,
                loc_Y: torch.Tensor,
                scale_Y: torch.Tensor,
                numeric_values: torch.Tensor,
                P_num: torch.Tensor,
                num_mask: torch.Tensor) -> torch.Tensor:
        """
        计算门控回归损失
        
        Args:
            loc_Y: [B, S] - 回归位置参数
            scale_Y: [B, S] - 回归尺度参数
            numeric_values: [B, S] - 真实数值
            P_num: [B, S] - <NUM> 词元的 OvR 概率
            num_mask: [B, S] - 数值位置掩码
        
        Returns:
            loss_gated: [B, S] - 门控回归损失
        """
        # 计算柯西分布的负对数似然
        # L_nll = log(π·scale) + log(1 + ((y_true - loc) / scale)²)
        residual = (numeric_values - loc_Y) / scale_Y  # [B, S]
        nll = torch.log(math.pi * scale_Y) + torch.log1p(residual ** 2)  # [B, S]
        
        # 计算门控权重
        # gate = num_mask * (alpha + (1 - alpha) * P_num)
        gate = num_mask * (self.alpha + (1 - self.alpha) * P_num)  # [B, S]
        
        # 应用门控
        loss_gated = gate * nll  # [B, S]
        
        return loss_gated

class CausalQwenLoss(nn.Module):
    """CausalQwen 的总损失计算模块"""
    
    def __init__(self, 
                 vocab_size: int,
                 num_token_id: int,
                 C_ovr: float = 100.0,
                 alpha: float = 0.0,
                 reg_weight: float = 1.0,
                 learnable_threshold: bool = False):
        super().__init__()
        self.num_token_id = num_token_id
        self.reg_weight = reg_weight
        
        # 子损失模块
        self.ovr_loss = OvrClassificationLoss(vocab_size, C_ovr, learnable_threshold)
        self.reg_loss = GatedRegressionLoss(alpha)
    
    def forward(self,
                loc_S: torch.Tensor,
                scale_S: torch.Tensor,
                loc_Y: torch.Tensor,
                scale_Y: torch.Tensor,
                labels: torch.Tensor,
                numeric_values: torch.Tensor,
                attention_mask: torch.Tensor,
                ignore_index: int = -100) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Returns:
            total_loss: 标量总损失
            loss_dict: 包含各部分损失的字典
        """
        # 1. 计算 OvR 分类损失
        cls_loss, valid_mask = self.ovr_loss(loc_S, scale_S, labels, ignore_index)
        
        # 2. 创建掩码
        cls_mask = attention_mask  # [B, S]
        num_mask = ((labels == self.num_token_id) & (attention_mask > 0)).float()  # [B, S]
        
        # 3. 计算 <NUM> 词元的预测概率
        loc_S_num = loc_S[:, :, self.num_token_id]  # [B, S]
        scale_S_num = scale_S[:, :, self.num_token_id]  # [B, S]
        
        # 使用与 OvR 损失相同的阈值
        if len(self.ovr_loss.C_ovr.shape) == 0:  # 标量
            C_num = self.ovr_loss.C_ovr
        else:  # 向量
            C_num = self.ovr_loss.C_ovr[self.num_token_id]
            
        z_num = (loc_S_num - C_num) / scale_S_num
        P_num = 0.5 + torch.atan(z_num) / math.pi  # [B, S]
        
        # 4. 计算门控回归损失
        reg_loss_gated = self.reg_loss(loc_Y, scale_Y, numeric_values, P_num, num_mask)
        
        # 5. 归约损失
        # 分类损失：在所有有效位置上平均
        n_cls = cls_mask.sum()
        cls_loss_mean = (cls_loss * cls_mask).sum() / n_cls.clamp(min=1)
        
        # 回归损失：仅在数值位置上平均
        n_reg = num_mask.sum()
        reg_loss_effective = reg_loss_gated.sum() / n_reg.clamp(min=1)
        
        # 6. 总损失
        total_loss = cls_loss_mean + self.reg_weight * reg_loss_effective
        
        return total_loss, {
            'cls_loss_mean': cls_loss_mean.item(),
            'reg_loss_effective': reg_loss_effective.item(),
            'total_loss': total_loss.item(),
            'n_cls': n_cls.item(),
            'n_reg': n_reg.item()
        }

class CausalQwen(nn.Module):
    """CausalQwen 主模型 - 整合所有子模块"""
    
    def __init__(self, qwen_model_path: str, 
                 ovr_threshold: float = 100.0, 
                 gamma_init: float = 10.0, 
                 alpha_gated_reg: float = 0.0,
                 reg_weight: float = 1.0,
                 learnable_threshold: bool = False):
        super().__init__()
        
        # 加载 Qwen 模型和配置
        self.qwen_config = AutoConfig.from_pretrained(qwen_model_path)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        
        self.vocab_size = self.qwen_config.vocab_size
        self.hidden_size = self.qwen_config.hidden_size
        self.causal_hidden_size = self.hidden_size  # C = H
        
        # 假设 <NUM> 是第一个预留词元
        self.num_token_id = self.vocab_size - 271  # Qwen 有 271 个预留位置
        
        # 1. 数值感知分词器
        self.numerical_tokenizer = NumericalAwareTokenizer(self.tokenizer, self.num_token_id)
        
        # 2. 数值感知嵌入
        self.numerical_aware_embedding = NumericalAwareEmbedding(
            self.qwen_model.model.embed_tokens,
            self.hidden_size
        )
        
        # 3. 特征提取网络
        self.qwen_transformer = QwenTransformer(self.qwen_model)
        
        # 4. 归因推断网络
        self.abduction_network = AbductionNetwork(
            self.hidden_size, 
            self.causal_hidden_size,
            gamma_init
        )
        
        # 5. 行动决策网络
        self.action_network = ActionNetwork(self.causal_hidden_size, self.vocab_size)
        
        # 6. 损失计算模块
        self.loss_module = CausalQwenLoss(
            self.vocab_size,
            self.num_token_id,
            ovr_threshold,
            alpha_gated_reg,
            reg_weight,
            learnable_threshold
        )
        
        # 初始化行动网络的分类头
        self._init_action_network_from_qwen()
    
    def _init_action_network_from_qwen(self):
        """从 Qwen 复制 lm_head 权重到行动网络的分类头"""
        with torch.no_grad():
            self.action_network.classification_head.weight.data.copy_(
                self.qwen_model.lm_head.weight.data
            )
            if self.qwen_model.lm_head.bias is not None:
                self.action_network.classification_head.bias.data.copy_(
                    self.qwen_model.lm_head.bias.data
                )
    
    def forward(self, 
                input_ids: torch.Tensor, 
                numeric_values: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            input_ids: [B, S] - 词元ID序列
            numeric_values: [B, S] - 对应的数值
            attention_mask: [B, S] - 注意力掩码
        
        Returns:
            六个张量：loc_U, scale_U, loc_S, scale_S, loc_Y, scale_Y
        """
        # 1. 数值感知嵌入
        enhanced_embeddings = self.numerical_aware_embedding(input_ids, numeric_values)
        
        # 2. 特征提取
        context_features = self.qwen_transformer(enhanced_embeddings, attention_mask)
        
        # 3. 归因推断
        loc_U, scale_U = self.abduction_network(context_features)
        
        # 4. 行动决策
        loc_S, scale_S, loc_Y, scale_Y = self.action_network(loc_U, scale_U)
        
        return loc_U, scale_U, loc_S, scale_S, loc_Y, scale_Y
    
    def calculate_loss(self, 
                      loc_S: torch.Tensor,
                      scale_S: torch.Tensor,
                      loc_Y: torch.Tensor,
                      scale_Y: torch.Tensor,
                      labels: torch.Tensor,
                      numeric_values: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      ignore_index: int = -100) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失"""
        if attention_mask is None:
            attention_mask = torch.ones_like(labels)
            
        return self.loss_module(
            loc_S, scale_S, loc_Y, scale_Y,
            labels, numeric_values, attention_mask, ignore_index
        )
    
    def deterministic_inference(self, 
                               loc_S: torch.Tensor,
                               scale_S: torch.Tensor,
                               loc_Y: torch.Tensor,
                               scale_Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        标准推理（确定性）
        
        Returns:
            predicted_token_ids: [B, S] - 预测的词元ID
            predicted_numeric_values: [B, S] - 预测的数值
        """
        # 计算所有词汇的 OvR 概率
        if hasattr(self.loss_module.ovr_loss, 'C_ovr'):
            C_ovr = self.loss_module.ovr_loss.C_ovr
            if len(C_ovr.shape) == 0:  # 标量
                C_ovr = C_ovr.unsqueeze(0).unsqueeze(0)
            else:  # 向量
                C_ovr = C_ovr.unsqueeze(0).unsqueeze(0)
        else:
            C_ovr = 100.0  # 默认值
            
        z = (loc_S - C_ovr) / scale_S
        P = 0.5 + torch.atan(z) / math.pi
        
        # 选择概率最高的词汇
        predicted_token_ids = torch.argmax(P, dim=-1)  # [B, S]
        
        # 回归预测使用位置参数
        predicted_numeric_values = loc_Y  # [B, S]
        
        return predicted_token_ids, predicted_numeric_values
    
    def causal_sampling(self, loc_U, scale_U, num_samples=1, epsilon_seed=None):
        """
        因果采样
        
        Args:
            loc_U, scale_U: 个体表征分布参数
            num_samples: 采样次数
            epsilon_seed: 随机种子（可选）
        
        Returns:
            sampled_token_ids: [B, S, num_samples] - 采样的词元ID
            sampled_numeric_values: [B, S, num_samples] - 采样的数值
            sampled_U: [B, S, C, num_samples] - 采样的个体表征
        """
        # MOCKER: 简单的重复和随机
        B, S, C = loc_U.shape
        device = loc_U.device
        
        # 模拟采样
        sampled_token_ids = torch.randint(0, self.vocab_size, (B, S, num_samples), device=device)
        sampled_numeric_values = torch.randn(B, S, num_samples, device=device)
        sampled_U = loc_U.unsqueeze(-1).expand(-1, -1, -1, num_samples)
        
        return sampled_token_ids, sampled_numeric_values, sampled_U

    def compatible_traditional_sampling(self, loc_S):
        """
        兼容传统采样（返回 softmax 概率）
        
        Args:
            loc_S: [B, S, V_full] - 分类位置参数
            
        Returns:
            softmax_probs: [B, S, V_full] - softmax 概率分布
        """
        softmax_probs = F.softmax(loc_S, dim=-1)
        return softmax_probs


class SyntheticNumericalDataset(Dataset):
    """合成数值数据集 - 生成包含数值运算的多样化句子"""
    
    def __init__(self, 
                 tokenizer_wrapper,
                 samples: List[str] = None,
                 max_length: int = 128):
        """
        Args:
            tokenizer_wrapper: CausalQwen 的 numerical_tokenizer
            samples: 样本列表，如果为 None 则使用默认样本
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length
        
        # 使用提供的样本或默认样本
        self.samples = samples if samples is not None else self._get_default_samples()
    
    def _get_default_samples(self) -> List[str]:
        """获取50个手动设计的高质量样本"""
        return [
            # 基础算术运算 (10个)
            "小明有5个苹果，小红给了他3个，现在他有8个苹果。",
            "教室里有24个学生，又来了6个，总共30个学生。",
            "商店里有100瓶水，卖出了45瓶，还剩55瓶。",
            "一本书有256页，我已经读了128页，还剩128页。",
            "公交车上有15人，下车7人，上车10人，现在有18人。",
            "妈妈买了3千克苹果，每千克12元，总共花了36元。",
            "一个班有40名学生，男生23人，女生17人。",
            "小华的零花钱有50元，买文具用了18元，还剩32元。",
            "一年有365天，已经过了200天，还剩165天。",
            "图书馆有1000本书，借出了234本，还有766本在馆。",
            
            # 百分比和折扣计算 (10个)
            "原价200元的衣服，打8折后是160元。",
            "考试满分100分，小明考了85分，得分率是85%。",
            "商品原价150元，优惠30元，折扣率是20%。",
            "全班50人，40人及格，及格率是80%。",
            "投篮20次，命中12次，命中率是60%。",
            "月收入8000元，房租2400元，占收入的30%。",
            "原价500元，现价350元，优惠了150元，打了7折。",
            "总预算10000元，已花费7500元，完成了75%。",
            "销售目标100万，今年实际完成120万，完成率120%。",
            "股票从100元涨到125元，涨幅25%。",
            
            # 单位转换和度量 (10个)
            "这条路长5000米，相当于5千米。",
            "水壶容量2升，等于2000毫升。",
            "汽车时速60公里，行驶3小时，共行驶180公里。",
            "房间面积120平方米，长12米，宽10米。",
            "今天气温32摄氏度，相当于89.6华氏度。",
            "包裹重3.5千克，等于3500克。",
            "跑步用时30分钟，相当于0.5小时。",
            "网速100兆比特每秒，下载1000兆需要10秒。",
            "油箱容量60升，每百公里耗油8升，可行驶750公里。",
            "屏幕分辨率1920x1080，总共2073600个像素。",
            
            # 时间和日期计算 (10个)
            "现在是14:30，2小时后是16:30。",
            "今天是15号，7天后是22号。",
            "会议9:00开始，11:30结束，持续2.5小时。",
            "项目工期30天，已进行18天，还需12天完成。",
            "早上6:30起床，晚上22:30睡觉，清醒16小时。",
            "2024年是闰年，2月有29天。",
            "火车18:45发车，延误25分钟，实际19:10发车。",
            "工作8小时，休息1小时，实际工作7小时。",
            "距离新年还有45天，大约1.5个月。",
            "电影时长135分钟，等于2小时15分钟。",
            
            # 混合场景和对比 (10个)
            "这个月电费150元，上个月120元，增加了30元。",
            "身高175厘米，体重70千克，BMI指数22.9。",
            "上午卖出商品50件，下午卖出80件，全天共130件。",
            "去年销售额500万，今年650万，增长30%。",
            "A产品99元，B产品149元，B比A贵50元。",
            "第一次考试78分，第二次92分，进步了14分。",
            "早高峰车速20公里/时，平时40公里/时，慢了一半。",
            "上周气温18度，本周25度，升高了7度。",
            "旧手机电池3000毫安时，新手机5000毫安时，增加66.7%。",
            "标准价368元，会员价294.4元，会员享受8折优惠。"
        ]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回处理后的样本
        
        Returns:
            dict: 包含以下键值
                - input_ids: [seq_len] 输入词元ID
                - numeric_values: [seq_len] 对齐的数值
                - attention_mask: [seq_len] 注意力掩码
                - labels: [seq_len] 目标词元ID（用于训练）
                - label_numeric_values: [seq_len] 目标数值（用于训练）
        """
        text = self.samples[idx]
        
        # 使用数值感知分词器处理
        processed = self.tokenizer.tokenize_with_numbers(text)
        input_ids = processed['input_ids']
        numeric_values = processed['numeric_values']
        
        # 创建标签（向左移动一位）
        if len(input_ids) > 1:
            labels = input_ids[1:].clone()
            label_numeric_values = numeric_values[1:].clone()
            
            # 输入去掉最后一个
            input_ids = input_ids[:-1]
            numeric_values = numeric_values[:-1]
        else:
            # 处理长度为1的边界情况
            labels = input_ids.clone()
            label_numeric_values = numeric_values.clone()
        
        # Padding到固定长度
        seq_len = len(input_ids)
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            numeric_values = torch.cat([numeric_values, torch.zeros(pad_len)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])  # -100 是 ignore_index
            label_numeric_values = torch.cat([label_numeric_values, torch.zeros(pad_len)])
            attention_mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        else:
            # 截断
            input_ids = input_ids[:self.max_length]
            numeric_values = numeric_values[:self.max_length]
            labels = labels[:self.max_length]
            label_numeric_values = label_numeric_values[:self.max_length]
            attention_mask = torch.ones(self.max_length)
        
        return {
            'input_ids': input_ids.long(),
            'numeric_values': numeric_values.float(),
            'attention_mask': attention_mask.long(),
            'labels': labels.long(),
            'label_numeric_values': label_numeric_values.float()
        }

def create_dataloaders(
    model: CausalQwen,
    train_samples: List[str] = None,
    val_samples: List[str] = None,
    batch_size: int = 8,
    max_length: int = 128,
    num_workers: int = 0,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        model: CausalQwen 模型实例
        train_samples: 训练样本列表
        val_samples: 验证样本列表
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 数据加载进程数
        val_split: 如果没有提供 val_samples，从 train_samples 中划分的比例
    
    Returns:
        train_loader, val_loader
    """
    # 如果没有提供样本，使用默认的50个样本
    if train_samples is None and val_samples is None:
        all_samples = SyntheticNumericalDataset(model.numerical_tokenizer)._get_default_samples()
        
        # 随机打乱并划分训练/验证集
        random.shuffle(all_samples)
        
        val_size = int(len(all_samples) * val_split)
        train_samples = all_samples[val_size:]
        val_samples = all_samples[:val_size]
    
    # 创建数据集
    train_dataset = SyntheticNumericalDataset(
        model.numerical_tokenizer, 
        samples=train_samples,
        max_length=max_length
    )
    
    val_dataset = SyntheticNumericalDataset(
        model.numerical_tokenizer,
        samples=val_samples,
        max_length=max_length
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Example Usage and Test
if __name__ == '__main__':
    # 超参数
    QWEN_MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B")
    BATCH_SIZE = 2
    SEQUENCE_LENGTH = 10

    # 实例化模型
    model = CausalQwen(QWEN_MODEL_PATH, ovr_threshold=100.0, gamma_init=10.0, alpha_gated_reg=0.0)
    print("模型初始化成功。")
    print(f"词汇表大小: {model.vocab_size}, 隐藏层维度: {model.hidden_size}")
    print(f"<NUM> 词元ID: {model.num_token_id}")

    # 创建虚拟输入数据
    input_ids = torch.randint(0, model.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH))
    numeric_values = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.float)
    attention_mask = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.long)
    
    # 随机设置一些位置为数值
    for i in range(BATCH_SIZE):
        num_positions = torch.randint(1, 3, (1,)).item()
        positions = torch.randperm(SEQUENCE_LENGTH)[:num_positions]
        input_ids[i, positions] = model.num_token_id
        numeric_values[i, positions] = torch.randn(num_positions) * 100

    # 创建真实标签用于损失计算
    labels = torch.randint(0, model.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH))
    true_numeric_values = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH) * 100
    
    # 确保标签中也有一些 NUM_TOKEN_ID
    for i in range(BATCH_SIZE):
        num_positions = torch.randint(1, 3, (1,)).item()
        positions = torch.randperm(SEQUENCE_LENGTH)[:num_positions]
        labels[i, positions] = model.num_token_id
        true_numeric_values[i, positions] = torch.randn(num_positions) * 100

    print("虚拟输入数据创建完成。")

    # 前向传播
    loc_U, scale_U, loc_S, scale_S, loc_Y, scale_Y = model(input_ids, numeric_values, attention_mask)
    print(f"\n前向传播输出形状:")
    print(f"  loc_U: {loc_U.shape}, scale_U: {scale_U.shape}")
    print(f"  loc_S: {loc_S.shape}, scale_S: {scale_S.shape}")
    print(f"  loc_Y: {loc_Y.shape}, scale_Y: {scale_Y.shape}")

    # 损失计算 - 使用正确的参数
    total_loss, loss_dict = model.calculate_loss(
        loc_S, scale_S, loc_Y, scale_Y, 
        labels, true_numeric_values, 
        attention_mask=attention_mask
    )
    print(f"\n损失:")
    print(f"  总损失: {loss_dict['total_loss']:.4f}")
    print(f"  分类损失: {loss_dict['cls_loss_mean']:.4f}")
    print(f"  回归损失: {loss_dict['reg_loss_effective']:.4f}")
    print(f"  分类位置数: {loss_dict['n_cls']:.0f}")
    print(f"  数值位置数: {loss_dict['n_reg']:.0f}")

    # 确定性推理
    predicted_token_ids, predicted_numeric_values = model.deterministic_inference(
        loc_S, scale_S, loc_Y, scale_Y
    )
    print(f"\n确定性推理输出形状:")
    print(f"  预测的词元ID: {predicted_token_ids.shape}")
    print(f"  预测的数值: {predicted_numeric_values.shape}")

    # 因果采样
    sampled_token_ids, sampled_numeric_values, sampled_U = model.causal_sampling(
        loc_U, scale_U, num_samples=3
    )
    print(f"\n因果采样输出形状 (3个样本):")
    print(f"  采样的词元ID: {sampled_token_ids.shape}")
    print(f"  采样的数值: {sampled_numeric_values.shape}")
    print(f"  采样的U: {sampled_U.shape}")

    # 兼容传统采样
    softmax_probs = model.compatible_traditional_sampling(loc_S)
    print(f"\n兼容传统采样 (Softmax概率) 形状: {softmax_probs.shape}")

    # 测试数值感知分词器
    print("\n测试数值感知分词器:")
    test_texts = ["价格是99.9元", "温度达到-15.5度", "今天天气真好"]
    tokenizer_output = model.numerical_tokenizer.batch_tokenize(test_texts, max_length=20)
    print(f"  批量分词输出形状:")
    print(f"    input_ids: {tokenizer_output['input_ids'].shape}")
    print(f"    numeric_values: {tokenizer_output['numeric_values'].shape}")
    print(f"    attention_mask: {tokenizer_output['attention_mask'].shape}")

    # 测试合成数据集
    print("\n测试合成数据集:")
    train_loader, val_loader = create_dataloaders(model, batch_size=4, max_length=64)
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 查看一个批次的数据
    batch = next(iter(train_loader))
    print(f"\n批次数据形状:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # 显示第一个样本的内容
    print(f"\n第一个样本:")
    first_sample_ids = batch['input_ids'][0]
    first_sample_values = batch['numeric_values'][0]
    first_sample_mask = batch['attention_mask'][0]
    
    # 解码并显示
    valid_length = first_sample_mask.sum().item()
    decoded_text = model.tokenizer.decode(first_sample_ids[:valid_length])
    print(f"  文本: {decoded_text}")
    print(f"  数值位置: {(first_sample_values[:valid_length] != 0).nonzero(as_tuple=True)[0].tolist()}")
    
    # 测试模型前向传播使用数据集的数据
    print(f"\n测试模型前向传播（使用数据集）:")
    with torch.no_grad():
        loc_U, scale_U, loc_S, scale_S, loc_Y, scale_Y = model(
            batch['input_ids'], 
            batch['numeric_values'], 
            batch['attention_mask']
        )
        
        # 计算损失
        total_loss, loss_dict = model.calculate_loss(
            loc_S, scale_S, loc_Y, scale_Y,
            batch['labels'], 
            batch['label_numeric_values'],
            batch['attention_mask']
        )
        
        print(f"  批次损失: {loss_dict['total_loss']:.4f}")
        print(f"  分类损失: {loss_dict['cls_loss_mean']:.4f}")
        print(f"  回归损失: {loss_dict['reg_loss_effective']:.4f}")
    
    print("\n所有核心功能测试成功。")
    print("这是一个细粒度模块化的版本 - 每个模块都可以独立测试和改进。")


