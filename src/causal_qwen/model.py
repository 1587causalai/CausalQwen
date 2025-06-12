"""CausalQwen 主模型实现

包含完整的模型架构和推理功能。
"""

import torch
from torch import nn
from typing import Optional, Dict, Any, Tuple, Union, List
from .modules import NumericalEmbedding, AbductionNetwork, ActionNetwork

class CausalQwen(nn.Module):
    """CausalQwen 主模型类
    
    实现了基于因果推理的语言模型，支持文本和数值的统一处理。
    完全兼容 Qwen 的接口。
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_vocab = config.num_vocab
        self.hidden_dim = config.hidden_dim
        self.num_token_id = config.num_token_id
        
        # 核心模块
        self.numerical_embedding = NumericalEmbedding(config)
        self.transformer = self._build_transformer(config)
        self.abduction = AbductionNetwork(config)
        self.action = ActionNetwork(config)
        
        # 需要设置 tokenizer（从外部传入或加载）
        self.tokenizer = None
        
    def set_tokenizer(self, tokenizer):
        """设置tokenizer"""
        self.tokenizer = tokenizer
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        num_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: token ids [batch_size, seq_len]
            num_values: 数值输入 [batch_size, seq_len]
            labels: 分类标签 [batch_size, seq_len]
            num_labels: 回归标签 [batch_size, seq_len]
            
        Returns:
            包含损失和预测的字典
        """
        # 1. 获取特征表示
        embeddings = self.numerical_embedding(input_ids, num_values)
        features = self.transformer(embeddings)  # [B, S, H]
        
        # 2. 获取因果表征的分布参数
        u_loc, u_scale = self.abduction(features)  # [B, S, H], [B, S, H]
        
        # 3. 从柯西分布中采样U
        u_samples = self._sample_cauchy(u_loc, u_scale)  # [B, S, N, H]
        
        # 4. 通过行动网络得到输出
        action_output = self.action(u_samples)
        class_scores = action_output['class_scores']  # [B, S, N, K]
        regression_values = action_output['regression_values']  # [B, S, N]
        
        # 5. 计算分类概率（OvR）
        class_probs = self.action.compute_ovr_probs(class_scores)
        
        # 6. 计算损失（如果提供了标签）
        loss = None
        if labels is not None and num_labels is not None:
            loss = self._compute_loss(class_scores, regression_values, labels, num_labels)
        
        return {
            'loss': loss,
            'class_scores': class_scores,
            'class_probs': class_probs,
            'regression_values': regression_values,
            'u_samples': u_samples
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_samples: int = 1,
        sampling_mode: str = "causal",  # "causal" or "traditional"
        return_dict_in_generate: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """生成新的token序列
        
        Args:
            input_ids: 输入序列 [batch_size, seq_len]
            num_values: 数值序列 [batch_size, seq_len]
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_k: top-k 采样参数
            top_p: top-p (nucleus) 采样参数
            num_samples: 每个位置的采样次数（仅用于causal模式）
            sampling_mode: 采样模式，"causal"或"traditional"
            return_dict_in_generate: 是否返回详细信息
            
        Returns:
            生成的序列 [batch_size, seq_len + max_new_tokens]
            或详细信息字典（如果return_dict_in_generate=True）
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 初始化
        generated_ids = input_ids.clone()
        generated_values = num_values.clone() if num_values is not None else torch.zeros_like(input_ids, dtype=torch.float)
        
        # 存储生成过程信息
        generation_info = {
            'u_distributions': [],  # 每步的U分布参数
            'token_probs': [],      # 每步的token概率
            'sampled_values': []    # 每步的数值预测
        } if return_dict_in_generate else None
        
        # 逐步生成
        for _ in range(max_new_tokens):
            # 获取下一个token
            if sampling_mode == "causal":
                next_token, next_value, step_info = self._causal_sample_next(
                    generated_ids, generated_values, temperature, top_k, top_p
                )
            else:
                next_token, next_value, step_info = self._traditional_sample_next(
                    generated_ids, generated_values, temperature, top_k, top_p
                )
            
            # 更新序列
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            generated_values = torch.cat([generated_values, next_value.unsqueeze(1)], dim=1)
            
            # 记录生成信息
            if return_dict_in_generate:
                for key, value in step_info.items():
                    if key in generation_info:
                        generation_info[key].append(value)
            
            # 检查是否全部结束（如果有EOS token）
            if hasattr(self.config, 'eos_token_id'):
                if torch.all(next_token == self.config.eos_token_id):
                    break
        
        if return_dict_in_generate:
            return {
                'sequences': generated_ids,
                'values': generated_values,
                'generation_info': generation_info
            }
        else:
            return generated_ids
    
    def _causal_sample_next(
        self,
        input_ids: torch.Tensor,
        num_values: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """因果采样下一个token
        
        在因果表征空间U中采样，然后通过行动网络得到确定性输出。
        使用 softmax(loc_S) 的兼容方法进行采样。
        """
        batch_size = input_ids.shape[0]
        
        # 1. 获取特征
        embeddings = self.numerical_embedding(input_ids, num_values)
        features = self.transformer(embeddings)  # [B, S, H]
        
        # 2. 获取最后一个位置的因果表征分布
        last_features = features[:, -1, :]  # [B, H]
        u_loc, u_scale = self.abduction(last_features.unsqueeze(1))  # [B, 1, H]
        u_loc, u_scale = u_loc.squeeze(1), u_scale.squeeze(1)  # [B, H]
        
        # 3. 从柯西分布采样U（无温度）
        cauchy_dist = torch.distributions.Cauchy(u_loc, u_scale)
        u_sample = cauchy_dist.sample()  # [B, H]
        
        # 4. 通过行动网络得到分类分数分布
        action_output = self.action(u_sample)
        class_scores_loc = action_output['class_scores']  # [B, K] - 分数的 loc
        regression_values = action_output['regression_values']  # [B]
        
        # 5. 使用 softmax(loc_S) 进行兼容采样（关键改进！）
        # 这与传统 LM 完全兼容
        logits = class_scores_loc  # 直接使用 loc 作为 logits
        
        # 6. 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 7. 计算概率分布
        probs = torch.softmax(logits, dim=-1)  # [B, K] - 标准概率分布
        
        # 8. 应用 top-k/top-p 过滤
        filtered_probs = self._filter_probs(probs, top_k, top_p)
        
        # 9. 从过滤后的分布中采样
        next_token = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)  # [B]
        
        # 10. 确定数值
        next_value = torch.where(
            next_token == self.num_token_id,
            regression_values,
            torch.zeros_like(regression_values)
        )
        
        # 11. 收集信息
        step_info = {
            'u_loc': u_loc,
            'u_scale': u_scale,
            'class_scores_loc': class_scores_loc,  # 原始分数 loc
            'logits': logits,  # 温度调整后的 logits
            'probs': probs,  # softmax 概率
            'regression_pred': regression_values
        }
        
        return next_token, next_value, step_info
    
    def _traditional_sample_next(
        self,
        input_ids: torch.Tensor,
        num_values: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """传统采样下一个token
        
        使用分布参数直接计算 OvR 概率，然后在词汇表上采样。
        """
        # 1. 前向传播获取分布参数
        output = self.forward(input_ids, num_values)
        
        # 2. 获取最后位置的预测
        ovr_probs = output['class_probs'][:, -1, :]  # [B, K] - OvR 概率
        reg_loc = output['regression_loc'][:, -1]  # [B]
        reg_scale = output['regression_scale'][:, -1]  # [B]
        
        # 3. 重新归一化 OvR 概率（关键！）
        normalized_probs = ovr_probs / ovr_probs.sum(dim=-1, keepdim=True)
        
        # 4. 温度调整（在归一化后的概率上）
        if temperature != 1.0:
            # 转换为 logits，应用温度，再转回概率
            epsilon = 1e-10
            logits = torch.log(normalized_probs + epsilon)
            logits = logits / temperature
            normalized_probs = torch.softmax(logits, dim=-1)
        
        # 5. 应用 top-k/top-p 过滤
        filtered_probs = self._filter_probs(normalized_probs, top_k, top_p)
        
        # 6. 采样 token
        next_token = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)  # [B]
        
        # 7. 确定数值（从回归分布采样，可以使用温度）
        cauchy_dist = torch.distributions.Cauchy(reg_loc, reg_scale * temperature)
        sampled_values = cauchy_dist.sample()
        next_value = torch.where(
            next_token == self.num_token_id,
            sampled_values,
            torch.zeros_like(sampled_values)
        )
        
        # 8. 收集信息
        step_info = {
            'ovr_probs': ovr_probs,
            'normalized_probs': normalized_probs,
            'regression_loc': reg_loc,
            'regression_scale': reg_scale
        }
        
        return next_token, next_value, step_info
    
    def _filter_probs(
        self,
        probs: torch.Tensor,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> torch.Tensor:
        """应用top-k和top-p过滤
        
        注意：输入的 probs 应该已经是归一化的概率分布。
        
        Args:
            probs: 概率分布 [batch_size, vocab_size]，满足 sum = 1
            top_k: 保留概率最高的k个token
            top_p: 保留累积概率达到p的token
            
        Returns:
            过滤后的概率分布 [batch_size, vocab_size]
        """
        filtered_probs = probs.clone()
        
        # Top-k 过滤
        if top_k is not None and top_k > 0:
            indices_to_remove = probs < torch.topk(probs, k=top_k, dim=-1)[0][..., -1, None]
            filtered_probs[indices_to_remove] = 0
        
        # Top-p (nucleus) 过滤
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 找到超过阈值的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留至少一个token
            sorted_indices_to_remove[..., 0] = False
            
            # 恢复原始顺序
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            filtered_probs[indices_to_remove] = 0
        
        # 重新归一化（保证概率和为1）
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        return filtered_probs
    
    def get_enhanced_embeddings(self, input_ids: torch.Tensor, num_values: Optional[torch.Tensor] = None):
        """获取数值增强的嵌入（供外部使用）"""
        return self.numerical_embedding(input_ids, num_values)
    
    def _build_transformer(self, config):
        """构建Transformer模块"""
        # 这里应该实现或调用实际的Transformer
        # 暂时返回一个占位符
        return nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        sampling_mode: str = "causal",
        **kwargs
    ) -> Union[str, Any]:
        """与 Qwen 兼容的 chat 接口
        
        Args:
            messages: 对话消息列表，每个消息包含 role 和 content
                [{"role": "user", "content": "你好"}]
            stream: 是否流式输出
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_k: top-k 采样
            top_p: top-p 采样
            sampling_mode: "causal" 或 "traditional"
            
        Returns:
            生成的回复文本（非流式）或生成器（流式）
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please call set_tokenizer() first.")
            
        # 构建输入prompt
        prompt = self._format_chat_prompt(messages)
        
        if stream:
            # 流式生成
            return self._stream_chat(
                prompt, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sampling_mode=sampling_mode,
                **kwargs
            )
        else:
            # 非流式生成
            full_response = self.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sampling_mode=sampling_mode,
                **kwargs
            )
            
            # 提取助手回复
            return self._extract_assistant_reply(full_response, prompt)
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话消息为prompt
        
        这里应该使用与Qwen相同的prompt模板
        """
        # 简化版本 - 实际应该使用Qwen的模板
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        # 添加助手提示
        prompt += "Assistant: "
        return prompt
    
    def _extract_assistant_reply(self, full_response: str, prompt: str) -> str:
        """从完整响应中提取助手回复"""
        if full_response.startswith(prompt):
            reply = full_response[len(prompt):]
        else:
            reply = full_response
            
        # 清理回复
        reply = reply.strip()
        
        # 如果包含下一轮对话，只取第一个回复
        if "\nUser:" in reply:
            reply = reply.split("\nUser:")[0].strip()
            
        return reply
    
    def _stream_chat(self, prompt: str, **kwargs):
        """流式生成实现"""
        # 这里应该实现真正的流式生成
        # 现在只是模拟
        response = self.generate_text(prompt, **kwargs)
        assistant_reply = self._extract_assistant_reply(response, prompt)
        
        # 模拟流式输出
        for i in range(0, len(assistant_reply), 5):  # 每次输出5个字符
            yield assistant_reply[i:i+5]
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """与 transformers 兼容的加载接口
        
        Args:
            model_path: 模型路径
            **kwargs: 其他参数
            
        Returns:
            加载的模型实例
        """
        # 这里应该实现真正的模型加载逻辑
        # 1. 加载配置
        # 2. 创建模型
        # 3. 加载权重
        # 4. 加载tokenizer
        
        raise NotImplementedError("from_pretrained 方法尚未实现")
    
    def save_pretrained(self, save_path: str):
        """保存模型
        
        Args:
            save_path: 保存路径
        """
        # 这里应该实现模型保存逻辑
        # 1. 保存配置
        # 2. 保存权重
        # 3. 保存tokenizer配置
        
        raise NotImplementedError("save_pretrained 方法尚未实现")