#!/usr/bin/env python
"""
Synthetic data generator for training and evaluating the causal language model.

This module provides functions to create synthetic datasets with controlled properties,
including:
- Basic text sequences
- Text sequences with numerical values
- Controlled complexity and length
"""

import numpy as np
import torch
from typing import List, Dict, Any


class SyntheticDataGenerator:
    """
    语料库生成器
    
    该类用于生成用于训练和评估因果语言模型的合成数据集。
    """
    
    def __init__(self, tokenizer, num_samples: int = 1000, max_length: int = 32):
        """
        初始化生成器。
        
        Args:
            tokenizer: 分词器，用于编码和解码文本
            num_samples (int): 要生成的样本数量
            max_length (int): 每个样本的最大长度
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        
    def _generate_basic_samples(self) -> List[Dict[str, Any]]:
        """生成基本文本样本（无数值）"""
        samples = []
        for _ in range(self.num_samples):
            # 随机生成长度在1到max_length之间的序列
            seq_length = np.random.randint(1, self.max_length + 1)
            input_ids = [np.random.randint(1, self.tokenizer.vocab_size) for _ in range(seq_length)]
            
            # 标签为输入的下一个词元（自回归任务）
            labels = input_ids[1:] + [self.tokenizer.pad_token_id]
            
            samples.append({
                'input_ids': input_ids,
                'labels': labels
            })
        
        return samples
    
    def _generate_numeric_samples(self) -> List[Dict[str, Any]]:
        """生成带有数值的文本样本"""
        samples = []
        for _ in range(self.num_samples):
            # 随机生成长度在3到max_length之间的序列，确保有足够的上下文
            seq_length = np.random.randint(3, self.max_length + 1)
            
            # 前半部分为文本token
            text_length = seq_length - 1  # 留出一个位置给<NUM>
            input_ids = [np.random.randint(1, self.tokenizer.vocab_size) for _ in range(text_length)]
            
            # 在随机位置插入<NUM>词元
            num_position = np.random.randint(0, text_length)
            input_ids.insert(num_position, self.tokenizer.num_token_id)
            
            # 标签为输入的下一个词元（自回归任务）
            labels = input_ids[1:] + [self.tokenizer.pad_token_id]
            
            # 数值信息：为<NUM>位置生成随机数值
            numerical_values = [0.0] * seq_length
            target_numerical_values = [0.0] * seq_length
            numerical_values[num_position] = np.random.uniform(1, 100)  # 随机数值
            target_numerical_values[num_position] = numerical_values[num_position]  # 目标数值与输入数值相同
            
            samples.append({
                'input_ids': input_ids,
                'labels': labels,
                'input_numerical_values': numerical_values,
                'target_numerical_values': target_numerical_values
            })
        
        return samples

    def _create_batch_tensors(self, samples):
        """将样本列表转换为批次张量"""
        max_len = max(len(sample['input_ids']) for sample in samples)
        batch_size = len(samples)
        
        # 初始化张量
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 用于忽略损失
        numerical_values = torch.zeros((batch_size, max_len), dtype=torch.float)  # 修复：使用0而非nan
        target_values = torch.zeros((batch_size, max_len), dtype=torch.float)    # 修复：使用0而非nan
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, sample in enumerate(samples):
            seq_len = len(sample['input_ids'])
            
            # 填充输入序列
            input_ids[i, :seq_len] = torch.tensor(sample['input_ids'])
            labels[i, :seq_len] = torch.tensor(sample['labels'])
            attention_mask[i, :seq_len] = 1
            
            # 填充数值信息
            for j, (inp_val, tgt_val) in enumerate(zip(sample['input_numerical_values'], 
                                                      sample['target_numerical_values'])):
                # 重要修复：只填充有效数值，无效位置保持0
                if not (inp_val != inp_val):  # 检查不是NaN
                    numerical_values[i, j] = inp_val
                if not (tgt_val != tgt_val):  # 检查不是NaN  
                    target_values[i, j] = tgt_val
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'numerical_values': numerical_values,
            'target_values': target_values,
            'attention_mask': attention_mask
        }

    def get_synthetic_dataset(self, data_type: str = 'basic'):
        """
        获取合成数据集
        
        Args:
            data_type (str): 数据类型，'basic'或'numeric'
        
        Returns:
            dict: 合成数据集的张量
        """
        if data_type == 'basic':
            samples = self._generate_basic_samples()
        elif data_type == 'numeric':
            samples = self._generate_numeric_samples()
        else:
            raise ValueError("Unsupported data type. Choose 'basic' or 'numeric'.")
        
        # 创建批次张量
        return self._create_batch_tensors(samples)