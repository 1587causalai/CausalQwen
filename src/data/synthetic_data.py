#!/usr/bin/env python
"""合成数据生成器"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
from ..data.tokenizer import QwenTokenizerWrapper


class SyntheticDataGenerator:
    """生成合成训练数据的类"""
    
    def __init__(self, tokenizer: QwenTokenizerWrapper, config, random_seed: int = 42):
        """初始化数据生成器
        
        Args:
            tokenizer: 分词器包装器
            config: 模型配置
            random_seed: 随机种子，用于确保可重复性
        """
        self.tokenizer = tokenizer
        self.config = config
        self.rng = np.random.RandomState(random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_samples(self, num_samples: int) -> Dict[str, List]:
        """生成合成样本
        
        Args:
            num_samples: 要生成的样本数量
            
        Returns:
            包含 input_ids, attention_mask, labels, numerical_labels 的字典
        """
        texts = []
        for _ in range(num_samples):
            # 生成随机句子
            num_words = self.rng.randint(5, 15)
            words = []
            
            for _ in range(num_words):
                if self.rng.random() < 0.3:  # 30% 概率生成数字
                    number = self.rng.uniform(0, 100)
                    words.append(f"{number:.2f}")
                else:
                    # 生成随机词
                    word = ''.join(self.rng.choice(list('abcdefghijklmnopqrstuvwxyz'), 
                                                  size=self.rng.randint(3, 8)))
                    words.append(word)
            
            text = ' '.join(words)
            texts.append(text)
        
        # 使用 tokenizer 编码
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        numerical_values = encoded.get('numerical_values', torch.zeros_like(input_ids, dtype=torch.float))
        
        # 创建标签（shifted input_ids）
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = -100
        
        # 创建数值标签
        numerical_labels = numerical_values.clone()
        numerical_labels[:, :-1] = numerical_values[:, 1:].clone()
        numerical_labels[:, -1] = -100
        
        # 只在 <NUM> token 位置保留数值
        num_mask = (labels != self.tokenizer.num_token_id)
        numerical_labels[num_mask] = -100
        
        return {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
            'labels': labels.numpy(),
            'numerical_labels': numerical_labels.numpy()
        }
    
    def create_training_dataloader(self, num_samples: int, batch_size: int, 
                                   mode: str = 'training', shuffle: bool = True) -> DataLoader:
        """创建训练数据加载器
        
        Args:
            num_samples: 生成的样本数量
            batch_size: 批次大小
            mode: 'training' 或 'evaluation'
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader: PyTorch 数据加载器
        """
        print(f"Generating {num_samples} synthetic {mode} samples...")
        
        # 生成数据
        samples = self.generate_samples(num_samples)
        
        # 创建 TensorDataset
        dataset = TensorDataset(
            torch.tensor(samples['input_ids']),
            torch.tensor(samples['attention_mask']),
            torch.tensor(samples['labels']),
            torch.tensor(samples['numerical_labels'])
        )
        
        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return dataloader
    
    # 为了向后兼容，添加别名
    def create_dataloader(self, num_samples: int, batch_size: int, 
                         mode: str = 'training', shuffle: bool = True):
        """创建数据加载器（向后兼容）"""
        return self.create_training_dataloader(num_samples, batch_size, mode, shuffle)