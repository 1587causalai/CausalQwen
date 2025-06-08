"""
合成数据生成器

生成用于训练和评估因果语言模型的合成数据。
"""

import torch
import random
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import Dataset


class TextWithNumbersGenerator:
    """
    生成包含数值的文本数据
    """
    
    def __init__(self, vocab_size: int = 1000, num_token_id: int = 999):
        self.vocab_size = vocab_size
        self.num_token_id = num_token_id
        
        # 基础模板
        self.templates = [
            "The price is {value} dollars.",
            "The temperature is {value} degrees.",
            "The distance is {value} kilometers.",
            "The weight is {value} kilograms.",
            "The height is {value} meters.",
            "The speed is {value} mph.",
            "The cost is {value} yuan.",
            "The time is {value} hours.",
        ]
        
        # 问答模板
        self.qa_templates = [
            ("The item costs {value} dollars. What is the price?", "{value}"),
            ("The journey takes {value} hours. How long does the journey take?", "{value}"),
            ("The building is {value} meters tall. What is the height?", "{value}"),
            ("The car weighs {value} kg. What is the weight?", "{value}"),
        ]
    
    def generate_basic_sample(self) -> Tuple[str, float]:
        """生成基础数值文本样本"""
        template = random.choice(self.templates)
        value = random.uniform(0, 100)
        text = template.format(value=value)
        return text, value
    
    def generate_qa_sample(self) -> Tuple[str, float]:
        """生成问答格式样本"""
        context_template, answer_template = random.choice(self.qa_templates)
        value = random.uniform(0, 100)
        text = context_template.format(value=value)
        return text, value
    
    def generate_extreme_value_sample(self) -> Tuple[str, float]:
        """生成极端数值样本"""
        template = random.choice(self.templates)
        
        # 80%常规值，20%极端值
        if random.random() < 0.8:
            value = random.uniform(0, 100)
        else:
            # 使用幂律分布生成极端值
            value = np.random.pareto(1.0) * 100 + 100
        
        text = template.format(value=value)
        return text, value
    
    def generate_boundary_sample(self) -> Tuple[str, float]:
        """生成边界值样本"""
        template = random.choice(self.templates)
        
        # 边界值：0, 1, 很小的数，很大的数
        boundary_values = [0.0, 1.0, 0.001, 0.1, 10.0, 100.0, 1000.0]
        value = random.choice(boundary_values)
        
        text = template.format(value=value)
        return text, value


class SimpleTokenizer:
    """
    简单的分词器实现
    """
    
    def __init__(self, vocab_size: int = 1000, num_token: str = "<NUM>"):
        self.vocab_size = vocab_size
        self.num_token = num_token
        
        # 构建词汇表
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            self.num_token: vocab_size - 1  # <NUM>词元放在最后
        }
        
        # 常用词汇
        common_words = [
            'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
            'price', 'temperature', 'distance', 'weight', 'height', 'speed',
            'cost', 'time', 'dollars', 'degrees', 'kilometers', 'kilograms',
            'meters', 'mph', 'yuan', 'hours', 'what', 'how', 'long', 'tall',
            'item', 'costs', 'journey', 'takes', 'building', 'car', 'weighs'
        ]
        
        # 添加常用词汇到词汇表
        for i, word in enumerate(common_words):
            if len(self.vocab) < vocab_size - 1:
                self.vocab[word] = len(self.vocab)
        
        # 填充剩余位置
        while len(self.vocab) < vocab_size - 1:
            self.vocab[f'word_{len(self.vocab)}'] = len(self.vocab)
        
        # 创建反向词汇表
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """编码文本为token ids"""
        # 简单的空格分词
        tokens = text.lower().replace('.', ' .').replace('?', ' ?').split()
        
        token_ids = []
        for token in tokens:
            # 检查是否为数字
            try:
                float(token)
                token_ids.append(self.vocab[self.num_token])
            except ValueError:
                token_ids.append(self.vocab.get(token, self.vocab['<UNK>']))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token ids为文本"""
        tokens = [self.id_to_token.get(id, '<UNK>') for id in token_ids]
        return ' '.join(tokens)


class CausalDataset(Dataset):
    """
    因果语言模型数据集
    """
    
    def __init__(self, data_type: str = 'basic', size: int = 1000, 
                 vocab_size: int = 1000, max_length: int = 32):
        self.data_type = data_type
        self.size = size
        self.max_length = max_length
        
        self.generator = TextWithNumbersGenerator(vocab_size)
        self.tokenizer = SimpleTokenizer(vocab_size)
        
        # 生成数据
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[str, float]]:
        """生成数据"""
        data = []
        
        for _ in range(self.size):
            if self.data_type == 'basic':
                text, value = self.generator.generate_basic_sample()
            elif self.data_type == 'qa':
                text, value = self.generator.generate_qa_sample()
            elif self.data_type == 'extreme':
                text, value = self.generator.generate_extreme_value_sample()
            elif self.data_type == 'boundary':
                text, value = self.generator.generate_boundary_sample()
            else:
                raise ValueError(f"Unknown data type: {self.data_type}")
            
            data.append((text, value))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, value = self.data[idx]
        
        # 编码文本
        token_ids = self.tokenizer.encode(text)
        
        # 截断或填充到固定长度
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.tokenizer.vocab['<PAD>']] * (self.max_length - len(token_ids)))
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'cls_target': torch.tensor(self.tokenizer.vocab[self.tokenizer.num_token], dtype=torch.long),
            'reg_target': torch.tensor(value, dtype=torch.float),
            'text': text,
            'value': value
        }


def create_dataloader(data_type: str = 'basic', batch_size: int = 32, 
                     size: int = 1000, vocab_size: int = 1000) -> torch.utils.data.DataLoader:
    """
    创建数据加载器
    
    Args:
        data_type: 数据类型 ('basic', 'qa', 'extreme', 'boundary')
        batch_size: 批次大小
        size: 数据集大小
        vocab_size: 词汇表大小
    
    Returns:
        数据加载器
    """
    dataset = CausalDataset(data_type, size, vocab_size)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    cls_targets = torch.stack([item['cls_target'] for item in batch])
    reg_targets = torch.stack([item['reg_target'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'cls_targets': cls_targets,
        'reg_targets': reg_targets,
        'texts': [item['text'] for item in batch],
        'values': [item['value'] for item in batch]
    }

