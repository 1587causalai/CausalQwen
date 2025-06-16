import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Dict, List, Tuple, Optional
import numpy as np

class SyntheticNumericalDataset(Dataset):
    """合成数值数据集 - 生成包含数值运算的多样化句子"""
    
    def __init__(self, 
                 tokenizer_wrapper,
                 num_samples: int = 1000,
                 max_length: int = 128,
                 seed: Optional[int] = 42):
        """
        Args:
            tokenizer_wrapper: 数值感知的分词器包装器
            num_samples: 数据集大小
            max_length: 最大序列长度
            seed: 随机种子
        """
        self.tokenizer = tokenizer_wrapper
        self.num_samples = num_samples
        self.max_length = max_length
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 预生成所有样本
        self.samples = [self._generate_sample() for _ in range(num_samples)]
    
    def _generate_sample(self) -> str:
        """生成单个样本 - MOCKER VERSION"""
        # 这里只是返回固定的示例，实际实现时会随机生成
        templates = [
            # 数值运算类
            "小明有5个苹果，小红给了他3个，现在他有8个苹果。",
            "商品原价100元，打8折后是80元。",
            "今天温度是25度，比昨天的20度高了5度。",
            "这本书有256页，我已经读了128页，还剩128页。",
            "火车以120公里/小时的速度行驶，3小时后行驶了360公里。",
            
            # 数值描述类
            "这栋楼有32层，每层高3.5米，总高度是112米。",
            "手机电池容量是4000毫安时，充电功率18瓦。",
            "显示器分辨率是1920x1080，刷新率144赫兹。",
            
            # 纯文本类（对比用）
            "今天天气真好，适合出去散步。",
            "人工智能技术正在改变我们的生活方式。"
        ]
        
        return random.choice(templates)
    
    def __len__(self) -> int:
        return self.num_samples
    
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
        labels = input_ids[1:].clone()
        label_numeric_values = numeric_values[1:].clone()
        
        # 输入去掉最后一个
        input_ids = input_ids[:-1]
        numeric_values = numeric_values[:-1]
        
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

class AdvancedSyntheticGenerator:
    """高级合成数据生成器 - 生成更复杂的数值运算样本"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_arithmetic_sample(self) -> str:
        """生成算术运算样本"""
        # 加法
        a, b = random.randint(1, 100), random.randint(1, 100)
        templates = [
            f"{a}加{b}等于{a+b}。",
            f"计算{a}+{b}的结果是{a+b}。",
            f"把{a}和{b}相加得到{a+b}。"
        ]
        return random.choice(templates)
    
    def generate_percentage_sample(self) -> str:
        """生成百分比计算样本"""
        base = random.randint(50, 500)
        percent = random.randint(10, 90)
        result = base * percent / 100
        templates = [
            f"{base}的{percent}%是{result:.1f}。",
            f"原价{base}元，打{100-percent}折后是{base*(100-percent)/100:.1f}元。",
            f"总数{base}，占{percent}%的部分是{result:.1f}。"
        ]
        return random.choice(templates)
    
    def generate_comparison_sample(self) -> str:
        """生成数值比较样本"""
        a, b = random.randint(10, 200), random.randint(10, 200)
        diff = abs(a - b)
        templates = [
            f"{a}比{b}{'大' if a > b else '小'}{diff}。",
            f"温度从{a}度变到{b}度，{'上升' if b > a else '下降'}了{diff}度。",
            f"距离{a}公里和{b}公里相差{diff}公里。"
        ]
        return random.choice(templates)
    
    def generate_unit_conversion_sample(self) -> str:
        """生成单位转换样本"""
        meters = random.randint(100, 5000)
        kilometers = meters / 1000
        templates = [
            f"{meters}米等于{kilometers:.1f}千米。",
            f"距离{kilometers:.1f}千米，换算成米是{meters}米。"
        ]
        return random.choice(templates)
    
    def generate_mixed_text_sample(self) -> str:
        """生成混合文本和数值的样本"""
        price = random.randint(50, 500)
        quantity = random.randint(2, 10)
        total = price * quantity
        templates = [
            f"每件商品{price}元，买{quantity}件需要{total}元。",
            f"单价{price}元的商品，{quantity}件总价是{total}元。"
        ]
        return random.choice(templates)
    
    def generate_pure_text_sample(self) -> str:
        """生成纯文本样本（用于对比）"""
        templates = [
            "机器学习是人工智能的一个重要分支。",
            "深度学习模型需要大量的训练数据。",
            "自然语言处理技术发展迅速。",
            "神经网络可以学习复杂的模式。"
        ]
        return random.choice(templates)
    
    def generate_batch(self, batch_size: int = 10) -> List[str]:
        """生成一批多样化的样本"""
        generators = [
            self.generate_arithmetic_sample,
            self.generate_percentage_sample,
            self.generate_comparison_sample,
            self.generate_unit_conversion_sample,
            self.generate_mixed_text_sample,
            self.generate_pure_text_sample
        ]
        
        # 确保每种类型都有一定比例
        samples = []
        for i in range(batch_size):
            if i < len(generators):
                # 前几个样本确保每种类型都有
                samples.append(generators[i]())
            else:
                # 剩余样本随机选择
                generator = random.choice(generators[:-1])  # 减少纯文本比例
                samples.append(generator())
        
        # 打乱顺序
        random.shuffle(samples)
        return samples

def create_dataloaders(
    tokenizer_wrapper,
    train_size: int = 1000,
    val_size: int = 100,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        tokenizer_wrapper: 数值感知分词器
        train_size: 训练集大小
        val_size: 验证集大小
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 数据加载进程数
    
    Returns:
        train_loader, val_loader
    """
    # 创建数据集
    train_dataset = SyntheticNumericalDataset(
        tokenizer_wrapper, 
        num_samples=train_size,
        max_length=max_length,
        seed=42
    )
    
    val_dataset = SyntheticNumericalDataset(
        tokenizer_wrapper,
        num_samples=val_size,
        max_length=max_length,
        seed=123  # 不同的种子
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

# 测试代码
if __name__ == "__main__":
    print("=== 测试合成数据生成器 ===\n")
    
    # 测试高级生成器
    generator = AdvancedSyntheticGenerator(seed=42)
    
    print("生成一批样本（10条）：")
    samples = generator.generate_batch(10)
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")
    
    print("\n=== 各类样本示例 ===")
    print(f"算术运算: {generator.generate_arithmetic_sample()}")
    print(f"百分比计算: {generator.generate_percentage_sample()}")
    print(f"数值比较: {generator.generate_comparison_sample()}")
    print(f"单位转换: {generator.generate_unit_conversion_sample()}")
    print(f"混合文本: {generator.generate_mixed_text_sample()}")
    print(f"纯文本: {generator.generate_pure_text_sample()}")
