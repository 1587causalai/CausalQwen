# 快速开始

本指南将帮助你快速上手CausalQwen-0.5B项目，了解基本用法和核心功能。

## 基本概念

在开始使用之前，了解以下核心概念会很有帮助：

- **因果语言模型**：将决策过程分解为推断和行动两个阶段的语言模型
- **特征网络**：提取输入特征的网络
- **推断网络**：从特征推断因果状态分布的网络
- **行动网络**：基于因果状态做出决策的网络
- **柯西分布**：用于表示因果状态不确定性的重尾分布
- **OvR分类**：One-vs-Rest分类策略，为每个类别提供独立的二分类决策

## 创建模型

### 使用Qwen-0.5B作为特征网络

推荐的使用方式是基于真实的Qwen-0.5B模型：

```python
import torch
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper

# 创建分词器
tokenizer = QwenTokenizerWrapper(
    model_path="~/models/Qwen2.5-0.5B",
    use_real_tokenizer=True
)

# 创建模型配置
config = CausalLMConfig(
    vocab_size=tokenizer.vocab_size,  # 使用Qwen词汇表大小
    num_token_id=tokenizer.num_token_id,  # <NUM>词元ID
    hidden_size=896,  # Qwen-0.5B的隐藏层大小
    causal_dim=64,    # 因果状态维度
    use_real_qwen=True,  # 使用真实Qwen模型
    qwen_model_path="~/models/Qwen2.5-0.5B"
)

# 创建模型
model = CausalLanguageModel(config)

# 查看模型结构
print(model)
```

### 使用模拟特征网络（用于测试）

如果你想快速测试而不下载大模型，可以使用模拟特征网络：

```python
import torch
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper

# 创建模拟分词器
tokenizer = QwenTokenizerWrapper(use_real_tokenizer=False, vocab_size=1000)

# 创建模型配置
config = CausalLMConfig(
    vocab_size=tokenizer.vocab_size,
    num_token_id=tokenizer.num_token_id,
    hidden_size=768,  # 模拟隐藏层大小
    causal_dim=64,    # 因果状态维度
    use_mock_feature_network=True,  # 使用模拟特征网络
    use_real_qwen=False
)

# 创建模型
model = CausalLanguageModel(config)
```

## 模型推理

### 基本推理

```python
# 准备输入文本
texts = ["The price is 42.5 dollars.", "The temperature is 25.3 degrees."]

# 使用分词器处理文本
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 模型前向传播
outputs = model(
    inputs['input_ids'], 
    inputs['numerical_values'], 
    inputs['attention_mask']
)

# 查看输出
print("因果状态位置参数:", outputs['causal_loc'].shape)
print("因果状态尺度参数:", outputs['causal_scale'].shape)
print("分类概率:", outputs['cls_probs'].shape)
print("回归预测分布:", outputs['reg_loc'].shape, outputs['reg_scale'].shape)

# 获取确定性预测结果
predictions = model.predict(
    inputs['input_ids'], 
    inputs['numerical_values'], 
    inputs['attention_mask']
)
print("预测的词元:", predictions['cls_pred'])
print("预测的数值:", predictions['reg_pred'])
print("<NUM>词元概率:", predictions['num_prob'])
```

### 探索性推理（随机采样）

```python
# 从因果状态分布中采样进行预测
sampled_predictions = model.sample_and_predict(
    inputs['input_ids'], 
    inputs['numerical_values'], 
    inputs['attention_mask']
)

print("采样预测的词元:", sampled_predictions['cls_pred'])
print("采样预测的数值:", sampled_predictions['reg_pred'])
print("采样的因果状态:", sampled_predictions['causal_sample'].shape)
```

## 模型训练

### 使用训练器

```python
from src.training.trainer import Trainer
from src.data.synthetic import TextWithNumbersGenerator
from torch.utils.data import DataLoader, TensorDataset

# 创建合成训练数据
generator = TextWithNumbersGenerator(seed=42)
texts, values = generator.generate_text(num_samples=1000)

# 使用分词器处理
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 准备目标
targets = torch.full((len(texts),), tokenizer.num_token_id, dtype=torch.long)
target_values = torch.tensor(values, dtype=torch.float32)

# 创建训练器
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    config=config,  # 传递配置对象
    learning_rate=1e-4,
    batch_size=16
)

# 训练模型
trainer.train(num_epochs=10, num_samples=1000)
```

## 模型评估

### 使用评估器

```python
from src.evaluation.evaluator import Evaluator
from src.data.evaluation_data import get_all_evaluation_datasets

# 创建评估器
evaluator = Evaluator(
    model=model,
    tokenizer=tokenizer,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    model_config=config  # 传递配置对象用于指标计算
)

# 获取评估数据集
evaluation_datasets = get_all_evaluation_datasets(tokenizer)

# 评估模型
for name, dataset in evaluation_datasets.items():
    print(f"\n=== 评估数据集: {name} ===")
    metrics = evaluator.evaluate(dataset, batch_size=16)
    
    # 打印关键指标
    print(f"分类F1: {metrics['cls_f1']:.4f}")
    print(f"分类准确率: {metrics['cls_accuracy']:.4f}")
    print(f"回归MAE: {metrics['reg_mae']:.4f}")
    print(f"回归MSE: {metrics['reg_mse']:.4f}")
    print(f"校准ECE: {metrics['calib_ece']:.4f}")
    print(f"回归PICP: {metrics['reg_picp']:.4f}")
```

## 运行完整实验

### 使用实验运行器

```python
import subprocess
import os

# 运行基础实验
result = subprocess.run([
    'python', 'src/run_experiments.py', 'basic',
    '--qwen_model_path', '~/models/Qwen2.5-0.5B',
    '--epochs', '5',
    '--num_samples', '500'
], capture_output=True, text=True)

print("实验输出:", result.stdout)
print("实验结果保存在:", "results/basic_[timestamp]/")
```

### 生成可视化图表

```python
# 假设你有一个消融实验结果目录
results_dir = "results/ablation_20231208_143000/"

# 生成对比图表
result = subprocess.run([
    'python', 'src/visualization/plotter.py', results_dir
], capture_output=True, text=True)

print("图表生成完成:", result.stdout)
```

## 处理自定义数据

### 创建自定义数据集

```python
from torch.utils.data import Dataset, TensorDataset

class CustomNumberDataset(Dataset):
    def __init__(self, texts, values, tokenizer):
        self.texts = texts
        self.values = values
        self.tokenizer = tokenizer
        
        # 预处理数据
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.numerical_values = inputs['numerical_values']
        
        # 目标标签
        self.targets = torch.full((len(texts),), tokenizer.num_token_id)
        self.target_values = torch.tensor(values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.numerical_values[idx],
            self.targets[idx],
            self.target_values[idx]
        )

# 使用自定义数据集
custom_texts = ["Stock price: 150.25", "Temperature: -5.8°C"]
custom_values = [150.25, -5.8]

custom_dataset = CustomNumberDataset(custom_texts, custom_values, tokenizer)

# 评估自定义数据
metrics = evaluator.evaluate(custom_dataset, batch_size=2)
print("自定义数据评估结果:", metrics)
```

## 保存和加载模型

### 保存模型

```python
import torch
import json

# 保存模型权重
model_path = 'trained_causal_lm.pth'
torch.save(model.state_dict(), model_path)

# 保存配置
config_path = 'model_config.json'
config_dict = {
    'vocab_size': config.vocab_size,
    'num_token_id': config.num_token_id,
    'hidden_size': config.hidden_size,
    'causal_dim': config.causal_dim,
    'use_real_qwen': config.use_real_qwen,
    'qwen_model_path': config.qwen_model_path
}
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"模型保存到: {model_path}")
print(f"配置保存到: {config_path}")
```

### 加载模型

```python
# 加载配置
with open('model_config.json', 'r') as f:
    config_dict = json.load(f)

# 重建配置对象
loaded_config = CausalLMConfig(**config_dict)

# 重建模型
loaded_model = CausalLanguageModel(loaded_config)

# 加载权重
loaded_model.load_state_dict(torch.load('trained_causal_lm.pth'))
loaded_model.eval()

print("模型加载完成")
```

## 下一步

现在你已经了解了CausalQwen-0.5B的基本用法，可以：

- 查看[实验设计](/experiments/experiment_design.md)，了解完整的实验框架
- 探索[数学理论](/math/mathematical_foundations.md)，深入了解背后的原理
- 查看[架构设计](/architecture/architecture_design.md)，了解系统的设计思想
- 运行完整的消融实验来验证模型的各个组件

