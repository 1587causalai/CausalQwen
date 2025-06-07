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

### 使用模拟特征网络

最简单的方式是使用模拟特征网络创建一个因果语言模型：

```python
import torch
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig

# 创建模型配置
config = CausalLMConfig(
    vocab_size=1000,  # 词汇表大小
    hidden_size=768,  # 隐藏层大小
    causal_dim=64,    # 因果状态维度
    use_mock_feature_network=True  # 使用模拟特征网络
)

# 创建模型
model = CausalLanguageModel(config)

# 查看模型结构
print(model)
```

### 使用Qwen-0.5B作为特征网络

如果你想使用Qwen-0.5B作为特征网络，可以这样做：

```python
import torch
from transformers import AutoModel, AutoTokenizer
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.models.feature_network import QwenFeatureNetwork

# 加载Qwen-0.5B模型
qwen_model = AutoModel.from_pretrained("Qwen/Qwen-0.5B")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-0.5B")

# 创建模型配置
config = CausalLMConfig(
    vocab_size=len(qwen_tokenizer),  # 使用Qwen词汇表大小
    hidden_size=qwen_model.config.hidden_size,  # 使用Qwen隐藏层大小
    causal_dim=64,  # 因果状态维度
    use_mock_feature_network=False,  # 不使用模拟特征网络
    num_token_id=qwen_tokenizer.convert_tokens_to_ids('<NUM>')  # 数值词元ID
)

# 创建模型
model = CausalLanguageModel(config)

# 创建Qwen特征网络
feature_network = QwenFeatureNetwork(qwen_model)

# 替换模型的特征网络
model.feature_network = feature_network

# 设置分词器
model.tokenizer = qwen_tokenizer
```

## 模型推理

### 基本推理

```python
# 准备输入
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 批大小为1，序列长度为5的输入

# 模型前向传播
outputs = model(input_ids)

# 查看输出
print("因果状态位置参数:", outputs['causal_loc'].shape)
print("因果状态尺度参数:", outputs['causal_scale'].shape)
print("分类概率:", outputs['cls_probs'].shape)
print("回归预测:", outputs['reg_loc'].shape)
print("回归不确定性:", outputs['reg_scale'].shape)

# 获取预测结果
predictions = model.predict(input_ids)
print("预测的词元:", predictions['cls_pred'])
print("预测的数值:", predictions['reg_pred'])
```

### 处理数值输入

如果输入包含数值，可以使用`numerical_values`参数：

```python
# 准备输入
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
numerical_values = torch.tensor([[0.0, 0.0, 10.5, 0.0, 0.0]])  # 第3个位置是数值10.5

# 模型前向传播
outputs = model(input_ids, numerical_values)

# 获取预测结果
predictions = model.predict(input_ids, numerical_values)
```

## 模型训练

### 准备数据

```python
from torch.utils.data import DataLoader, TensorDataset

# 准备训练数据
input_ids = torch.randint(0, 1000, (100, 10))  # 100个样本，每个长度为10
numerical_values = torch.zeros_like(input_ids, dtype=torch.float32)
targets = torch.randint(0, 1000, (100,))  # 目标词元
target_values = torch.rand(100)  # 目标数值

# 创建数据集
dataset = TensorDataset(input_ids, numerical_values, targets, target_values)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### 训练循环

```python
import torch.optim as optim

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        input_ids, numerical_values, targets, target_values = batch
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids, numerical_values)
        
        # 计算损失
        loss = model.compute_loss(outputs, targets, target_values)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

## 模型评估

```python
from src.evaluate import evaluate_model

# 准备测试数据
test_input_ids = torch.randint(0, 1000, (50, 10))
test_numerical_values = torch.zeros_like(test_input_ids, dtype=torch.float32)
test_targets = torch.randint(0, 1000, (50,))
test_target_values = torch.rand(50)

test_dataset = TensorDataset(test_input_ids, test_numerical_values, test_targets, test_target_values)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# 评估模型
model.eval()
metrics = evaluate_model(model, test_dataloader)

# 打印评估指标
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

## 可视化

```python
from src.utils.visualization import (
    visualize_causal_state,
    visualize_decision_boundary,
    visualize_regression_performance
)

# 可视化因果状态
sample_input = {'input_ids': test_input_ids[:1]}
visualize_causal_state(model, sample_input, output_dir='results')

# 可视化决策边界
visualize_decision_boundary(model, output_dir='results')

# 可视化回归性能
predictions = model.predict(test_input_ids, test_numerical_values)
visualize_regression_performance(
    predictions['reg_pred'].numpy(),
    test_target_values.numpy(),
    outputs['reg_scale'].numpy(),
    output_dir='results'
)
```

## 保存和加载模型

```python
# 保存模型
torch.save(model.state_dict(), 'causal_lm_model.pth')

# 加载模型
new_model = CausalLanguageModel(config)
new_model.load_state_dict(torch.load('causal_lm_model.pth'))
```

## 下一步

现在你已经了解了CausalQwen-0.5B的基本用法，可以：

- 查看[API参考](/guide/api_reference.md)，了解详细的API文档
- 探索[示例](/guide/examples.md)，学习更多高级用法
- 阅读[数学理论](/math/mathematical_foundations.md)，深入了解背后的原理
- 查看[架构设计](/architecture/architecture_design.md)，了解系统的设计思想

