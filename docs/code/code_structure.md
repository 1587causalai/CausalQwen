# 代码结构

本文档介绍CausalQwen-0.5B项目的代码结构和实现细节，帮助开发者理解项目的组织方式和各个模块的职责。

## 目录结构

项目的主要目录结构如下：

```
causal-lm-project/
├── docs/                  # 文档
│   ├── architecture/      # 架构设计文档
│   ├── experiments/       # 实验设计文档
│   └── math/              # 数学理论文档
├── examples/              # 示例代码
│   └── basic_example.py   # 基本使用示例
├── src/                   # 源代码
│   ├── data/              # 数据处理模块
│   │   ├── dataset.py     # 数据集类
│   │   ├── synthetic.py   # 合成数据生成
│   │   └── tokenizer.py   # 分词器
│   ├── models/            # 模型定义
│   │   ├── abduction_network.py  # 推断网络
│   │   ├── action_network.py     # 行动网络
│   │   ├── causal_lm.py          # 因果语言模型
│   │   └── feature_network.py    # 特征网络
│   ├── utils/             # 工具函数
│   │   ├── distributions.py  # 分布工具
│   │   ├── losses.py         # 损失函数
│   │   ├── metrics.py        # 评估指标
│   │   └── visualization.py  # 可视化工具
│   ├── evaluate.py        # 评估脚本
│   ├── run_experiments.py # 实验运行脚本
│   └── train.py           # 训练脚本
└── tests/                 # 测试代码
    └── test_causal_lm.py  # 模型测试
```

## 核心模块

### 1. 因果语言模型 (`src/models/causal_lm.py`)

因果语言模型是项目的核心，它整合了特征网络、推断网络和行动网络，实现了完整的因果语言模型架构。

主要类和函数：

- `CausalLMConfig`：模型配置类，包含各种超参数和设置
- `CausalLanguageModel`：因果语言模型的主类，实现了模型的前向传播、损失计算和预测功能

关键方法：

- `forward`：模型的前向传播，从输入到输出的完整流程
- `compute_loss`：计算模型的损失函数
- `predict`：生成预测结果

### 2. 特征网络 (`src/models/feature_network.py`)

特征网络负责从输入提取特征表示，是模型的第一个组件。

主要类：

- `FeatureNetwork`：特征网络的基类，定义了接口
- `MockFeatureNetwork`：模拟特征网络，用于测试和验证
- `QwenFeatureNetwork`：基于Qwen-0.5B的特征网络

### 3. 推断网络 (`src/models/abduction_network.py`)

归因推断网络负责从特征表示推断个体因果表征分布，是模型的第二个组件。

主要类：

- `AbductionNetwork`：归因推断网络类，实现了从特征到个体因果表征分布的映射

关键方法：

- `forward`：从特征到个体因果表征分布的映射
- `sample`：从个体因果表征分布中采样

### 4. 行动网络 (`src/models/action_network.py`)

行动网络负责基于个体因果表征分布做出决策，是模型的第三个组件。

主要类：

- `ActionNetwork`：行动网络类，包含了分类头和回归头

关键方法：

- `forward`：从个体因果表征分布到决策的映射
- `classify`：生成分类预测
- `regress`：生成回归预测

### 5. 分布工具 (`src/utils/distributions.py`)

分布工具提供了处理柯西分布的各种函数，支持模型中的不确定性表示和传播。

主要函数：

- `cauchy_logpdf`：计算柯西分布的对数概率密度
- `cauchy_sample`：从柯西分布中采样
- `cauchy_cdf`：计算柯西分布的累积分布函数
- `linear_transform_cauchy`：对柯西分布进行线性变换

### 6. 损失函数 (`src/utils/losses.py`)

损失函数模块实现了模型的训练目标，包括OvR分类损失和门控回归损失。

主要函数：

- `ovr_classification_loss`：计算One-vs-Rest分类损失
- `gated_regression_loss`：计算门控回归损失
- `combined_loss`：组合分类损失和回归损失

### 7. 数据处理 (`src/data/`)

数据处理模块提供了数据加载、预处理和合成数据生成的功能。

主要类和函数：

- `NumAwareTokenizer`：数值感知分词器，能够识别和处理文本中的数值
- `TextWithNumbersGenerator`：文本数值混合数据生成器
- `SyntheticDataset`：合成数据集类

### 8. 评估和实验 (`src/evaluate.py`, `src/run_experiments.py`)

评估和实验模块提供了模型评估和实验运行的功能。

主要函数：

- `evaluate_model`：评估模型性能
- `run_comprehensive_evaluation`：运行全面评估
- `compare_models`：比较不同模型的性能

## 代码示例

### 创建和使用模型

```python
import torch
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig

# 创建模型配置
config = CausalLMConfig(
    vocab_size=1000,
    hidden_size=768,
    causal_dim=64,
    use_mock_feature_network=True
)

# 创建模型
model = CausalLanguageModel(config)

# 准备输入
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# 模型前向传播
outputs = model(input_ids)

# 获取预测结果
predictions = model.predict(input_ids)
print(f"预测的词元: {predictions['cls_pred']}")
print(f"预测的数值: {predictions['reg_pred']}")
```

### 计算损失

```python
# 准备目标
targets = torch.tensor([42])  # 目标词元
target_values = torch.tensor([3.14])  # 目标数值

# 计算损失
loss = model.compute_loss(outputs, targets, target_values)
print(f"损失: {loss.item()}")
```

### 训练模型

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 准备训练数据
input_ids = torch.randint(0, 1000, (100, 10))
numerical_values = torch.zeros_like(input_ids, dtype=torch.float32)
targets = torch.randint(0, 1000, (100,))
target_values = torch.rand(100)

# 创建数据集和数据加载器
dataset = TensorDataset(input_ids, numerical_values, targets, target_values)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        input_ids, numerical_values, targets, target_values = batch
        
        optimizer.zero_grad()
        outputs = model(input_ids, numerical_values)
        loss = model.compute_loss(outputs, targets, target_values)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

## 扩展和定制

### 自定义特征网络

你可以通过继承`FeatureNetwork`基类来创建自定义的特征网络：

```python
from src.models.feature_network import FeatureNetwork

class CustomFeatureNetwork(FeatureNetwork):
    def __init__(self, config):
        super().__init__()
        # 初始化自定义网络
        self.hidden_size = config.hidden_size
        # ...
    
    def forward(self, input_ids, numerical_values=None, attention_mask=None):
        # 实现自定义的前向传播
        # ...
        return features
```

### 自定义归因推断网络

你可以通过修改`AbductionNetwork`类来创建自定义的归因推断网络：

```python
from src.models.abduction_network import AbductionNetwork

class CustomAbductionNetwork(AbductionNetwork):
    def __init__(self, hidden_size, causal_dim):
        super().__init__(hidden_size, causal_dim)
        # 自定义初始化
        # ...

    def forward(self, features):
        # 在这里实现你自己的前向传播逻辑
        x = self.custom_layer(features)
        return super().forward(x)
```

### 自定义行动网络

你可以通过修改`ActionNetwork`类来创建自定义的行动网络：

```python
from src.models.action_network import ActionNetwork

class CustomActionNetwork(ActionNetwork):
    def __init__(self, config):
        super().__init__(config)
        # 添加或修改网络层
        # ...
    
    def forward(self, causal_loc, causal_scale):
        # 实现自定义的前向传播
        # ...
        return outputs
```

## 最佳实践

### 代码风格

项目遵循以下代码风格：

- 使用类型注解提高代码可读性
- 为所有类和函数提供文档字符串
- 使用一致的命名约定（小写下划线命名法）
- 遵循PEP 8风格指南

### 测试

项目包含全面的测试套件，确保代码的正确性和稳定性：

- 使用pytest运行测试：`pytest tests/`
- 每个核心组件都有对应的测试
- 测试覆盖正常情况和边缘情况

### 性能优化

为了提高性能，项目采用了以下优化策略：

- 使用向量化操作代替循环
- 避免不必要的内存复制
- 使用无采样训练，减少计算开销
- 支持批处理，提高并行度

## 常见问题

### Q: 如何添加新的分布类型？

A: 你可以在`src/utils/distributions.py`中添加新的分布函数，然后在`CausalLMConfig`中添加一个配置选项，最后在`归因推断网络 (AbductionNetwork)` 和`ActionNetwork`中添加对应的处理逻辑。

### Q: 如何支持多GPU训练？

A: 你可以使用PyTorch的`DataParallel`或`DistributedDataParallel`来支持多GPU训练：

```python
model = torch.nn.DataParallel(model)
```

### Q: 如何处理超长序列？

A: 你可以使用滑动窗口或分块处理的方式来处理超长序列，或者修改特征网络以支持更长的序列。

## 下一步

- 查看[API参考](/guide/api_reference.md)，了解详细的API文档
- 探索[示例](/guide/examples.md)，学习更多高级用法
- 阅读[架构设计](/architecture/architecture_design.md)，了解系统的设计思想

