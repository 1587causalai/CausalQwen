# CausalQwen-0.5B: 因果语言模型架构

## 项目简介

CausalQwen-0.5B是一个基于因果语言模型架构的项目，实现了最简单的因果语言模型架构，并提供了将标准大语言模型（如Qwen-0.5B）改造为因果语言模型的方法。

本项目的主要特点包括：

- **简洁的因果语言模型架构**：实现了最简单的因果语言模型架构，包含特征网络、推断网络和行动网络三个核心组件。
- **灵活的改造方案**：提供了将标准大语言模型（如Qwen-0.5B）改造为因果语言模型的详细方法。
- **详细的数学理论**：包含柯西分布、推断-行动范式、OvR分类和门控损失函数等核心概念的详细数学推导。
- **清晰的代码架构**：采用模块化设计，职责分离清晰，便于理解和扩展。
- **完整的实验框架**：提供了合成数据生成、模型验证、评估指标和可视化工具等完整的实验验证框架。

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/causal-lm-project.git
cd causal-lm-project

# 安装依赖
pip install -e .
```

### 基本使用

```python
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

# 使用模型进行预测
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
outputs = model(input_ids)
predictions = model.predict(input_ids)

print(f"预测的词元: {predictions['cls_pred']}")
print(f"预测的数值: {predictions['reg_pred']}")
```

## 项目结构

```
causal-lm-project/
├── docs/                  # 文档
│   ├── architecture/      # 架构设计文档
│   ├── experiments/       # 实验设计文档
│   └── math/              # 数学理论文档
├── examples/              # 示例代码
├── src/                   # 源代码
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   └── utils/             # 工具函数
└── tests/                 # 测试代码
```

## 核心概念

### 因果语言模型架构

因果语言模型将传统语言模型的决策过程分解为两个阶段：

1. **推断阶段**：从输入特征推断潜在的因果状态分布
2. **行动阶段**：基于因果状态分布做出决策（分类或回归）

这种分解使模型能够更好地处理不确定性，并在统一的框架下处理分类和回归任务。

### 柯西分布

柯西分布是一种重尾分布，具有以下特性：

- 无限方差，更适合表示高度不确定性
- 线性变换封闭性，便于传播不确定性
- 无需采样即可训练，提高计算效率

### OvR分类

One-vs-Rest (OvR) 分类相比传统的Softmax分类有以下优势：

- 独立的二分类决策，更灵活
- 支持多标签分类
- 每个类别有独立的不确定性估计

### 门控损失函数

门控损失函数实现了"先分类，再回归"的学习策略：

- 分类损失用于所有样本
- 回归损失仅用于数值样本
- 确保预测一致性并支持不确定性传播

## 文档导航

- [项目概述](/overview.md)：详细了解项目的背景、目标和特点
- [数学理论](/math/mathematical_foundations.md)：深入了解因果语言模型的数学基础
- [架构设计](/architecture/architecture_design.md)：了解系统的整体架构和组件设计
- [代码实现](/code/code_structure.md)：探索代码结构和实现细节
- [实验设计](/experiments/experiment_design.md)：了解实验设计和结果分析
- [使用指南](/guide/installation.md)：获取安装和使用指南

## 贡献

欢迎贡献代码、报告问题或提出改进建议！请查看[贡献指南](/contributing.md)了解更多信息。

## 许可证

本项目采用MIT许可证。详见[LICENSE](https://github.com/yourusername/causal-lm-project/blob/main/LICENSE)文件。

