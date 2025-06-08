# CausalQwen: 因果语言模型精简版

一个基于柯西分布的因果语言模型实现，专注于核心理论和关键功能。

## 核心特点

- **柯西分布建模**：使用柯西分布表示认知不确定性
- **推断-行动范式**：先推断因果状态分布，再基于分布进行预测
- **无采样训练**：利用柯西分布的线性组合性质实现高效训练
- **数值预测能力**：专门优化的数值理解和预测框架

## 项目结构

```
causal-qwen-refined/
├── src/
│   ├── models/           # 核心模型实现
│   ├── utils/            # 工具函数
│   └── data/             # 数据处理
├── docs/
│   ├── math/             # 数学理论
│   └── experiments/      # 实验设计
├── examples/             # 使用示例
└── tests/                # 测试代码
```

## 快速开始

```python
from src.models.causal_lm import CausalLanguageModel

# 创建模型
model = CausalLanguageModel(
    vocab_size=1000,
    hidden_size=512,
    causal_dim=64
)

# 训练和推理
# 详见 examples/ 目录
```


解压缩附件中的zip文件，安装依赖：`pip install -r requirements.txt`
运行数学验证：`python tests/test_math.py`
查看训练示例：`python examples/basic_training.py`
阅读核心理论：`docs/math/mathematical_foundations.md` 
阅读实验设计：`docs/experiments/experiment_design.md`


## 理论基础

详细的数学理论请参考 `docs/math/mathematical_foundations.md`

## 许可证

MIT License

