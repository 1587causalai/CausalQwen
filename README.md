# CausalQwen-0.5B: 因果语言模型架构

## 项目概述

CausalQwen-0.5B 是一个简化的因果语言模型架构，旨在将标准的大语言模型（如Qwen-0.5B）改造为一个**因果语言模型**。该项目实现了一个理论完备且路径清晰的架构，将LLM强大的符号推理能力与结构化的数值因果推断框架相结合。

本项目是**V3"推断-行动"范式**的直接体现，优先考虑工程实现的简洁性和因果原生设计。每一个组件都经过精心简化，以便能够快速验证其核心理论。

## 核心特点

1. **简化的因果语言模型架构**：实现了最简单的因果语言模型架构，使用模拟器（mockers）替代复杂组件，遵循"非必要勿增实体"的原则。

2. **可扩展的代码架构**：设计使得后续可以轻松将标准大语言模型（如Qwen-0.5B）改造为因果语言模型。

3. **详细的理论说明**：包含完整的数学理论说明、架构文档和代码说明。

4. **清晰的实验设计**：提供了验证模型性能的实验设计和分析框架。

5. **组织良好的文档**：所有文档通过docsify网站进行组织，方便查阅和分享。

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+（如果使用GPU）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/causal-lm-project.git
cd causal-lm-project

# 安装依赖
pip install -e .
```

### 验证安装

```bash
# 运行测试
pytest tests/

# 运行示例
python examples/basic_example.py
```

## 快速开始

### 创建模型

```python
import torch
from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.tokenizer import QwenTokenizerWrapper

# 创建分词器
tokenizer = QwenTokenizerWrapper(model_path="~/models/Qwen2.5-0.5B", use_real_tokenizer=True)

# 创建模型配置
config = CausalLMConfig(
    vocab_size=tokenizer.vocab_size,
    num_token_id=tokenizer.num_token_id,
    hidden_size=896,  # For Qwen-0.5B
    causal_dim=64,
    use_real_qwen=True,
    qwen_model_path="~/models/Qwen2.5-0.5B"
)

# 创建模型
model = CausalLanguageModel(config)

# 准备输入
texts = ["The price is 42.5 dollars."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 模型前向传播
outputs = model(inputs['input_ids'], inputs['numerical_values'], inputs['attention_mask'])

# 获取预测结果
predictions = model.predict(inputs['input_ids'], inputs['numerical_values'], inputs['attention_mask'])
print(f"预测的词元: {predictions['cls_pred']}")
print(f"预测的数值: {predictions['reg_pred']}")
```

## 项目结构

```
causal-lm-project/
├── src/                  # 源代码
│   ├── models/           # 模型实现
│   │   ├── causal_lm.py          # 因果语言模型
│   │   ├── feature_network.py    # 特征网络
│   │   ├── abduction_network.py  # 推断网络
│   │   └── action_network.py     # 行动网络
│   ├── data/             # 数据处理
│   │   ├── tokenizer.py          # 分词器
│   │   ├── synthetic.py          # 合成数据生成
│   │   └── evaluation_data.py    # 评估数据集
│   ├── training/         # 训练模块
│   │   └── trainer.py            # 训练器
│   ├── evaluation/       # 评估模块
│   │   └── evaluator.py          # 评估器
│   ├── utils/            # 工具函数
│   │   ├── distributions.py      # 分布工具
│   │   ├── losses.py             # 损失函数
│   │   └── metrics.py            # 评估指标
│   ├── visualization/    # 可视化工具
│   │   └── plotter.py            # 绘图工具
│   └── run_experiments.py        # 统一实验运行器
├── docs/                 # 文档
│   ├── math/             # 数学理论
│   ├── architecture/     # 架构设计
│   ├── experiments/      # 实验说明
│   └── guide/            # 使用指南
├── tests/                # 测试代码
├── examples/             # 示例代码
└── results/              # 实验结果
```

## 核心哲学

本项目基于几个区别于标准LLM的根本性原则：

1. **"推断-行动"范式**：模型首先推断潜在高维因果表征的概率分布，然后从该分布中采样进行预测。

2. **柯西分布**：使用柯西分布为个体因果表征建模，其极重尾特性确保了即使是极小概率的个体因果表征也拥有非零的概率密度。

3. **`<NUM>`词元**：引入特殊词元作为统一任务的桥梁，使模型能够将"是否应该输出数值"视为标准的"下一词元预测"问题。

## 文档网站

项目包含一个完整的docsify网站，提供详细的文档：

```bash
# 启动文档网站
cd docs
docsify serve .
```

然后在浏览器中访问 http://localhost:3000 查看文档。

文档网站包含以下内容：

- **项目概述**：项目的背景、目标和特点
- **数学理论**：详细的数学推导和理论基础
- **架构设计**：系统架构和组件设计
- **代码实现**：代码结构和实现细节
- **实验设计**：实验方法和结果分析
- **使用指南**：安装、配置和使用说明
- **API参考**：详细的API文档
- **常见问题**：常见问题解答
- **贡献指南**：如何参与项目开发

## 实验运行

### 基础实验

### 快速训练

```bash
python scripts/quick_train.py --use_real_qwen --num_epochs 5 --batch_size 8 --num_samples 500
```

或者：

```bash
# 运行基本实验（验证基础功能）
python src/run_experiments.py basic

# 运行综合实验（多数据集评估）
python src/run_experiments.py comprehensive

# 运行对比实验（超参数敏感性分析）
python src/run_experiments.py comparison

# 运行消融实验（架构组件贡献度验证）
python src/run_experiments.py ablation
```

### 使用真实 Qwen2.5-0.5B 模型

确保您已经下载了 Qwen2.5-0.5B 模型到本地：

```bash
# 指定 Qwen 模型路径
python src/run_experiments.py ablation --qwen_model_path ~/models/Qwen2.5-0.5B

# 调整训练参数
python src/run_experiments.py basic --epochs 5 --batch_size 8 --num_samples 500

# 只运行评估，跳过训练
python src/run_experiments.py comprehensive --no_train
```

### 实验参数说明

- `--qwen_model_path`: Qwen 模型路径 (默认: ~/models/Qwen2.5-0.5B)
- `--hidden_size`: 隐藏层大小 (默认: 896，匹配 Qwen-0.5B)
- `--causal_dim`: 个体因果表征维度 (默认: 64)
- `--epochs`: 训练轮数 (默认: 10)
- `--batch_size`: 批处理大小 (默认: 16)
- `--num_samples`: 训练样本数量 (默认: 1000)
- `--lr`: 学习率 (默认: 1e-4)
- `--no_train`: 跳过训练，只运行评估

### 生成实验图表

```bash
# 对消融实验结果生成对比图表
python src/visualization/plotter.py results/ablation_20231208_143000/

# 对超参数对比实验结果生成图表
python src/visualization/plotter.py results/comparison_20231208_143000/
```

## 实验结果

实验结果将保存在 `results/` 目录下，包含：

- `results.json`: 详细的评估指标数据
- `model_*.pth`: 训练好的模型权重
- `*.png`: 自动生成的对比图表（消融和对比实验）

关键评估指标包括：
- **分类指标**: `cls_accuracy`, `cls_f1`, `cls_precision`, `cls_recall`
- **回归指标**: `reg_mse`, `reg_mae`
- **校准指标**: `calib_ece`（分类校准）, `reg_picp`（回归校准）

## 贡献

欢迎贡献代码、报告问题或提出改进建议！请查看[贡献指南](./docs/contributing.md)了解更多信息。

## 许可证

MIT

## 联系方式

如果你有任何问题或需要帮助，可以在GitHub上提交Issue或联系项目维护者。

### 本地预览

```bash
cd docs
docsify serve
```

