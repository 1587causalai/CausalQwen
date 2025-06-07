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

### 使用Qwen-0.5B作为特征网络

```python
from transformers import AutoModel, AutoTokenizer
from src.models.feature_network import QwenFeatureNetwork

# 加载Qwen-0.5B模型
qwen_model = AutoModel.from_pretrained("Qwen/Qwen-0.5B")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-0.5B")

# 创建Qwen特征网络
feature_network = QwenFeatureNetwork(qwen_model)

# 替换模型的特征网络
model.feature_network = feature_network
model.tokenizer = qwen_tokenizer
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
│   ├── utils/            # 工具函数
│   │   ├── distributions.py      # 分布工具
│   │   ├── losses.py             # 损失函数
│   │   ├── metrics.py            # 评估指标
│   │   └── visualization.py      # 可视化工具
│   └── data/             # 数据处理
│       ├── dataset.py            # 数据集类
│       ├── tokenizer.py          # 分词器
│       └── synthetic.py          # 合成数据生成
├── docs/                 # 文档
│   ├── math/             # 数学理论
│   ├── architecture/     # 架构设计
│   ├── experiments/      # 实验说明
│   └── api/              # API文档
├── docs-site/            # docsify网站
├── tests/                # 测试代码
├── examples/             # 示例代码
├── experiments/          # 实验代码和结果
└── data/                 # 数据集
```

## 核心哲学

本项目基于几个区别于标准LLM的根本性原则：

1. **"推断-行动"范式**：模型首先推断潜在高维因果表征的概率分布，然后从该分布中采样进行预测。

2. **柯西分布**：使用柯西分布为因果状态建模，其极重尾特性确保了即使是极小概率的因果状态也拥有非零的概率密度。

3. **`<NUM>`词元**：引入特殊词元作为统一任务的桥梁，使模型能够将"是否应该输出数值"视为标准的"下一词元预测"问题。

## 文档网站

项目包含一个完整的docsify网站，提供详细的文档：

```bash
# 启动文档网站
cd docs-site
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

### 使用模拟模型运行实验

```bash
# 运行基本实验
python src/run_experiments.py --experiment basic

# 运行综合实验
python src/run_experiments.py --experiment comprehensive

# 运行模型比较实验
python src/run_experiments.py --experiment comparison

# 运行消融实验
python src/run_experiments.py --experiment ablation
```

### 使用真实 Qwen2.5-0.5B 模型运行实验

确保您已经下载了 Qwen2.5-0.5B 模型到 `~/models/Qwen2.5-0.5B` 目录。

```bash
# 使用便捷脚本运行（推荐）
python src/run_qwen_experiments.py --experiment basic
python src/run_qwen_experiments.py --experiment comprehensive
python src/run_qwen_experiments.py --experiment comparison
python src/run_qwen_experiments.py --experiment ablation

# 或者使用原脚本的Qwen选项
python src/run_experiments.py --experiment basic --use_real_qwen --qwen_model_path ~/models/Qwen2.5-0.5B

# 如果模型在其他位置，指定路径
python src/run_qwen_experiments.py --experiment basic --qwen_model_path /path/to/your/qwen/model

# 调整批处理大小和样本数量（如果遇到内存问题）
python src/run_qwen_experiments.py --experiment basic --batch_size 8 --num_samples 200
```

### 实验参数说明

**模拟模型实验参数:**
- `--num_samples`: 评估样本数量 (默认: 1000)
- `--batch_size`: 批处理大小 (默认: 32)
- `--hidden_size`: 隐藏层大小 (默认: 768)

**真实 Qwen 模型实验参数:**
- `--num_samples`: 评估样本数量 (默认: 500，因为真实模型更慢)
- `--batch_size`: 批处理大小 (默认: 16，因为真实模型需要更多内存)
- `--hidden_size`: 隐藏层大小 (默认: 896，匹配 Qwen2.5-0.5B)
- `--qwen_model_path`: Qwen 模型路径 (默认: ~/models/Qwen2.5-0.5B)

### 实验结果对比

使用真实 Qwen 模型的实验会生成以 `qwen_` 为前缀的结果目录，例如：
- `results/qwen_basic_20231208_143000/`
- `results/qwen_comprehensive_20231208_143000/`

这样可以方便地将真实模型和模拟模型的结果进行对比分析。

## 贡献

欢迎贡献代码、报告问题或提出改进建议！请查看[贡献指南](./docs-site/contributing.md)了解更多信息。

## 许可证

MIT

## 联系方式

如果你有任何问题或需要帮助，可以在GitHub上提交Issue或联系项目维护者。

