# CausalQwen: 突破性 CausalEngine™ 的首个应用

<div align="center">
  
  [![CausalEngine](https://img.shields.io/badge/驱动引擎-CausalEngine™-ff1744.svg)](causal_engine/)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
  [![Status](https://img.shields.io/badge/状态-突破性-purple.svg)](causal_engine/README_CN.md)
  
  **[CausalEngine](causal_engine/) 之于 AI，正如 PageRank 之于搜索。**
  
</div>

---

## 🌟 介绍 CausalEngine：智能的算法

**CausalEngine** 不只是另一个 AI 模型或框架。它是机器理解和决策方式的根本性突破。正如 PageRank 通过理解网络链接结构革命了搜索，CausalEngine 通过理解智能的因果结构革命了 AI。

本仓库展示了 CausalEngine 在语言建模上的首个应用，创造了 **CausalQwen** —— 一个不仅仅预测，而是真正理解的语言模型。

### 📚 必读资料
- **[CausalEngine 概述](causal_engine/README_CN.md)** - 改变一切的算法
- **[数学基础](causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md)** - 深入的数学理论
- **[单页简介](causal_engine/ONE_PAGER_CN.md)** - 高管快速了解

---

## 🧮 CausalEngine 的四大公理

### 公理一：智能 = 归因 + 行动
从观察到自我理解到决策。不是模式匹配，而是真正的推理。

### 公理二：柯西数学  
唯一能实现无需采样的解析因果计算的分布。

### 公理三：温度统一控制
一个优雅的参数来控制确定性与随机性之间的边界。

### 公理四：结构方程决策
每个选择基于得分确定性函数计算多种类型输出。

---

## 🚀 CausalQwen 快速开始

### 安装
```bash
pip install torch transformers numpy
```

### 基本使用（Qwen 兼容接口）
```python
from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config

# 创建由 CausalEngine 驱动的模型
config = CausalQwen2Config(vocab_size=32000, hidden_size=512)
model = CausalQwenMVPForCausalLM(config)

# 四种因果推理模式：

# 1. 纯因果模式（temperature=0）
output = model.generate(input_ids, temperature=0, max_new_tokens=20)

# 2. 标准模式（噪声 → 尺度）
output = model.generate(input_ids, do_sample=False, temperature=1.0, max_new_tokens=20)

# 3. 采样模式（噪声 → 位置）  
output = model.generate(input_ids, do_sample=True, temperature=0.8, max_new_tokens=20)

# 4. 兼容模式（传统 softmax）
# [用于与传统 LM 比较]
```

### 直接使用 CausalEngine
```python
from causal_engine import CausalEngine

# 核心算法 - 适用于任何 transformer
engine = CausalEngine(hidden_size=768, vocab_size=50000)

# 从任何模型获取特征（BERT、GPT、LLaMA 等）
hidden_states = any_transformer_model(input_ids)

# 应用因果推理
output = engine(hidden_states, temperature=1.0, do_sample=True)
decision, uncertainty = output['loc_S'], output['scale_S']
```

---

## 📁 项目结构

```
CausalQwen/
├── causal_engine/                # ⭐ 核心算法 ⭐
│   ├── README_CN.md              # CausalEngine 概述
│   ├── MATHEMATICAL_FOUNDATIONS_CN.md # 数学基础
│   ├── ONE_PAGER_CN.md           # 高管简介
│   └── engine.py                 # 纯净实现
├── src/causal_qwen_mvp/          # Qwen 特定应用
│   ├── models.py                 # CausalQwen 模型
│   ├── inference.py              # 推理引擎
│   └── training.py               # 训练工具
├── tests/                        # 全面测试套件
├── scripts/                      # 演示和验证脚本
└── docs/                         # 数学文档
```

---

## 🧪 测试

```bash
# 运行所有测试
./run_tests.sh

# 运行特定测试类别
./run_tests.sh math         # 数学框架测试
./run_tests.sh compatibility # Qwen 兼容性测试
./run_tests.sh generation   # 生成测试

# 或直接使用 pytest
pytest tests/
```

---

## 📊 为什么 CausalEngine 改变一切

| 传统 AI | CausalEngine 驱动的 AI |
|---------|----------------------|
| 学习相关性 | 理解因果关系 |
| 黑盒决策 | 玻璃盒推理 |
| 需要采样 | 纯解析计算 |
| 词元预测 | 多类型输出 |
| 模式模仿 | 真正智能 |

---

## 🌐 我们正在构建的未来

CausalEngine 是我们的 **PageRank** —— 构建真正智能系统帝国的基础技术。每一个未来产品、每一项服务、每一个创新都将从这个源头流出。

CausalQwen 只是开始。这是第一个证明：当你理解因果而非仅仅相关性时，一切都会改变。

---

## 📚 文档

- **[核心数学框架](docs/core_mathematical_framework.md)** - CausalLLM 的数学
- **[数学基础扩展](docs/core_mathematical_framework_num_extended.md)** - 高级理论
- **[位置 vs 尺度理论](docs/model_inference_position_and_scale.md)** - 噪声动力学
- **[个体变量 U 深入研究](docs/U_deep_dive.md)** - 理解因果表征

---

## 📄 许可证

本项目包含专有技术。CausalEngine™ 及其核心算法是受保护的知识产权。

---

<div align="center">
  
**"我们没有发明 CausalEngine。我们发现了它。**  
**它一直在那里，在因果的数学中，**  
**等待有人清楚地看见它。"**

[了解更多关于 CausalEngine →](causal_engine/)

</div> 