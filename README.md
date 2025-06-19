# CausalQwen: 首个原生因果大语言模型架构

> **🎯 最小可行版本**: 专注核心数学框架，完全兼容Qwen  
> **🚀 数学创新**: 位置vs尺度的精妙差异  

---

## 🧮 核心数学创新

### 模型原生因果推理能力

传统语言模型仅有一种生成模式，CausalQwen V2引入了**原生因果推理能力**：

```
do_sample=False: U' ~ Cauchy(μ, γ + |b_noise|)     # 噪声影响尺度参数
do_sample=True:  U' ~ Cauchy(μ + T·|b_noise|·ε, γ) # 噪声影响位置参数
```

**深层含义**:
- **确定性模式** (`do_sample=False`): 噪声增加决策的不确定性，但不改变决策中心，因果表征，外生噪声和结构方程共同决定结果。
- **采样模式** (`do_sample=True`): 噪声扰动个体因果表征，产生不同的决策，进一步温度参数为0时，相当于完全因果表征进行推理。

---

## 🚀 快速开始

### 安装依赖
```bash
pip install torch transformers numpy
```

### 基本使用（与Qwen完全相同）
```python
from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config

# 创建模型
config = CausalQwen2Config(vocab_size=32000, hidden_size=512)
model = CausalQwenMVPForCausalLM(config)

# 确定性生成（噪声影响尺度参数）
output = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=False
)

# 采样生成（噪声影响位置参数）
output = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

---

## 🧪 测试

### 标准化测试套件

使用pytest框架的标准化测试：

```bash
# 运行所有测试（不需要Qwen模型）
./run_tests.sh

# 运行特定测试
./run_tests.sh math         # 数学框架测试
./run_tests.sh compatibility # 兼容性测试
./run_tests.sh generation   # 生成功能测试

# 生成测试覆盖率报告
./run_tests.sh coverage

# 或直接使用pytest
pytest tests/              # 运行所有测试
pytest tests/ -m "not requires_qwen"  # 跳过需要Qwen的测试
```

### 快速验证脚本

运行核心测试脚本快速验证：

```bash
# 1. 核心数学框架验证
python scripts/test_core_math_framework.py

# 2. Qwen接口兼容性验证  
python scripts/test_qwen_interface_compatibility.py

# 3. 基本使用演示
python scripts/demo_basic_usage.py

# 4. 与原版Qwen对比测试（需要Qwen模型）
python scripts/test_vs_original_qwen.py
```


---

## 📁 项目结构

```
CausalQwen/
├── src/causal_qwen_mvp/          # 核心实现
│   ├── __init__.py               # 包初始化
│   ├── models.py                 # CausalQwen V2模型
│   ├── inference.py              # 推理引擎  
│   └── training.py               # 训练工具
├── tests/                        # 标准化测试套件
│   ├── conftest.py               # pytest配置和fixtures
│   ├── test_math_framework.py    # 核心数学框架测试
│   ├── test_compatibility.py     # Qwen接口兼容性测试
│   ├── test_generation.py        # 生成功能测试
│   ├── test_comparison.py        # 与Qwen对比测试
│   └── README.md                 # 测试说明文档
├── scripts/                      # 快速验证脚本
│   ├── test_core_math_framework.py         # 核心数学框架验证
│   ├── test_qwen_interface_compatibility.py # Qwen接口兼容性测试
│   ├── demo_basic_usage.py                 # 基本使用演示
│   ├── test_vs_original_qwen.py            # 与原版Qwen对比
│   └── TEST_INDEX.md                       # 测试说明
├── docs/                         # 核心数学文档
│   ├── core_mathematical_framework.md                  # CausalLLM 核心数学框架
│   ├── core_mathematical_framework_num_extended.md     # 同时进行分类和回归的CausalLLM 核心数学框架
│   ├── model_inference_position_and_scale.md           # 位置vs尺度理论
│   ├── init_pretraining_alignment.md                   # 预训练对齐
│   └── U_deep_dive.md                                  # U变量深入研究
├── run_tests.sh                  # 测试运行脚本
└── README.md                     # 本文档
```

---

## 🎯 核心优势

### 1. 数学严谨性
- **柯西分布线性稳定性**: 严格的数学基础
- **位置vs尺度差异**: 革命性的噪声影响机制
- **温度选择性生效**: 仅在采样模式下影响噪声强度

### 2. 完全兼容Qwen
- **零学习成本**: 使用方式与Qwen完全相同
- **无缝替换**: 可直接替代现有Qwen代码
- **完整参数支持**: do_sample, temperature, top_k, top_p等

### 3. 最小可行实现
- **核心专注**: 仅保留essential组件
- **清晰架构**: 易于理解和扩展
- **充分验证**: 核心测试100%覆盖

---

## 📚 数学理论

详细数学推导请参考：
- [核心数学框架](docs/core_mathematical_framework.md)
- [位置vs尺度理论](docs/model_inference_position_and_scale.md)

### ActionNetwork统一框架

```python
def forward(self, loc_U, scale_U, do_sample=False, temperature=1.0):
    if do_sample:
        # 采样模式：噪声影响位置参数
        epsilon = torch.tan(torch.pi * (torch.rand_like(loc_U) - 0.5))
        loc_U_noisy = loc_U + temperature * torch.abs(self.b_noise) * epsilon
        loc_S = self.lm_head(loc_U_noisy)
        scale_S = scale_U @ torch.abs(self.lm_head.weight).T
    else:
        # 确定性模式：噪声影响尺度参数  
        scale_U_noisy = scale_U + torch.abs(self.b_noise)
        loc_S = self.lm_head(loc_U)
        scale_S = scale_U_noisy @ torch.abs(self.lm_head.weight).T
    
    return loc_S, scale_S
```

---

## 📄 许可证

MIT License
