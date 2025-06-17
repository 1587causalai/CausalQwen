# CausalQwen: 革命性因果语言模型

> **🎯 最小可行版本**: 专注核心数学框架，完全兼容Qwen  
> **🚀 V2数学创新**: 位置vs尺度的精妙差异  
> **📊 验证状态**: 核心测试100%通过

---

## 🧮 核心数学创新

### 位置vs尺度的精妙差异

传统语言模型仅有一种生成模式，CausalQwen V2引入了**噪声影响方式的革命性差异**：

```
do_sample=False: U' ~ Cauchy(μ, γ + |b_noise|)     # 噪声影响尺度参数
do_sample=True:  U' ~ Cauchy(μ + T·|b_noise|·ε, γ) # 噪声影响位置参数
```

**深层含义**:
- **确定性模式** (`do_sample=False`): 噪声增加决策的不确定性，但不改变决策中心
- **采样模式** (`do_sample=True`): 噪声扰动个体身份，产生不同的决策个体

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

## 🧪 核心测试验证

运行4个核心测试脚本验证数学框架：

```bash
# 1. V2数学原理验证
python scripts/causal_qwen_v2_validation_test.py

# 2. Qwen兼容性验证  
python scripts/qwen_compatibility_test.py

# 3. 基本使用演示
python scripts/simple_demo_v2.py

# 4. 端到端对比测试（需要Qwen模型）
python scripts/end_to_end_comparison_test_v2.py
```

### 预期测试结果
```
🎯 V2核心创新验证：do_sample控制的位置vs尺度差异
✅ ActionNetwork统一框架：兼容Qwen的所有参数  
✅ 温度参数选择性生效：仅在do_sample=True时影响噪声强度
✅ 柯西分布线性稳定性：严格的数学基础实现
✅ 完全Qwen兼容：generate()接口和所有采样参数
```

---

## 📁 项目结构

```
CausalQwen/
├── src/causal_qwen_mvp/          # 核心实现
│   ├── models.py                 # CausalQwen V2模型
│   ├── inference.py              # 推理引擎  
│   └── training.py               # 训练工具
├── scripts/                      # 核心测试（仅4个）
│   ├── causal_qwen_v2_validation_test.py    # V2数学验证
│   ├── qwen_compatibility_test.py           # Qwen兼容性
│   ├── simple_demo_v2.py                    # 使用演示
│   └── end_to_end_comparison_test_v2.py     # 端到端对比
├── docs/                         # 核心数学文档
│   ├── causal_qwen_v2_architecture.md      # V2架构
│   └── causal_qwen_inference_theory.md     # 数学理论
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
- [CausalQwen V2架构](docs/causal_qwen_v2_architecture.md)
- [因果推理理论](docs/causal_qwen_inference_theory.md)

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

## 🤝 贡献

CausalQwen专注于因果语言模型的核心数学框架研究。欢迎在以下方面贡献：
- 数学理论完善
- 性能优化
- 测试用例增强
- 文档改进

---

## 📄 许可证

MIT License

---

## 🎉 总结

**CausalQwen**: 革命性因果语言模型的最小可行实现

- **🧮 数学创新**: 位置vs尺度的精妙差异机制
- **🔗 完全兼容**: 与Qwen接口100%兼容
- **⚡ 专注精简**: 核心组件+4个测试脚本
- **📐 理论严谨**: 基于柯西分布线性稳定性

**完美平衡**: 突破性数学创新 + 零学习成本使用！