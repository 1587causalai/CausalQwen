# CausalQwen: 首个原生因果语言模型

> 🏆 **MVP v0.1.0** - 核心框架已验证，测试6/6通过  
> 🎯 **分支**: `causal-mvp` - 专注因果理念验证

## 核心理念

将语言生成从"概率采样"转向"个体决策"：

```
传统LM: 文本生成 = 从P(Y|X)随机采样
CausalQwen: 文本生成 = 个体在因果律下的必然表达
```

**数学框架**: `Y = f(U, ε)` 其中U是个体选择变量，ε是外生噪声，f是普适因果机制

## 当前状态

### ✅ 已实现 (MVP v0.1.0)
- 核心架构：4个模块集成完成
- 推理模式：标准/因果/兼容三种模式
- 验证框架：完整测试套件通过
- HuggingFace兼容：继承Qwen2架构

### 🔄 进行中 (v0.2.0)  
- 数学完善：Cauchy分布数值稳定性
- 权重初始化：从真实Qwen复制权重


## 快速开始

```bash
# 安装
git clone -b causal-mvp https://github.com/yourusername/CausalQwen.git
cd CausalQwen
pip install torch transformers

# 验证
python scripts/check_everything_works.py
# 期望: 🎉 所有测试通过！MVP框架基础功能正常
```

### 基础使用

```python
from src.causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config

# 小型配置
config = CausalQwen2Config(vocab_size=1000, hidden_size=128, ...)
model = CausalQwenMVPForCausalLM(config)

# 三种推理模式
input_ids = torch.randint(0, 1000, (1, 10))
standard_out = model.inference(input_ids, mode='standard')
causal_out = model.inference(input_ids, mode='causal')
compatible_out = model.inference(input_ids, mode='compatible')
```

## 项目结构

```
CausalQwen/
├── src/causal_qwen_mvp/           # 核心MVP实现
├── scripts/check_everything_works.py  # 框架测试 (6/6通过)
├── docs/mvp_design.md             # MVP设计文档
├── design-docs/causal_qwen.md  # 完整理论文档
└── archive/                       # 已清理的旧代码
```

## 核心文档

- [完整设计文档](design-docs/causal_qwen.md) - 937行理论与实现细节
- [MVP设计](docs/mvp_design.md) - 当前阶段范围与标准  
- [实现指南](docs/implementation_plan.md) - 技术实现路线

## 贡献

欢迎参与！当前重点：数值稳定性优化、权重初始化完善。

## 许可证

MIT License

---

🎯 **核心洞察**: 文本不是随机采样的结果，而是特定个体在因果律下的必然表达

