# 消融实验 (Ablation Studies) - 2024更新版

本目录包含基于官方基准测试协议的CausalEngine消融实验，重点验证**固定噪声vs自适应噪声学习**的核心假设。

## 🧪 核心实验设计

基于基准测试协议 (`causal_engine/misc/benchmark_strategy.md`) 的科学对照实验：

### 实验控制的优雅性
通过简单的 `b_noise.requires_grad` 布尔开关实现严格的实验控制：
- **实验组A**: `requires_grad = False` (固定噪声)
- **实验组B**: `requires_grad = True` (自适应噪声学习)
- **对照组**: 传统MLP基准

### 核心假设
> **"让模型自主学习全局环境噪声"相比固定噪声能够带来性能提升**

## 📊 实验架构

### 三层对比框架
```
传统MLP基准 ←→ CausalEngine(固定噪声) ←→ CausalEngine(自适应噪声)
     ↓                    ↓                        ↓
  标准损失函数        b_noise固定值            b_noise可学习参数
```

### 固定噪声强度测试
- 测试噪声值: `[0.01, 0.05, 0.1, 0.2, 0.5]`
- 寻找最优固定噪声区间
- 分析噪声强度-性能曲线

### 自适应噪声学习
- 初始值: `0.1`
- 学习过程: 与主网络同时优化
- 分析学到的噪声值合理性

## 🎯 实验文件

### 主要实验
- **`fixed_vs_adaptive_noise_study.py`**: 🔥**核心消融实验** (基于2024基准协议)
- `comprehensive_comparison.py`: 传统三组对比实验 (遗留)

### 支持文件
- `ABLATION_UPDATE.md`: 消融实验设计更新说明
- 生成的报告和可视化文件

## 🔬 科学方法论

### 标准化配置 (遵循基准协议)
- **优化器**: AdamW
- **学习率**: 1e-4 (线性warm-up + cosine decay)
- **权重衰减**: 0.01
- **早停**: patience=10
- **随机种子**: 42 (确保可复现性)

### 评估指标
- **分类任务**: Accuracy, Precision, Recall, F1-Score
- **回归任务**: R², MAE, RMSE, MdAE
- **噪声分析**: 学习曲线，收敛值，合理性检验

## 🚀 快速开始

### 运行核心消融实验
```bash
# 完整的固定vs自适应噪声实验
python fixed_vs_adaptive_noise_study.py
```

### 查看结果
```bash
# 可视化结果
open fixed_vs_adaptive_noise_results.png

# 详细报告
cat ablation_report.md
```

## 📈 预期发现

### 理论预期
1. **噪声敏感性**: 性能对噪声强度呈非线性关系
2. **自适应优势**: 学到的噪声值接近或优于最佳固定值
3. **任务特异性**: 不同任务可能需要不同的最优噪声

### 科学验证
- **假设验证**: 自适应学习 > 最佳固定噪声
- **机制理解**: 噪声参数的因果解释
- **方法论**: 布尔开关控制的实验优雅性

## 🎯 实验价值

### 学术贡献
- 首次系统性验证CausalEngine噪声机制
- 建立因果AI消融实验的标准范式
- 为理论假设提供实证支持

### 工程意义
- 指导噪声参数的实际配置
- 验证自适应学习的实用价值
- 为模型优化提供科学依据

## 📚 相关资源

- 🧪 **基准协议**: `causal_engine/misc/benchmark_strategy.md`
- 📐 **数学理论**: `causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md`
- 🎯 **应用示例**: `tutorials/01_classification/`, `tutorials/02_regression/`
- 🏗️ **理论基础**: `tutorials/00_getting_started/theoretical_foundations.py`

---

🔬 **通过严谨的科学实验，验证CausalEngine"自主学习全局噪声"的核心假设！**