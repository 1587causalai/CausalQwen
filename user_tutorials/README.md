# CausalEngine 用户教程 - 轻松上手因果推理

欢迎使用 CausalEngine！这是一个专为普通用户设计的简化教程，让您能够快速上手使用因果推理引擎完成传统机器学习任务。

## 🎯 适合人群

- **数据科学初学者**：想要体验因果推理的强大能力
- **机器学习实践者**：希望在现有项目中集成因果推理
- **产品经理/分析师**：需要更准确的预测模型
- **任何对AI感兴趣的人**：想了解下一代AI技术

## 🚀 为什么选择 CausalEngine？

与传统机器学习相比，CausalEngine 提供：

- **🧠 更智能的推理**：不仅仅是模式识别，真正理解因果关系
- **🎯 更准确的预测**：基于因果机制的预测更加稳定可靠
- **🔍 更好的解释性**：清楚地知道"为什么"会有这个预测结果
- **🛡️ 更强的泛化能力**：在新环境中依然保持良好性能

## ⚡ 5分钟快速开始

```python
# 安装依赖
pip install torch scikit-learn matplotlib

# 运行第一个示例
python user_tutorials/01_quick_start/first_example.py
```

## 📚 学习路径

### 第一步：快速开始 (5分钟)
- `01_quick_start/` - 环境配置和第一个完整示例

### 第二步：分类任务 (15分钟)
- `02_classification/synthetic_data.py` - 使用合成数据学习分类
- `02_classification/iris_dataset.py` - 经典鸢尾花数据集
- `02_classification/tips_and_tricks.py` - 分类任务最佳实践

### 第三步：回归任务 (15分钟)
- `03_regression/synthetic_data.py` - 使用合成数据学习回归
- `03_regression/boston_housing.py` - 经典房价预测
- `03_regression/tips_and_tricks.py` - 回归任务最佳实践

### 第四步：真实应用与基准测试 (30分钟)
- `04_real_world_examples/classification_benchmark.py` - 4个真实分类数据集基准测试
- `04_real_world_examples/regression_benchmark.py` - 4个真实回归数据集基准测试
- `04_real_world_examples/README.md` - 基准测试详细说明

## 🎨 教程特色

### 🔰 零门槛入门
- 不需要了解复杂的数学理论
- 不需要深入理解算法细节
- 专注于如何使用和应用

### 📝 注释详细
每个示例都包含：
- 详细的中文注释
- 参数说明和使用建议
- 常见问题和解决方案

### 🎯 即用即得
- 提供预设的最佳配置
- 只需要替换您的数据
- 复制粘贴即可运行

### 🔧 实用工具
- 简化的模型接口
- 自动数据预处理
- 结果可视化工具

## 🏆 基准测试

想要客观评估 CausalEngine 的性能？我们提供了全面的基准测试：

### 📊 运行基准测试
```bash
# 从项目根目录运行
python run_benchmarks.py

# 或者单独运行
cd user_tutorials
python 04_real_world_examples/classification_benchmark.py
python 04_real_world_examples/regression_benchmark.py
```

**包含的测试**：
- 4 个真实分类数据集 (Adult Census, Bank Marketing, Credit Default, Mushroom)
- 4 个真实回归数据集 (Bike Sharing, Wine Quality, Ames Housing, California Housing)
- 对比 5 种传统机器学习方法
- 自动生成性能报告和可视化图表

**基准测试特色**：
- ✅ 客观公正的第三方数据集
- ✅ 业界标准的评估指标
- ✅ 自动化的结果分析
- ✅ 精美的可视化报告

## 🆚 与传统方法对比

| 特性 | 传统机器学习 | CausalEngine |
|------|-------------|-------------|
| 学习方式 | 模式匹配 | 因果推理 |
| 预测稳定性 | 依赖数据分布 | 基于因果机制 |
| 解释能力 | 黑盒模型 | 清晰因果链 |
| 泛化能力 | 容易过拟合 | 更好泛化 |
| 不确定性 | 难以量化 | 精确量化 |

## 🎯 典型使用场景

### 商业应用
- **客户行为预测**：预测客户购买意向、流失风险
- **市场分析**：理解影响销售的真正因素
- **风险评估**：金融风控、保险定价

### 科研应用
- **医疗诊断**：基于症状预测疾病
- **教育评估**：学生成绩预测和干预建议
- **环境监测**：气候变化影响预测

## 🤔 常见问题

### Q: 我需要很强的数学背景吗？
A: 不需要！这个教程专门为非技术用户设计，重点是如何使用，而不是算法原理。

### Q: 与 scikit-learn 相比有什么优势？
A: CausalEngine 理解因果关系，而 scikit-learn 只是找模式。在数据分布变化时，CausalEngine 更稳定。

### Q: 能处理哪些类型的数据？
A: 支持表格数据（如 CSV）、数值特征、分类特征等常见的机器学习数据格式。

### Q: 性能如何？
A: 通常比传统方法准确率更高，尤其在数据量不大或存在分布偏移的情况下。可以运行我们的基准测试脚本客观验证：
```bash
python run_benchmarks.py
```

### Q: 如何验证 CausalEngine 确实比传统方法好？
A: 我们提供了完整的基准测试，在多个真实数据集上与 5 种主流机器学习方法进行公平对比，包括详细的性能分析报告。

## 🔗 进阶学习

如果您想深入了解技术细节：
- 开发者教程：`tutorials/` 目录
- 数学原理：`causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md`
- 技术文档：`causal_engine/README.md`

## 💬 获取帮助

- 查看示例代码中的详细注释
- 阅读 `tips_and_tricks.py` 文件
- 参考 `04_real_world_examples/` 中的完整项目

---

🎉 **现在就开始您的因果推理之旅吧！**

从 `01_quick_start/first_example.py` 开始，5分钟内体验 CausalEngine 的强大功能！