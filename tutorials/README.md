# CausalEngine 实际应用教程

欢迎来到 CausalEngine 实际应用教程！这个教程集合通过具体的例子展示了 CausalEngine 在各种机器学习任务中的强大能力。

## 🎯 教程目标

通过这些教程，你将学会：
- ✅ 如何在真实数据集上使用 CausalEngine
- ✅ 理解 CausalEngine 相比传统方法的优势
- ✅ 掌握三种推理模式的使用场景
- ✅ 学会配置不同的激活模式（分类、回归、有序分类）
- ✅ 实现多任务学习和混合输出
- ✅ 评估和可视化模型性能

## 📚 教程结构

### 🟢 基础应用教程（单任务）

#### [01. 分类任务：情感分析](01_classification/)
- **📁 文件**: `sentiment_analysis.py`
- **🎯 任务**: 电影评论情感分类（正面/负面）
- **💡 重点展示**:
  - CausalEngine 分类激活函数
  - OvR (One-vs-Rest) 概率计算
  - 不确定性量化能力
  - 三种推理模式对比
- **📊 数据**: 模拟 IMDb 电影评论 (TF-IDF 特征)
- **⭐ 亮点**: 与传统 Logistic Regression 的性能对比

#### [02. 回归任务：房价预测](02_regression/)
- **📁 文件**: `house_price_prediction.py`
- **🎯 任务**: California Housing 房价预测
- **💡 重点展示**:
  - CausalEngine 回归激活函数
  - 柯西分布的异常值鲁棒性
  - 预测区间和不确定性量化
  - Cauchy NLL 损失函数
- **📊 数据**: California Housing Dataset
- **⭐ 亮点**: 90% 预测区间覆盖率分析

#### [03. 有序分类：评分预测](03_ordinal/)
- **📁 文件**: `rating_prediction.py`  
- **🎯 任务**: 电影星级评分预测（1-5星）
- **💡 重点展示**:
  - **新功能**: 离散有序激活 (CausalEngine v2.0.4)
  - 类别间顺序关系保持
  - 阈值自动学习机制
  - 有序分类特殊评估指标
- **📊 数据**: 模拟电影评分数据
- **⭐ 亮点**: 相邻准确率 vs 精确准确率分析

### 🔥 高级应用教程（多任务）

#### [04. 混合任务：电商评论综合分析](04_multitask/)
- **📁 文件**: `ecommerce_analysis.py`
- **🎯 任务**: 同时预测情感(分类) + 评分(有序) + 有用性(回归)
- **💡 重点展示**:
  - **核心特色**: 混合激活模式
  - 共享因果表征的优势
  - 多任务协同学习效应
  - 动态任务权重平衡
- **📊 数据**: 模拟电商评论多维数据
- **⭐ 亮点**: 单个模型处理三种不同输出类型

#### [05. 语言模型：CausalQwen 应用](05_language_model/) ⏳
- **📁 文件**: `causal_qwen_demo.py`
- **🎯 任务**: 可控文本生成 + 情感控制
- **💡 重点展示**:
  - CausalQwen 完整工作流程
  - 语言模型中的因果推理
  - 生成质量 vs 多样性权衡
- **📊 数据**: 小规模中文语料
- **⭐ 亮点**: 真实语言模型应用案例

### 🔬 深度分析教程

#### [06. 对比分析：传统方法 vs CausalEngine](06_comparison/) ⏳
- **📁 文件**: `comprehensive_comparison.py`
- **🎯 目标**: 全面性能基准测试
- **💡 重点展示**:
  - 多个数据集上的系统对比
  - 收敛速度和稳定性分析
  - 计算效率对比
  - 不确定性校准质量评估
- **📊 数据**: 多个公开基准数据集
- **⭐ 亮点**: 科学严谨的对比实验

## 🚀 快速开始

### 环境准备
```bash
# 安装依赖
pip install torch transformers numpy matplotlib seaborn scikit-learn

# 克隆仓库
cd /path/to/CausalQwen
```

### 运行第一个教程
```bash
# 情感分析教程
cd tutorials/01_classification/
python sentiment_analysis.py
```

### 预期输出
每个教程都会：
1. 📊 创建/加载相应数据集
2. 🏗️ 构建 CausalEngine 模型
3. 🚀 训练并对比基线方法
4. 🔍 评估三种推理模式
5. 📈 生成详细的可视化分析
6. 📋 输出性能总结报告

## 📋 教程特色

### ✨ 统一的设计模式
每个教程遵循一致的结构：
- **数据准备**: 真实或高质量模拟数据
- **模型构建**: 清晰的 CausalEngine 配置
- **训练过程**: 完整的训练循环和验证
- **性能评估**: 多维度指标和可视化
- **对比分析**: 与传统方法的科学对比

### 🎯 重点展示的 CausalEngine 优势
1. **数学严谨性**: 基于柯西分布的解析计算
2. **不确定性量化**: 原生支持预测不确定性
3. **推理模式多样性**: 因果/标准/采样三种模式
4. **任务类型丰富**: 分类/回归/有序分类/混合任务
5. **异常值鲁棒性**: 柯西分布的天然优势
6. **参数效率**: 多任务共享表征

### 📊 可视化分析
每个教程包含丰富的可视化：
- 📈 训练曲线和收敛分析
- 📊 性能对比条形图和热力图
- 🎯 预测 vs 真实值散点图
- 📐 不确定性分布和预测区间
- 🔍 混淆矩阵和分类报告
- 🎨 共享表征空间可视化

## 🔧 自定义和扩展

### 修改数据集
每个教程都支持简单的数据集替换：
```python
# 在 create_sample_data() 或 load_and_prepare_data() 中
# 替换为你自己的数据加载逻辑
features, labels = your_data_loading_function()
```

### 调整模型架构
```python
# 修改 CausalEngine 配置
engine = CausalEngine(
    hidden_size=256,  # 增加隐藏层大小
    vocab_size=num_classes,
    activation_modes=your_activation_modes,
    # ... 其他参数
)
```

### 添加新的评估指标
```python
# 在相应的评估函数中添加新指标
def evaluate_model(predictions, targets):
    # 现有指标
    accuracy = accuracy_score(targets, predictions)
    
    # 你的新指标
    your_metric = your_metric_function(targets, predictions)
    
    return {'accuracy': accuracy, 'your_metric': your_metric}
```

## 🎓 学习路径建议

### 初学者路径
1. 🟢 **01-分类教程**: 理解基本概念和分类激活
2. 🟢 **02-回归教程**: 学习回归任务和不确定性
3. 🟢 **03-有序分类**: 掌握新的有序激活功能

### 进阶路径  
4. 🔥 **04-多任务教程**: 理解混合激活和共享表征
5. 🔥 **05-语言模型**: 学习 CausalQwen 应用
6. 🔬 **06-对比分析**: 深入理解性能优势

## 📞 获取帮助

### 常见问题
- **Q**: 为什么我的结果与教程略有不同？
- **A**: 这是正常的！随机种子和数据生成可能导致小幅差异。关注整体趋势和相对性能。

- **Q**: 如何在自己的数据上使用 CausalEngine？
- **A**: 参考教程中的数据预处理部分，确保特征标准化和标签格式正确。

- **Q**: 哪种推理模式最适合我的任务？
- **A**: 一般建议：纯因果模式用于稳定预测，标准模式平衡性能和不确定性，采样模式用于探索多样性。

### 技术支持
- 📖 查看 [CausalEngine 文档](../causal_engine/README.md)
- 🔍 阅读 [数学基础](../causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md)  
- 🏗️ 参考 [架构决策](../causal_engine/misc/ARCHITECTURE_DECISION.md)

## 🌟 贡献

我们欢迎社区贡献更多教程和改进建议！如果你：
- 🆕 开发了新的应用场景教程
- 🔧 发现了可以改进的地方
- 📊 有更好的可视化想法
- 🐛 发现了 Bug

请通过 GitHub Issues 或 Pull Requests 与我们联系。

---

**让我们一起探索 CausalEngine 的强大潜力，体验真正的因果推理在机器学习中的魅力！** 🚀✨