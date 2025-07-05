# Causal-Sklearn - 因果机器学习革命

基于突破性CausalEngine™算法的scikit-learn兼容实现 - 将因果推理带入传统机器学习生态系统。

## 项目概述

Causal-Sklearn将强大的因果推理能力引入到熟悉的scikit-learn生态系统中。基于革命性的CausalEngine™算法构建，它提供了传统ML估计器的直接替代品，能够理解因果关系而不仅仅是相关性。

### 🎯 核心突破
- **因果vs相关**: 超越传统模式匹配，实现真正的因果关系理解
- **鲁棒性优势**: 在噪声和异常值存在时表现出色，远超传统方法
- **数学创新**: 以柯西分布为核心的全新数学框架
- **sklearn兼容**: 完美融入现有ML工作流，无需改变使用习惯

## 🌟 核心特性

- **🔧 Scikit-learn兼容**: 完全兼容sklearn接口，可直接替换`MLPRegressor`和`MLPClassifier`
- **🧠 因果推理**: 超越模式匹配，理解数据背后的因果关系
- **🛡️ 鲁棒性卓越**: 在标签噪声和异常值环境下性能远超传统方法
- **📊 分布预测**: 提供完整的分布输出，而非仅点估计
- **⚙️ 多模式推理**: 支持deterministic、standard、sampling等多种推理模式
- **🎯 真实世界验证**: 在加州房价等真实数据集上展现显著优势

## 🚀 快速开始

### 基础使用示例

```python
from causal_sklearn import MLPCausalRegressor, MLPCausalClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

# 回归示例
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = MLPCausalRegressor(mode='standard', random_state=42)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# 分类示例
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = MLPCausalClassifier(mode='standard', random_state=42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

### 🚀 快速验证测试

运行快速测试脚本，同时验证回归和分类性能：

```bash
# 运行完整的快速测试（回归+分类）
python scripts/quick_test_causal_engine.py

# 这个脚本将：
# 1. 生成合成数据（回归：4000样本×12特征，分类：4000样本×10特征×3类）
# 2. 在30%噪声下比较8种方法性能
# 3. 统一标准化策略确保公平比较
# 4. 生成完整的性能分析报告和可视化图表
```

**快速测试亮点**：
- ⚡ 快速验证：几分钟内完成全面测试
- 🔄 双任务支持：同时测试回归和分类能力
- 🎯 8种方法对比：sklearn/PyTorch基线 + CausalEngine四种模式
- 📊 科学实验设计：无数据泄露的标准化策略
- 🛡️ 噪声鲁棒性：在30%噪声环境下验证优势

### 🏠 真实世界教程 - 加州房价预测

运行完整的真实世界回归教程，展示CausalEngine的强大性能：

```bash
# 运行真实世界回归教程（sklearn-style版本）
python examples/comprehensive_causal_modes_tutorial_sklearn_style.py

# 这个教程将：
# 1. 加载加州房价数据集（20,640个样本）
# 2. 在30%标签噪声下比较13种方法
# 3. 测试CausalEngine所有4种推理模式
# 4. 生成标准版和扩展版性能对比图表
```

**真实世界教程亮点**：
- 🌍 真实数据：加州房价数据集的完整分析
- 🔬 全面对比：13种方法（传统ML + 稳健回归 + CausalEngine四模式）
- 📊 双重可视化：标准版（9种核心方法）+ 扩展版（13种全方法）
- 🎯 CausalEngine专项：四种推理模式的深度对比分析
- 🛡️ 鲁棒性验证：在30%噪声环境下的真实表现

### 📈 鲁棒性测试脚本

测试算法在不同噪声水平下的鲁棒性表现：

```bash
# 回归算法鲁棒性测试（真实数据集）
python scripts/regression_robustness_real_datasets.py

# 分类算法鲁棒性测试（真实数据集）
python scripts/classification_robustness_real_datasets.py

# 这些脚本将：
# 1. 使用sklearn内置真实数据集
# 2. 测试0%-100%噪声梯度（11个级别）
# 3. 比较多种算法的鲁棒性曲线
# 4. 支持多次运行取平均，提高结果稳定性
```

#

## 🔬 基准测试与性能验证

### 使用我们的基准测试框架

```python
from causal_sklearn.benchmarks import BaselineBenchmark
from sklearn.datasets import fetch_california_housing

# 创建基准测试实例
benchmark = BaselineBenchmark()

# 加载真实数据
data = fetch_california_housing()
X, y = data.data, data.target

# 运行完整的基准测试比较
results = benchmark.compare_models(
    X=X, y=y,
    task_type='regression',
    anomaly_ratio=0.25,  # 25%标签噪声
    causal_modes=['deterministic', 'standard'],
    verbose=True
)

# 快速测试版本 (使用较小数据集)
# results = benchmark.compare_models(X=X[:1000], y=y[:1000], ...)

# 打印详细结果
benchmark.print_results(results, 'regression')
```

### 🎯 性能亮点

在加州房价数据集（25%标签噪声）上的真实运行结果：

| 方法 | 验证集 MAE | 验证集 MdAE | 验证集 RMSE | 验证集 R² | 测试集 MAE | 测试集 MdAE | 测试集 RMSE | 测试集 R² |
|------|-----------|-------------|-------------|-----------|-----------|-------------|-------------|-----------|
| sklearn MLP | 0.5230 | 0.3992 | 0.7196 | 0.5025 | 0.5056 | 0.3803 | 0.7085 | 0.4922 |
| PyTorch MLP | 0.4868 | 0.3696 | 0.6626 | 0.5782 | 0.4912 | 0.3716 | 0.6705 | 0.5451 |
| CausalEngine (Det) | 0.5581 | 0.4434 | 0.7485 | 0.4617 | 0.5429 | 0.4278 | 0.7349 | 0.4536 |
| **CausalEngine (Std)** | **0.3446** | **0.2174** | **0.5482** | **0.7112** | **0.3326** | **0.2055** | **0.5112** | **0.7356** |

**真实数据验证的关键优势**：
- 🏆 **性能领先**: CausalEngine (Standard) 在测试集上R²达到0.7356，显著超过传统方法
- 🎯 **误差控制**: 测试集MAE仅0.3326，比sklearn MLP降低34.2%
- 🛡️ **噪声鲁棒**: 在25%标签噪声环境下仍保持最优性能
- 📊 **一致性强**: 验证集和测试集表现高度一致，证明模型泛化能力
- 🧠 **因果理解**: 通过因果推理实现更深层的数据理解

## 📚 文档与理论基础

### 🧮 数学理论基础
- **[🌟 数学基础 (中文)](docs/MATHEMATICAL_FOUNDATIONS_CN.md)** - **最核心文档** 完整的CausalEngine理论框架
- **[One-Pager Summary](docs/ONE_PAGER.md)** - Executive summary of CausalEngine

## 📄 许可证

本项目采用Apache License 2.0 - 详见[LICENSE](LICENSE)文件。


## 📖 学术引用

如果您在研究中使用了Causal-Sklearn，请引用：

```bibtex
@software{causal_sklearn,
  title={Causal-Sklearn: Scikit-learn Compatible Causal Regression and Classification},
  author={Heyang Gong},
  year={2025},
  url={https://github.com/1587causalai/CausalQwen/tree/causal-sklearn-mvp},
  note={基于CausalEngine™核心的因果回归和因果分类算法的scikit-learn兼容实现}
}
```

---

<div align="center">

**🌟 CausalEngine™ - 重新定义机器学习的因果推理革命 🌟**

*从相关性到因果性，从模式匹配到因果理解*

</div>