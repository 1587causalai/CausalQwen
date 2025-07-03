# CausalQwen 示例脚本使用指南

欢迎来到 `CausalQwen` 的示例目录！本文档旨在帮助您理解本目录下的各类示例脚本，并为您选择最适合您需求的脚本提供指导。

## 核心设计理念：两种实现范式

为了满足不同开发者的需求，本目录中的核心示例均提供了两种实现版本。这两种版本在功能上是等价的，但在 API 调用方式和代码组织结构上有着本质的区别。

---

### 范式一：端到端原生API (Legacy End-to-End Style)

这类脚本通常不包含 `_sklearn_style` 后缀，例如 `real_world_regression_tutorial.py`。

-   **核心特点**: 通过一个高度封装的辅助类（如 `BaselineBenchmark`）和单一的函数调用（`compare_models`）来完成从数据处理、模型训练、到性能评估的整个实验流程。
-   **设计目的**: 这种"端到端"的方式旨在提供一个"开箱即用"的快速通道，让用户可以便捷地复现和评估模型的核心性能，而无需关心过多的中间实现细节。它非常适合用于快速、标准化的性能评测。我们称之为"Legacy"（早期）版本，因为它代表了项目早期的实现思路。

### 范式二：Scikit-learn 风格API (Scikit-learn Style)

这类脚本通常带有 `_sklearn_style` 后缀，例如 `real_world_regression_tutorial_sklearn_style.py`。

-   **核心特点**: 将 `CausalQwen` 的模型（如 `MLPCausalRegressor`）封装成与 `scikit-learn` 完全兼容的估计器（Estimator）。整个实验流程遵循标准的 `.fit()` / `.predict()` 范式，数据处理和评估步骤都是显式、透明的。
-   **设计目的**: 这种方式的核心是为了"方便调用"，提供完全透明、灵活的使用体验。它使得 `CausalQwen` 可以无缝对接到广大的 `scikit-learn` 生态中，例如与 `Pipeline`、`GridSearchCV` 等工具结合使用，是进行深度开发和自定义工作流的最佳选择。

---

## 核心差异速查表

| 对比维度         | 端到端原生API (`*.py`)                                 | Scikit-learn 风格 (`*_sklearn_style.py`)                         |
| :--------------- | :----------------------------------------------------- | :--------------------------------------------------------------- |
| **核心抽象**     | 一个高级辅助类 (`BaselineBenchmark`)                   | 与 sklearn 兼容的模型类 (`MLPCausalRegressor`)                   |
| **数据处理**     | 隐藏在 `compare_models` 函数内部                       | 用户显式调用 `train_test_split`, `StandardScaler` 等             |
| **模型训练**     | 通过函数传参间接触发                                   | 显式调用 `model.fit(X_train, y_train)`                           |
| **模型预测**     | 无独立的预测步骤，结果直接返回                         | 显式调用 `model.predict(X_test)`                                 |
| **代码结构**     | 核心逻辑封装在"黑盒"函数中，关注输入与输出             | 标准的机器学习实验流程，逻辑清晰，步骤透明                       |
| **灵活性与集成** | 适合快速运行标准评测，但自定义与集成较为困难           | 灵活性极高，易于修改和集成到 `scikit-learn` 工作流中             |

---

## 如何选择合适的示例？

-   **如果您希望快速评估 `CausalQwen` 在标准数据集上的性能表现**，请选择 **端到端原生API** 版本的脚本。
-   **如果您熟悉 `scikit-learn`，希望将 `CausalQwen` 集成到您现有的项目中，或者需要对实验流程进行深度定制**，请选择 **Scikit-learn 风格** 的脚本。

---

## 示例脚本清单

本目录下的脚本成对出现，分别对应上述两种实现范式。

-   **真实世界回归任务 (California Housing)**
    -   `real_world_regression_tutorial.py`
    -   `real_world_regression_tutorial_sklearn_style.py`
    -   *描述*: 在加州房价数据集上，演示了如何在含有标签噪声的真实回归任务中，对比 `CausalEngine` 与传统机器学习方法的性能。

-   **扩展版真实世界回归任务 (California Housing Extended)**
    -   `real_world_regression_tutorial_extended.py`
    -   `real_world_regression_tutorial_extended_sklearn_style.py`
    -   *描述*: 这是对基础回归教程的扩展，可能包含了更复杂的分析或模型配置。

-   **CausalEngine 因果模式详解 (Classification)**
    -   `comprehensive_causal_modes_tutorial.py`
    -   `comprehensive_causal_modes_tutorial_sklearn_style.py`
    -   *描述*: 在一个分类任务中，详细介绍并对比了 `CausalEngine` 支持的多种不同因果模式（如 `deterministic`, `standard`, `exogenous` 等）的工作方式和效果。

---

*注：随着项目的迭代，未来新增的示例也将遵循这种双版本的设计，此文档会同步更新。* 