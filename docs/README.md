# Causal-Sklearn 数学文档

本目录包含 causal-sklearn 中 CausalEngine 算法的完整数学基础和理论框架。

## 核心数学文档

为了明确当前开发阶段的重点，我们将文档分为两类：

### 🎯 CausalML for Sklearn (当前分支核心)

这些文档构成了 `causal-sklearn` 实现的直接理论和数学基础，专注于解决常规的分类与回归任务。

- **[ONE_PAGER.md](ONE_PAGER.md)** - 算法概览与高管摘要
- **[MATHEMATICAL_FOUNDATIONS_CN.md](MATHEMATICAL_FOUNDATIONS_CN.md)** - 🌟 **最核心** CausalEngine 数学基础 (中文完整版)
- **[mathematical_equivalence_deep_dive.md](mathematical_equivalence_deep_dive.md)** - 与传统机器学习模型的数学等价性深度分析

### 🚀 CausalLLM (未来探索方向)

这些文档为项目最终目标——将 CausalEngine 与大语言模型（LLM）结合——提供前瞻性的理论探索。
- **[core_mathematical_framework.md](core_mathematical_framework.md)** - CausalQwen 核心数学框架实现细节
- **[core_mathematical_framework_num_extended.md](core_mathematical_framework_num_extended.md)** - CausalQwen 扩展数值理论

## 关键数学概念

### CausalEngine 四大公理

1. **智能 = 归因 + 行动**：从观测到自我理解再到决策
2. **柯西数学**：唯一支持因果推理解析计算的分布
3. **结构方程决策**：每个选择都由确定性函数计算

### 核心数学框架

CausalEngine 算法基于结构因果方程：

$$
Y = f(U, E)
$$

其中：
- $U$：个体因果表征（从上下文 $X$ 学习得到）
- $E$：外生噪声（独立随机扰动）
- $f$：普适因果机制（确定性函数）

### 三阶段架构

1. **归因推断阶段**：$X → U$（证据到个体表征）
   - AbductionNetwork：将观测映射到因果表征
   - 使用柯西分布实现解析不确定性传播

2. **行动决策阶段**：$U → S$（个体表征到决策得分）
   - ActionNetwork：将表征映射到决策潜能
   - 利用柯西分布线性稳定性实现解析计算

3. **任务激活阶段**：$S → Y$（决策得分到任务输出）
   - ActivationHead：将潜能转换为具体输出（分类/回归）
   - 支持多种推理模式和任务类型

### 数学特性

- **解析计算**：利用柯西分布特性无需采样
- **重尾鲁棒性**：自然处理异常值和极端事件
- **未定义矩**：与真实不确定性哲学对齐
- **尺度不变性**：跨不同尺度的一致行为

## 实现中的使用

这些数学文档作为权威参考用于：

1. **正确性验证**：确保实现与理论框架匹配
2. **参数理解**：所有超参数的数学含义
3. **调试指导**：排查实现问题的理论基础
4. **功能扩展**：添加新特性的数学基础

## 阅读顺序

### 对于实现者和开发者：
1. 从 `ONE_PAGER.md` 开始了解高层概览
2. 阅读 `MATHEMATICAL_FOUNDATIONS_CN.md` 获得完整理论（**最重要**）
3. 参考 `core_mathematical_framework.md` 了解详细方程
4. 使用 `MATHEMATICAL_IMPLEMENTATION_VERIFICATION.md` 进行验证

### 对于研究者和理论家：
1. 从 `MATHEMATICAL_FOUNDATIONS_CN.md` 开始（**核心文档**）
2. 深入研究 `mathematical_equivalence_deep_dive.md`
3. 学习 `core_mathematical_framework_num_extended.md` 的高级理论

## 重要说明

> **🎯 当前分支目标**：值得注意的是，本项目的最终目标是将因果推理引擎与大语言模型（LLM）结合。然而，当前 `causal-sklearn-mvp` 分支的焦点是**将因果引擎应用于常规的分类和回归任务**，为 `sklearn` 生态提供一个功能强大、理论完备的因果模型。

> **📋 权威规范**：这些文档是 CausalEngine 的权威数学规范。causal-sklearn 中的任何实现都必须严格遵循这些数学定义。
> 
> **🌟 核心文档**：`MATHEMATICAL_FOUNDATIONS_CN.md` 是最核心、最完整、最准确的数学基础文档，包含最新的理论更新和图解说明。
> 
> **🔍 验证标准**：所有代码实现的正确性都应以这些数学文档为标准进行验证。