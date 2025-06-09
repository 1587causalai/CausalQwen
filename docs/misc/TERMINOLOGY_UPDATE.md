# 重要术语更正：从"因果状态"到"个体因果表征"

## 背景

基于对 `design-docs/U_deep_dive.md` 的深入理解，项目中一直使用的"因果状态"术语是**不准确**的，需要系统性地更正为正确的术语。

## 核心理解

根据 `U_deep_dive.md` 的核心思想：

### U 的本质
- **U 不是"状态"，而是"个体"**
- U 是一个"个体选择变量"（Individual Selection Variable）
- U=u 代表从所有可能的个体中选择出了一个独一无二的个体
- u 本身是一个高维向量，即"个体因果表征"（Individual Causal Representation）

### 推断过程的本质
- 我们不是在推断某种"状态"
- 我们是在**根据观测证据推断个体可能属于哪个子群体**
- P(U|x) 是对这个子群体的数学刻画

## 术语更正映射

| 错误术语 | 正确术语 |
|---------|---------|
| 因果状态 | 个体因果表征 |
| 潜在因果状态 | 潜在个体因果表征 |
| 因果状态分布 | 个体因果表征分布 |
| 因果状态维度 | 个体因果表征维度 |
| 因果状态空间 | 个体因果表征空间 |
| 推断因果状态 | 推断个体因果表征 |

## 已更正的文档

1. `design-docs/core-design.md` ✅
2. `design-docs/architecture/architecture_design.md` ✅ 
3. `docs/experiments/model_initialization_strategy.md` ✅

## 需要继续更正的文档

由于项目中有大量文档使用了错误术语，建议按优先级逐步更正：

### 高优先级（核心设计文档）
- `design-docs/experiments/experiment_design.md`
- `design-docs/PROJECT_BLUEPRINT.md`
- `design-docs/architecture/project_plan.md`

### 中优先级（用户面向文档）
- `README.md`
- `docs/README.md`
- `docs/overview.md`
- `docs/faq.md`

### 低优先级（其他文档）
- `todo.md`
- `docs/code/code_structure.md`
- `docs/architecture/architecture_design.md`

## 重要性

这个术语更正不仅仅是用词问题，它反映了对项目核心理论的深度理解：
- 我们的模型不是在处理抽象的"状态"
- 我们的模型是在进行**个体选择和表征学习**
- 这个理解对于模型的设计、训练和解释都具有根本性意义

正确的术语使用体现了对项目底层数学逻辑的真正理解。 