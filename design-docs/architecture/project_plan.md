# CausalQwen-0.5B 项目规划

## 代码架构设计

本项目采用模块化设计，清晰分离各个组件的职责，以便于后续扩展和维护。

### 核心模块结构

1. **FeatureNetwork（特征提取网络）**
   - **功能**: 从原始输入（如文本、表格数据）中提取高维特征表示 `z`。
   - **实现**: 直接复用预训练的 `Qwen-0.5B` 模型的主干部分。

2. **归因推断网络（AbductionNetwork）**
   - **功能**: 将特征 `z` 映射为因果表征 `U` 的分布参数（`loc` 和 `scale`）。
   - **实现**: 一个简单的线性层，将 `hidden_size` 映射到 `causal_dim * 2`。

3. **ActionNetwork（行动网络）**
   - 职责：从个体因果表征生成分类和回归预测
   - 子组件：
     - 分类头：生成词元预测概率
     - 回归头：生成数值预测
   - 接口：`forward(causal_state) -> (classification_logits, regression_value)`

4. **CausalDistributions（因果分布）**
   - 职责：实现柯西分布相关操作
   - 功能：密度函数、累积分布函数、采样、重参数化等
   - 接口：`sample(loc, scale)`, `cdf(x, loc, scale)`, `pdf(x, loc, scale)`等

5. **LossFunction（损失函数）**
   - 职责：计算训练损失
   - 组件：
     - 分类损失：基于OvR的二元交叉熵
     - 回归损失：柯西负对数似然
     - 门控机制：根据`<NUM>`预测概率调整回归损失权重
   - 接口：`compute_loss(predictions, targets)`

### 数据流设计

```mermaid
graph TD
    A[输入序列] --> B{分词器};
    B --> C[FeatureNetwork<br>(Qwen-0.5B)];
    C --"特征z"--> D[AbductionNetwork<br>(归因推断)];
    D --"U的分布参数"--> E((采样U));
    E --"因果表征u"--> F[ActionNetwork];
    F --> G{输出};
    G --> H1[分类结果<br>OvR概率];
    G --> H2[回归结果<br>数值];
```

**简化流程**:
输入序列 → 分词器 → FeatureNetwork → 归因推断网络 (AbductionNetwork) →
采样 → ActionNetwork → OvR分类/回归 → 损失计算

### 训练与推理流程

1. **训练流程**
   - 前向传播：计算个体因果表征分布参数，解析计算输出分布
   - 损失计算：分类损失 + 门控回归损失
   - 反向传播：更新网络参数

2. **推理流程**
   - 确定性推理：使用分布参数的中位数作为预测
   - 随机推理：从分布中采样，模拟真实世界的随机性

## 模块职责详细说明

### 1. 模型组件 (`src/models/`)

- **feature.py**: `FeatureNetwork` 的封装。
- **abduction.py**: `归因推断网络 (AbductionNetwork)` 的实现。
- **action.py**: `ActionNetwork` 的实现。
- **causal_lm.py**: `