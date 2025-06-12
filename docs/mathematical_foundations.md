# 因果语言模型数学概览

本文档旨在为读者提供 CausalQwen 模型核心数学思想的直观概览。我们的目标是解释"为什么"和"是什么"，而将更深层次的"如何推导"留给 `design-docs` 目录下的专业文档。

## 1. 核心动机：超越关联，拥抱因果

传统机器学习模型擅长学习**关联** ($P(y|x)$)，但难以回答**反事实**问题（"如果...会怎样?"）。为了真正实现因果推理，我们需要一个能够对个体的内在特性进行建模的框架。

本项目的理论基石 ([arXiv:2401.15911](https://arxiv.org/abs/2401.15911)) 从数学上证明，为了构建一个能够灵活表达反事实的因果模型，引入一个外生的**"个体选择变量" $U$** 是必要的。

> 深度解读请参见: [`design-docs/U_deep_dive.md`](design-docs/U_deep_dive.md)

## 2. $U$ 的双重角色：选择与表征

变量 $U$ 是理解本模型所有魔法的关键。它有两个核心身份：

1.  **个体选择变量 (Individual Selection Variable)**：一次具体的赋值 $U=u$ 代表着从所有可能的个体中"选中"了某一个特定个体 `u`。
2.  **个体因果表征 (Individual Causal Representation)**：被选中的向量 $u$ 本身，就包含了该个体所有内在的、驱动其行为的潜在属性。

**核心思想**：普适的因果律 ($Y=f(T;u)$) 应用于不同的个体 ($u$)，从而产生了不同的反事实结果 ($Y(t)$)。$U$ 是所有个体性差异的最终来源。

## 3. 推断-行动范式 (Abduction-Action Paradigm)

基于 $U$ 的概念，我们将决策过程分解为两个逻辑步骤，这个过程在序列的**每一个位置**上独立进行。

#### 步骤 1: 归因推断 (Abduction) - "你是谁？"

归因推断网络接收上下文特征 $z_i$，推断该位置的因果表征分布：
\[
U_i | z_i \sim \text{Cauchy}(\text{loc}(z_i), \text{scale}(z_i))
\]

#### 步骤 2: 行动 (Action) - "你会做什么？"

行动网络基于因果表征分布，直接计算分类和回归结果：
- **分类决策分数**: $S_{k,i} = \vec{A}_k \cdot U_i + B_k$
- **回归值**: $Y_i = \vec{W} \cdot U_i + b$

## 4. 数学引擎：柯西分布的威力

我们选择**柯西分布(Cauchy Distribution)** 的核心优势在于其**线性稳定性**：

**柯西分布的线性封闭性**：
如果 $U \sim \text{Cauchy}(\mu, \gamma)$，则：
\[
Y = aU + b \sim \text{Cauchy}(a\mu + b, |a|\gamma)
\]

这使得从 $U$ 到 $S_k$ 和 $Y$ 的整个前向传播可以**完全在分布参数层面进行解析计算**，无需采样。

## 5. 训练策略：门控损失函数

**OvR 分类概率**：
\[
P_{k,i} = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)
\]

**混合门控回归损失**：
\[
\mathcal{L}_{\text{reg\_gated},i} = m_i \cdot \left(\alpha + (1-\alpha) \cdot P_{\text{<NUM>},i}\right) \cdot \mathcal{L}_{\text{cauchy\_nll},i}
\]

其中：
- $m_i$ 是位置 $i$ 的数值掩码（0或1）
- $\alpha$ 是门控系数（1.0=无门控，0.0=完全门控）
- $P_{\text{<NUM>},i}$ 是位置 $i$ 预测为 `<NUM>` 的概率

**总损失**：
\[
\mathcal{L}_{\text{total}} = \sum_i \left( \mathcal{L}_{\text{cls},i} + \lambda \cdot \mathcal{L}_{\text{reg\_gated},i} \right)
\]

## 6. 数值感知的统一表示

对于序列中的**每个位置** $i$，我们在**输入层**就进行数值感知处理：

$$e_i = \text{embed}(x_i) + \phi(v_i)$$

其中：
- $\text{embed}(x_i)$ 是词元 $x_i$ 的基础嵌入
- 当 $x_i = \text{<NUM>}$ 时，$v_i$ 是对应的实际数值；否则 $v_i = 0$
- $\phi(v) = \text{sign}(v) \cdot \ln(1 + |v|) \cdot \vec{e}$ 是数值编码函数

**关键性质**：$\phi(0) = 0$，非数值位置自然退化为 $e_i = \text{embed}(x_i)$。

增强嵌入序列 $(e_1, ..., e_S)$ 然后通过 Transformer 得到特征：
$$z = h(e_1, ..., e_S) = (z_1, ..., z_S)$$

为什么选择对数编码？
1. **数值稳定性**：将大范围的数值压缩到合理区间
2. **相对误差保持**：对数空间中的等距对应原空间的等比
3. **自然退化**：零值自动消失，无需特殊处理

## 7. 并行化计算的核心思想

CausalQwen的并行化发生在**因果推断-行动阶段**，而非特征提取阶段。

**阶段1：特征提取**（串行，不可并行）
- 输入：增强嵌入 $(e_1, ..., e_S)$
- 处理：通过Transformer的自注意力机制
- 输出：特征序列 $z = (z_1, ..., z_S)$
- 特性：位置间有依赖关系

**阶段2：因果推断-行动**（并行）
- 输入：特征 $z_i$（每个位置独立）
- 处理：$U_i \sim \text{Cauchy}(\text{loc}(z_i), \text{scale}(z_i))$
- 输出：$P_{k,i}$ 和 $(\text{loc}_{Y_i}, \text{scale}_{Y_i})$
- 特性：给定 $z$ 后，位置间完全独立


**条件独立性**：给定特征序列 $z$，位置 $i$ 的输出仅依赖 $z_i$：
$$P(U_i, S_{k,i}, Y_i | z) = P(U_i, S_{k,i}, Y_i | z_i)$$

这保证了：
- 所有位置可以同时进行因果推断
- 损失计算可以完全向量化
- 梯度计算可以并行进行

## 8. 总结

CausalQwen 的数学框架基于三个核心洞察：

1. **因果表征**：通过 $U$ 建模个体差异，实现真正的因果推理
2. **分布计算**：利用柯西分布的线性性质，实现无采样训练
3. **统一架构**：通过巧妙的数学设计，统一处理文本和数值

更详细的数学推导请参考：[`design-docs/math/mathematical_foundations.md`](design-docs/math/mathematical_foundations.md)

