# 超越词元预测范式：CausalLLM 核心数学框架与架构设计

> **📋 文档用途**: 构建最小可行的因果语言模型，专注于验证同时能进行回归和分类的CausalLLM的核心哲学和数学框架
> **🎯 目标读者**: 项目负责人，用于验证AI实现是否符合理论预期  
> **📖 内容定位**: 详细的理论框架、数学原理、架构设计的权威参考

> 本文档详细描述了 CausalLLM 因果语言模型的核心设计理念、数学框架和实现细节。CausalLLM 是首个将个体选择变量 U 引入语言生成的因果推理模型，实现了从"概率采样"到"个体决策"的范式转变。本文中，我们选择Qwen2.5作为基础构建 CausalQwen 作为示范。

## 1.核心创新：引入个体选择变量 U

为了真正实现因果推理，我们需要一个能够对个体的内在基因进行建模的框架。本项目的理论基石 ([arXiv:2401.15911](https://arxiv.org/abs/2401.15911)) 从数学上证明，为了构建一个能够灵活表达反事实的因果模型，引入一个外生的 **"个体选择变量" $U$** 是必要的。 $U$ 是理解本模型所有魔法的关键。它有两个核心身份：

1.  **个体选择变量 (Individual Selection Variable)**：一次具体的赋值 $U=u$ 代表着从所有可能的个体中"选中"了某一个特定个体 `u`。
2.  **个体因果表征 (Individual Causal Representation)**：被选中的向量 $u$ 本身，就包含了该个体所有内在的、驱动其行为的潜在属性。

**核心思想**：普适的因果律 ($Y=f(t;u, \text{noise})$) 应用于不同的个体 ($u$) 与外生噪声 ($\text{noise}$)，从而产生了不同的反事实结果 ($Y(t)$)。$U$ 是所有个体性系统性差异的最终来源，而 $\text{noise}$ 则代表了不可控的、非系统性的随机扰动。

> 深度解读请参见: [`design-docs/U_deep_dive.md`](design-docs/U_deep_dive.md)

## 2.训练阶段：前向传播 (Forward Pass)

模型训练的核心是执行一个完整的前向传播，计算预测值与真实标签之间的损失，然后通过反向传播更新模型参数。整个前向传播过程可以分解为五个核心模块。

> 我们用 B 代表批次大小, S 代表序列长度, H 代表模型隐藏维度, C 代表因果表征维度, V_full 代表 Qwen 的词汇表总大小。Qwen 的词汇表包含 K 个已使用词汇和 271 个预留位置，即 V_full = K + 271。CausalQwen 使用第一个预留位置（ID = K）作为 `<NUM>` 词元。

> **设计决策**: 在当前实现中，我们设定因果表征维度 `C` 与模型隐藏层维度 `H` 相等，即 **`C = H`**。这简化了归因推断网络的初始化。

### 2.1 模块一：数值感知嵌入 (Numerical-aware Embedding)
这一模块的目标是将混合了文本和数值的原始输入，转化为一个统一的、数值感知的特征向量序列。这个过程包含三个关键步骤, *输入示例**: 原始字符串文本 `"价格是99.9元"`:

#### 1.分词与数值识别
分词器处理原始文本，识别并替换数值：

1.  **数值识别**: 分词器扫描文本，识别数值模式（如 `99.9`）
2.  **词元替换**: 将识别出的数值替换为特殊词元 `<NUM>`
3.  **数值保存**: 将原始数值单独保存，与词元序列保持位置对齐

-   **输出**: 
    - `input_ids` $[x_1, ..., x_S]$: `['价格', '是', '<NUM>', '元']` → `[12345, 67890, <NUM_ID>, 11111]` (形状: `[B, S]`)
    - `numeric_values` $[v_1, ..., v_S]$: `[0.0, 0.0, 99.9, 0.0]` (形状: `[B, S]`)

#### 2.词元嵌入
将词元ID序列转换为基础嵌入向量：

-   **输入**: `input_ids` (形状: `[B, S]`)
-   **处理**: 通过嵌入层查找每个词元的向量表示
    $$\text{base\_embed}_i = \text{EmbeddingLayer}(\text{input\_ids}_i)$$
-   **输出**: `base_embeddings` (形状: `[B, S, H]`)

#### 3.数值编码与融合
结合词元的基础嵌入和数值的对数编码，计算出最终的增强嵌入：

-   **输入**: 
    - `base_embeddings` (形状: `[B, S, H]`)
    - `numeric_values` (形状: `[B, S]`)
-   **处理**: 对每个位置 $i$，计算增强嵌入：
    $$e_i = \text{base\_embed}_i + \phi(v_i)$$
    数值编码函数：
    $$\phi(v) = \text{sign}(v) \cdot \ln(1 + |v|) \cdot \vec{w}_{\text{num}}$$
    其中 $v_i$ 是位置 $i$ 的数值（非数值位置为 0），$\vec{w}_{\text{num}} \in \mathbb{R}^H$ 是数值感知嵌入模块的可学习参数向量。
-   **输出**: 
    - `e`: 增强嵌入张量 (形状: `[B, S, H]`)

**关键洞察**：
1. **自然退化**: 对于非数值位置，$v_i = 0$ 导致 $\phi(0) = 0$，因此 $e_i = \text{base\_embed}_i$，自然退化为标准词元嵌入
2. **统一处理**: 所有位置使用相同的计算公式，无需条件分支
3. **位置对齐**: 数值信息与词元序列严格对齐，确保语义的连贯性

**完整示例**:
```
原始文本: "价格是99.9元"
     ↓ (分词器)
input_ids: [12345, 67890, <NUM_ID>, 11111]
numeric_values: [0.0, 0.0, 99.9, 0.0]
     ↓ (嵌入层)
     
base_embeddings: [[e1], [e2], [e3], [e4]]  # 每个ei是H维向量
     ↓ (数值编码)
φ(numeric_values): [[φ(0)], [φ(0)], [φ(99.9)], [φ(0)]]  # φ(99.9) = ln(100.9) * ê
     ↓ (融合)
enhanced_embeddings: [[e1], [e2], [e3 + φ(99.9)], [e4]]
```

> **设计动机**: 选择对数编码 $\phi(v)$ 是因为它具有三大优势：1) **数值稳定性**，将大范围数值压缩到合理区间；2) **相对误差保持**，对数空间中的等距对应原空间的等比；3) **自然退化**，由于$\phi(0)=0$，非数值位置自然退化为标准词元嵌入，无需特殊处理。

### 2.2 模块二：特征提取网络 (Feature Extraction Network)
该模块使用一个标准的 Transformer 网络（如Qwen）作为主干，来深度理解序列的上下文信息。

-   **输入**: `e`: 增强嵌入张量 (形状: `[B, S, H]`)
-   **处理**: 通过 $L$ 层 Transformer 进行特征提取：
    $$z = \text{QwenTransformer}(e)$$
    
    由于完全继承 Qwen 权重，当 $e \approx e_{\text{Qwen}}$ 时，$z \approx z_{\text{Qwen}}$。
-   **输出**: `z`: 上下文特征张量 (形状: `[B, S, H]`)

> **训练策略**: 在训练的初期阶段，QwenTransformer 的参数保持冻结，仅在后续阶段考虑使用 LoRA 等技术进行微调。这既保证了快速收敛，又维持了与基座模型的可比性。

### 2.3 模块三：归因推断网络 (Abduction Network)
该模块从上下文特征中推断出每个位置的个体因果表征分布。

-   **输入**: 上下文特征 `z` (形状: `[B, S, H]`)
-   **处理**: 通过线性层计算因果表征的分布参数：
    $$\text{loc}_{U_i} = W_{\text{loc}} \cdot z_i + b_{\text{loc}}$$
    $$\text{scale}_{U_i} = \text{softplus}(W_{\text{scale}} \cdot z_i + b_{\text{scale}})$$
    
    其中 $\text{softplus}(x) = \log(1 + \exp(x))$，保证尺度参数严格为正。
    
-   **输出**: 
    - `loc_U`: 因果表征分布的位置参数 (形状: `[B, S, C]`)
    - `scale_U`: 因果表征分布的尺度参数 (形状: `[B, S, C]`)

> **灵活性设计**: 归因推断网络是最灵活的组件，可以针对不同领域采用不同的网络结构。但需要满足约束：$D_{KL}(P(U|X) || P_{\text{base}}(U|X)) \leq \epsilon$，以防止灾难性遗忘。

### 2.4 模块四：行动网络 (Action Network)

该模块是模型的核心决策单元，体现了**普适的线性因果律**。在训练阶段，它通过注入一个外生噪声，并将个体表征 $U$ 映射到并行的决策头（分类和回归），从而产生最终的决策潜能。

-   **输入**: `loc_U` (形状: `[B, S, C]`), `scale_U` (形状: `[B, S, C]`)
-   **内部参数**: 一个可学习的参数向量 `b_noise` (形状: `[C]`)
-   **处理**:
    1.  **注入外生噪声**:
        -   **基本原理**: 核心思想是对个体表征 $U$ 注入一个标准柯西分布的噪声 $\varepsilon \sim \text{Cauchy}(0, 1)$，其强度由一个可学习的参数向量 $\mathbf{b}_{\text{noise}}$ 控制。变换后的随机变量 $U'$ 为：
            $$U' = U + \mathbf{b}_{\text{noise}} \cdot \varepsilon$$
        -   **解析推导与计算实现**: 根据柯西分布的线性稳定性，可以推导出 $U' \sim \text{Cauchy}(\text{loc}_U, \text{scale}_U + |\mathbf{b}_{\text{noise}}|)$。这个推导允许我们在计算中完全避免采样，直接通过对尺度参数进行加法操作来高效地实现噪声注入。

    2.  **并行决策 (Parallel Decision Making)**：基于包含了噪声的分布 $U'$，通过两个并行的线性"头"来计算分类和回归的决策分布。

        -   **分类头 (Classification Head)**：
            $$\text{loc}_S = \text{lm\_head}(\text{loc}_U)$$
            $$\text{scale}_S = (\text{scale}_U + |\mathbf{b}_{\text{noise}}|) \cdot |\text{lm\_head.weight}^T|$$

        -   **回归头 (Regression Head)**：
            $$\text{loc}_Y = \text{reg\_head}(\text{loc}_U)$$
            $$\text{scale}_Y = (\text{scale}_U + |\mathbf{b}_{\text{noise}}|) \cdot |\text{reg\_head.weight}^T|$$

-   **输出**:
    - 分类决策分布参数: `loc_S` (形状: `[B, S, V_full]`), `scale_S` (形状: `[B, S, V_full]`)
    - 回归决策分布参数: `loc_Y` (形状: `[B, S]`), `scale_Y` (形状: `[B, S]`)

> **核心引擎：柯西分布的线性稳定性**
> 如果 $X_1, X_2, ..., X_n$ 是独立的柯西随机变量，$X_j \sim \text{Cauchy}(\mu_j, \gamma_j)$，那么对于权重 $w_j$：
> $$\sum_{j=1}^n w_j X_j \sim \text{Cauchy}\left(\sum_{j=1}^n w_j \mu_j, \sum_{j=1}^n |w_j| \gamma_j\right)$$
> 这一定理是整个行动网络能够以解析方式（无采样）对分布参数进行精确变换的数学基石。

---
**重要连接**: 上述训练过程产生的、包含了外生噪声影响的最终决策潜能分布 $S$ 和 $Y$，将被传递给任务激活头（ActivationHead）进行最终的、任务特定的处理。

### 2.5 模块五：损失计算 (Loss Calculation)

#### 1. OvR 分类损失
对每个类别计算独立的二元分类概率：
$$P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

其中 $C_k$ 是类别 $k$ 的可学习阈值参数。

然后计算所有类别的二元交叉熵之和：
$$L_{\text{cls}, i} = -\sum_{k=0}^{V_{\text{full}}-1} [y_{k,i} \log P_{k,i} + (1-y_{k,i}) \log (1 - P_{k,i})]$$

其中 $y_{k,i}$ 是 one-hot 编码的真实标签。

#### 2. 门控回归损失
柯西分布的负对数似然及其门控损失：
$$\mathcal{L}_{\text{nll},i} = \log(\pi \cdot \text{scale}_{Y_i}) + \log\left(1 + \left(\frac{y_{\text{true},i} - \text{loc}_{Y_i}}{\text{scale}_{Y_i}}\right)^2\right) \\
\mathcal{L}_{\text{reg\_gated},i} = \text{num\_mask}_i \cdot \left(\alpha + (1-\alpha) P_{\text{<NUM>},i} \right) \cdot \mathcal{L}_{\text{nll},i}$$

门控权重（$\alpha=0$ 和 $\alpha=1$ 时）：
$$
\mathcal{L}_{\text{reg\_gated},i} = \text{num\_mask}_i \cdot P_{\text{<NUM>},i} \cdot \mathcal{L}_{\text{nll},i} \\
\mathcal{L}_{\text{reg\_gated},i} = \text{num\_mask}_i \cdot \mathcal{L}_{\text{nll},i}
$$

其中 `num_mask` 指示位置 $i$ 的真实标签是否为 `<NUM>`。

#### 3. 总损失
引入两种掩码：
- `cls_mask = attention_mask`：用于分类损失计算
- `num_mask = (labels == NUM_TOKEN_ID) & attention_mask`：用于回归损失计算

总损失为：
$$\mathcal{L}_{\text{total}} = \underbrace{\frac{\sum_i L_{\text{cls}, i} \cdot \text{cls\_mask}_i}{\sum_i \text{cls\_mask}_i}}_{\text{平均分类损失}} + \lambda \cdot \underbrace{\frac{\sum_i \mathcal{L}_{\text{reg\_gated},i}}{\sum_i \text{num\_mask}_i}}_{\text{有效回归损失}}$$

## 3. 推理阶段：对噪声的灵活调制

在推理阶段，我们可以通过 `temperature` 和 `do_sample` 两个参数，灵活地**调制**已经学习到的外生噪声 $\mathbf{b}_{\text{noise}}$，以实现不同的生成策略。

### 3.1 核心思想：温度统一的噪声控制

推理时，`temperature` 作为噪声强度的统一控制器，`do_sample` 选择噪声的作用方式：

-   **温度 = 0**: 关闭外生噪声，实现**纯因果生成**。
    -   数学原理: `U' ~ Cauchy(μ, γ)`
-   **温度 > 0**:
    -   **非采样模式** (`do_sample=False`): 噪声增加**尺度参数**，扩大决策不确定性。
        -   数学原理: `U' ~ Cauchy(μ, γ + T·|b_noise|)`
    -   **采样模式** (`do_sample=True`): 噪声扰动**位置参数**，改变个体身份。
        -   数学原理: `U' ~ Cauchy(μ + T·|b_noise|·ε, γ)`

这种**温度统一控制**的设计实现了完美的对称性和直观性。

### 3.2 推理模式详解

CausalQwen 提供四种核心推理模式，通过 `do_sample` 和 `temperature` 参数的组合实现，每种模式对应不同的因果生成哲学。

| 模式名称 | `do_sample` | `temperature` | 数学原理 | 哲学含义 |
| :--- | :--- | :--- | :--- | :--- |
| **因果模式 (Causal)** | `any` | `0` | `U' ~ Cauchy(μ, γ)` | 纯因果生成，无外生噪声，个体的必然表达 |
| **标准模式 (Standard)** | `False` | `> 0` | `U' ~ Cauchy(μ, γ+T·\|b_noise\|)` | 噪声增加决策不确定性，保持个体身份稳定 |
| **采样模式 (Sampling)** | `True` | `> 0` | `U'~Cauchy(μ+T·\|b_noise\|·ε,γ)` | 噪声扰动个体身份，探索决策空间多样性 |
| **兼容模式 (Compatible)** | `N/A` | `any` | 标准 Softmax 概率计算 | 用于与传统LM进行基准比较 |

#### 3.2.1 因果模式 (Causal Mode) (`temperature = 0`)

这是对因果理论最纯粹的表达，无论 `do_sample` 取值如何。当温度为零时，完全没有外生噪声，生成过程基于个体自身的因果表征。决策分布完全由 `U ~ Cauchy(μ, γ)` 决定，精确对应了理论公式 **`Y = f(U)`**，即输出完全是个体 `U` 在普适因果律 `f` 下的必然表达。这种模式提供了最纯粹的因果生成，适用于需要高度一致性和可解释性的场景。

#### 3.2.2 标准模式 (Standard Mode) (`do_sample=False, temperature > 0`)

这是模型的默认确定性推理模式。外生噪声 `T·|b_noise|` 被融合到尺度参数中，增加了决策的不确定性（分布更宽），但保持了决策中心（位置参数）的稳定。哲学含义是环境噪声使得个体的判断变得更加模糊，但个体的核心身份保持不变。最终通过 `argmax` 选择概率最高的词元，过程完全确定。

#### 3.2.3 采样模式 (Sampling Mode) (`do_sample=True, temperature > 0`)

当采用随机生成时，外生噪声 `ε` 经过温度 `T` 调节后，扰动**位置参数**。这改变了个体的身份表征，从而探索不同的决策结果。这相当于在探索"如果这个个体受到随机扰动偏离典型状态，会做出什么不同的决策？"。温度越高，扰动越大，生成的多样性越强。

#### 3.2.4 兼容模式 (Compatible Mode)

此模式下，模型会忽略所有因果模块（尤其是尺度参数 `scale_S`），仅使用决策的位置参数 `loc_S` 作为传统 logits，并应用标准的 Softmax 函数进行采样。这使得 CausalQwen 可以与任何传统语言模型在相同的设置下进行公平比较。

## 4. 初始化策略：知识迁移

为了使 CausalQwen 能够无缝继承基座模型的强大语言能力，我们采用了一种**精简而有效**的初始化策略。核心思想是：**在训练开始时，CausalQwen 的行为应与原始的 Qwen 尽可能一致**。

#### 步骤1：数值感知嵌入 → 标准初始化

- **`<NUM>` 词元嵌入**：直接继承 Qwen 的第一个保留词元嵌入：
  $$\text{embed}(\text{<NUM>}) \leftarrow \text{embed}_{\text{Qwen}}(\text{<NUM>})$$

- **数值编码向量**：使用标准的向量初始化：
  $$\vec{w}_{\text{num}} \sim \mathcal{N}(0, 1/\sqrt{H})$$
  
  这是标准的 Xavier 初始化，确保前向传播时方差稳定。

#### 步骤2：归因推断网络 → 恒等映射初始化

为了确保知识迁移，归因推断网络应该初始化为近似恒等映射，使得初始的因果表征分布直接反映 Qwen 的特征：

- **位置网络**：设置为恒等映射
  $$W_{\text{loc}} \leftarrow I_H, \quad b_{\text{loc}} \leftarrow 0$$
  这样 $\text{loc}_{U_i} = z_i$，即因果表征的位置参数直接等于 Qwen 的输出特征。

- **尺度网络**：设置为产生常数尺度
  $$W_{\text{scale}} \leftarrow 0, \quad b_{\text{scale}} \leftarrow \gamma_{\text{init}}$$
  其中 $\gamma_{\text{init}} = 1.0$ 或类似的正数。这样 $\gamma_i = \text{softplus}(\gamma_{\text{init}})$ 是一个与输入无关的常数，提供了宽泛的先验分布。

> **关键洞察**: 这种初始化策略确保了：
> 1. 因果表征的位置参数完全继承了 Qwen 的知识表示
> 2. 尺度参数初始为常数，表示对所有位置的不确定性有相同的先验认识
> 3. 模型将在训练过程中学习到哪些位置需要更高或更低的不确定性

**数学推导**：在初始化状态下，对于位置 $i$：
- 归因推断网络输出：
  $$\text{loc}_{U_i} = W_{\text{loc}} \cdot z_i + b_{\text{loc}} = I_H \cdot z_i + 0 = z_i$$
  $$\text{scale}_{U_i} = \text{softplus}(W_{\text{scale}} \cdot z_i + b_{\text{scale}}) = \text{softplus}(0 \cdot z_i + \gamma_{\text{init}}) = \text{softplus}(\gamma_{\text{init}}) \cdot \mathbf{1}_C = \gamma_0 \cdot \mathbf{1}_C$$
  
- 因此，初始的因果表征分布为：
  $$U_i \sim \text{Cauchy}(z_i, \gamma_0 \cdot \mathbf{1}_C)$$

#### 步骤3：行动网络(分类) → 复制 Qwen 权重

$$W_{\text{cls}} \leftarrow W_{\text{Qwen\_lm\_head}}, \quad b_{\text{cls}} = 0$$

这确保了初始分类输出与 Qwen 一致。由于我们完整复制了 Qwen 的 `lm_head` 权重矩阵（维度为 `[V_full, H]`），所有词汇（包括已使用的和预留的）都有对应的权重。

**分类决策分布推导**：
- 融合输入分布（加入噪声）：
  $$U'_i \sim \text{Cauchy}(z_i, \gamma_0 \cdot \mathbf{1}_C + |b_{\text{noise}}|)$$

  其中 $\gamma_0 \cdot \mathbf{1}_C + |b_{\text{noise}}|$ 表示对逐元素相加, $b_\text{noise}$ 初始化成某个常数值即可。

- 对于词汇 $k$，经过权重向量 $W_{\text{cls},k} \in \mathbb{R}^C$ 的线性变换（内积）后：
  $$S_{k,i} = W_{\text{cls},k} \cdot U'_i + b_{\text{cls},k} \sim \text{Cauchy}\left(W_{\text{cls},k} \cdot z_i, |W_{\text{cls},k}| \cdot (\gamma_0 \cdot \mathbf{1}_C + |b_{\text{noise}}|)\right)$$
  
- 因此：
  $$\text{loc}_{S_{k,i}} = W_{\text{cls},k} \cdot z_i = W_{\text{Qwen},k} \cdot z_i$$
  $$\text{scale}_{S_{k,i}} = |W_{\text{cls},k}| \cdot (\gamma_0 \cdot \mathbf{1}_C + |b_{\text{noise}}|)$$

这表明位置参数与原始 Qwen 的 logits 完全相同。

#### 步骤4：行动网络(回归) → 标准初始化

使用标准的 Xavier 或 He 初始化。模型将在训练中学习合适的回归映射。

**回归决策分布推导**：
- 基于融合输入分布 $U'_i \sim \text{Cauchy}(z_i, \gamma_0 \cdot \mathbf{1}_C + |b_{\text{noise}}|)$
- 经过回归权重向量 $W_{\text{reg}} \in \mathbb{R}^C$ 的线性变换：
  $$Y_i = W_{\text{reg}} \cdot U'_i + b_{\text{reg}} \sim \text{Cauchy}(\mu_{\text{reg},i}, \gamma_{\text{reg},i})$$
  
其中：
$$\mu_{\text{reg},i} = W_{\text{reg}} \cdot z_i + b_{\text{reg}}$$
$$\gamma_{\text{reg},i} = |W_{\text{reg}}| \cdot (\gamma_0 \cdot \mathbf{1}_C + |b_{\text{noise}}|)$$

这里 $|W_{\text{reg}}| \cdot (\gamma_0 \cdot \mathbf{1}_C + |b_{\text{noise}}|)$ 表示先对权重向量逐元素取绝对值，再与尺度向量进行内积。

通过上述初始化步骤，CausalQwen 在训练开始时具有以下性质：

-   **因果表征**: 对于每个位置 $i$，因果表征 $U_i$ 服从分布 $U_i \sim \text{Cauchy}(z_i, \gamma_0 \cdot \mathbf{1}_C)$，其中 $z_i \in \mathbb{R}^C$ 是 Qwen 的输出特征，$\gamma_0 = \text{softplus}(\gamma_{\text{init}})$ 是初始的常数尺度。
-   **分类决策**: 由于完整复制了 Qwen 的 lm_head 权重，分类输出的位置参数与 Qwen 完全一致：$\text{loc}_{S_{k,i}} = W_{\text{Qwen},k} \cdot z_i$，对所有 $k \in [0, V_{\text{full}})$。
-   **回归决策**: 位置参数 $\mu_{\text{reg},i} = W_{\text{reg}} \cdot z_i + b_{\text{reg}}$ 接近零均值，尺度参数由权重向量与尺度向量的内积决定。
-   **知识保持**: 当使用兼容性采样（基于 `loc_S` 的 softmax）时，模型的输出分布与原始 Qwen 完全相同，确保了知识的完美迁移。


## 5. 训练监控指标 (Training Monitoring Metrics)

为了有效评估和调试 CausalQwen 模型的训练过程，我们利用 Weights & Biases (wandb) 平台进行实时监控。本章提炼了 [`wandb_monitoring_metrics.md`](./experiments/wandb_monitoring_metrics.md) 中的核心指标，所有指标的计算都严格遵循本文档定义的数学原理。

### 5.1. 前提：基于确定性推理的评估

所有性能评估指标 (以 `eval/` 为前缀) **均基于确定性推理 (Deterministic Inference) 模式计算**。这保证了评估结果的**可复现性**和**稳定性**，为不同实验提供了可靠的基准。

### 5.2. 核心损失指标 (`train/*`)
-   **`train/accuracy`**: 在所有真实词元（应用 `attention_mask`）上计算分类准确率，衡量基础语言能力。
-   **`train/total_loss`**: 由平均分类损失 (`cls_loss_mean`) 和有效回归损失 (`reg_loss_effective`) 加权构成，是最终的优化目标。
-   **`train/cls_loss_mean`**: 在所有真实词元（应用 `attention_mask`）上计算的 OvR 分类损失的平均值，衡量基础语言能力。
-   **`train/reg_loss_effective`**: 仅在真实数值词元（应用数值掩码 `m`）上计算的门控回归损失的平均值，确保回归信号不被稀释。

### 5.3. 模型性能指标 (`eval/*`)

-   **词元预测准确率(`eval/accuracy`)** 
    - Perplexity 也可以考虑计算，但是 OvR 非归一化多分类概率，所以需要特殊处理才能使用。
-   **数值词元预测 (`eval/num_*`):
    -   **`num_precision`, `num_recall`, `num_f1`**: 全面评估模型在有效位置上辨别 `<NUM>` 词元的能力，是**门控机制性能的关键**。

-   **回归性能 (`eval/reg_*`)**:
    -   **`reg_mae` (平均绝对误差)**: 传统的误差度量，对异常值敏感。
    -   **`reg_mdae` (中位绝对误差)**: 对异常值稳健的误差度量。当 `mae` 远大于 `mdae` 时，表明存在少数极端错误的预测。

### 5.4. 内部状态分布指标 (`dist/*`)

这些指标的统计数据（`mean`, `median`, `std`, `iqr`）均在**有效词元位置（应用 `attention_mask`）** 上计算。

-   **因果表征 `U` (`dist/U_*`)**:
    -   通过对比 `U_loc` 和 `U_scale` 的 `mean`/`std` 与 `median`/`iqr`，我们可以深入分析其分布的**偏斜度**和**尾部重量**，从而诊断模型是否对特定词元学习到了特化表征，或是否明智地在困难样本上表达了更高的不确定性。

-   **OvR 校准 (`dist/ovr_prob_sum_*`)**:
    -   **标准推理模式**：`dist/ovr_prob_sum_median_standard` - 监控标准推理下的概率和
    -   **因果采样模式**：`dist/ovr_prob_sum_median_causal` - 监控固定个体下的概率和
    
    对于同一个确定的个体，其下一个词元只能有一个真值，因此因果采样模式下的概率和应该**更接近 1**，这是模型校准性的重要指标。

## 6. 使用流程图理解


### 图 1：CausalQwen 总体架构概览

这张图展示了模型最高层级的四大核心步骤，从输入到输出的完整流程。

```mermaid
graph TD
    A["<b>步骤 1: 数值感知嵌入</b><br>处理文本与数值输入"] --> B;
    B["<b>步骤 2: 特征提取</b><br>使用 Qwen 主干网络理解上下文"];
    B --> C["<b>步骤 3: 因果推断与决策</b><br>推断个体表征 U 并决定行动"];
    C --> D["<b>步骤 4: 输出预测</b><br>生成分类与回归结果"];

    style A fill:#e3f2fd
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#fce4ec
```

---

### 图 2：详解步骤 1 - 数值感知嵌入

这张图详细描绘了第一个模块如何将混合了文本和数值的原始输入，转化为统一的向量表示 `e`。

```mermaid
graph TD
    A["原始输入<br><i>'价格是99.9元'</i>"] --> B{"分词器"};
    B --> C["词元 ID 序列<br>input_ids"];
    B --> D["对齐的数值<br>numeric_values"];
    
    C --> E["词元嵌入层"];
    E --> F["基础嵌入<br>base_embeddings"];
    
    D --> G["数值编码函数<br>φ(v) = sign(v)·ln(1+|v|)·w_num"];
    
    F & G --> H["融合"];
    H --> I["<b>增强嵌入 e</b><br>[B, S, H]"];

    style I fill:#e3f2fd,stroke:#1b5e20,stroke-width:2px
```

---

### 图 3：详解步骤 2 & 3 - 因果核心流程

这张图展示了模型的核心机制：如何从上下文特征 `z` 推断出代表个体的因果分布 `U`，并基于 `U` 和内部噪声 `noise` 产生决策分布 `S` 和 `Y`。

```mermaid
graph TD
    A["增强嵌入 e<br>[B, S, H]"] --> B["<b>Qwen 特征网络</b>"];
    B --> C["上下文特征 z<br>[B, S, H]"];
    C --> D["<b>归因推断网络 (Abduction)</b>"];
    D --> E["<b>个体因果表征 U<br>Uᵢ ~ Cauchy(loc, scale)</b>"];
    
    E --> F{"<b>行动网络 (Action)</b><br>基于U与内部噪声<br>进行决策"};
    F --> G["分类决策分布 S<br>S_{k,i} ~ Cauchy(...)"];
    F --> H["回归决策分布 Y<br>Y_i ~ Cauchy(...)"];

    style E fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style G fill:#fbe9e7
    style H fill:#fbe9e7
```

---

### 图 4：详解步骤 4 


#### 4.1 CausalQwen 三种推理模式

这张图展示了CausalQwen的三种推理模式，从输入文本到最终预测的完整流程。

```mermaid
graph TB
    Start["输入文本"] --> Forward["前向传播<br>(数值感知嵌入 → Qwen → 归因推断)"]
    Forward --> U["个体因果表征<br>U ~ Cauchy(loc_U, scale_U)"]
    
    U --> Mode1["标准模式 (Standard)<br>do_sample=False, T>0"]
    U --> Mode2["采样模式 (Sampling)<br>do_sample=True, T>0"]
    U --> Mode3["因果模式 (Causal)<br>T=0"]
    U --> Mode4["兼容模式 (Compatible)<br>传统Softmax"]
    
    subgraph "标准模式"
        Mode1 --> Noise1["噪声融合到尺度参数<br>U' ~ Cauchy(μ, γ + T·|b_noise|)"]
        Noise1 --> Action1["行动网络<br>(解析分布变换)"]
        Action1 --> S1["S_k ~ Cauchy(loc_S_k, scale_S_k)"]
        Action1 --> Y1["Y ~ Cauchy(loc_Y, scale_Y)"]
        S1 --> OvR1["计算OvR概率<br>P_k = P(S_k > C_k)"]
        OvR1 --> Cls1["分类: argmax_k P_k"]
        Y1 --> Reg1["回归: loc_Y"]
    end
    
    subgraph "采样模式"
        Mode2 --> Sample2["采样噪声扰动位置<br>ε ~ Cauchy(0,1)"]
        Sample2 --> Noise2["U' ~ Cauchy(μ + T·|b_noise|·ε, γ)"]
        Noise2 --> Action2["行动网络"]
        Action2 --> S2["S_k ~ Cauchy(...)"]
        Action2 --> Y2["Y ~ Cauchy(...)"]
        S2 --> OvR2["计算OvR概率"]
        OvR2 --> Cls2["分类: argmax_k P_k"]
        Y2 --> Reg2["回归: loc_Y"]
    end
    
    subgraph "因果模式"
        Mode3 --> Pure3["纯因果表征<br>U' ~ Cauchy(μ, γ)"]
        Pure3 --> Action3["行动网络"]
        Action3 --> S3["S_k ~ Cauchy(...)"]
        Action3 --> Y3["Y ~ Cauchy(...)"]
        S3 --> OvR3["计算OvR概率"]
        OvR3 --> Cls3["分类: argmax_k P_k"]
        Y3 --> Reg3["回归: loc_Y"]
    end
    
    subgraph "兼容模式"
        Mode4 --> Skip4["跳过因果机制"]
        Skip4 --> Logits4["直接使用 loc_S<br>作为 logits"]
        Logits4 --> Softmax4["Softmax归一化"]
        Softmax4 --> TopK4["Top-k/Top-p 采样"]
    end
    
    Cls1 & Reg1 --> Output1["输出预测"]
    Cls2 & Reg2 --> Output2["输出预测"]
    Cls3 & Reg3 --> Output3["输出预测"]
    TopK4 --> Output4["输出预测"]
    
    style U fill:#fff3e0,stroke:#e65100,stroke-width:3px
    style Output1 fill:#fce4ec,stroke:#880e4f
    style Output2 fill:#fce4ec,stroke:#880e4f
    style Output3 fill:#fce4ec,stroke:#880e4f
    style Output4 fill:#fce4ec,stroke:#880e4f
```

---

#### 4.3 序列因果采样：共享外生噪声

这张图展示了使用固定外生噪声实例进行序列因果采样的完整流程，探索系统性偏差对生成结果的影响。

```mermaid
graph TB
    Start["初始化序列生成"] --> Loop["开始自回归循环<br>t = 1, 2, ..."]
    InitNoise["生成固定噪声实例<br>ε_noise ~ Cauchy(0, I)"]
    
    subgraph "因果采样生成主循环"
        Loop --> Forward["归因推断网络 U_t"]
        
        Forward --> NoiseInject["核心：噪声注入<br>U'_t ~ Cauchy(loc_U_t + T·|b_noise|·ε_noise, scale_U_t)"]
        
        NoiseInject --> Action["行动决策网络"]
        
        Action --> Classify["OvR分类预测<br>P_k,t = P(S_k,t > C_k)"]
        Action --> Regress["回归预测<br>pred_value_t = loc_Y_t"]
        
        Classify --> Decision{预测词元类型}
        Regress --> Decision
        
        Decision -->|数值| NumUpdate["使用回归值:<br>input_ids.append(NUM token)<br>numeric_values.append(pred_value_t)<br>生成文本.append(str(pred_value_t))"]
        Decision -->|文本| TextUpdate["使用分类结果:<br>input_ids.append(pred_token_id)<br>numeric_values.append(0.0)<br>生成数值.append(token_text)"]
        
        NumUpdate --> Check{"结束条件?<br>(EOS或最大长度)"}
        TextUpdate --> Check
        
        Check -->|否| Next["t = t + 1"]
        Check -->|是| End["输出完整生成序列"]
        
        Next --> Forward
    end
    
    InitNoise --> NoiseInject

    style Start fill:#e8f5e9
    style InitNoise fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style NoiseInject fill:#fbe9e7,stroke:#e91e63,stroke-width:3px
    style Decision fill:#e1f5fe
    style NumUpdate fill:#fbe9e7
    style TextUpdate fill:#e3f2fd
    style End fill:#e8f5e9
```

**关键特点**：
1. **固定噪声实例**：整个序列使用同一个 `ε_noise`，保证系统性一致
2. **因果注入机制**：每个位置的 `U'_t` 都受到相同噪声影响，但通过温度参数 `T` 控制强度
3. **双通道决策**：同时进行分类和回归预测
4. **序列一致性**：维持 `input_ids` 和 `numeric_values` 的对齐

**应用场景**：反事实分析、风格一致性生成、因果干预实验。



### 图 5：损失流程图



#### 图 5.1：分类损失 (`L_cls`) 的计算

这张图展示了如何从模型对全部词汇的预测分布，计算出每个位置的分类总损失。

```mermaid
graph TD
    subgraph "输入"
        A["<b>loc_S</b><br>形状: [B, S, V_full]"]
        B["<b>scale_S</b><br>形状: [B, S, V_full]"]
        C["<b>C_k 阈值参数</b><br>形状: [V_full]"]
        D["<b>真实标签 labels</b><br>形状: [B, S]"]
    end

    subgraph "OvR 概率计算"
        A & B & C --> E["对每个词汇 k 计算:<br>P_{k,i} = 1/2 + (1/π)arctan((loc_S_{k,i} - C_k)/scale_S_{k,i})"]
        E --> F["<b>OvR 概率张量 P</b><br>形状: [B, S, V_full]"]
    end
    
    subgraph "损失计算"
        D --> G["转换为 one-hot 编码 y<br>形状: [B, S, V_full]"]
        F & G --> H["计算二元交叉熵:<br>-[y·log(P) + (1-y)·log(1-P)]"]
        H --> I["对词汇维度求和"]
    end

    I --> J["<b>分类损失 L_cls</b><br>形状: [B, S]"]
    
    style J fill:#e3f2fd,stroke:#1b5e20,stroke-width:2px
```

**关键点**：
- 使用 OvR（One-versus-Rest）方法，每个词汇独立计算二元分类概率
- 引入可学习的阈值参数 `C_k`，允许模型为不同词汇学习不同的决策边界
- 最终损失是所有词汇的二元交叉熵之和

---

#### 图 5.2：门控回归损失 (`L_reg_gated`) 的计算

这张图详细展示了门控机制如何结合分类概率和回归损失。

```mermaid
graph TD
    subgraph "输入"
        A["<b>loc_Y</b><br>形状: [B, S]"]
        B["<b>scale_Y</b><br>形状: [B, S]"]
        C["<b>numeric_values</b><br>(真实数值)<br>形状: [B, S]"]
        D["<b>P_&lt;NUM&gt;</b><br>(来自分类)<br>形状: [B, S]"]
        E["<b>num_mask</b><br>(labels == NUM_TOKEN_ID)<br>形状: [B, S]"]
    end

    subgraph "路径 A：柯西负对数似然"
        A & B & C --> F["L_nll = log(π·scale_Y) +<br>log(1 + ((y_true - loc_Y)/scale_Y)²)"]
        F --> G["<b>基础回归损失 L_nll</b><br>形状: [B, S]"]
    end

    subgraph "路径 B：门控权重计算"
        D & E --> H["Gate = num_mask ×<br>(α + (1-α)·P_&lt;NUM&gt;)"]
        H --> I["<b>门控权重 Gate</b><br>形状: [B, S]"]
    end

    G & I --> J["L_reg_gated = Gate × L_nll<br>(逐元素相乘)"]
    J --> K["<b>门控回归损失 L_reg_gated</b><br>形状: [B, S]"]

    style K fill:#fff3e0,stroke:#e65100,stroke-width:2px
```



#### 图 5.1：分类损失 (`L_cls`) 的计算

这张图展示了如何从模型对全部词汇的预测分布，计算出每个位置的分类总损失。

```mermaid
graph TD
    subgraph "输入"
        A["<b>loc_S</b><br>形状: [B, S, V_full]"]
        B["<b>scale_S</b><br>形状: [B, S, V_full]"]
        C["<b>C_k 阈值参数</b><br>形状: [V_full]"]
        D["<b>真实标签 labels</b><br>形状: [B, S]"]
    end

    subgraph "OvR 概率计算"
        A & B & C --> E["对每个词汇 k 计算:<br>P_{k,i} = 1/2 + (1/π)arctan((loc_S_{k,i} - C_k)/scale_S_{k,i})"]
        E --> F["<b>OvR 概率张量 P</b><br>形状: [B, S, V_full]"]
    end
    
    subgraph "损失计算"
        D --> G["转换为 one-hot 编码 y<br>形状: [B, S, V_full]"]
        F & G --> H["计算二元交叉熵:<br>-[y·log(P) + (1-y)·log(1-P)]"]
        H --> I["对词汇维度求和"]
    end

    I --> J["<b>分类损失 L_cls</b><br>形状: [B, S]"]
    
    style J fill:#e3f2fd,stroke:#1b5e20,stroke-width:2px
```

**关键点**：
- 使用 OvR（One-versus-Rest）方法，每个词汇独立计算二元分类概率
- 引入可学习的阈值参数 `C_k`，允许模型为不同词汇学习不同的决策边界
- 最终损失是所有词汇的二元交叉熵之和

---

#### 图 5.2：门控回归损失 (`L_reg_gated`) 的计算

这张图详细展示了门控机制如何结合分类概率和回归损失。

```mermaid
graph TD
    subgraph "输入"
        A["<b>loc_Y</b><br>形状: [B, S]"]
        B["<b>scale_Y</b><br>形状: [B, S]"]
        C["<b>numeric_values</b><br>(真实数值)<br>形状: [B, S]"]
        D["<b>P_&lt;NUM&gt;</b><br>(来自分类)<br>形状: [B, S]"]
        E["<b>num_mask</b><br>(labels == NUM_TOKEN_ID)<br>形状: [B, S]"]
    end

    subgraph "路径 A：柯西负对数似然"
        A & B & C --> F["L_nll = log(π·scale_Y) +<br>log(1 + ((y_true - loc_Y)/scale_Y)²)"]
        F --> G["<b>基础回归损失 L_nll</b><br>形状: [B, S]"]
    end

    subgraph "路径 B：门控权重计算"
        D & E --> H["Gate = num_mask ×<br>(α + (1-α)·P_&lt;NUM&gt;)"]
        H --> I["<b>门控权重 Gate</b><br>形状: [B, S]"]
    end

    G & I --> J["L_reg_gated = Gate × L_nll<br>(逐元素相乘)"]
    J --> K["<b>门控回归损失 L_reg_gated</b><br>形状: [B, S]"]

    style K fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

**门控机制解释**（当 α=0 时）：
- `Gate = num_mask × P_<NUM>` 
- 只有当真实标签是数值（`num_mask=1`）且模型预测为数值的概率高时，回归损失才被充分计算
- 这种软门控避免了硬性的 0/1 切换，使梯度流更加平滑

---

#### 图 5.3：总损失 (`L_total`) 的合并

这张图展示了如何正确地将分类损失和回归损失合并为最终的训练目标。

```mermaid
graph TD
    subgraph "损失张量输入"
        A["<b>L_cls</b><br>分类损失<br>[B, S]"]
        B["<b>L_reg_gated</b><br>门控回归损失<br>[B, S]"]
        C["<b>cls_mask</b><br>(= attention_mask)<br>[B, S]"]
        D["<b>num_mask</b><br>(labels == NUM_TOKEN_ID & attention_mask)<br>[B, S]"]
    end

    subgraph "分类损失归约"
        A & C --> E["L_cls_masked = L_cls × cls_mask"]
        E --> F["sum(L_cls_masked) / sum(cls_mask)"]
        F --> G["<b>L_cls_mean</b><br>平均分类损失<br>(标量)"]
    end

    subgraph "回归损失归约"
        B --> H["L_reg_gated"]
        H --> I["sum(L_reg_gated) / sum(num_mask)"]
        I --> J["<b>L_reg_effective</b><br>有效回归损失<br>(标量)"]
    end

    G & J --> K["L_total = L_cls_mean + λ × L_reg_effective"]
    K --> L["<b>L_total</b><br>最终总损失<br>(标量)"]
    D --> I

    style L fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style C fill:#f1f8e9
    style D fill:#f1f8e9
```

**关键设计决策**：
1. **分类损失**：在所有有效词元上平均（使用 `cls_mask`）
2. **回归损失**：仅在数值词元上平均（使用 `num_mask`），避免稀疏性导致的信号稀释
3. **加权系数 λ**：平衡两个任务的重要性，是一个重要的超参数

这种设计确保了：
- 分类任务覆盖所有词元，保持语言建模能力
- 回归任务专注于数值预测，不被非数值位置稀释
- 两个任务通过门控机制协同学习

## 7. 核心洞察与总结

CausalQwen 的数学框架三个特色：

1.  **因果表征**：通过 $U$ 建模个体因果性差异
2.  **分布计算**：利用柯西分布的线性性质，实现无采样训练
3.  **统一架构**：设计为 Qwen 的子类，通过扩展而非重构增加数值处理能力

### 7.1 ⚖️ CausalQwen vs. 标准 Qwen 对比清单

为了清晰地展示 CausalQwen 的创新之处，我们将其与标准的 Qwen 模型在几个核心维度上进行直接比较。

| 对比维度 (Dimension) | 标准 Qwen (Standard Qwen) | CausalQwen |
| :--- | :--- | :--- |
| **核心假设** | **关联性**：学习输入 $X$ 和输出 $Y$ 之间的条件概率分布 $P(Y\|X)$。 | **因果性**：学习一个普适的因果函数 $Y = f(t; u, \text{noise})$，其中 $u$ 是个体属性，$\text{noise}$ 是外生噪声。 |
| **数值处理** 🔢<br>Numerical Handling | **视为纯文本 (As Plain Text)**<br>将数字（如 "99.9"）当作普通词元处理，缺乏内在的数值概念。 | **双通道处理 (Dual-Channel)**<br>文本部分走词元嵌入，数值部分走独立的**回归通道**，真正理解数值大小。 |
| **输出架构** 🏛️<br>Output Architecture | **单一 Logits 输出 (Single Logits Output)**<br>输出一个维度为词汇表大小的 logits 向量，用于 Softmax。 | **双重分布输出 (Dual Distribution Output)**<br>输出独立的**分类 OvR 分布**和**回归柯西分布**，分别处理文本与数值。 |
| **损失函数** 🧮<br>Loss Function | **Softmax 交叉熵 (Softmax Cross-Entropy)**<br>在整个词汇表上进行归一化，计算单一正确答案的损失。 | **OvR + 门控回归损失 (Gated Reg Loss)**<br>分类上进行独立二元判断，回归上由分类结果**智能门控**，实现多任务学习。 |
| **核心创新** ✨<br>Key Innovation | 强大的语言建模与上下文理解能力。 | 引入外生**个体选择变量 $U$**，显式建模**外生噪声 $\text{noise}$**，并利用柯西分布的数学特性，构建了一个可高效训练的因果生成框架。 |


