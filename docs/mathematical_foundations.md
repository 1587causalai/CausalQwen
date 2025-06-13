# 因果语言模型数学概览

本文档旨在为读者提供 CausalQwen 模型核心数学思想的直观概览。

## 1. 核心创新：引入个体选择变量 U

为了真正实现因果推理，我们需要一个能够对个体的内在基因进行建模的框架。本项目的理论基石 ([arXiv:2401.15911](https://arxiv.org/abs/2401.15911)) 从数学上证明，为了构建一个能够灵活表达反事实的因果模型，引入一个外生的 **"个体选择变量" $U$** 是必要的。 $U$ 是理解本模型所有魔法的关键。它有两个核心身份：

1.  **个体选择变量 (Individual Selection Variable)**：一次具体的赋值 $U=u$ 代表着从所有可能的个体中"选中"了某一个特定个体 `u`。
2.  **个体因果表征 (Individual Causal Representation)**：被选中的向量 $u$ 本身，就包含了该个体所有内在的、驱动其行为的潜在属性。

**核心思想**：普适的因果律 ($Y=f(T;u)$) 应用于不同的个体 ($u$)，从而产生了不同的反事实结果 ($Y(t)$)。$U$ 是所有个体性差异的最终来源。

> 深度解读请参见: [`design-docs/U_deep_dive.md`](design-docs/U_deep_dive.md)

## 2. 训练阶段：前向传播 (Forward Pass)

模型训练的核心是执行一个完整的前向传播，计算预测值与真实标签之间的损失，然后通过反向传播更新模型参数。整个前向传播过程可以分解为五个核心模块。

> 我们用 B 代表批次大小, S 代表序列长度, H 代表模型核心维度 (即词嵌入和隐藏层维度), C 代表因果表征维度, K 代表基座模型 Qwen 的已用词汇表大小, V_full代表总词汇表大小, CausalQwen 的已用词汇表大小为 K+1 (K+1 包含基座模型 Qwen 的已用词汇表大小 K 和 CausalQwen 的额外词汇 `<NUM>`) 

### 2.1 模块一：数值感知嵌入 (Numerical-aware Embedding)
这一模块的目标是将混合了文本和数值的原始输入，转化为一个统一的、数值感知的特征向量序列。这个过程分为两步：

1.  **分词**: 接收原始文本，通过分词器识别数值并替换为`<NUM>`词元，最终输出`input_ids`和`numerical_values`。
2.  **嵌入**: 结合词元的基础嵌入和数值的对数编码，计算出增强嵌入。

-   **输入**: 
    - `input_ids` (形状: `[B, S]`), 经过分词器处理的词元ID序列。
    - `numerical_values` (形状: `[B, S]`), 与词元序列对齐的数值序列。
-   **处理**: 通过结合词元的基础嵌入和数值的对数编码，计算出增强嵌入：
    \[
    e_i = \text{embed}(x_i) + \phi(v_i) \quad \text{,其中 } \phi(v) = \text{sign}(v) \cdot \ln(1 + |v|) \cdot \vec{e}
    \]
-   **输出**: 
    - `e`: 增强嵌入张量 (形状: `[B, S, H]`)

> **设计动机**: 选择对数编码 $\phi(v)$ 是因为它具有三大优势：1) **数值稳定性**，将大范围数值压缩到合理区间；2) **相对误差保持**，对数空间中的等距对应原空间的等比；3) **自然退化**，由于$\phi(0)=0$，非数值位置自然退化为标准词元嵌入，无需特殊处理。

### 2.2 模块二：特征提取网络 (Feature Extraction Network)
该模块使用一个标准的 Transformer 网络（如Qwen）作为主干，来深度理解序列的上下文信息。

-   **输入**: 
    - `e`: 增强嵌入张量 (形状: `[B, S, H]`)
-   **处理**: 增强嵌入序列 `e` 被送入Qwen的Transformer主干网络中。通过多层自注意力机制，网络为每个位置的词元计算出融合了全局上下文信息的深层特征：
    \[
    z = \text{FeatureNetwork}(e)
    \]
-   **输出**: 
    - `z`: 上下文特征张量 (形状: `[B, S, H]`)

### 2.3 模块三：归因推断网络 (Abduction Network)
该模块从上下文特征中推断出每个位置的、更深层次的个体因果表征。这对应着"你是谁？"的归因过程。

-   **输入**: 
    - `z`: 上下文特征张量 (形状: `[B, S, H]`)
-   **处理**: 一个线性层（或一个小型MLP）作为归因网络，为每个位置独立地计算出因果表征 $U_i$ 所服从的柯西分布的参数：
    \[
    (\text{loc}_{U_i}, \log(\text{scale}_{U_i})) = \text{AbductionNetwork}(z_i)
    \]
-   **输出**: 
    - `loc_U`: 因果表征分布的位置参数 (形状: `[B, S, C]`)
    - `scale_U`: 因果表征分布的尺度参数 (形状: `[B, S, C]`)

### 2.4 模块四：行动决策网络 (Action Network)
该模块基于推断出的因果表征分布，进行并行的分类和回归决策。这对应着"你会做什么？"的行动过程。

-   **输入**: 
    - `loc_U` (形状: `[B, S, C]`) 和 `scale_U` (形状: `[B, S, C]`)
-   **处理**: 通过两个独立的线性变换，将因果表征分布映射到分类和回归的决策空间。
    \[
    \begin{aligned}
    (\text{loc}_{S_k}, \text{scale}_{S_k}) &= \text{Action}_{\text{cls}}(\text{loc}_U, \text{scale}_U) \\
    (\text{loc}_Y, \text{scale}_Y) &= \text{Action}_{\text{reg}}(\text{loc}_U, \text{scale}_U)
    \end{aligned}
    \]
-   **输出**:
    - 分类决策分布参数: `loc_S` (形状: `[B, S, V_full]`), `scale_S` (形状: `[B, S, V_full]`)
    - 回归决策分布参数: `loc_Y` (形状: `[B, S]`), `scale_Y` (形状: `[B, S]`)

> **核心引擎：柯西分布的线性稳定性**
> 整个归因-行动流程之所以能高效運作，完全得益于柯西分布的**线性稳定性**。如果一个随机变量 $U \sim \text{Cauchy}(\mu, \gamma)$，那么它的任何线性变换 $Y = aU + b$ 之后，依然服从柯西分布 $Y \sim \text{Cauchy}(a\mu + b, |a|\gamma)$。
> 这意味着，行动网络中的线性变换可以直接作用于分布的参数（`loc` 和 `scale`），而无需进行任何耗时的随机采样。这是模型能够被高效训练的关键。

### 2.5 模块五：损失计算 (Loss Calculation)
此模块计算模型预测与真实标签之间的差异，为反向传播提供依据。它由两部分组成：

#### 1. OvR 分类损失
我们不使用标准的 Softmax，而是对每个类别进行独立的"一对多"（One-vs-Rest, OvR）判断。

-   **输入**: 
    - 分类决策分布参数: `loc_S`, `scale_S` (形状: `[B, S, V_full]`)
-   **处理**: 
    1.  利用柯西分布的累积分布函数（CDF）计算每个类别的概率：
        \[
        P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)
        \]
    2.  基于此概率计算标准的二元交叉熵损失。
-   **输出**: 
    - `L_cls`: 序列的分类总损失。

#### 2. 门控回归损失
我们只希望在模型确定当前位置是数值（`<NUM>`）时，才对其回归预测的准确性进行惩罚。

-   **输入**: 
    - 回归决策分布参数: `loc_Y`, `scale_Y` (形状: `[B, S]`)
    - 真实数值: `true_numerical_values` (形状: `[B, S]`)
    - `<NUM>`词元的预测概率: `P_<NUM>,i`
-   **处理**: 计算由分类概率加权的柯西负对数似然损失：
    \[
    \mathcal{L}_{\text{reg\_gated},i} = m_i \cdot \left(\alpha + (1-\alpha) \cdot P_{\text{<NUM>},i}\right) \cdot \mathcal{L}_{\text{cauchy\_nll},i}
    \]
    其中 $m_i$ 是指示真实标签是否为`<NUM>`的掩码, $\alpha$ 是一个可调的门控系数。
-   **输出**: 
    - `L_reg`: 序列的回归总损失。

最终的总损失是这两者的加权和: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{reg}}$。

## 3. 推理阶段：生成预测 (Inference)

在模型训练完成后，我们使用它来生成预测。CausalQwen 提供两种推理模式。

### 3.1 确定性推理 (Deterministic Inference)
这是默认的、最高效的推理模式。它完全基于解析计算，不涉及任何随机采样。
- **分类预测**: 直接使用前向传播计算出的各类 OvR 概率，并选择概率最高的类别。
    \[
    \hat{y}_{\text{cls},i} = \arg\max_k P_{k,i}
    \]
- **回归预测**: 直接使用回归值分布的位置参数（中位数），这是对柯西分布最稳健的点估计。
    \[
    \hat{y}_{\text{reg},i} = \text{loc}_{Y_i}
    \]

### 3.2 因果采样 (Causal Sampling)
这是一种新颖的随机推理范式，它不直接对不确定的"结果"进行采样，而是对不确定的"原因"（即个体因果表征 U）进行采样。这是一种比`top-k`/`top-p`更符合因果直觉的采样方法。

1.  **采样"原因"**: 根据上下文推断出的因果表征分布 `Cauchy(loc_U, scale_U)`，从中采样一个**具体的**表征向量 $u_i$。
2.  **观察"结果"**: 将这个确定的 $u_i$ 传入**行动网络**，得到**确定性的**分类分数和回归值，并据此做出最终预测。

传统大模型需要 Top-P 和 Top-K 采样，是因为它们提供了一种在保真度（fidelity）和多样性（diversity）之间进行权衡的有效手段。我们类似可以使用似然截断来来实现对应的功能， 也就是说概率密度 $p_{U_i}(u_i)$ 需要大于某个数值或者等于$\frac{1}{\pi \gamma_{U_i}}$(Cauchy密度最大值)的 $u_i$ 才被保留。



### 3.3 兼容传统采样 (Compatibility with Traditional Sampling)
除了独有的因果采样，CausalQwen 在设计上完全兼容传统语言模型（如Qwen）的 `top-k`/`top-p` 采样方法。

行动网络输出的决策位置参数 `loc_S` (形状: `[B, S, V_full]`) 可以被直接视作标准语言模型输出的 logits。通过对 `loc_S` 应用 `Softmax` 函数，我们可以得到一个归一化的词汇表概率分布：
\[
P_{\text{softmax}}(y_i=k|x) = \frac{\exp(\text{loc}_{S_{k,i}})}{\sum_{j=1}^{V_{\text{full}}} \exp(\text{loc}_{S_{j,i}})}
\]
随后，便可在这组概率上执行标准的 `top-k`/`top-p` 采样。这保证了 CausalQwen 可以作为 Qwen 的一个直接替代和功能超集来使用。
### 3.4 (可选)共享随机性：因果采样的高级模式

因果采样的一个核心优势，在于通过**重参数化技巧 (Reparameterization Trick)**，实现了对生成过程随机性的精细控制。

为了从推断出的因果表征分布 $\text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i})$ 中采样，我们不直接进行随机抽取，而是执行一个确定性变换：
1.  首先，从标准均匀分布中采样一个**固定的**随机向量 $\vec{\epsilon} \sim U(0, 1)^C$。
2.  然后，使用此 $\vec{\epsilon}$ 为序列中的**每一个**位置 $i$ 计算其因果表征 $u_i$：
    \[
    u_i = \text{loc}_{U_i} + \text{scale}_{U_i} \odot \tan\left(\pi \left(\vec{\epsilon} - 0.5\right)\right)
    \]
    其中 $\odot$ 表示逐元素相乘。

**深刻的含义：分离与共享**

这个技巧的巧妙之处在于它将模型的内在逻辑（由上下文决定的 $\text{loc}_{U_i}$ 和 $\text{scale}_{U_i}$）与外在的随机性（由 $\vec{\epsilon}$ 代表）完全分离。更重要的是，它允许我们在一次完整的生成任务（如生成一段回复）中，**只采样一次 $\vec{\epsilon}$ 并持续使用它**。这意味着：
- **随机性来源是统一的**：整个序列的所有词元共享同一个"随机种子"或"灵感来源" $\vec{\epsilon}$。
- **上下文驱动多样性**：词元间的差异完全由模型根据上下文动态计算出的 $\text{loc}_{U_i}$ 和 $\text{scale}_{U_i}$ 决定。

这与传统 `top-k`/`top-p` 采样在每一步都独立进行随机抽样的方式形成了鲜明对比。CausalQwen 的方法更贴近人类的表达方式：一旦确定了某种"谈话风格"或"核心意图"（由 $\vec{\epsilon}$ 固化），整个句子都会围绕它连贯地展开，而不是每个词都随机地偏离主题。这为生成更具一致性和个性化风格的文本提供了坚实的数学基础。

## 4. 初始化策略：知识迁移

为了使 CausalQwen 能够无缝继承基座模型的强大语言能力，我们采用了一种**简单而精确**的初始化策略。其核心思想是：**在训练开始时，CausalQwen 的行为应与原始的 Qwen 完全一致**。设计原则如下：

1. **数学恒等**：新增模块在初始时表现为恒等映射或零映射
2. **知识保持**：完整继承 Qwen 的语言表征能力  
3. **渐进激活**：新功能在训练过程中逐步被"唤醒"

总共四步初始化：

#### 步骤1：数值感知嵌入 → 保守初始化 (数值感知嵌入层)

数值感知嵌入层的初始化需要确保数值编码不会对原有的词元嵌入造成过大干扰。

- **`<NUM>` 词元嵌入处理**：因为我们将 `<NUM>` 设置成在 Qwen 词汇表中的第一个保留词元，直接继承 Qwen 的 `<NUM>` 嵌入,无需额外初始化：$$\text{embed}(\text{<NUM>}) \leftarrow \text{embed}_{\text{Qwen}}(\text{<NUM>})$$

- **方向向量初始化**(后续可以考虑让他是可学习参数)：
$$\vec{e} \sim \mathcal{N}(0, \sigma_e^2 I), \quad \text{然后归一化: } \vec{e} \leftarrow \frac{\vec{e}}{\|\vec{e}\|}$$
其中 $\sigma_e$ 是小的标准差（如 $\sigma_e = 0.02$）。

最终，数值($v_i$)感知嵌入层的计算公式为：
$$e_i = \text{embed}(x_i) + \phi(v_i), \quad \text{where }  \phi(v) = \text{sign}(v) \cdot \ln(1 + |v|) \cdot \vec{e}$$

#### 步骤2：归因推断网络 → 恒等映射

设定归因网络的权重和偏置，使得：
$$\text{loc}_{U_i} = z_i, \quad \text{scale}_{U_i} = \gamma \text{ (大常数)}$$

**数学实现**：
- 位置参数：$W_{\text{loc}} = I$（恒等矩阵），$b_{\text{loc}} = 0$
- 尺度参数：$W_{\text{scale}} = 0$，$b_{\text{scale}} = \log(\gamma)$，其中 $\gamma$ 是大常数（如 $\gamma = 100$）

**效果**：因果表征 $U_i$ 的分布为宽泛分布 $U_i \sim \text{Cauchy}(z_i, \gamma)$，当 $\gamma$ 很大时，柯西分布近似**均匀先验**，归因网络在保持恒等映射的同时提供了高不确定性的表征。

**数学直觉**：大尺度的柯西分布 $\text{Cauchy}(\mu, \gamma)$ 当 $\gamma \gg 1$ 时，在较大范围内近似均匀分布，这为模型提供了"无知先验"——模型开始时对个体差异保持最大的不确定性。

#### 步骤3：行动网络(分类) → 复制 Qwen 权重

直接将 Qwen 的词汇表预测头权重复制到分类行动网络：
$$W_{\text{cls}} \leftarrow W_{\text{Qwen\_lm\_head}}, \quad b_{\text{cls}} = 0$$

**数学保证**：由于 $\text{loc}_{U_i} = z_i$，我们的分类 logits 与 Qwen 的原始 logits **完全相等**：
$$\text{loc}_{S_{k,i}} = W_{\text{cls}}[k, :] \cdot z_i + b_{\text{cls}}[k] = W_{\text{Qwen}}[k, :] \cdot z_i$$

**关键结论**：在兼容传统采样模式下，CausalQwen 的 Softmax 概率分布与 Qwen 的输出**数学上完全一致**：
$$P_{\text{CausalQwen}}^{\text{softmax}}(y_i=k|\mathbf{x}) = P_{\text{Qwen}}(y_i=k|\mathbf{x})$$

这确保了 CausalQwen 在初始化时不仅行为类似 Qwen，而是**精确地复制了 Qwen 的语言建模能力**。

#### 步骤4：行动网络(回归) → 常规初始化

将回归行动网络使用标准的小权重初始化，如 Xavier 或 Kaiming 初始化。

**数学效果**：由于 $\|W_{\text{reg}}\|$ 很小，结合大尺度的因果表征分布 $U_i \sim \text{Cauchy}(z_i, \gamma)$，回归预测的分布为：
$$Y_i \sim \text{Cauchy}(W_{\text{reg}} \cdot z_i, \gamma \cdot \|W_{\text{reg}}\|)$$

当 $\gamma \gg \|W_{\text{reg}}\|$ 时，回归输出近似为以 $0$ 为中心的宽泛分布，提供了**无偏的回归先验**。

#### 步骤5：OvR 阈值 → 统一设置

在 OvR 分类中，阈值 $C_k$ 决定了柯西分布决策分数超过阈值的概率计算：

$$P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

**阈值初始化**：所有类别使用相同的常数阈值
$$C_k = C_{\text{OvR}}, \quad \forall k \in \{0, 1, \ldots, V_{\text{full}}-1\}$$

其中 $C_{\text{OvR}}$ 是预设常数（如 100.0）, 后续可以考虑让它是可学习参数。

**数学效果**：
- $C_{\text{OvR}} = 0$: 初始概率接近 0.5，无明显偏好
- $C_{\text{OvR}} = 10$: 初始概率普遍较低，创造稀疏激活
- $C_{\text{OvR}} \geq 100$: 极度稀疏的初始概率分布

**推荐设置**：$C_{\text{OvR}} = 100.0$，这提供了良好的起点.

通过上述初始化步骤，CausalQwen 在训练开始时具有以下性质：

-   **因果表征**: 对于每个位置 $i$，因果表征 $U_i$ 服从宽泛的柯西分布 $U_i \sim \text{Cauchy}(z_i, \gamma)$，其中 $\gamma$ 是大常数。
-   **分类决策**: 分类行动网络的输出与 Qwen 的原始输出完全一致，即 $\text{loc}_{S_{k,i}} = W_{\text{Qwen}}[k, :] \cdot z_i$。
-   **回归决策**: 由于回归网络的标准初始化，回归预测的分布为 $Y_i \sim \text{Cauchy}(W_{\text{reg}} \cdot z_i, \gamma \cdot \|W_{\text{reg}}\|)$，当 $\gamma$ 很大时，近似为以 0 为中心的宽泛分布。
-   **阈值设置**: 所有类别共享相同的初始阈值 $C_k = C_{\text{OvR}}$，这提供了一个适度稀疏且数值稳定的初始概率分布。


## 5. 核心洞察与总结

CausalQwen 的数学框架三个特色：

1.  **因果表征**：通过 $U$ 建模个体因果性差异
2.  **分布计算**：利用柯西分布的线性性质，实现无采样训练
3.  **统一架构**：设计为 Qwen 的子类，通过扩展而非重构增加数值处理能力


---
更详细的数学推导请参考：[`design-docs/math/mathematical_foundations.md`](design-docs/math/mathematical_foundations.md)

