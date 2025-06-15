# 因果语言模型数学概览

cmt: 我修改了行动网络，增加了一个外生噪声注入层，以更加符合 DiscoSCM(arXiv:2401.15911) 框架设定。

cmt: 行动网络，我更片好的表达是行动决策网络

cmt: 流程图部分我还没有开始去仔细思考和更改的，我觉得先要把文字部分数学部分给讨论清楚。

本文档旨在为读者提供 CausalQwen 模型核心数学思想的直观概览。

## 1.核心创新：引入个体选择变量 U

为了真正实现因果推理，我们需要一个能够对个体的内在基因进行建模的框架。本项目的理论基石 ([arXiv:2401.15911](https://arxiv.org/abs/2401.15911)) 从数学上证明，为了构建一个能够灵活表达反事实的因果模型，引入一个外生的 **"个体选择变量" $U$** 是必要的。 $U$ 是理解本模型所有魔法的关键。它有两个核心身份：

1.  **个体选择变量 (Individual Selection Variable)**：一次具体的赋值 $U=u$ 代表着从所有可能的个体中"选中"了某一个特定个体 `u`。
2.  **个体因果表征 (Individual Causal Representation)**：被选中的向量 $u$ 本身，就包含了该个体所有内在的、驱动其行为的潜在属性。

**核心思想**：普适的因果律 ($Y=f(t;u, \text{noise})$) 应用于不同的个体 ($u$) 与外生噪声 ($\text{noise}$)，从而产生了不同的反事实结果 ($Y(t)$)。$U$ 是所有个体性系统性差异的最终来源，而 $\text{noise}$ 则代表了不可控的的外生随机扰动。  


## 2.训练阶段：前向传播 (Forward Pass)

模型训练的核心是执行一个完整的前向传播，计算预测值与真实标签之间的损失，然后通过反向传播更新模型参数。整个前向传播过程可以分解为五个核心模块。

> 我们用 B 代表批次大小, S 代表序列长度, H 代表模型核心维度 (即词嵌入和隐藏层维度), C 代表因果表征维度, K 代表基座模型 Qwen 的已用词汇表大小, V_full代表其总词汇表配额（V_full = K + 271, 271 是 Qwen 的预留词汇大小）, CausalQwen 的已用词汇表大小为 K+1 (K+1 包含基座模型 Qwen 的已用词汇表大小 K 和 CausalQwen 的额外词汇 `<NUM>`) 

cmt: 优化一下表达，和全文更一致的表达。

> **设计决策**: 在当前实现中，我们设定因果表征维度 `C` 与模型隐藏层维度 `H` 相等，即 **`C = H`**。这方便了我们进行归因推断网络的初始化。

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
    $$\phi(v) = \text{sign}(v) \cdot \ln(1 + |v|) \cdot \vec{e}$$
    其中 $v_i$ 是位置 $i$ 的数值（非数值位置为 0），$\vec{e}$ 是归一化的方向向量，$\|\vec{e}\| = 1$。
-   **输出**: 
    - `e`: 增强嵌入张量 (形状: `[B, S, H]`)

cmt: 这个没有说清楚这个 $\vec{e}$ 的含义其实是数值感知嵌入模块可学习的参数，需要优化一下表达，考虑换一个数学符号都可以， 比如 $\vec{e}$ 可以换成 $\vec{w}$。

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


cmt: 这个提醒一下我们后续学习不会改变这里参数，只有到非常非常后面的阶段才会考虑对其参数进行 Lora 微调， 既可以让我们 CausalQwen 快速学习到效果，也可以和 Qwen 有一个合理对比。 

### 2.3 模块三：归因推断网络 (Abduction Network)
该模块从上下文特征中推断出每个位置的个体因果表征分布。

-   **输入**: 上下文特征 `z` (形状: `[B, S, H]`)
-   **处理**: 通过线性层计算因果表征的分布参数：
    $$\text{loc}_{U_i} = W_{\text{loc}} \cdot z_i + b_{\text{loc}}$$
    $$\text{scale}_{U_i} = \text{softplus}(W_{\text{scale}} \cdot z_i + b_{\text{scale}})$$
    
    其中 $\text{softplus}(x) = \log(1 + \exp(x))$，保证尺度参数严格为正。
    
    **初始化后**：$\text{loc}_{U_i} = z_i$，$\text{scale}_{U_i} = \gamma$
-   **输出**: 
    - `loc_U`: 因果表征分布的位置参数 (形状: `[B, S, C]`)
    - `scale_U`: 因果表征分布的尺度参数 (形状: `[B, S, C]`)

cmt: 注意这里的参数是我们最优先微调， 多个原因导致 QwenTransformer 的参数一般不会被微调，对于新的领域数据，后续的 Action Network 的参数代表因果结果方程也一般不会进行微调。 Abduction Network 是最灵活最灵活的, 它甚至可以改变网络结构, 每个不同领域的额外数据来一个不同的网络结构, 但是微调的时候它是要保证 $DL(U, U_\text{base}) \leq \epsilon$ 的, 这样才能防止发生灾难性遗忘问题。

### 2.4 模块四：行动决策网络 (Action Network)

该模块是模型的核心决策单元。其内部包含一个可学习的噪声参数 $b_{noise}$，并且其工作流程分为两步：

1.  **噪声注入 (Noise Infusion)**：首先，网络将上游推断出的个体表征分布 $U_i \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i})$ 与内部的、代表不可控随机性的外生噪声分布 $\epsilon_i \sim \text{Cauchy}(0, |b_{noise}|)$ 进行叠加，形成一个"有效输入分布"：
    $$U'_{\text{effective}, i} \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i} + |b_\text{noise}|)$$
2.  **并行决策 (Decision Making)**：然后，网络基于这个包含了两种不确定性来源的有效分布，进行并行的分类和回归决策。



-   **输入**: `loc_U`, `scale_U` 
-   **处理**: 
    - **分类**：每个词汇 $k$ 有独立的线性变换。该变换作用于有效输入分布 $U'_{\text{effective}, i}$ 上：
      $$\text{loc}_{S_{k,i}} = W_{\text{cls},k} \cdot \text{loc}_{U_i} + b_{\text{cls},k} \\ 
      \text{scale}_{S_{k,i}} = |W_{\text{cls},k}| \cdot(\text{scale}_{U_i} + |b_\text{noise}|)$$
      
      **初始化后**：$\text{loc}_{S_{k,i}} = W_{\text{Qwen},k} \cdot z_i$（与 Qwen 相同的 logits）
      
    - **回归**：单一的线性变换，同样作用于有效输入分布 $U'_{\text{effective}, i}$：
      $$\text{loc}_{Y_i} = W_{\text{reg}} \cdot \text{loc}_{U_i} + b_{\text{reg}} \\ 
      \text{scale}_{Y_i} = \|W_{\text{reg}}\|_1 \cdot (\text{scale}_{U_i} + |b_{noise}|)$$
      
      **初始化后**：$Y_i \sim \text{Cauchy}(0, \epsilon\gamma)$（近似无偏先验）
      
-   **输出**:
    - 分类决策分布参数: `loc_S` (形状: `[B, S, V_full]`), `scale_S` (形状: `[B, S, V_full]`)
    - 回归决策分布参数: `loc_Y` (形状: `[B, S]`), `scale_Y` (形状: `[B, S]`)


从数学上看，由于噪声注入和决策本身都是线性变换，整个行动网络可以被视为一个单一的、作用于增广输入 $[U_i, \epsilon_i]$ 上的线性变换，这保证了我们可以持续利用柯西分布的线性稳定性进行解析计算。

cmt:被$b_\text{noise}$ 调制的线性变换, 你看看有什么更好的表达, 该参数的可学习性也没有很好的表达，换成 $w_\text{noise}$? 我希望你优化一下。


> **核心引擎：柯西分布的线性稳定性**
> 如果 $U \sim \text{Cauchy}(\mu, \gamma)$，那么 $aU + b \sim \text{Cauchy}(a\mu + b, |a|\gamma)$。这让我们无需采样就能计算变换后的分布。

### 2.5 模块五：损失计算 (Loss Calculation)

#### 1. OvR 分类损失
对每个类别计算独立的二元分类概率：
$$P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

然后计算所有类别的二元交叉熵之和：
$$L_{\text{cls}, i} = -\sum_{k=0}^{V_{\text{full}}-1} [y_{k,i} \log P_{k,i} + (1-y_{k,i}) \log (1 - P_{k,i})]$$

其中 $y_{k,i}$ 是 one-hot 编码的真实标签。

#### 2. 门控回归损失
柯西分布的负对数似然：
$$\mathcal{L}_{\text{nll},i} = \log(\pi \cdot \text{scale}_{Y_i}) + \log\left(1 + \left(\frac{y_{\text{true},i} - \text{loc}_{Y_i}}{\text{scale}_{Y_i}}\right)^2\right)$$

门控权重（$\alpha=0$ 时）：
$$\mathcal{L}_{\text{reg\_gated},i} = m_i \cdot P_{\text{<NUM>},i} \cdot \mathcal{L}_{\text{nll},i}$$

其中 $m_i$ 指示位置 $i$ 的真实标签是否为 `<NUM>`。

#### 3. 总损失
$$\mathcal{L}_{\text{total}} = \underbrace{\frac{\sum_i L_{\text{cls}, i} \cdot \text{mask}_i}{\sum_i \text{mask}_i}}_{\text{平均分类损失}} + \lambda \cdot \underbrace{\frac{\sum_i \mathcal{L}_{\text{reg\_gated},i}}{\sum_i m_i}}_{\text{有效回归损失}}$$

cmt: 我建议是优化一下表达, 提一个看法呀, 我们引入 cls_mask, num_mask 这样的概念，他们都是一种 attention_mask 的变种，然后我们用 cls_mask 来计算分类损失， 用 num_mask 来计算回归损失。 这样我们就可以避免计算那些没有意义的损失。如果你采用我这个建议，我会思考 padding token 有关逻辑如何处理？ 还有 ignore_index 有关逻辑如何处理？


## 3.推理阶段：生成预测 (Inference)

在模型训练完成后，我们使用它来生成预测。CausalQwen 提供了一个层次化的推理方法体系，从最高效的标准预测，到能体现因果性的生成式采样，再到用于深度反事实分析的高级模式，允许用户根据不同需求选择合适的工具。

### 3.1 确定性推理 (Deterministic Inference)
这是默认的、最高效的推理模式。它完全基于解析计算，不涉及任何随机采样。其目标是给出在综合考虑了所有不确定性（个体 `U` 和噪声 `noise`）之后，最稳健（robust）的点估计。
- **分类预测**: 直接使用前向传播计算出的各类 OvR 概率，并选择概率最高的类别。
    $$
    \hat{y}_{\text{cls},i} = \arg\max_k P_{k,i}
    $$
- **回归预测**: 直接使用回归值分布的位置参数（中位数），这是对柯西分布最稳健的点估计。
    $$
    \hat{y}_{\text{reg},i} = \text{loc}_{Y_i}
    $$


cmt: Deterministic Inference 的表达当然会误导，没有随机性， 你有什么比较好的名字吗？ normal inference 那也不太好, top-1 inference 那也不太好？point inference 那也不太好, 你有什么比较好的名字吗？quick inference 那也不太好, 你有什么比较好的名字吗？non-random inference 那也不太好, 你有什么比较好的名字吗？

### 3.2 因果采样 (Causal Sampling)
这是一种混合了随机性与确定性的高级推理模式，它深刻地体现了模型的因果哲学：分离可控的个体因素与不可控的随机噪声。其过程分为三步：

1.  **第一步：采样"个体" (Sample the Individual)**: 首先，我们对代表"个体性"的变量 $U$ 进行一次随机采样，从其后验分布 `Cauchy(loc_U, scale_U)` 中得到一个**具体的**个体因果表征向量 $u_i$。

2.  **第二步：构建"情境"分布 (Construct the Situational Distribution)**: 接下来，我们将这个确定的个体 $u_i$ 与模型中固有的、不可控的**外生噪声分布** `Cauchy(0, |b_noise|)` 相结合。其结果是一个新的、代表了"特定个体在随机情境中"的柯西分布：
    $$ U'_{\text{situational}, i} \sim \text{Cauchy}(u_i, |b_{noise}|) $$

> cmt: 这里需要优化一下表达，直接说构建决策输入？ 这个Situational 的表达我肯定是不想要的，因为它会引起一些不必要的误解，嗯，数学公式描述也不精确。

3.  **第三步：执行"确定性"决策 (Perform Deterministic Decision)**: 最后，我们将这个新的"情境分布" $U'_{\text{situational}, i}$ 作为输入，传入行动网络的决策层（即线性变换部分）。由于柯西分布的线性稳定性，我们可以**解析地**计算出最终的分类和回归决策分布，并像在标准确定性推理中一样，通过取分布的中位数（`loc`参数）来得到最终的预测值。

**核心思想**: 这种方法的精妙之处在于，它只在代表"个体选择"的步骤引入随机性，而将"环境噪声"保持为一种不确定性（一个分布）。这使得我们能够在探索不同个体（不同的采样 $u_i$）可能产生的结果的同时，仍然对每种结果做出最稳健（确定性）的预测。

### 3.3 兼容传统采样 (Compatibility with Traditional Sampling)
除了独有的因果采样，CausalQwen 在设计上完全兼容传统语言模型（如Qwen）的 `top-k`/`top-p` 采样方法。

行动网络输出的决策位置参数 `loc_S` (形状: `[B, S, V_full]`) 可以被直接视作标准语言模型输出的 logits。通过对 `loc_S` 应用 `Softmax` 函数，我们可以得到一个归一化的词汇表概率分布：
$$
P_{\text{softmax}}(y_i=k|x) = \frac{\exp(\text{loc}_{S_{k,i}})}{\sum_{j=1}^{V_{\text{full}}} \exp(\text{loc}_{S_{j,i}})}
$$
随后，便可在这组概率上执行标准的 `top-k`/`top-p` 采样, 这保证了 CausalQwen 可以作为 Qwen 的一个直接替代和功能超集来使用。

另一种可选的归一化方法是直接对所有类别的 OvR 概率进行求和，并以此为分母进行归一化，这为评估模型提供了不同的视角

### 3.4 高级（序列）因果采样：分解并固定随机性来源

CausalQwen 的设计允许我们对生成过程中的“随机性”进行前所未有的精细控制。传统的采样（如 top-k）在每一步都从一个临时的概率分布中抽取结果，导致随机性是无记忆、逐词元独立的。

高级（序列）因果采样模式则基于一个核心思想：**在一次完整的生成任务中，将模型两大不确定性来源（个体 `U` 或噪声 `noise`）之一的随机实例固定下来，并观察其在整个生成过程中如何持续地与另一不确定性来源相互作用。** 这使得我们能进行两种深刻的反事实探究。


#### 3.4.1 模式一：共享“个体选择因子” (Fixing the Individual, Observing its Interaction with Noise)

* **研究目标**：此模式旨在回答：“如果我们预先选定**一个确切的个体**，然后观察这个个体在面对一系列**不可预测的、逐-词元-独立的随机噪声**时，其行为会如何展开？”
* **实现机制**:

  1. **采样个体**: 在生成任务开始时，采样一个固定的“个体选择因子” $\vec{\epsilon}_{\text{seed}} \sim U(0, 1)^C$。在第 $i$ 步，我们用它计算出该上下文中的**具体个体表征** $u_i$：
     $$
     u_i = \text{loc}_{U_i} + \text{scale}_{U_i} \odot \tan\left(\pi \left(\vec{\epsilon}_{\text{seed}} - 0.5\right)\right)
     $$
  2. **构建情境分布**: 这个确定的个体 $u_i$ 随即进入一个充满不确定性的“决策情境”。其有效的输入心智状态不再是一个确定的向量，而是一个**包含了外生随机噪声的分布**：
     $$
     U'_{\text{effective}, i} \sim \text{Cauchy}(u_i, |b_{noise}|)
     $$
  3. **解析决策**: 我们将这个**有效输入分布** $U'_{\text{effective}, i}$ 传入行动网络。由于柯西分布的线性稳定性，我们可以**解析地**计算出最终的分类和回归决策分布，并取其位置参数（中位数）作为该步骤的最终预测。
* 随机性的来源被分解：代表“个体选择”的随机性（来自 $U$）在任务开始时被一次性固定。而代表“不可控扰动”的随机性（来自 `noise`）则在**每一步的决策中都保持其分布形态**，代表了每一token 生成决策时都会遇到的、全新的、不可预测的微小扰动。


#### 3.4.2 模式二：共享"系统性噪声实例" (Fixing the Noise, Observing the Uncertain Individual)

* **研究目标**：此模式旨在回答：“如果一个系统存在一种**持续不变的、系统性的随机影响**，那么一个本身具有内在不确定性的**个体（一个分布，而非一个实体）** 在这种影响下，其行为会如何展开？”
* **实现机制**:

  1. **采样噪声**: 在生成任务开始时，采样一个固定的“系统性噪声实例” $\vec{\epsilon}_{\text{noise}} \sim \text{Cauchy}(0, I)$。
  2. **构建受扰动的心智分布**: 在第 $i$ 步，我们不选定具体个体，而是取其完整的**个体因果表征分布** $U_i \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i})$。然后，我们将固定的噪声实例作用于这个分布的定位上。根据柯西分布的加法稳定性，我们得到一个新的、被系统性偏移的有效输入分布：
     $$
     U'_{\text{effective}, i} \sim \text{Cauchy}(\text{loc}_{U_i} + |b_{noise}| \cdot \vec{\epsilon}_{\text{noise}}, \text{scale}_{U_i})
     $$
  3. **解析决策**: 同样，我们将这个新的**有效输入分布** $U'_{\text{effective}, i}$ 传入行动网络，解析地计算出最终决策分布，并取其位置参数作为预测。
* 随机性的来源被再次分解：代表“不可控扰动”的随机性被一次性采样并固定下来，成为一种贯穿始终的“风格”或“偏差”。而代表“个体选择”的随机性则在**每一步都保持其完整的分布形态**（由变化的 `loc_U` 和 `scale_U` 定义），代表了个体在不同上下文下自身固有的不确定性。


## 4.初始化策略：知识迁移

为了使 CausalQwen 能够无缝继承基座模型的强大语言能力，我们采用了一种**简单而精确**的初始化策略。其核心思想是：**在训练开始时，CausalQwen 的行为应与原始的 Qwen 完全一致**。设计原则如下：

1. **数学恒等**：新增模块在初始时表现为恒等映射或零映射
2. **知识保持**：完整继承 Qwen 的语言表征能力  
3. **渐进激活**：新功能在训练过程中逐步被"唤醒"

总共四步初始化：

#### 步骤1：数值感知嵌入 → 保守初始化 (数值感知嵌入层)

数值感知嵌入层的初始化需要确保数值编码不会对原有的词元嵌入造成过大干扰。

- **`<NUM>` 词元嵌入处理**：因为我们将 `<NUM>` 设置成在 Qwen 词汇表中的第一个保留词元(Qwen2.5-0.5B有271个保留词元)，直接继承 Qwen 的 `<NUM>` 嵌入,无需额外初始化：$$\text{embed}(\text{<NUM>}) \leftarrow \text{embed}_{\text{Qwen}}(\text{<NUM>})$$

- **方向向量初始化**(后续可以考虑让他是可学习参数)：
$$\vec{e} \sim \mathcal{N}(0, \sigma_e^2 I), \quad \text{然后归一化: } \vec{e} \leftarrow \frac{\vec{e}}{\|\vec{e}\|}$$

其中 $\sigma_e$ 是小的标准差（如 $\sigma_e = 0.02$）。最终，位置 $i$ 的数值感知嵌入层的计算公式为：
$$e_i = \text{embed}(x_i) + \phi(v_i), \quad \text{where }  \phi(v) = \text{sign}(v) \cdot \ln(1 + |v|) \cdot \vec{e}$$

使用它作为输出，经过 Qwen 主干网络，得到高维的特征表征 $z_i$(形状: [B, S, H]), 作为后续归因推断网络的输入。

cmt: 我是觉得这个又是归一化又是什么是非必要的复杂，就像一般的神经网络权重一样进行设置？直接一个最常见的高维向量初始化（一般就是随机抽样，然后除以根号维度之类）？


#### 步骤2：归因推断网络 → 恒等映射

设定权重使得：
$$\text{loc}_{U_i} = z_i, \quad \text{scale}_{U_i} = \gamma$$

**实现**：
- 位置网络：$W_{\text{loc}} = I$（单位矩阵），$b_{\text{loc}} = 0$
- 尺度网络：$W_{\text{scale}} = 0$，$b_{\text{scale}} = \text{softplus}^{-1}(\gamma) = \log(\exp(\gamma) - 1)$

其中 $\gamma$ 是大常数（如 10），提供宽泛的初始分布。

> **注意**：当 $\gamma$ 较大时，$\text{softplus}^{-1}(\gamma) \approx \gamma - \log(2)$

#### 步骤3：行动决策网络(分类) → 复制 Qwen 权重

$$W_{\text{cls}} \leftarrow W_{\text{Qwen\_lm\_head}}, \quad b_{\text{cls}} = 0$$

由于 $\text{loc}_{U_i} = z_i$，所以：
$$\text{loc}_{S_{k,i}} = W_{\text{cls},k} \cdot z_i = W_{\text{Qwen},k} \cdot z_i$$

这确保了初始分类输出与 Qwen 完全一致。

cmt: 我本着一个非必要不进行复杂设置的原则, 我觉得一个初始的方案不要去考虑提供宽泛的初始分布这个事情，你就用最常规的初始化， 然后让模型在训练过程中去学习这个分布。


#### 步骤4：行动决策网络(回归) → 常规初始化


**数学效果**：由于 $\|W_{\text{reg}}\|_1$ 很小，结合大尺度的因果表征分布 $U_i \sim \text{Cauchy}(z_i, \gamma_i)$，位置 $i$ 的回归预测分布为：
$$Y_i \sim \text{Cauchy}(\mu_{\text{reg},i}, \gamma_{\text{reg},i}),  \\
\mu_{\text{reg},i} = W_{\text{reg}} \cdot z_i + b_{\text{reg}},  \gamma_{\text{reg},i} = |W_{\text{reg}}|_1 \cdot (\gamma_i + |b_{noise}|)$$

其中 $|W_{\text{reg}}|_1$ 是 $W_{\text{reg}}$ 权重的 L1 范数。在初始化阶段，回归输出近似为以 $0$ 为中心的宽泛分布，提供了**无偏的回归先验** ($\mu_{\text{reg}}$ 和 $\gamma_{\text{reg}}$ 的张量形状都是 [B, S]) 。

cmt: 我觉得所有的可学习参数的都可以使用最常规的初始化, 宽泛分布嗯的初始化自然是很好，但是有多大作用呢？值得我们特意去设置吗？不过这个地方有一个特别重要的点是，随着学习的进行，因果表征能解释的越来越多了， 外生噪声的影响力会越来越小， 所以噪声项的系数是越来越小才合理，你觉得呢？


#### 步骤5：噪声参数初始化 → 零初始化
为保证初始化的确定性，我们将噪声项的系数初始化为零：
$$b_{noise} \leftarrow 0$$
这确保了在训练开始时，模型行为与无噪声版本完全一致，噪声的影响力将在训练中被逐步学习。


cmt: 这个初始化在我看来就是非必要的复杂, 原因是你把结果的随机性刚开始的时候全部归因于因果表征， 假定一旦因果表征确定了，结果就完全确定了， 这是一个相对极端的情况，不符合核心理论 DiscoSCM 的设定。

#### 步骤6：OvR 阈值 → 统一设置

在 OvR 分类中，阈值 $C_k$ 决定了柯西分布决策分数超过阈值的概率计算：

$$P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

得一个形状是 [B, S, V_full] 的概率张量 $\mathbf{P}$。

**阈值初始化**：
- **标量形式**：所有类别使用相同的常数阈值
  $$C_k = C_{\text{OvR}}, \quad \forall k \in \{0, 1, \ldots, V_{\text{full}}-1\}$$
  
- **向量形式**：为将来的可学习参数预留接口
  $$\mathbf{C} = [C_0, C_1, ..., C_{V_{\text{full}}-1}]$$
  
  初始化时可设为：$C_k = C_{\text{OvR}} + \epsilon_k$，其中 $\epsilon_k$ 是小的随机扰动

其中 $C_{\text{OvR}}$ 是预设常数（如 100.0）。

**数学效果**：
- $C_{\text{OvR}} = 0$: 初始概率接近 0.5，无明显偏好
- $C_{\text{OvR}} = 10$: 初始概率普遍较低，创造稀疏激活
- $C_{\text{OvR}} \geq 100$: 极度稀疏的初始概率分布

**推荐设置**：$C_{\text{OvR}} = 100.0$，这提供了良好的起点。

> **未来扩展**：当 $\mathbf{C}$ 成为可学习参数后，模型可以自动调整每个词汇的"激活阈值"，为频繁词汇学习较低阈值，为罕见词汇保持较高阈值。

通过上述初始化步骤，CausalQwen 在训练开始时具有以下性质：

-   **因果表征**: 对于每个位置 $i$，因果表征 $U_i$ 服从宽泛的柯西分布 $U_i \sim \text{Cauchy}(z_i, \gamma_i)$，其中 $\gamma_i$ 是大数。
-   **分类决策**: 分类行动网络的输出与 Qwen 的原始输出完全一致，即 $\text{loc}_{S_{k,i}} = \mathbf{W}_{\text{Qwen}}[k, :] \cdot z_i$。
-   **回归决策**: 由于回归网络的标准初始化以及 $b_{noise}$ 初始化为 0，回归预测的分布为 $Y_i \sim \text{Cauchy}(W_{\text{reg}} \cdot z_i + b_{\text{reg}},  \|W_{\text{reg}}\|_1 \cdot \gamma_i)$，当 $\gamma_i$ 很大时，近似为以 0 为中心的宽泛分布。
-   **阈值设置**: 所有类别共享相同的初始阈值 $C_k = C_{\text{OvR}}$，这提供了一个适度稀疏且数值稳定的初始概率分布。


cmt: 如果说前面外生噪声系数零初始化设计是非必要的复杂初始化， 那么这个部分的阈值初始化可能是我认为相对来说是必要的，并不是说我会拒绝去把这个初始化给去掉， 只是说这个初始化比其他的初始化保留的优先级更高。 我觉得要不直接把它变成可学习的参数吧？ 因为逻辑上来说，我觉得是有这个必要的，而且实际上也并不复杂？ 你对我的建议怎么看？

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
-   **数值词元预测 (`eval/num_*`)**:
    -   **`num_precision`, `num_recall`, `num_f1`**: 全面评估模型在有效位置上辨别 `<NUM>` 词元的能力，是**门控机制性能的关键**。

-   **回归性能 (`eval/reg_*`)**:
    -   **`reg_mae` (平均绝对误差)**: 传统的误差度量，对异常值敏感。
    -   **`reg_mdae` (中位绝对误差)**: 对异常值稳健的误差度量。当 `mae` 远大于 `mdae` 时，表明存在少数极端错误的预测。

### 5.4. 内部状态分布指标 (`dist/*`)

这些指标的统计数据（`mean`, `median`, `std`, `iqr`）均在**有效词元位置（应用 `attention_mask`）** 上计算。

-   **因果表征 `U` (`dist/U_*`)**:
    -   通过对比 `U_loc` 和 `U_scale` 的 `mean`/`std` 与 `median`/`iqr`，我们可以深入分析其分布的**偏斜度**和**尾部重量**，从而诊断模型是否对特定词元学习到了特化表征，或是否明智地在困难样本上表达了更高的不确定性。

-   **OvR 校准 (`dist/ovr_prob_sum_median`)**:
    -   监控所有词元的 OvR 概率之和的中位数。一个充分校准的模型，其概率和应**收敛于 1**，这反映了模型学习到了词元预测任务的内在互斥性。

cmt: 这个地方你还要考虑一下 causal sampling 下， ovr_prob_sum_median 会不会接近 1？ 我觉得这个监控好像同样重要？甚至监控他逻辑上可能更加合理？对同一个个体，他才会 next token 只能有一个真值！你这个可以跟我好好讨论一下，期待咱们把这个事情给确认下来，找到数学上逻辑上原理上更加合适的监控指标。



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
    
    D --> G["数值编码函数<br>φ(v)"];
    
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

这张图从决策分布 `S` 和 `Y` 出发，清晰地展示了三种不同的预测生成方式。
```mermaid
graph TD
    A["<b>个体因果表征 U</b><br>Uᵢ ~ Cauchy(loc, scale)"];

    subgraph "行动网络 (Action Network)"
        A -- "分类行动决策" --> B["分类分布 S"];
        A -- "回归行动决策" --> C["回归分布 Y"];
    end

    subgraph "模式三：确定性推理 (默认)"
        B -- "计算 OvR 概率" --> D["<b>分类预测 ŷ_cls </b><br> argmax_k P(S_{k,i} > C_k)"];
        C -- "取位置参数" --> E["<b>回归预测 ŷ_reg </b><br> loc_{Y_i}"];
    end
    subgraph "模式二：兼容传统采样"
        B -- "取 loc_S 作为 logits" --> H["Softmax(loc_S)"];
        H --> I["<b>Top-k / Top-p 采样预测</b> <br> ŷ_cls"];
    end

    subgraph "模式一：因果采样"
        A -- "（阈值）采样**原因**" --> F["得到具体个体 uᵢ"];
        F -- "传入'行动网络'" --> G["<b>确定性的分类/回归预测</b> <br> ŷ_cls, ŷ_reg"];
    end



    style A fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style D fill:#fce4ec,stroke:#880e4f
    style E fill:#fce4ec,stroke:#880e4f
    style G fill:#fce4ec,stroke:#880e4f
    style I fill:#fce4ec,stroke:#880e4f
```

#### 4.2 CausalQwen 自回归生成流程图

```mermaid
graph TD
    A["<b>开始：初始输入 (Prompt)</b>"] --> B{"Tokenize"};
    B --> C["<b>生成初始序列对</br>词元序列: [x_1, ..., x_T]<br> 数值序列: [v_1, ..., v_T]"];
    C --> D["<b>进入循环</b>"] --> Loop;

    subgraph "自回归循环 (t = T, T+1, ...)"
        Loop["<b>1.将当前<u>序列对</u>送入模型</b><br> input_ids, numeric_values"];
        Loop --> E["<b>2.模型前向传播</b>"];
        E --> F["<b>3.获取下一词元(t+1)的预测</b><br>ŷ_cls (预测词元ID)<br>ŷ_reg (预测回归值)"];
        
        F --> G{"预测词元ŷ_cls是NUM词元？"};

        G -- "<b>是 (Yes)</b>" --> H["<b>更新序列对:</b><br>input_ids.append(&lt;NUM_ID&gt;)<br>numeric_values.append(<b>ŷ_reg</b>)"];
        H --> I["(同时，将数值 ŷ_reg 转为文本追加到生成结果)"];

        G -- "<b>否 (No)</b>" --> J["<b>更新序列对:</b><br>input_ids.append(ŷ_cls)<br>numeric_values.append(<b>0.0</b>)"];
        J --> I["(同时，将词元ŷ_cls转为文本追加到生成结果)"];
        
        I --> K{"是EOS词元或<br>已达最大长度？"};
        
        K -- "否 (No), 继续循环" --> L["t = t + 1"];
        L --> Loop;
    end
    
    K -- "是 (Yes), 结束循环" --> M["<b>结束：输出完整生成文本</b>"];

    style C fill:#e8f5e9,stroke:#1b5e20
    style H fill:#fff3e0,stroke:#e65100
    style J fill:#e3f2fd,stroke:#1b5e20
    style M fill:#e8f5e9,stroke:#1b5e20

``` 


```mermaid
graph LR
    start_point["共享随机性实例生哼<br>（个体选择因子或噪声实例）"]

    start_point --> C1
    start_point --> C2

    subgraph "模式一：共享个体选择因子<br>ε_seed ~ U(0,1)^C"
        C1["循环: t=1,2,..."]
        C1 --> D1["计算 U_t 分布<br>U_t ~ Cauchy(loc_U_t, scale_U_t)"]
        D1 --> E1["使用固定因子计算个体<br>u_t = loc_U_t + scale_U_t ⊙ tan(π(ε_seed - 0.5))"]
        E1 --> F1["U'_t ~ Cauchy(u_t, |b_noise|)"]
        F1 --> G1["生成下一词元"]
        G1 --> H1{继续生成?}
        H1 -->|是| C1
        H1 -->|否| I1["结束"]
    end
    
    subgraph "模式二：共享外生噪声实例<br>ε_noise ~ Cauchy(0, I)"
        C2["循环: t=1,2,..."]
        C2 --> D2["计算 U_t 分布<br>U_t ~ Cauchy(loc_U_t, scale_U_t)"]
        D2 --> E2["U'_t ~ Cauchy(loc_U_t + |b_noise|·ε_noise, scale_U_t)"]
        E2 --> F2["生成下一词元"]
        F2 --> H2{继续生成?}
        H2 -->|是| C2
        H2 -->|否| I2["结束"]
    end
    
    style E1 fill:#fbe9e7
    style E2 fill:#fbe9e7
```

### 图 5：损失流程图



#### 图 5.1：分类损失 (`L_cls`) 的计算

这张图展示了如何从模型对全部词汇的预测分布，计算出每个位置的分类总损失。

```mermaid
graph TD
    subgraph "输入"
        A["<b>loc_S</b><br>形状: [B, S, V_full]"]
        B["<b>scale_S</b><br>形状: [B, S, V_full]"]
        C["<b>真实类别标签 (y)</b><br>形状: [B, S, V_full]"]
    end

    D[计算 OvR 概率 P]
    A & B --> D
    
    subgraph "计算过程"
        direction LR
        D --> E["<b>概率张量 P</b><br>形状: [B, S, V_full]"]
        E & C --> F["计算 OvR 二元交叉熵"]
        F -- "对词汇表维度(V_full)求和" --> G
    end

    G["<b>分类损失 L_cls</b><br>形状: [B, S]"]
    
    style G fill:#e3f2fd,stroke:#1b5e20,stroke-width:2px
```
* **解读**：模型输出的 `loc_S` 和 `scale_S` 首先被用来计算词汇表中每个词的 OvR 概率，得到一个形状为 `[B, S, V_full]` 的概率张量 `P`。然后，这个概率张量与同样形状的真实标签 `y` 一起计算交叉熵损失。最后，将每个位置上所有词汇的损失相加（在 `V_full` 维度上求和），最终得到一个形状为 `[B, S]` 的张量 `L_cls`，代表了批次中每个序列在每个位置的分类损失。

---

#### 图 5.2：门控回归损失 (`L_reg_gated`) 的计算

这张图是整个损失计算中最精巧的部分，详细解释了门控机制的数据流。

```mermaid
graph TD
    subgraph "输入"
        A["<b>loc_Y</b><br>形状: [B, S]"]
        B["<b>scale_Y</b><br>形状: [B, S]"]
        C["<b>真实数值</b><br>形状: [B, S]"]
        D["<b>P_&lt;NUM&gt; 概率</b><br>(来自 L_cls 计算过程)<br>形状: [B, S]"]
        E["<b>数值位置掩码 (m)</b><br>形状: [B, S]"]
    end

    subgraph "路径 A：计算基础回归损失"
        A & B & C --> F["计算柯西负对数似然"]
        F --> G["<b>基础回归损失 L_nll</b><br>形状: [B, S]"]
    end

    subgraph "路径 B：计算门控权重"
        D & E --> H["计算门控权重<br>Gate = m * (α + (1-α)P_&lt;NUM&gt;)"]
        H --> I["<b>门控权重 Gate</b><br>形状: [B, S]"]
    end

    J[逐元素相乘]
    G & I --> J
    J --> K["<b>门控回归损失 L_reg_gated</b><br>形状: [B, S]"]


    style K fill:#fff3e0,stroke:#e65100,stroke-width:2px
```
* **解读**：此流程有两个并行的路径。**路径 A** 使用回归分布参数和真实数值计算出一个基础的回归损失张量 `L_nll`。**路径 B** 利用分类任务中得到的 `<NUM>` 词元概率，结合一个指示真实标签是否为数值的掩码 `m`，计算出一个同样形状的 `Gate` 张量。最后，将 `L_nll` 和 `Gate` **逐元素相乘**，得到最终的门控回归损失 `L_reg_gated`。这种设计（默认 $\alpha=0$）确保了只有在真实标签是数值 `(m=1)` 且模型有一定把握认为是数值 `(P_<NUM> > 0)` 的位置，回归损失才会被有效计算。

---

#### 图 5.3：总损失 (`L_total`) 的合并

这张最终的图展示了如何将前两步计算出的损失张量合并，并得到最终用于反向传播的标量损失值。

```mermaid
graph TD
    subgraph "Inputs"
        A["<b>分类损失 L_cls</b><br>[B, S]"]
        B["<b>门控回归损失 L_reg_gated</b><br>[B, S]"]
        C["<b>注意力掩码 attention_mask</b><br>[B, S]"]
        D["<b>数值位置掩码 m</b><br>[B, S]"]
    end

    subgraph "分类损失路径"
        A & C --> E["带掩码的分类损失<br>L_cls * attention_mask"]
        E -- "求和后除以 attention_mask.sum()" --> F["<b>平均分类损失 L_cls_mean</b><br>(标量)"]
    end

    subgraph "回归损失路径"
        B & D --> G["门控回归损失<br>(已包含掩码 m)"]
        G -- "求和后除以 m.sum()" --> H["<b>有效回归损失 L_reg_eff</b><br>(标量)"]
    end

    subgraph "合并"
        F & H --> I["加权求和<br>L_total = L_cls_mean + λ * L_reg_eff"]
        I --> J["<b>最终总损失 L_total</b><br>(标量)"]
    end

    style J fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style C fill:#f1f8e9
    style D fill:#f1f8e9
```
* **关键修正与解读**：我们**不能**简单地将 `L_cls` 和 `L_reg_gated` 逐元素相加再求平均。由于数值 (`<NUM>`) 词元在文本中是**稀疏**的，`L_reg_gated` 张量中绝大部分元素为零。如果直接求平均，回归损失的信号会被严重"稀释"，无法有效指导模型优化。
* **正确流程**：正确的做法是分别对两个损失进行归约，然后再合并：
   1.  **分类损失**：对 `L_cls` 张量应用 `attention_mask` 来排除填充词元，然后求平均，得到一个标量 `L_cls_mean`。
   2.  **回归损失**：对 `L_reg_gated` 张量应用掩码 `m`，只对真实数值词元求和再求平均，得到一个标量 `L_reg_eff`。
   3.  **总损失**：将 `L_cls_mean` 和 `L_reg_eff` 进行加权求和，得到最终的标量损失 `L_total`。

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
| **采样范式** 🎲<br>Sampling Paradigm | **对"结果"采样 (Sampling the "Effect")**<br>在最终的 logits 分布上使用 `top-k`/`top-p` 进行随机采样。 | **对"原因"采样 (Sampling the "Cause")**<br>引入**因果采样**，直接对"个体" $U$ 进行采样，得到更多样且风格一致的生成结果。 |
| **核心创新** ✨<br>Key Innovation | 强大的语言建模与上下文理解能力。 | 引入外生**个体选择变量 $U$**，显式建模**外生噪声 $\text{noise}$**，并利用柯西分布的数学特性，构建了一个可高效训练的因果生成框架。 |


> cmt: 采样范式方面肯定是需要仔细斟酌他的写法的, 

