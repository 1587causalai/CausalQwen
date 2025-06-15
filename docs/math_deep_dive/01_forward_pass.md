# 前向传播详解

本文档提供 CausalQwen 训练阶段前向传播的完整数学推导和实现细节。

> 返回主文档：[`../mathematical_foundations.md`](../mathematical_foundations.md)

## 符号约定

我们用 B 代表批次大小, S 代表序列长度, H 代表模型隐藏维度, C 代表因果表征维度, V_full 代表 Qwen 的词汇表总大小。Qwen 的词汇表包含 K 个已使用词汇和 271 个预留位置，即 V_full = K + 271。CausalQwen 使用第一个预留位置（ID = K）作为 `<NUM>` 词元。

> **设计决策**: 在当前实现中，我们设定因果表征维度 `C` 与模型隐藏层维度 `H` 相等，即 **`C = H`**。这简化了归因推断网络的初始化。

## 前向传播总览

```mermaid
graph TD
    A["<b>原始输入</b><br>'价格是99.9元'"] --> B["<b>模块一：数值感知嵌入</b><br>处理混合文本/数值输入"];
    B --> |"e: [B,S,H]"| C["<b>模块二：特征提取网络</b><br>Qwen Transformer"];
    C --> |"z: [B,S,H]"| D["<b>模块三：归因推断网络</b><br>推断个体表征U分布"];
    D --> |"loc_U, scale_U"| E["<b>模块四：行动决策网络</b><br>并行分类+回归决策"];
    E --> |"loc_S, scale_S, loc_Y, scale_Y"| F["<b>模块五：损失计算</b><br>OvR分类 + 门控回归"];
    F --> G["<b>总损失</b><br>L_total"];

    style A fill:#f9f9f9
    style B fill:#e3f2fd
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f3e5f5
    style G fill:#ffebee
```

## 1. 模块一：数值感知嵌入 (Numerical-aware Embedding)

```mermaid
graph TD
    A["原始输入<br><i>'价格是99.9元'</i>"] --> B{"分词器"};
    B --> C["词元 ID 序列<br>input_ids: [12345, 67890, &lt;NUM_ID&gt;, 11111]"];
    B --> D["对齐的数值<br>numeric_values: [0.0, 0.0, 99.9, 0.0]"];
    
    C --> E["词元嵌入层"];
    E --> F["基础嵌入<br>base_embeddings"];
    
    D --> G["数值编码函数<br>φ(v) = sign(v)·ln(1+|v|)·w_num"];
    
    F & G --> H["融合: e_i = base_embed_i + φ(v_i)"];
    H --> I["<b>增强嵌入 e</b><br>[B, S, H]"];

    style I fill:#e3f2fd,stroke:#1b5e20,stroke-width:2px
```

这一模块的目标是将混合了文本和数值的原始输入，转化为一个统一的、数值感知的特征向量序列。

### 1.1 分词与数值识别
分词器处理原始文本，识别并替换数值：

1.  **数值识别**: 分词器扫描文本，识别数值模式（如 `99.9`）
2.  **词元替换**: 将识别出的数值替换为特殊词元 `<NUM>`
3.  **数值保存**: 将原始数值单独保存，与词元序列保持位置对齐

-   **输出**: 
    - `input_ids` $[x_1, ..., x_S]$: `['价格', '是', '<NUM>', '元']` → `[12345, 67890, <NUM_ID>, 11111]` (形状: `[B, S]`)
    - `numeric_values` $[v_1, ..., v_S]$: `[0.0, 0.0, 99.9, 0.0]` (形状: `[B, S]`)

### 1.2 词元嵌入
将词元ID序列转换为基础嵌入向量：

-   **输入**: `input_ids` (形状: `[B, S]`)
-   **处理**: 通过嵌入层查找每个词元的向量表示
    $$\text{base\_embed}_i = \text{EmbeddingLayer}(\text{input\_ids}_i)$$
-   **输出**: `base_embeddings` (形状: `[B, S, H]`)

### 1.3 数值编码与融合
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
φ(numeric_values): [[φ(0)], [φ(0)], [φ(99.9)], [φ(0)]]  # φ(99.9) = ln(100.9) * w_num
     ↓ (融合)
enhanced_embeddings: [[e1], [e2], [e3 + φ(99.9)], [e4]]
```

> **设计动机**: 选择对数编码 $\phi(v)$ 是因为它具有三大优势：1) **数值稳定性**，将大范围数值压缩到合理区间；2) **相对误差保持**，对数空间中的等距对应原空间的等比；3) **自然退化**，由于$\phi(0)=0$，非数值位置自然退化为标准词元嵌入，无需特殊处理。

## 2. 模块二：特征提取网络 (Feature Extraction Network)

该模块使用一个标准的 Transformer 网络（如Qwen）作为主干，来深度理解序列的上下文信息。

-   **输入**: 
    - `e`: 增强嵌入张量 (形状: `[B, S, H]`)
-   **处理**: 增强嵌入序列 `e` 被送入Qwen的Transformer主干网络中。通过多层自注意力机制，网络为每个位置的词元计算出融合了全局上下文信息的深层特征：
    $$z = \text{FeatureNetwork}(e) = \text{QwenTransformer}(e)$$
    
    具体而言，对于第 $l$ 层的 Transformer：
    $$h^{(l)} = \text{TransformerLayer}^{(l)}(h^{(l-1)}), \quad h^{(0)} = e$$
    
    最终输出：
    $$z = h^{(L)} = [z_1, z_2, ..., z_S]$$
    
    其中 $L$ 是 Transformer 的层数。

-   **输出**: 
    - `z`: 上下文特征张量 (形状: `[B, S, H]`)

**初始化后的效果**：由于我们完全继承了 Qwen 的 Transformer 权重，当输入的增强嵌入 $e$ 接近原始 Qwen 嵌入时（数值编码的影响很小），输出 $z$ 也将非常接近原始 Qwen 的特征表示。

### 2.1 Transformer 处理流程图

```mermaid
graph TD
    subgraph "输入"
        A["增强嵌入 e<br>[B, S, H]"]
    end
    
    subgraph "Qwen Transformer 层级结构"
        A --> B1["Layer 1: Self-Attention + FFN"]
        B1 --> B2["Layer 2: Self-Attention + FFN"]
        B2 --> B3["..."]
        B3 --> BL["Layer L: Self-Attention + FFN"]
    end
    
    subgraph "输出"
        BL --> Z["上下文特征 z<br>[B, S, H]"]
    end
    
    style A fill:#e3f2fd
    style Z fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
```

## 3. 模块三：归因推断网络 (Abduction Network)

该模块从上下文特征中推断出每个位置的个体因果表征分布。

-   **输入**: 
    - `z`: 上下文特征张量 (形状: `[B, S, H]`)
-   **处理**: 通过两个独立的线性层，分别计算因果表征的位置和尺度参数：
    
    **位置参数**：
    $$\text{loc}_{U_i} = W_{\text{loc}} \cdot z_i + b_{\text{loc}}$$
    
    **尺度参数**（使用 softplus 保证正值）：
    $$\text{scale}_{U_i} = \text{softplus}(W_{\text{scale}} \cdot z_i + b_{\text{scale}})$$
    
    其中 $\text{softplus}(x) = \log(1 + \exp(x))$ 是一个平滑的 ReLU 近似，具有以下性质：
    - 当 $x \to -\infty$ 时，$\text{softplus}(x) \to 0$
    - 当 $x \to +\infty$ 时，$\text{softplus}(x) \to x$
    - 导数：$\text{softplus}'(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}$
    
-   **输出**: 
    - `loc_U`: 因果表征分布的位置参数 (形状: `[B, S, C]`)
    - `scale_U`: 因果表征分布的尺度参数 (形状: `[B, S, C]`)

**初始化后的具体计算**：根据初始化策略（$W_{\text{loc}} = I$, $b_{\text{loc}} = 0$, $W_{\text{scale}} = 0$, $b_{\text{scale}} = \sigma_{\text{init}}$）：

$$\text{loc}_{U_i} = I \cdot z_i + 0 = z_i$$
$$\text{scale}_{U_i} = \text{softplus}(0 \cdot z_i + \sigma_{\text{init}}) = \text{softplus}(\sigma_{\text{init}}) = \gamma_0 \cdot \mathbf{1}_C$$

因此，初始化后每个位置的因果表征服从：
$$U_i \sim \text{Cauchy}(z_i, \gamma_0 \cdot \mathbf{1}_C)$$

其中 $\gamma_0 = \text{softplus}(\sigma_{\text{init}})$，$\mathbf{1}_C$ 是 C 维的全 1 向量。

### 3.1 归因推断网络结构图

```mermaid
graph LR
    subgraph "输入"
        Z["上下文特征 z<br>[B, S, H]"]
    end
    
    subgraph "并行线性变换"
        Z --> L1["位置网络<br>W_loc · z + b_loc"]
        Z --> L2["尺度网络<br>W_scale · z + b_scale"]
    end
    
    subgraph "激活与输出"
        L1 --> LOC["loc_U = W_loc · z + b_loc<br>[B, S, C]"]
        L2 --> ACT["softplus 激活"]
        ACT --> SCALE["scale_U = softplus(...)<br>[B, S, C]"]
    end
    
    LOC --> U["U ~ Cauchy(loc_U, scale_U)"]
    SCALE --> U
    
    style Z fill:#e8f5e9
    style U fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

### 3.2 初始化效果可视化

```mermaid
graph TD
    subgraph "初始化参数"
        P1["W_loc = I_H<br>b_loc = 0"]
        P2["W_scale = 0<br>b_scale = σ_init"]
    end
    
    subgraph "初始化后的效果"
        Z["特征 z_i"] --> LOC["loc_U_i = z_i<br>(恒等映射)"]
        Z --> SCALE["scale_U_i = γ_0 · 1_C<br>(常数尺度)"]
    end
    
    LOC --> U["U_i ~ Cauchy(z_i, γ_0 · 1_C)"]
    SCALE --> U
    
    P1 --> LOC
    P2 --> SCALE
    
    style U fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

## 4. 模块四：行动决策网络 (Action Network)

该模块是模型的核心决策单元。其内部包含一个可学习的噪声参数 $b_{\text{noise}} \in \mathbb{R}^C$，工作流程分为两步：

1.  **噪声注入 (Noise Infusion)**：网络将上游推断出的个体表征分布 $U_i \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i})$ 与代表不可控随机性的外生噪声分布 $\epsilon \sim \text{Cauchy}(0, |b_{\text{noise}}|)$ 进行逐元素的独立叠加，形成融合输入分布：
    $$U'_{i} \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i} + |b_{\text{noise}}|)$$
    
    这里 $|b_{\text{noise}}|$ 表示对 $b_{\text{noise}} \in \mathbb{R}^C$ 逐元素取绝对值。

2.  **并行决策 (Parallel Decision Making)**：基于包含两种不确定性的融合输入分布，进行分类和回归决策。

-   **输入**: `loc_U` (形状: `[B, S, C]`), `scale_U` (形状: `[B, S, C]`)
-   **处理**: 
    - **分类**：每个词汇 $k$ 有独立的线性变换：
      $$\text{loc}_{S_{k,i}} = W_{\text{cls},k} \cdot \text{loc}_{U_i} + b_{\text{cls},k}$$
      $$\text{scale}_{S_{k,i}} = |W_{\text{cls},k}| \cdot (\text{scale}_{U_i} + |b_{\text{noise}}|)$$
      
      其中 $W_{\text{cls},k} \in \mathbb{R}^C$ 是词汇 $k$ 对应的权重向量，$\cdot$ 表示内积运算，$|W_{\text{cls},k}|$ 表示对权重向量逐元素取绝对值。
      
    - **回归**：单一的线性变换：
      $$\text{loc}_{Y_i} = W_{\text{reg}} \cdot \text{loc}_{U_i} + b_{\text{reg}}$$
      $$\text{scale}_{Y_i} = |W_{\text{reg}}| \cdot (\text{scale}_{U_i} + |b_{\text{noise}}|)$$
      
      其中 $W_{\text{reg}} \in \mathbb{R}^C$ 是回归权重向量，$|W_{\text{reg}}|$ 表示对权重向量逐元素取绝对值。

-   **输出**:
    - 分类决策分布参数: `loc_S` (形状: `[B, S, V_full]`), `scale_S` (形状: `[B, S, V_full]`)
    - 回归决策分布参数: `loc_Y` (形状: `[B, S]`), `scale_Y` (形状: `[B, S]`)

### 4.1 行动网络的双步骤流程

```mermaid
graph TB
    subgraph "Step 1: 噪声注入"
        U["个体表征分布<br>U ~ Cauchy(loc_U, scale_U)"] --> Plus["\+"]
        Noise["噪声分布<br>ε ~ Cauchy(0, |b_noise|)"] --> Plus
        Plus --> U_prime["融合分布<br>U' ~ Cauchy(loc_U, scale_U + |b_noise|)"]
    end
    
    subgraph "Step 2: 并行决策"
        U_prime --> CLS["分类线性层<br>W_cls, b_cls"]
        U_prime --> REG["回归线性层<br>W_reg, b_reg"]
        
        CLS --> S["分类决策分布<br>S_k ~ Cauchy(loc_S_k, scale_S_k)"]
        REG --> Y["回归决策分布<br>Y ~ Cauchy(loc_Y, scale_Y)"]
    end
    
    style U fill:#fff3e0
    style U_prime fill:#fbe9e7,stroke:#ff6f00,stroke-width:2px
    style S fill:#e3f2fd
    style Y fill:#e8f5e9
```

### 4.2 噪声注入的数学原理

```mermaid
graph LR
    subgraph "个体不确定性"
        U1["U_i 的第 j 维<br>U_ij ~ Cauchy(loc_ij, scale_ij)"]
    end
    
    subgraph "环境不确定性"
        E1["ε_j ~ Cauchy(0, |b_noise_j|)"]
    end
    
    subgraph "柯西分布加法性质"
        U1 --> Add["独立相加"]
        E1 --> Add
        Add --> U_prime["U'_ij ~ Cauchy(loc_ij, scale_ij + |b_noise_j|)"]
    end
    
    U_prime --> Note["注：逐元素独立操作"]
    
    style U1 fill:#fff3e0
    style E1 fill:#fce4ec
    style U_prime fill:#fbe9e7
```

## 5. 模块五：损失计算 (Loss Calculation)

### 5.1 OvR 分类损失

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

我们不使用标准的 Softmax，而是对每个类别进行独立的"一对多"（One-vs-Rest, OvR）判断。

#### 输入处理
-   **输入**: 
    - 分类决策分布参数: `loc_S`, `scale_S` (形状: `[B, S, V_full]`)
    - 真实标签: `y` (形状: `[B, S]`)，其中 `ignore_index=-100` 表示不需要计算损失的位置
    - 注意力掩码: `attention_mask` (形状: `[B, S]`)
    - OvR 阈值: `C_ovr` - 可以是标量或形状为 `[V_full]` 的张量

#### 损失计算步骤

1. **创建有效位置掩码**：
   $$\text{valid\_mask}_{i} = \mathbb{1}[y_{i} \neq \text{ignore\_index}]$$
   
   其中 $\mathbb{1}[\cdot]$ 是指示函数。

2. **One-hot 编码**（仅对有效位置）：
   $$y_{\text{onehot}, k, i} = \begin{cases}
   1 & \text{if } k = y_{i} \text{ and } \text{valid\_mask}_{i} = 1 \\
   0 & \text{otherwise}
   \end{cases}$$

3. **计算 OvR 概率**：
   $$P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

   这里利用了柯西分布的 CDF 简洁形式。

4. **计算二元交叉熵损失**：
   $$\mathcal{L}_{\text{bce}, k, i} = -[y_{\text{onehot}, k, i} \log P_{k,i} + (1-y_{\text{onehot}, k, i}) \log (1 - P_{k,i})]$$

5. **聚合并应用掩码**：
   $$L_{\text{cls}, i} = \text{valid\_mask}_{i} \cdot \sum_{k=0}^{V_{\text{full}}-1} \mathcal{L}_{\text{bce}, k, i}$$

    -   **输出**: 
        - `L_cls`: 有效分类损失 (形状: `[B, S]`)，仅在真实标签位置有效

#### 实现示例（PyTorch 风格）

```python
def compute_ovr_classification_loss(loc_S, scale_S, labels, C_ovr=100.0, ignore_index=-100):
    """
    计算 OvR 分类损失，支持 ignore_index
    
    Args:
        loc_S: [B, S, V_full] - 分类位置参数
        scale_S: [B, S, V_full] - 分类尺度参数
        labels: [B, S] - 真实标签，包含 ignore_index
        C_ovr: float 或 Tensor[V_full] - OvR 阈值（标量或每类独立）
        ignore_index: int - 忽略的标签值
    
    Returns:
        loss: [B, S] - 每个位置的损失，忽略位置为0
        valid_mask: [B, S] - 有效位置掩码
    """
    B, S, V = loc_S.shape
    
    # 创建有效位置掩码
    valid_mask = (labels != ignore_index).float()  # [B, S]
    
    # 处理 C_ovr 的不同形式
    if isinstance(C_ovr, (int, float)):
        # 标量：广播到所有类别
        C_ovr = torch.full((V,), C_ovr, device=loc_S.device)
    else:
        # 张量：确保形状正确
        assert C_ovr.shape == (V,), f"C_ovr shape must be ({V},), got {C_ovr.shape}"
    
    # 计算 OvR 概率
    # 广播 C_ovr: [V] -> [1, 1, V]
    C_ovr = C_ovr.unsqueeze(0).unsqueeze(0)
    z = (loc_S - C_ovr) / scale_S  # [B, S, V]
    P = 0.5 + torch.atan(z) / math.pi  # [B, S, V]
    
    # 创建 one-hot 标签（仅对有效位置）
    y_onehot = torch.zeros_like(loc_S)  # [B, S, V]
    valid_labels = labels.clone()
    valid_labels[labels == ignore_index] = 0  # 临时设置为0避免索引错误
    y_onehot.scatter_(2, valid_labels.unsqueeze(-1), 1) # cmt: 上一步的设置能避免索引错误
    y_onehot = y_onehot * valid_mask.unsqueeze(-1)  # 应用掩码
    
    # 计算二元交叉熵（数值稳定版本）
    eps = 1e-7
    bce = -(y_onehot * torch.log(P + eps) + 
            (1 - y_onehot) * torch.log(1 - P + eps))  # [B, S, V]
    
    # 对词汇表维度求和
    loss = bce.sum(dim=-1)  # [B, S]
    
    # 应用有效位置掩码
    loss = loss * valid_mask  # [B, S]
    
    return loss, valid_mask
```

#### 阈值设计的灵活性

| 阈值形式 | 用途 | 优势 |
|---------|------|------|
| **标量阈值** | 所有类别共享 | 简单，参数少 |
| **向量阈值** | 每类独立阈值 | 灵活，可学习 |
| **可学习参数** | 通过梯度优化 | 自适应调整 |

> **设计展望**：将 `C_ovr` 设计为可学习参数后，模型可以：
> - 为高频词设置较低阈值（更容易被选中）
> - 为低频词设置较高阈值（需要更强信号）
> - 自动学习词汇的"激活难度"分布

### 5.2 门控回归损失

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
        A & B & C --> F["L_nll = log(π·scale_Y) +<br>log(1 + ((v_true - loc_Y)/scale_Y)²)"]
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

我们只在目标为 `<NUM>` 的位置计算回归损失。

-   **输入**: 
    - 回归决策分布参数: `loc_Y`, `scale_Y` (形状: `[B, S]`)
    - 真实数值: `numeric_values` (形状: `[B, S]`)
    - `<NUM>`词元的预测概率: $P_{\text{<NUM>},i}$
    - 数值位置掩码: `num_mask` (形状: `[B, S]`)
-   **处理**： 
    
    柯西分布的负对数似然：
    $$\mathcal{L}_{\text{nll},i} = \log(\pi \cdot \text{scale}_{Y_i}) + \log\left(1 + \left(\frac{v_{\text{true},i} - \text{loc}_{Y_i}}{\text{scale}_{Y_i}}\right)^2\right)$$
    
    门控机制（$\alpha$ 默认为0）：
    $$\text{gate}_i = m_i \cdot \left(\alpha + (1-\alpha) \cdot P_{\text{<NUM>},i}\right)$$
    
    最终损失：
    $$\mathcal{L}_{\text{reg\_gated},i} = \text{gate}_i \cdot \mathcal{L}_{\text{nll},i}$$
    
    其中 $m_i = \mathbb{1}[y_i = \text{<NUM>}]$ 是数值位置掩码。

### 5.3 总损失的合并

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
        B --> H["注意: L_reg_gated 已包含 num_mask"]
        H --> I["sum(L_reg_gated) / sum(num_mask)"]
        I --> J["<b>L_reg_effective</b><br>有效回归损失<br>(标量)"]
    end

    G & J --> K["L_total = L_cls_mean + λ × L_reg_effective"]
    K --> L["<b>L_total</b><br>最终总损失<br>(标量)"]

    style L fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style C fill:#f1f8e9
    style D fill:#f1f8e9
```

### 5.3.1 损失归约的数学细节

```mermaid
graph LR
    subgraph "为什么分离归约？"
        Issue["问题：数值词元稀疏<br>典型比例 < 5%"]
        Issue --> Solution["解决：使用不同的归约基数"]
    end
    
    subgraph "归约策略"
        Solution --> Cls["分类：sum(L_cls) / N_all<br>N_all = sum(cls_mask)"]
        Solution --> Reg["回归：sum(L_reg) / N_num<br>N_num = sum(num_mask)"]
    end
    
    subgraph "效果"
        Cls --> Effect1["保证语言建模信号充足"]
        Reg --> Effect2["避免回归信号被稀释"]
    end
    
    style Issue fill:#ffebee
    style Solution fill:#e8f5e9
```

## 实现要点总结

1. **数值感知嵌入**：通过对数编码 $\phi(v)$ 实现了文本和数值的统一表示
2. **柯西分布的线性稳定性**：使得整个前向传播可以在参数空间完成，无需采样
3. **OvR 分类**：独立的二元判断避免了 softmax 的归一化约束
4. **门控回归**：智能地将分类置信度用于回归损失的加权
5. **分离归约**：不同的平均基数确保回归信号不被稀释

这种设计使得 CausalQwen 能够在单一架构中同时处理文本生成和数值预测任务。

### 总体数据流与维度变化图

```mermaid
graph TD
    subgraph "输入层"
        Text["文本: '价格是99.9元'"] --> IDs["input_ids [B,S]"]
        Text --> Nums["numeric_values [B,S]"]
    end
    
    subgraph "嵌入层"
        IDs --> Embed["base_embed [B,S,H]"]
        Nums --> Phi["φ(v) [B,S,H]"]
        Embed --> E["e = base + φ(v)<br>[B,S,H]"]
        Phi --> E
    end
    
    subgraph "特征提取"
        E --> Z["z = Qwen(e)<br>[B,S,H]"]
    end
    
    subgraph "归因推断"
        Z --> LocU["loc_U [B,S,C]"]
        Z --> ScaleU["scale_U [B,S,C]"]
    end
    
    subgraph "决策输出"
        LocU --> LocS["loc_S [B,S,V_full]"]
        ScaleU --> ScaleS["scale_S [B,S,V_full]"]
        LocU --> LocY["loc_Y [B,S]"]
        ScaleU --> ScaleY["scale_Y [B,S]"]
    end
    
    subgraph "损失计算"
        LocS --> Loss["L_total (标量)"]
        ScaleS --> Loss
        LocY --> Loss
        ScaleY --> Loss
    end
    
    style E fill:#e3f2fd
    style Z fill:#e8f5e9
    style LocU fill:#fff3e0
    style ScaleU fill:#fff3e0
    style Loss fill:#fce4ec
```
