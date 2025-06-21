# CausalLLM: 核心数学框架与架构设计

> - 📋 文档用途: 构建最小可行的因果语言模型，专注于验证CausalLLM的核心哲学和数学框架
> - 🎯 目标读者: 项目负责人，用于验证AI实现是否符合理论预期  
> - 📖 内容定位: 详细的理论框架、数学原理、架构设计的权威参考

本文档详细描述了 CausalLLM 因果语言模型的核心设计理念、数学框架和实现细节。CausalLLM 是首个将个体选择变量 U 引入语言生成的因果推理模型，实现了从"概率采样"到"个体决策"的范式转变。本文中，我们选择Qwen2.5作为基础构建 CausalQwen 作为示范。

> 在反事实世界里面，一切皆有可能。Everything is possible in the counterfactual world.

## 技术概述

CausalQwen 代表了语言模型设计的重大突破，其核心创新包括：

1. **因果理论基础**：基于 [Distribution-consistency Structural Causal Models](https://arxiv.org/abs/2401.15911) 理论框架
2. **个体选择变量 U**：首次在语言生成中引入个体选择的概念，实现可控一致生成
3. **柯西分布族**：利用重尾分布诚实表达"开放世界"的不确定性
4. **OvR 分类机制**：摆脱 Softmax 归一化约束，支持独立的二元判断
5. **解析化计算**：基于柯西分布的线性稳定性，整个前向传播无需采样

**核心范式转变**：从"概率分布采样"转向"个体在因果律下的必然表达"。

## 1. 核心数学框架

### 1.1 因果生成模型的数学基础

CausalQwen 的核心创新是将传统的条件生成 $P(Y|X)$ 分解为因果结构：

$$Y = f(U, \epsilon)$$

其中：
- $U$ 是个体选择变量，其分布是**从上下文 $X$ 推断出的个体因果表征分布**
- $\epsilon$ 是**外生噪声**（不可控的随机扰动）
- $f$ 是**普适因果机制**（对所有个体一致的决策规律）

#### 1.1.1 个体选择变量 $U$ 的双重身份

根据 [Distribution-consistency Structural Causal Models](https://arxiv.org/abs/2401.15911) 的理论基础，$U$ 具有双重含义：

1. **个体选择变量**：$U=u$ 代表从所有可能个体中"选择"了特定个体 $u$
2. **个体因果表征**：向量 $u$ 包含了该个体所有内在的、驱动其行为的潜在属性

**关键洞察**：
- **表征的复杂性**：从混乱的表面证据 $X$ 推断真正的因果表征 $U$ 是高度非线性的
- **规律的简洁性**：一旦找到正确的表征 $u$，因果规律 $f$ 本身是简单线性的
- **因果一致性**：函数 $f$ 对所有个体普适，个体差异完全体现在 $u$ 中
- **不确定性分解**：随机性分为个体选择的不确定性和外生噪声的随机性

#### 1.1.2 CausalQwen vs 传统 Qwen：本质对比

| **维度** | **传统 Qwen** | **CausalQwen** |
|----------|---------------|----------------|
| **数学假设** | 学习条件分布 $P(Y\|X)$ | 学习因果机制 $Y = f(U, \epsilon)$ |
| **生成哲学** | 从分布采样结果 | 个体在规律下的必然表达 |
| **随机性来源** | 仅输出层 Softmax 采样 | 个体选择 $U$ + 外生噪声 $\epsilon$ |
| **分类方式** | Softmax（竞争性归一化） | OvR（独立二元判断） |
| **不确定性表达** | 隐式（通过概率分布） | 显式（柯西分布的 scale 参数） |
| **可解释性** | 黑盒概率模型 | 个体+规律+噪声的清晰分解 |


### 1.2 柯西分布：开放世界的数学基础

我们选择**柯西分布**作为核心分布族的三重考量：

#### 1.2.1 诚实的不确定性表达

柯西分布作为**重尾分布**，为"黑天鹅"事件保留不可忽略的概率，诚实表达"开放世界"的深层不确定性：

> "任何观测到的伟大成就，任何人都有非零的概率做出来"

#### 1.2.2 数学上的"深刻未知"

柯西分布的期望和方差数学上无定义，恰好对应了"我们永远无法完全知道一个个体到底是什么样的"这一哲学事实。

#### 1.2.3 线性稳定性（计算魔法）

**加法稳定性**：
$$X_1 \sim \text{Cauchy}(\mu_1, \gamma_1), X_2 \sim \text{Cauchy}(\mu_2, \gamma_2) \Rightarrow X_1 + X_2 \sim \text{Cauchy}(\mu_1 + \mu_2, \gamma_1 + \gamma_2)$$

**缩放稳定性**：
$$w \cdot X \sim \text{Cauchy}(w \cdot \mu, |w| \cdot \gamma)$$

**线性组合稳定性**：
$$\sum_{i=1}^n w_i X_i \sim \text{Cauchy}\left(\sum_{i=1}^n w_i \mu_i, \sum_{i=1}^n |w_i| \gamma_i\right)$$

这使得我们可以在整个前向传播中**保持分布形式不变，无需采样**，实现高效的解析计算。

## 2. 模型架构

### 2.1 总体架构：从证据到个体到决策

#### 2.1.1 核心数据流与维度变化

```mermaid
graph LR
    Input["原始输入<br>[B, S]"] --> Embed["词元嵌入<br>[B, S, H]"]
    Embed --> Transform["特征提取<br>[B, S, H]"]
    Transform --> Abduct["归因推断<br>[B, S, C]×2"]
    Abduct --> Action["行动决策<br>[B, S, V]×2"]
    Action --> Output["OvR分类<br>输出词元"]
    
    style Input fill:#f9f9f9
    style Embed fill:#e3f2fd
    style Transform fill:#e8f5e9
    style Abduct fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Action fill:#fce4ec,stroke:#d81b60,stroke-width:2px
    style Output fill:#ffebee
```

**关键维度变化**：`[B,S]` → `[B,S,H]` → `[B,S,C]` → `[B,S,V]` → `词元输出`

#### 2.1.2 因果哲学的体现

```mermaid
graph LR
    subgraph "现实世界"
        Reality["真实个体 u*<br>（不可观测）"]
        Evidence["观察证据 X<br>（有限、有偏）"]
    end
    
    subgraph "模型推理"
        Evidence --> Inference["归因推断<br>P(U|X)"]
        Inference --> Population["个体子群体<br>所有符合证据的个体"]
        Population --> Law["线性因果律<br>f(U, ε) → Y"]
    end
    
    subgraph "生成结果"
        Law --> Output["文本生成<br>个体在规律下的表达"]
    end
    
    Reality -.->|不可知| Population
    
    style Reality fill:#ffebee,stroke:#c62828,stroke-width:2px
    style Population fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Law fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
```

### 2.2 维度约定

| 符号 | 含义 | 说明 |
|------|------|------|
| `B` | 批次大小 | Batch size |
| `S` | 序列长度 | Sequence length |
| `H` | 隐藏维度 | 继承自 Qwen |
| `C` | 因果表征维度 | 设计决策：`C = H` |
| `V` | 词汇表大小 | Vocabulary size |

## 3. 详细模块设计

### 3.1 模块1：标准词元嵌入

我们直接使用预训练模型的词元嵌入：

```mermaid
graph LR
    A["input_ids<br>[B, S]"] --> B["词元嵌入层<br>embed_tokens"]
    B --> C["embeddings<br>[B, S, H]"]
    
    style C fill:#e3f2fd
```

**数学表示**：
$$e_i = \text{EmbedTokens}(x_i), \quad x_i \in \{0, 1, ..., V-1\}$$

**实现要点**：
- 直接复用预训练模型的嵌入层，保持与原始 Qwen 的完全兼容
- 标准的词元到向量映射，为后续模块提供稳定的特征基础

### 3.2 模块2：特征提取网络

使用标准 Transformer 架构提取上下文特征：

```mermaid
graph LR
    E["embeddings<br>[B, S, H]"] --> T["QwenTransformer<br>L层 MHA+FFN"]
    T --> Z["context_features<br>[B, S, H]"]
    
    style T fill:#e8f5e9
```

**数学表示**：$z = \text{QwenTransformer}(e)$，完全继承预训练权重。

### 3.3 模块3：归因推断网络（核心创新）

#### 3.3.1 从证据圈定个体子群体

归因推断网络的本质工作：**根据观察证据 $X$，在茫茫人海中圈定出一个所有成员都符合该证据的子群体**

```mermaid
graph TD
    subgraph "输入：证据特征"
        Z["上下文特征 z<br>[B, S, H]<br>（深度理解的证据）"]
    end
    
    subgraph "推断网络：双头架构"
        Z --> L1["群体中心推断<br>Linear(H → C)<br>W_loc, b_loc"]
        Z --> L2["群体多样性推断<br>Linear(H → C)<br>W_scale, b_scale"]
    end
    
    subgraph "子群体描述"
        L1 --> LOC["群体典型代表<br>μ = W_loc·z + b_loc<br>[B, S, C]"]
        L2 --> PRE["多样性预激活<br>W_scale·z + b_scale"]
        PRE --> SCALE["群体内部多样性<br>γ = softplus(...)<br>[B, S, C]"]
    end
    
    subgraph "个体子群体分布"
        LOC --> DIST["符合证据的个体群体<br>U ~ Cauchy(μ, γ)"]
        SCALE --> DIST
    end
    
    style Z fill:#e8eaf6,color:#000
    style DIST fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    style LOC fill:#e1f5fe,color:#000
    style SCALE fill:#f3e5f5,color:#000
```

#### 3.3.2 数学公式：分布参数推断

对于序列中的每个位置 $i$（对应不同的证据上下文）：

**群体中心**（典型代表）：
$$\text{loc}_{U_i} = W_{\text{loc}} \cdot z_i + b_{\text{loc}} \in \mathbb{R}^C$$

**群体多样性**（不确定性范围）：
$$\text{scale}_{U_i} = \text{softplus}(W_{\text{scale}} \cdot z_i + b_{\text{scale}}) \in \mathbb{R}^C_+$$

**个体子群体分布**：
$$U_i \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i})$$

#### 3.3.3 代码实现：归因推断

```python
class AbductionNetwork(nn.Module):
    """从证据推断个体子群体"""
    
    def __init__(self, hidden_size: int, causal_size: int = None):
        super().__init__()
        causal_size = causal_size or hidden_size  # C = H
        
        # 推断群体中心（典型代表）
        self.loc_net = nn.Linear(hidden_size, causal_size)
        # 推断群体多样性（内部不确定性）
        self.scale_net = nn.Linear(hidden_size, causal_size)
        
        # 知识继承初始化
        self._init_weights()
    
    def _init_weights(self):
        # 恒等映射：直接使用 Qwen 特征作为个体中心
        nn.init.eye_(self.loc_net.weight)
        nn.init.zeros_(self.loc_net.bias)
        
        # 常数多样性：初始对所有位置给予相同的先验不确定性
        nn.init.zeros_(self.scale_net.weight)
        nn.init.constant_(self.scale_net.bias, 0.0)  # softplus(0) = ln(2) ≈ 0.69 (初始 γ)
    
    def forward(self, context_features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            context_features: [B, S, H] 上下文特征
        Returns:
            loc_U: [B, S, C] 个体群体的中心
            scale_U: [B, S, C] 个体群体的多样性
        """
        # 推断符合证据的个体群体
        loc_U = self.loc_net(context_features)
        scale_U = F.softplus(self.scale_net(context_features))
        
        return loc_U, scale_U
```

#### 3.3.4 关键洞察：推断的本质

1. **不是寻找唯一个体**：真实的 $u^*$ 永远不可观测
2. **而是圈定子群体**：所有可能产生观察证据 $X$ 的个体集合
3. **位置参数** $\text{loc}_{U_i}$：该群体的"平均画像"或"典型代表"
4. **尺度参数** $\text{scale}_{U_i}$：群体内部的多样性，证据强则小，证据弱则大
5. **上下文依赖**：不同位置基于不同证据推断出不同的个体群体

### 3.4 模块4：行动决策网络（普适因果律）

#### 3.4.1 训练阶段的前向传播：注入外生噪声

行动网络体现了我们的**线性因果律假说**：一旦找到正确的个体表征 $U$，因果规律本身是简单线性的。在训练的前向传播中，它通过注入一个外生噪声，并将个体表征应用到普适的线性变换上，从而产生最终的决策潜能。

```mermaid
graph TB
    subgraph "Step 1: 注入外生噪声"
        U["个体群体<br>U ~ Cauchy(μ, γ)"] --> Fusion["噪声融合"]
        Epsilon["外生噪声<br><b>b_noise</b>"] --> Fusion
        Fusion --> U_prime["融合了噪声的分布<br>U' ~ Cauchy(μ, γ + |<b>b_noise</b>|)"]
    end
    
    subgraph "Step 2: 应用普适线性因果律"
        U_prime --> Law["因果律 f(u', ε)<br>线性变换 W_cls, b_cls"]
        Law --> Decision["决策分布<br>S_k ~ Cauchy(loc_S_k, scale_S_k)"]
    end
    
    subgraph "Step 3: 词汇级决策"
        Decision --> Words["对每个词汇 k ∈ {1,...,V}<br>独立的因果决策"]
    end
    
    style U fill:#fff3e0,color:#000
    style U_prime fill:#fbe9e7,stroke:#ff6f00,stroke-width:2px,color:#000
    style Law fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    style Decision fill:#e3f2fd,color:#000
```

#### 3.4.2 数学推导：训练时的噪声传播

##### Step 1: 外生噪声融合

-   **基本原理**: 核心思想是对个体表征 $U_i$ 注入一个标准柯西分布的噪声 $\varepsilon \sim \text{Cauchy}(0, 1)$，其强度由一个可学习的参数向量 $\mathbf{b}_{\text{noise}}$ 控制。变换后的随机变量 $U'_i$ 为：
    $$U'_i = U_i + \mathbf{b}_{\text{noise}} \cdot \varepsilon$$
-   **解析推导**: 根据柯西分布的线性稳定性，我们可以推导出 $U'_i$ 的分布。
    -   首先，我们有 $U_i \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i})$。
    -   其次，缩放后的噪声项 $\mathbf{b}_{\text{noise}} \cdot \varepsilon \sim \text{Cauchy}(0, |\mathbf{b}_{\text{noise}}|)$。
    -   因此，两个独立的柯西变量之和的分布为：
        $$U'_i \sim \text{Cauchy}(\text{loc}_{U_i} + 0, \text{scale}_{U_i} + |\mathbf{b}_{\text{noise}}|) = \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i} + |\mathbf{b}_{\text{noise}}|)$$
-   **计算实现**: 这个推导允许我们在计算中完全避免采样，直接通过对尺度参数进行加法操作来高效地实现噪声注入。

##### Step 2: 线性因果律应用
对每个词汇 $k$，对这个包含了噪声的分布 $U'_i$ 应用相同的线性规律：
$$S_{k,i} = W_{\text{cls},k} \cdot U'_i + b_{\text{cls},k}$$

##### Step 3: 决策分布推导
由柯西分布的线性稳定性，最终的决策分布参数为：
$$S_{k,i} \sim \text{Cauchy}(\text{loc}_{S_{k,i}}, \text{scale}_{S_{k,i}})$$

其中：
- **位置参数**：$\text{loc}_{S_{k,i}} = W_{\text{cls},k} \cdot \text{loc}_{U_i} + b_{\text{cls},k}$
- **尺度参数**：$\text{scale}_{S_{k,i}} = (\text{scale}_{U_i} + |\mathbf{b}_{\text{noise}}|) \cdot |W_{\text{cls},k}|$

通过反向传播，模型会自动学习噪声强度参数 $\mathbf{b}_{\text{noise}}$ 的大小，从而为不同任务适配最优的不确定性。

#### 3.4.3 设计洞察：两层哲学

1. **表征层的复杂性**：从证据到个体表征是高度非线性的（归因推断网络）
2. **规律层的简洁性**：从个体表征到决策是简单线性的（行动决策网络）

这种设计体现了深刻的世界观：
> 真正的挑战在于**学会如何看待世界**（学习正确的表征），一旦学会了，世界的规律将以极为优雅和简单的方式呈现。

---
**重要连接**: 上述训练过程产生的、包含了外生噪声影响的最终决策潜能分布 $S$，将被传递给任务激活头（ActivationHead）进行最终的、任务特定的处理。

## 4. 损失函数与训练

### 4.1 OvR (One-vs-Rest) 分类

与传统的 Softmax 不同，OvR 对每个词汇进行独立的二元判断：

```mermaid
graph TD
    subgraph "输入"
        S["S_k ~ Cauchy(loc_S_k, scale_S_k)"]
        C["阈值 C_ovr"]
        Y["真实标签 y"]
    end
    
    subgraph "概率计算"
        S --> P["P_k = P(S_k > C_ovr)"]
        C --> P
        P --> F["P_k = 1/2 + arctan((loc_S_k - C_ovr)/scale_S_k)/π"]
    end
    
    subgraph "损失计算"
        F --> BCE["二元交叉熵"]
        Y --> OH["one-hot 编码"]
        OH --> BCE
        BCE --> L["L_cls = Σ_k BCE_k"]
    end
    
    style L fill:#e3f2fd
```

### 4.2 完整损失计算

**Step 1: OvR 概率计算**
$$P_{k,i} = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_{\text{ovr}}}{\text{scale}_{S_{k,i}}}\right)$$

**Step 2: 掩码损失计算**  
$$L_{\text{cls},i} = -\sum_{k=1}^V [y_{k,i} \log P_{k,i} + (1-y_{k,i}) \log(1-P_{k,i})] \cdot \text{mask}_i$$

**Step 3: 总损失**
$$\mathcal{L} = \frac{\sum_{i=1}^S L_{\text{cls},i}}{\sum_{i=1}^S \text{mask}_i}$$

其中 $\text{mask}_i \in \{0,1\}$ 表示位置 $i$ 是否为有效位置（非 padding）。

**实现要点**：
- **掩码处理**：确保 padding 位置不参与损失计算
- **数值稳定性**：使用 `torch.clamp` 避免 log(0) 
- **OvR 优势**：独立判断，支持不确定性表达

## 5. 推理模式：对噪声的灵活调制

在推理阶段，我们可以通过 `temperature` 和 `do_sample` 两个参数，灵活地**调制**已经学习到的外生噪声 $\mathbf{b}_{\text{noise}}$，以实现不同的生成策略。CausalQwen 的 `ActionNetwork` 的 `forward` 函数完美地实现了这个统一框架。

### 5.1 核心思想：温度统一的噪声控制

推理时，`temperature` 作为噪声强度的统一控制器，`do_sample` 选择噪声的作用方式：

-   **温度 = 0**: 关闭外生噪声，实现**纯因果生成**。
    -   数学原理: `U' ~ Cauchy(μ, γ)`
-   **温度 > 0**:
    -   **非采样模式** (`do_sample=False`): 噪声增加**尺度参数**，扩大决策不确定性。
        -   数学原理: `U' ~ Cauchy(μ, γ + T·|b_noise|)`
    -   **采样模式** (`do_sample=True`): 噪声扰动**位置参数**，改变个体身份。
        -   数学原理: `U' ~ Cauchy(μ + T·|b_noise|·ε, γ)`

### 5.2 ActionNetwork 的推理实现

```python
class ActionNetwork(nn.Module):
    """行动网络：从个体表征到决策分布"""
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        
        self.lm_head = nn.Linear(config.causal_size, config.vocab_size, bias=True)
        self.b_noise = nn.Parameter(torch.zeros(config.causal_size))
        
        self._init_from_original_lm_head()
    
    def _init_from_original_lm_head(self):
        """从原始lm_head复制权重，符合知识继承原则"""
        nn.init.constant_(self.b_noise, self.config.b_noise_init)
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)
        
    def forward(self, loc_U, scale_U=None, do_sample=False, temperature=1.0):
        """前向传播：温度统一控制噪声强度
        
        核心创新：温度参数统一控制噪声强度，do_sample控制噪声作用方式
        
        temperature=0时两种模式都自动退化为纯因果模式:
        ├─ U' ~ Cauchy(μ, γ) 
        └─ 无外生噪声，个体的必然表达
        
        temperature>0 且 do_sample=False (标准模式):
        ├─ U' ~ Cauchy(μ, γ + T·|b_noise|)
        └─ 噪声增加决策不确定性，保持个体身份
        
        temperature>0 且 do_sample=True (采样模式):
        ├─ ε ~ Cauchy(0, 1) 标准噪声采样
        ├─ U' ~ Cauchy(μ + T·|b_noise|·ε, γ)
        └─ 噪声扰动个体身份，探索多样性
        """
        if scale_U is None:
            scale_U = torch.zeros_like(loc_U)
        
        if temperature == 0:
            # 🎯 纯因果模式：无噪声影响
            loc_U_final = loc_U
            scale_U_final = scale_U
            
        elif do_sample:
            # 🎲 采样模式：噪声影响位置参数
            uniform_sample = torch.rand_like(loc_U)
            epsilon = torch.tan(torch.pi * (uniform_sample - 0.5))
            
            loc_U_final = loc_U + temperature * torch.abs(self.b_noise) * epsilon
            scale_U_final = scale_U

        else:
            # 🔧 标准模式：噪声影响尺度参数
            loc_U_final = loc_U
            scale_U_final = scale_U + temperature * torch.abs(self.b_noise)
        
        # 线性因果律应用
        loc_S = self.lm_head(loc_U_final)
        scale_S = scale_U_final @ torch.abs(self.lm_head.weight).T
        
        return loc_S, scale_S
```

### 5.3 推理模式详解

CausalQwen 提供四种核心推理模式，通过 `do_sample` 和 `temperature` 参数的组合实现，每种模式对应不同的因果生成哲学。

| 模式名称 | `do_sample` | `temperature` | 数学原理 | 哲学含义 |
| :--- | :--- | :--- | :--- | :--- |
| **因果模式 (Causal)** | `any` | `0` | `U' ~ Cauchy(μ, γ)` | 纯因果生成，无外生噪声，个体的必然表达 |
| **标准模式 (Standard)** | `False` | `> 0` | `U' ~ Cauchy(μ, γ+T·\|b_noise\|)` | 噪声增加决策不确定性，保持个体身份稳定 |
| **采样模式 (Sampling)** | `True` | `> 0` | `U'~Cauchy(μ+T·\|b_noise\|·ε,γ)` | 噪声扰动个体身份，探索决策空间多样性 |
| **兼容模式 (Compatible)** | `N/A` | `any` | 标准 Softmax 概率计算 | 用于与传统LM进行基准比较 |

#### 5.3.1 因果模式 (Causal Mode) (`temperature = 0`)

这是对因果理论最纯粹的表达，无论 `do_sample` 取值如何。当温度为零时，完全没有外生噪声，生成过程基于个体自身的因果表征。决策分布完全由 `U ~ Cauchy(μ, γ)` 决定，精确对应了理论公式 **`Y = f(U)`**，即输出完全是个体 `U` 在普适因果律 `f` 下的必然表达。这种模式提供了最纯粹的因果生成，适用于需要高度一致性和可解释性的场景。

#### 5.3.2 标准模式 (Standard Mode) (`do_sample=False, temperature > 0`)

这是模型的默认确定性推理模式。外生噪声 `T·|b_noise|` 被融合到尺度参数中，增加了决策的不确定性（分布更宽），但保持了决策中心（位置参数）的稳定。哲学含义是环境噪声使得个体的判断变得更加模糊，但个体的核心身份保持不变。最终通过 `argmax` 选择概率最高的词元，过程完全确定。

#### 5.3.3 采样模式 (Sampling Mode) (`do_sample=True, temperature > 0`)

当采用随机生成时，外生噪声 `ε` 经过温度 `T` 调节后，扰动**位置参数**。这改变了个体的身份表征，从而探索不同的决策结果。这相当于在探索"如果这个个体受到随机扰动偏离典型状态，会做出什么不同的决策？"。温度越高，扰动越大，生成的多样性越强。
- TODO: 多步预测的时候可以考虑共享噪声 instance，生成某种一致性的结果。

#### 5.3.4 兼容模式 (Compatible Mode)

此模式下，模型会忽略所有因果模块（尤其是尺度参数 `scale_S`），仅使用决策的位置参数 `loc_S` 作为传统 logits，并应用标准的 Softmax 函数进行采样。这使得 CausalQwen 可以与任何传统语言模型在相同的设置下进行公平比较。

## 6. 自回归序列生成

自回归生成过程是标准的，但在每一步选择下一词元时，可以灵活运用四种推理模式。

### 6.1 生成流程图

```mermaid
graph TD
    Prompt["输入 Prompt"] --> Forward["前向传播<br>获得 loc_U, scale_U"]
    
    Forward --> TempCheck{temperature}
    
    %% 推理模式分支
    TempCheck -->|T = 0| Causal["🎯 因果模式<br>无外生噪声"]
    TempCheck -->|T > 0| SampleCheck{do_sample}
    
    SampleCheck -->|False| Standard["🔧 标准模式<br>噪声→尺度参数"]
    SampleCheck -->|True| Sampling["🎲 采样模式<br>噪声→位置参数"]
    
    %% 决策计算
    Causal --> Action["线性因果律<br>loc_S, scale_S"]
    Standard --> Action
    Sampling --> Action
    
    Action --> OvR["OvR 概率计算"]
    OvR --> Selection{选择策略}
    
    Selection -->|因果推理| ArgMax["argmax(P_k)"]
    Selection -->|兼容模式| Softmax["Softmax 采样"]
    
    ArgMax --> NextToken["下一词元"]
    Softmax --> NextToken
    NextToken --> EndCheck{结束?}
    
    EndCheck -->|否| Forward
    EndCheck -->|是| Done["完成"]
    
    style Causal fill:#e8f5e9
    style Standard fill:#e3f2fd
    style Sampling fill:#fce4ec
    style Action fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Done fill:#e8f5e9
```

### 6.2 与传统语言模型的生成对比

新架构在保留与传统语言模型比较能力的同时，提供了更丰富的因果生成选项。

#### 6.2.1 生成哲学对比

| 方面 | 传统 LM (如 GPT/Qwen) | CausalQwen |
|---|---|---|
| **核心假设** | 学习条件分布 $P(Y\|X)$ | 学习因果机制 $Y = f(U, \epsilon)$ |
| **随机性来源** | 仅输出层采样 | **位置/尺度** 的噪声选择性注入 |
| **一致性** | 无显式机制 | 通过**共享因果机制**保证 |
| **可解释性** | 黑盒概率 | 个体/规律/噪声的清晰分解 |

#### 6.2.2 数学框架对比

```mermaid
graph
    subgraph "Traditional LM"
        X1["上下文 X"] --> P1["P(Y|X)"]
        P1 --> Sample1["Softmax 采样"]
        Sample1 --> Y1["输出 Y"]
    end
    
    subgraph "Causal LLM"
        X2["上下文 X"] --> U2["推断 U|X<br>Cauchy(μ,γ)"]
        U2 --> F2["Y = f(U,ε)<br>噪声影响μ或γ"]
        Noise["外生噪声 ε"] --> F2
        F2 --> Y2["输出 Y"]
    end
    
    style U2 fill:#fff3e0
    style F2 fill:#e8f5e9
```
*注：CausalQwen 将随机性的作用机制从单一的输出层采样，深化为对因果链条内部不同参数（位置或尺度）的精准干预，从而实现了对生成过程多样性与确定性之间更精细的控制。通过温度参数统一控制噪声强度，实现了四种推理模式的对称设计。* 

## 7. 实现要点

### 7.1 模块化设计

```mermaid
graph LR
    subgraph "核心模块"
        E[StandardEmbedding]
        T[QwenTransformer]
        A[AbductionNetwork]
        N[ActionNetwork]
    end
    
    subgraph "损失计算"
        O[OvrClassificationLoss]
    end
    
    E --> T --> A --> N --> O
    
    style A fill:#fff3e0
    style N fill:#fce4ec
    style O fill:#e3f2fd
```

### 7.2 关键设计决策

1. **因果表征维度**：$C = H$（简化设计，充分利用预训练知识）
2. **分布族选择**：柯西分布（数学性质优美，计算高效）
3. **分类方式**：OvR 而非 Softmax（独立判断，灵活性高）
4. **初始化策略**：最大程度继承预训练权重

### 7.3 初始化策略详解

为了使 CausalQwen 能够无缝继承基座模型的强大语言能力，我们采用精心设计的初始化策略。**核心思想**：在训练开始时，CausalQwen 的行为应与原始 Qwen 完全一致。

#### 7.3.1 初始化总览

```mermaid
graph LR
    Start["开始初始化"] --> Qwen["加载预训练 Qwen 模型"]
    
    Qwen --> Module1["模块1: 词元嵌入"]
    Qwen --> Module2["模块2: 特征提取网络"]
    Qwen --> Module3["模块3: 归因推断网络"]
    Qwen --> Module4["模块4: 行动网络"]
    
    subgraph "词元嵌入初始化"
        Module1 --> Inherit["完全继承 Qwen 嵌入权重"]
    end
    
    subgraph "特征提取网络初始化"
        Module2 --> Freeze["完全继承 Qwen Transformer<br>初期可选择性冻结"]
    end
    
    subgraph "归因推断网络初始化"
        Module3 --> Loc["位置网络: W_loc = I, b_loc = 0"]
        Module3 --> Scale["尺度网络: W_scale = 0, b_scale = γ_init"]
    end
    
    subgraph "行动网络初始化"
        Module4 --> Cls["分类头: 复制 Qwen LM head"]
        Module4 --> Noise["噪声参数: 小常数初始化"]
    end
    
    Inherit & Freeze & Loc & Scale & Cls & Noise --> Verify["验证: CausalQwen ≈ Qwen"]
    
    Verify --> End["初始化完成"]
    
    style Start fill:#e8f5e9
    style End fill:#e8f5e9
    style Verify fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

#### 7.3.2 各模块初始化详解

**模块1：词元嵌入**
- 完全继承 Qwen 的词元嵌入权重
- 保持与原始模型的完全兼容性

**模块2：特征提取网络**  
- 直接复制 Qwen Transformer 的所有权重
- 初期可选择性冻结参数，逐步解冻

**模块3：归因推断网络**（关键）
- **位置网络**：$W_{\text{loc}} = I_H$, $b_{\text{loc}} = \mathbf{0}$ → 恒等映射
- **尺度网络**：$W_{\text{scale}} = \mathbf{0}$, $b_{\text{scale}} = \gamma_{\text{init}}$ → 常数输出

初始化后的效果：
$$\text{loc}_{U_i} = z_i \quad \text{（直接使用 Qwen 特征）}$$
$$\text{scale}_{U_i} = \text{softplus}(\gamma_{\text{init}}) = \gamma_0 \cdot \mathbf{1}_C$$

**模块4：行动网络**
- **分类头**：$W_{\text{cls}} \leftarrow W_{\text{Qwen\_lm\_head}}$，完整继承
- **噪声参数**：$b_{\text{noise}} = c \cdot \mathbf{1}_C$，小常数初始化（如 $c = 0.1$）

#### 7.3.3 数学保证

初始化后，对于每个位置 $i$ 和词汇 $k$：

1. **个体表征分布**：
   $$U_i \sim \text{Cauchy}(z_i, \gamma_0 \cdot \mathbf{1}_C)$$

2. **融合后分布**（加入外生噪声）：
   $$U'_i \sim \text{Cauchy}(z_i, (\gamma_0 + c) \cdot \mathbf{1}_C)$$

3. **分类决策分布**：
   $$S_{k,i} \sim \text{Cauchy}(W_{\text{Qwen},k} \cdot z_i, \text{scale}_{S_{k,i}})$$

**关键洞察**：分类决策的位置参数 $W_{\text{Qwen},k} \cdot z_i$ 与原始 Qwen 的 logits 完全一致，确保了初始行为的等价性。

#### 7.3.4 初始化验证

```python
def verify_initialization(model, qwen_model, test_input):
    """验证初始化效果"""
    with torch.no_grad():
        # CausalQwen 前向传播
        causal_outputs = model(test_input)
        
        # Qwen 前向传播  
        qwen_outputs = qwen_model(test_input)
        
        # 比较分类头的位置参数
        causal_logits = causal_outputs['loc_S']  # [B, S, V]
        qwen_logits = qwen_outputs.logits       # [B, S, V]
        
        diff = torch.norm(causal_logits - qwen_logits)
        print(f"初始化验证: |CausalQwen_logits - Qwen_logits| = {diff:.6f}")
        
        return diff < 1e-3  # 应该几乎相等
```

### 7.4 核心实现示例

```python
class CausalQwen(nn.Module):
    def __init__(self, qwen_model_path):
        super().__init__()
        # 加载预训练 Qwen
        self.qwen = AutoModel.from_pretrained(qwen_model_path)
        hidden_size = self.qwen.config.hidden_size
        vocab_size = self.qwen.config.vocab_size
        
        # CausalQwen 特有组件
        self.abduction = AbductionNetwork(hidden_size)
        self.action = ActionNetwork(hidden_size, vocab_size)
        
        # 初始化：复制 Qwen 的 lm_head
        self.action.lm_head.weight.data = self.qwen.lm_head.weight.data.clone()
    
    def forward(self, input_ids):
        # 1. 标准嵌入
        embeddings = self.qwen.embed_tokens(input_ids)
        
        # 2. 特征提取（使用 Qwen）
        features = self.qwen.model(embeddings=embeddings).last_hidden_state
        
        # 3. 推断个体分布
        loc_U, scale_U = self.abduction(features)
        
        # 4. 基于个体进行决策
        loc_S, scale_S = self.action(loc_U, scale_U)
        
        return loc_S, scale_S, loc_U, scale_U

class AbductionNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.loc_net = nn.Linear(hidden_size, hidden_size)
        self.scale_net = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, features):
        loc_U = self.loc_net(features)
        scale_U = F.softplus(self.scale_net(features))
        return loc_U, scale_U

class ActionNetwork(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.b_noise = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, loc_U, scale_U=None, do_sample=False, temperature=1.0):
        if do_sample:
            # 🎲 采样模式：噪声影响位置参数
            
            # Step 1: 采样标准柯西噪声 ε ~ Cauchy(0, I)
            uniform_sample = torch.rand_like(loc_U)
            epsilon = torch.tan(torch.pi * (uniform_sample - 0.5))
            
            # Step 2: 温度调节的噪声注入到位置参数
            loc_U_final = loc_U + temperature * torch.abs(self.b_noise) * epsilon
            scale_U_final = scale_U

        else:
            # 🔧 标准模式：噪声影响尺度参数
            loc_U_final = loc_U
            scale_U_final = scale_U + temperature * torch.abs(self.b_noise)
        
        # 线性因果律应用
        loc_S = self.lm_head(loc_U_final)
        scale_S = scale_U_final @ torch.abs(self.lm_head.weight).T
        
        return loc_S, scale_S
```

## 8. 训练策略

### 8.1 预训练对齐阶段：高效离线特征蒸馏

虽然第 7.3 节的静态初始化提供了一个良好的起点，但我们通过一个专门的预训练对齐阶段，可以更高效、更鲁棒地将教师模型（Qwen）的知识迁移到CausalQwen的学生模块中。我们采纳一种**两阶段离线特征蒸馏**的范式，它将预训练转化为一个高度优化的监督学习问题，专门训练 `AbductionNetwork` 和 `ActionNetwork`。

#### 8.1.1 阶段一：离线数据生成

此阶段一次性地创建一个高质量的蒸馏数据集。我们选取一个大型通用语料库，将其输入固定的Qwen教师模型。对于语料库中的每个词元位置，我们抽取出两样东西：
1.  **输入特征 `z`**: Qwen最后一层Transformer的输出 `last_hidden_state`。
2.  **输出目标**: 教师的`lm_head`所预测的**Top-K**词元索引及其对应的Softmax概率。

最终，我们获得一个大规模的 `(z, Top-K_Probs)` 映射数据集，该数据集将用于下一阶段的训练。

#### 8.1.2 阶段二：高效对齐训练

在这个阶段，我们不再需要运行庞大的教师模型。训练循环直接从离线数据集中采样批次的 `(z, Top-K_Probs)` 对。
- 特征 `z` 被直接送入 `AbductionNetwork`。
- CausalQwen的模块进行前向传播，计算出OvR概率。
- 使用Top-K对齐损失函数进行反向传播。

#### 8.1.3 Top-K 对齐损失函数

对齐是通过优化以下损失函数来实现的，该函数旨在最小化学生和教师在教师最自信的K个词元上的概率差异：

$$ \mathcal{L}_{\text{Top-K}} = \sum_{i \in \text{batch}} \sum_{k \in \mathcal{K}_{\text{teacher}, i}} \left( P_{\text{student}, i,k}^{\text{OvR}} - P_{\text{teacher}, i,k}^{\text{Softmax}} \right)^2 $$

其中 $\mathcal{K}_{\text{teacher}, i}$ 是教师模型在位置 $i$ 预测的Top-K词元索引集。

#### 8.1.4 预期成果与优化目标

通过在此目标上训练，CausalQwen的学生模块（`AbductionNetwork` 和 `ActionNetwork`）将学会模仿教师的决策逻辑。这种方法不仅保证了Top-K预测行为上的一致性，还传递了更丰富的"暗知识"。

此阶段的优化目标包括`AbductionNetwork`和`ActionNetwork`的所有权重，以及可训练的外生噪声参数 `b_noise` 和全局OvR决策阈值 `C_ovr`。这为下游的主要因果目标微调提供了一个经过充分对齐和校准的、高质量的初始化模型。


### 8.2 训练目标

CausalQwen 基于 OvR 分类损失进行端到端训练，优化期望损失：

$$ \mathcal{L} = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \frac{1}{|S|} \sum_{i=1}^{|S|} L_{\text{cls}}(y_i | x_{<i}) \right] $$

### 8.3 优化目标

训练过程同时优化：
1. **归因推断能力**：学习从上下文推断合理的个体分布
2. **决策一致性**：学习个体到输出的稳定映射
3. **不确定性校准**：学习何时应该有高/低确定性

### 8.4 训练监控指标

参考 [`mathematical_foundations.md`](core_mathematical_framework_num_extended.md) 第5节的监控体系：

- **核心损失指标** (`train/*`)
  - `train/accuracy`: 分类准确率
  - `train/loss`: 总优化目标

- **内部状态分布** (`dist/*`)
  - `dist/U_loc_*`: 个体表征位置参数统计
  - `dist/U_scale_*`: 个体表征尺度参数统计
  - `dist/ovr_prob_sum_*`: OvR 概率校准指标

## 9. 总结与展望

### 9.1 核心贡献

CausalQwen 的核心创新在于：

1.  **因果分解**：将生成过程分解为"个体推断"和"基于个体的决策"
2.  **数学优雅**：利用柯西分布的性质实现高效计算
3.  **因果生成**：通过去除噪声，实现确定性的因果生成
4.  **理论基础**：基于严格的因果推理理论




### 9.2 技术创新意义

这一设计代表了语言模型的重要进展：

```mermaid
graph TB
    subgraph "传统语言模型"
        T1["概率分布 P(Y|X)"]
        T2["Softmax 采样"]
        T3["随机性输出"]
    end
    
    subgraph "CausalQwen 创新"
        C1["个体选择变量 U"]
        C2["因果机制 f(U,ε)"]
        C3["OvR 独立判断"]
        C4["因果推理"]
    end
    
    T1 --> C1
    T2 --> C2
    T3 --> C3
    T2 --> C4
    
    style C1 fill:#fff3e0
    style C2 fill:#e8f5e9
    style C3 fill:#e3f2fd
    style C4 fill:#fce4ec
```

### 9.3 理论意义

这种设计不仅是技术创新，更是对"语言生成本质"的全新理解：

> **文本不是从概率分布中随机抽取的结果，而是特定"个体"在特定"环境"下的必然表达。**

**实用价值**：可控生成、可解释性、一致性保证、反事实推理。

CausalQwen 开辟了因果语言生成的新范式，为下一代可控、可解释的人工智能系统奠定了坚实的理论与实践基础。