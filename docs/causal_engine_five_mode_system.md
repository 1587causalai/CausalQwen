# CausalEngine 五模式系统设计

> **文档目标**: 定义纯回归和分类任务的五种建模模式，作为建模层面的统一参数  
> **适用范围**: MLPCausalRegressor, MLPCausalClassifier（不涉及LLM+CausalEngine）  
> **核心设计**: mode 参数同时控制训练、推理、损失计算的统一框架

---

## 1. 五模式系统概述

### 1.1 设计哲学

CausalEngine 的五模式系统基于对随机性来源的不同建模假设：

```mermaid
graph TB
    subgraph Philosophy["建模哲学框架"]
        direction TB
        
        subgraph Source["随机性来源"]
            direction LR
            S1["个体选择差异<br/>（内在）"]
            S2["环境噪声扰动<br/>（外在）"]
        end
        
        subgraph Effects["噪声作用方式"]
            direction LR
            E1["位置参数扰动<br/>改变个体身份"]
            E2["尺度参数增强<br/>扩大决策不确定性"]
            E3["完全消除噪声<br/>纯因果确定性"]
        end
        
        Source --> Effects
    end
    
    subgraph Modes["五种建模模式"]
        direction TB
        M1["Deterministic Mode<br/>γ_U=0, b_noise=0<br/>确定性因果"]
        M2["Exogenous Mode<br/>γ_U=0, b_noise≠0<br/>外生噪声因果"]
        M3["Endogenous Mode<br/>γ_U≠0, b_noise=0<br/>内生因果推理"]
        M4["Standard Mode<br/>b_noise→scale<br/>混合因果推理"]
        M5["Sampling Mode<br/>b_noise→location<br/>随机探索因果"]
    end
    
    Philosophy --> Modes
    
    classDef philosophyStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef modeStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class Philosophy,Source,Effects philosophyStyle
    class Modes,M1,M2,M3,M4,M5 modeStyle
```

### 1.2 五模式数学定义

| 模式 | 参数设置 | 数学表述 | 损失计算 | 哲学含义 |
|------|----------|----------|----------|----------|
| **Deterministic** | $\gamma_U=0, b_{noise}=0$ | $U' = \mu_U$ (确定性) | MSE/CrossEntropy (等价性验证) | 数学计算上等价于传统MLP |
| **Exogenous** | $\gamma_U=0, b_{noise} \neq 0$ | $U' \sim \text{Cauchy}(\mu_U, \|b_{noise}\|)$ | Cauchy NLL / OvR概率 | 外生噪声驱动的因果推理 |
| **Endogenous** | $\gamma_U \neq 0, b_{noise}=0$ | $U' \sim \text{Cauchy}(\mu_U, \gamma_U)$ | Cauchy NLL / OvR概率 | 内生个体不确定性驱动的因果推理 |
| **Standard** | $\gamma_U \neq 0, b_{noise} \neq 0$ (作用于尺度) | $U' \sim \text{Cauchy}(\mu_U, \gamma_U + \|b_{noise}\|)$ | Cauchy NLL / OvR概率 | 内生+外生混合，噪声增强决策不确定性 |
| **Sampling** | $\gamma_U \neq 0, b_{noise} \neq 0$ (作用于位置) | $U' \sim \text{Cauchy}(\mu_U + b_{noise}\varepsilon, \gamma_U)$ | Cauchy NLL / OvR概率 | 随机性扰动个体身份，探索性因果推理 |


五种模式的核心区别在于如何计算 $U'$ 的分布。基于柯西分布的线性稳定性，我们可以解析地推导出每种模式下 $U'$ 的分布：

**基础设定**：
- 个体表征：$U \sim \text{Cauchy}(\mu_U, \gamma_U)$
- 外生噪声：$\varepsilon \sim \text{Cauchy}(0, 1)$
- 统一公式：$U' = U + b_{noise} \varepsilon$

**各模式的分布推导**：

1. **Deterministic Mode** ($\gamma_U=0, b_{noise}=0$)：
   $$U' = U + 0 \cdot \varepsilon = \mu_U \quad \text{(确定性)}$$

2. **Exogenous Mode** ($\gamma_U=0, b_{noise} \neq 0$)：
   $$U' = \mu_U + b_{noise} \varepsilon \sim \text{Cauchy}(\mu_U, |b_{noise}|)$$
   
3. **Endogenous Mode** ($\gamma_U \neq 0, b_{noise}=0$)：
   $$U' = U + 0 \cdot \varepsilon = U \sim \text{Cauchy}(\mu_U, \gamma_U)$$

4. **Standard Mode** ($\gamma_U \neq 0, b_{noise} \neq 0$，噪声作用于尺度)：
   
   虽然统一公式是 $U' = U + b_{noise} \varepsilon$，但在实现中噪声被融合到尺度参数：
   $$U' \sim \text{Cauchy}(\mu_U, \gamma_U + |b_{noise}|)$$
   
   这利用了柯西分布的加法稳定性：如果 $X \sim \text{Cauchy}(\mu, \gamma)$ 且 $Y \sim \text{Cauchy}(0, |b|)$，则 $X + Y \sim \text{Cauchy}(\mu, \gamma + |b|)$。

5. **Sampling Mode** ($\gamma_U \neq 0, b_{noise} \neq 0$，噪声作用于位置)：
   
   首先采样噪声 $\varepsilon$，然后计算：
   $$U' \sim \text{Cauchy}(\mu_U + b_{noise}\varepsilon, \gamma_U)$$
   
   其中 $\varepsilon$ 是从标准柯西分布采样的具体值。

**关键洞察**：Standard 和 Sampling 模式使用相同的统一公式 $U' = U + b_{noise} \varepsilon$，但通过不同的实现方式达到不同的数学效果：
- **Standard**：解析地将噪声融合到尺度参数，避免采样
- **Sampling**：显式采样噪声，扰动位置参数

### 1.3 核心设计原则：为什么不对U采样？

**设计原则**: 对噪声采样是深度学习中广泛存在的实践，符合领域惯例。更多的采样方式，我们会后续进行探索。

**当前设计**: CausalEngine保持个体表征分布 $U \sim \text{Cauchy}(\mu_U, \gamma_U)$ 的完整信息，仅对外生噪声 $\epsilon$ 进行采样（在Sampling模式中），以平衡信息保存与计算效率。

---

## 2. 建模层面的mode参数设计

### 2.1 mode作为统一控制参数

**核心设计原则**：mode参数不仅仅是推理时的配置，而是贯穿整个建模过程的统一参数：

```mermaid
graph LR
    subgraph ModelingProcess["建模全流程"]
        direction TB
        
        subgraph Training["训练阶段"]
            direction TB
            T1["前向传播<br/>根据mode调整噪声"]
            T2["损失计算<br/>mode影响损失函数选择"]
            T3["反向传播<br/>参数更新策略"]
        end
        
        subgraph Inference["推理阶段"]
            direction TB
            I1["特征提取<br/>相同的MLP backbone"]
            I2["因果推理<br/>mode控制噪声机制"]
            I3["输出生成<br/>概率计算方式"]
        end
        
        subgraph Loss["损失计算"]
            direction TB
            L1["Traditional: MSE/CrossE"]
            L2["Causal: 柯西似然"]
            L3["Standard: 混合损失"]
            L4["Sampling: 探索性损失"]
        end
    end
    
    Mode[["mode参数<br/>(统一控制)"]] --> Training
    Mode --> Inference  
    Mode --> Loss
    
    classDef processStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef modeStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    
    class Training,Inference,Loss,T1,T2,T3,I1,I2,I3,L1,L2,L3,L4 processStyle
    class Mode modeStyle
```

### 2.2 mode参数的API设计

```python
class MLPCausalRegressor:
    def __init__(self, mode='standard', **kwargs):
        """
        初始化时设定建模模式，影响整个建模流程
        
        Parameters:
        -----------
        mode : str, default='standard'
            建模模式选择：
            - 'deterministic': 确定性因果（等价于sklearn MLP）
            - 'exogenous': 外生噪声因果建模
            - 'endogenous': 内生因果建模
            - 'standard': 标准因果建模（默认）
            - 'sampling': 探索性因果建模
        """
        self.mode = mode
        self._setup_mode_configuration()
    
    def fit(self, X, y, mode=None):
        """
        训练模型，可覆盖初始化时的mode设置
        
        Parameters:
        -----------
        mode : str, optional
            临时覆盖建模模式（仅对当前训练有效）
        """
        effective_mode = mode or self.mode
        return self._fit_with_mode(X, y, effective_mode)
    
    def predict(self, X, mode=None, enable_flexibility=True):
        """
        预测，支持推理时的模式灵活切换
        
        Parameters:
        -----------
        mode : str, optional
            推理模式（可与训练模式不同）
        enable_flexibility : bool, default=True
            是否允许推理时切换模式
        """
        if enable_flexibility:
            inference_mode = mode or self.mode
        else:
            inference_mode = self.mode  # 强制使用训练时的模式
            
        return self._predict_with_mode(X, inference_mode)
```

---

## 3. 五模式详细设计

### 3.1 Deterministic Mode (确定性模式)

**设计目标**: 完全等价于sklearn MLP，提供基线比较

```mermaid
graph TB
    subgraph Deterministic["Deterministic Mode 数学流程"]
        direction TB
        
        Input["输入 X"]
        MLP["MLP特征提取<br/>H = MLP(X)"]
        Direct["直接线性映射<br/>y = W·H + b"]
        Output["输出 y"]
        
        Input --> MLP --> Direct --> Output
        
        subgraph Config["配置要求"]
            direction TB
            C1["γ_U = 0 (无尺度参数)"]
            C2["b_noise = 0 (无外生噪声)"]
            C3["绕过 AbductionNetwork"]
            C4["使用传统损失函数"]
        end
    end
    
    classDef deterministicStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef configStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class Input,MLP,Direct,Output deterministicStyle
    class Config,C1,C2,C3,C4 configStyle
```

**数学表述**:
$$y = W \cdot \text{MLP}(X) + b$$

**实现关键点**:
- AbductionNetwork: 设置为恒等映射或完全绕过
- ActionNetwork: 简化为线性层
- ActivationHead: 恒等映射
- 损失函数: MSE (回归) / CrossEntropy (分类)

### 3.2 Exogenous Mode (外生模式)

**设计目标**: 确定性个体推断，外生噪声驱动随机性

```mermaid
graph TB
    subgraph Exogenous["Exogenous Mode 数学流程"]
        direction TB
        
        Input["输入 X"]
        MLP["MLP特征提取<br/>H = MLP(X)"]
        Deterministic["确定性个体推断<br/>U = μ_U = H (γ_U = 0)"]
        Noise["噪声注入<br/>U' = U + b_noise*ε"]
        Action["行动决策<br/>ActionNetwork(U')"]
        Activation["任务激活<br/>y = f(S)"]
        Output["输出 y"]
        
        Input --> MLP --> Deterministic --> Noise --> Action --> Activation --> Output
        
        subgraph Config["配置要求"]
            direction TB
            C1["γ_U = 0 (无个体不确定性)"]
            C2["b_noise ≠ 0 (外生噪声)"]
            C3["完全确定性推断个体"]
            C4["噪声直接作用于输出"]
        end
    end
    
    classDef exogenousStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef configStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class Input,MLP,Deterministic,Noise,Action,Activation,Output exogenousStyle
    class Config,C1,C2,C3,C4 configStyle
```

**数学表述**:
- **确定性推断**: $U = \mu_U = \text{MLP}(X), \gamma_U = 0$
- **噪声注入**: $U' = \mu_U + b_{noise} \varepsilon$，其中 $\varepsilon \sim \text{Cauchy}(0,1)$
- **行动网络**: ActionNetwork 接收 $U'$ 作为输入

**哲学含义**: 通过观察证据可以完全确定个体是谁（$\gamma_U = 0$），但环境中存在无法预测的外生随机因素

**关键特性**:
- 个体表征完全确定性
- 外生噪声独立于个体特征
- 适用于"能力确定但结果随机"的场景

### 3.3 Endogenous Mode (内生模式)

**设计目标**: 纯内生因果推理，无外生随机扰动

```mermaid
graph TB
    subgraph Endogenous["Endogenous Mode 数学流程"]
        direction TB
        
        Input["输入 X"]
        MLP["MLP特征提取<br/>H = MLP(X)"]
        Abduction["归因推断<br/>U ~ Cauchy(μ_U, γ_U)"]
        NoiseApply["噪声计算<br/>U' = U + 0 = U"]
        Action["行动决策<br/>ActionNetwork(U')"]
        Activation["任务激活<br/>y = f(S)"]
        Output["输出 y"]
        
        Input --> MLP --> Abduction --> NoiseApply --> Action --> Activation --> Output
        
        subgraph Config["配置要求"]
            direction TB
            C1["b_noise = 0 (无外生噪声)"]
            C2["γ_U > 0 (保持尺度参数)"]
            C3["使用柯西CDF激活"]
            C4["可选择传统或柯西损失"]
        end
    end
    
    classDef endogenousStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef configStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    
    class Input,MLP,Abduction,NoiseApply,Action,Activation,Output endogenousStyle
    class Config,C1,C2,C3,C4 configStyle
```

**数学表述**:
- **归因**: $U \sim \text{Cauchy}(\mu_U(H), \gamma_U(H))$
- **噪声计算**: $U' = U + 0 = U$（无外生噪声）
- **行动网络**: ActionNetwork 接收 $U'$ 作为输入
- **激活**: $P_k = \frac{1}{2} + \frac{1}{\pi}\arctan\left(\frac{\text{loc}_{S_k}}{\text{scale}_{S_k}}\right)$ (分类)

**关键特性**:
- 完全确定性的因果推理
- 保留柯西分布的解析性质
- 适用于高一致性需求场景

### 3.4 Standard Mode (标准模式)

**设计目标**: 噪声作用于尺度参数，扩大决策不确定性

```mermaid
graph TB
    subgraph Standard["Standard Mode 数学流程"]
        direction TB
        
        Input["输入 X"]
        MLP["MLP特征提取<br/>H = MLP(X)"]
        Abduction["归因推断<br/>U ~ Cauchy(μ_U, γ_U)"]
        NoiseScale["噪声计算<br/>U' = U + b_noise*ε"]
        Action["行动决策<br/>ActionNetwork(U')"]
        Note["注：尺度模式下<br/>噪声作用于尺度参数"]
        Activation["任务激活<br/>y = f(S)"]
        Output["输出 y"]
        
        Input --> MLP --> Abduction --> NoiseScale --> Action --> Activation --> Output
        NoiseScale -.-> Note
        
        subgraph Config["配置要求"]
            direction TB
            C1["b_noise ≠ 0 (外生噪声)"]
            C2["噪声作用于尺度参数"]
            C3["保持个体身份不变"]
            C4["混合损失函数"]
        end
    end
    
    classDef standardStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef configStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class Input,MLP,Abduction,NoiseScale,Action,Activation,Output standardStyle
    class Note configStyle
    class Config,C1,C2,C3,C4 configStyle
```

**数学表述**:
- **统一公式**: $U' = U + b_{noise} \varepsilon$，其中 $\varepsilon \sim \text{Cauchy}(0,1)$
- **尺度模式**: 噪声作用于尺度参数，扩大决策不确定性
- **行动网络**: ActionNetwork 接收 $U'$ 作为输入

**哲学含义**: 环境噪声使个体决策更加模糊，但核心身份（位置参数）保持不变

### 3.5 Sampling Mode (采样模式)

**设计目标**: 噪声作用于位置参数，扰动个体身份

```mermaid
graph TB
    subgraph Sampling["Sampling Mode 数学流程"]
        direction TB
        
        Input["输入 X"]
        MLP["MLP特征提取<br/>H = MLP(X)"]
        Abduction["归因推断<br/>U ~ Cauchy(μ_U, γ_U)"]
        NoiseSample["噪声采样<br/>ε ~ Cauchy(0, 1)"]
        NoiseLocation["噪声计算<br/>U' = U + b_noise*ε"]
        Action["行动决策<br/>ActionNetwork(U')"]
        Note2["注：采样模式下<br/>噪声作用于位置参数"]
        Activation["任务激活<br/>y = f(S)"]
        Output["输出 y"]
        
        Input --> MLP --> Abduction --> NoiseSample --> NoiseLocation --> Action --> Activation --> Output
        NoiseLocation -.-> Note2
        
        subgraph Config["配置要求"]
            direction TB
            C1["b_noise ≠ 0 (外生噪声)"]
            C2["噪声作用于位置参数"]
            C3["随机扰动个体身份"]
            C4["探索性损失函数"]
        end
    end
    
    classDef samplingStyle fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    classDef configStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class Input,MLP,Abduction,NoiseSample,NoiseLocation,Action,Activation,Output samplingStyle
    class Note2 configStyle
    class Config,C1,C2,C3,C4 configStyle
```

**数学表述**:
- **统一公式**: $U' = U + b_{noise} \varepsilon$，其中 $\varepsilon \sim \text{Cauchy}(0,1)$
- **采样模式**: 噪声作用于位置参数，扰动个体身份
- **行动网络**: ActionNetwork 接收 $U'$ 作为输入

**哲学含义**: 探索个体在随机扰动下的非典型行为，用于多样性生成

---

## 4. 损失函数设计

### 4.1 mode相关的损失函数策略

每种模式采用最适合其哲学含义的损失函数：

```mermaid
graph TB
    subgraph LossStrategy["损失函数策略"]
        direction 
        
        subgraph Deterministic["Deterministic Mode"]
            direction TB
            TL1["回归: MSE Loss"]
            TL2["分类: CrossEntropy Loss"]
            TL3["完全等价于sklearn"]
        end
        
        subgraph Exogenous["Exogenous Mode"]
            direction TB
            NL1["回归: Cauchy NLL"]
            NL2["分类: OvR概率"]
            NL3["外生噪声因果推理"]
        end
        
        subgraph Endogenous["Endogenous Mode"]
            direction TB
            CL1["回归: Cauchy NLL"]
            CL2["分类: OvR概率"]
            CL3["内生因果推理"]
        end
        
        subgraph Standard["Standard Mode"]
            direction TB
            SL1["回归: Cauchy NLL"]
            SL2["分类: OvR概率"]
            SL3["混合因果推理"]
        end
        
        subgraph Sampling["Sampling Mode"]
            direction TB
            SML1["回归: Cauchy NLL"]
            SML2["分类: OvR概率"]
            SML3["探索性因果推理"]
        end
    end
    
    classDef deterministicStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef exogenousStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef endogenousStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef standardStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef samplingStyle fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    
    class Deterministic,TL1,TL2,TL3 deterministicStyle
    class Exogenous,NL1,NL2,NL3 exogenousStyle
    class Endogenous,CL1,CL2,CL3 endogenousStyle
    class Standard,SL1,SL2,SL3 standardStyle
    class Sampling,SML1,SML2,SML3 samplingStyle
```

### 4.2 损失函数数学定义

#### Deterministic Mode损失 (等价性验证用)
```python
def deterministic_loss(predictions, targets, task_type):
    """与 sklearn 完全等价的损失函数"""
    if task_type == 'regression':
        return F.mse_loss(predictions, targets)
    elif task_type == 'classification':
        return F.cross_entropy(predictions, targets)
```

#### Exogenous/Endogenous/Standard/Sampling Mode统一损失
```python
def causal_loss(loc_S, scale_S, targets, task_type):
    """模式2-5的统一损失函数（完全相同）"""
    if task_type == 'regression':
        # Cauchy负对数似然
        return -cauchy_log_pdf(targets, loc_S, scale_S).mean()
    elif task_type == 'classification':
        # OvR Cauchy概率
        probs = cauchy_cdf(0, loc_S, scale_S)  # P(S > 0)
        return F.binary_cross_entropy(probs, targets_one_hot)
```

#### 重要说明：损失函数统一性

用户需要明确的是，**第2种模式到第5种模式，它的损失函数都是一模一样的**。它们的区别仅在于 $U'$ 的计算方式：

- **Exogenous Mode**: $U' = \mu_U + b_{noise} \varepsilon$
- **Endogenous Mode**: $U' = U + 0 = U \sim \text{Cauchy}(\mu_U, \gamma_U)$
- **Standard Mode**: $U' = U + b_{noise} \varepsilon$ (噪声作用于尺度)
- **Sampling Mode**: $U' = U + b_{noise} \varepsilon$ (噪声作用于位置)

但它们都使用相同的 Cauchy NLL / OvR 概率损失函数，然后行动网络的输入是 $U'$。

---

## 5. 推理与应用策略

### 5.1 五模式的应用场景

**核心设计原则**: 每种模式都有其特定的应用场景和理论意义

```mermaid
graph TB
    subgraph Applications["五模式应用场景"]
        
        subgraph Deterministic["🎯 Deterministic Mode"]
            direction TB
            D1["等价性验证<br/>与sklearn基线对比"]
            D2["调试与开发<br/>确定性行为"]
            D3["基础功能验证<br/>算法正确性检查"]
        end
        
        subgraph Exogenous["🌍 Exogenous Mode"]
            direction TB
            E1["外生冲击建模<br/>市场波动、自然灾害"]
            E2["环境噪声场景<br/>传感器误差、测量噪声"]
            E3["确定性个体<br/>但外部随机干扰"]
        end
        
        subgraph Endogenous["🧠 Endogenous Mode"]
            direction TB
            EN1["纯因果推理<br/>内在不确定性建模"]
            EN2["个体差异分析<br/>认知能力分布"]
            EN3["高可解释性需求<br/>医疗诊断、金融风控"]
        end
        
        subgraph Standard["⚡ Standard Mode"]
            direction TB
            S1["生产环境部署<br/>平衡性能与理论"]
            S2["一般应用场景<br/>默认推荐模式"]
            S3["混合因果建模<br/>内生+外生并存"]
        end
        
        subgraph Sampling["🎲 Sampling Mode"]
            direction TB
            SA1["探索性数据分析<br/>发现隐藏模式"]
            SA2["多样性生成<br/>创意推荐系统"]
            SA3["研究与实验<br/>因果发现"]
        end
    end
    
    classDef deterministicStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef exogenousStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef endogenousStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef standardStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef samplingStyle fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    
    class Deterministic,D1,D2,D3 deterministicStyle
    class Exogenous,E1,E2,E3 exogenousStyle
    class Endogenous,EN1,EN2,EN3 endogenousStyle
    class Standard,S1,S2,S3 standardStyle
    class Sampling,SA1,SA2,SA3 samplingStyle
```

### 5.2 模式选择决策树

```mermaid
graph TD
    Start(["开始选择模式"]) --> Question1{"需要等价性验证？"}
    
    Question1 -->|是| Deterministic["🎯 Deterministic Mode<br/>与sklearn完全等价"]
    Question1 -->|否| Question2{"个体表征确定性？"}
    
    Question2 -->|完全确定| Question3{"存在外生噪声？"}
    Question2 -->|有不确定性| Question4{"存在外生噪声？"}
    
    Question3 -->|是| Exogenous["🌍 Exogenous Mode<br/>确定个体+外生噪声"]
    Question3 -->|否| Deterministic
    
    Question4 -->|否| Endogenous["🧠 Endogenous Mode<br/>纯内生因果推理"]
    Question4 -->|是| Question5{"应用场景？"}
    
    Question5 -->|生产环境| Standard["⚡ Standard Mode<br/>噪声增强不确定性"]
    Question5 -->|探索研究| Sampling["🎲 Sampling Mode<br/>噪声扰动身份"]
    
    classDef questionStyle fill:#f9f9f9,stroke:#666,stroke-width:2px
    classDef deterministicStyle fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    classDef exogenousStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef endogenousStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    classDef standardStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    classDef samplingStyle fill:#fff8e1,stroke:#f9a825,stroke-width:3px
    
    class Start,Question1,Question2,Question3,Question4,Question5 questionStyle
    class Deterministic deterministicStyle
    class Exogenous exogenousStyle
    class Endogenous endogenousStyle
    class Standard standardStyle
    class Sampling samplingStyle
```

### 5.3 统一API设计

```python
class MLPCausalRegressor:
    def __init__(self, mode='standard', **kwargs):
        """
        五模式统一接口
        
        Parameters:
        -----------
        mode : str, default='standard'
            建模模式选择：
            - 'deterministic': γ_U=0, b_noise=0 (等价sklearn)
            - 'exogenous': γ_U=0, b_noise≠0 (外生噪声)
            - 'endogenous': γ_U≠0, b_noise=0 (内生因果)
            - 'standard': γ_U≠0, b_noise≠0 (噪声→尺度)
            - 'sampling': γ_U≠0, b_noise≠0 (噪声→位置)
        """
        self.mode = mode
        self._configure_mode_parameters()
    
    def _configure_mode_parameters(self):
        """根据模式配置参数"""
        if self.mode == 'deterministic':
            self.gamma_U_enabled = False
            self.b_noise_enabled = False
            self.loss_type = 'traditional'  # MSE/CrossEntropy
        elif self.mode == 'exogenous':
            self.gamma_U_enabled = False
            self.b_noise_enabled = True
            self.loss_type = 'causal'  # Cauchy NLL
        elif self.mode == 'endogenous':
            self.gamma_U_enabled = True
            self.b_noise_enabled = False
            self.loss_type = 'causal'  # Cauchy NLL
        elif self.mode in ['standard', 'sampling']:
            self.gamma_U_enabled = True
            self.b_noise_enabled = True
            self.loss_type = 'causal'  # Cauchy NLL
    
    def predict(self, X, return_uncertainty=False):
        """
        统一预测接口
        
        根据训练时的模式自动选择正确的U'计算方式
        """
        # 1. 特征提取（所有模式统一）
        H = self.mlp_backbone(X)
        
        # 2. 个体推断（根据模式调整）
        if self.gamma_U_enabled:
            U = self.abduction(H)  # U ~ Cauchy(μ_U, γ_U)
        else:
            U = H  # U = μ_U (确定性)
        
        # 3. 计算U'（核心差异）
        if self.mode == 'deterministic':
            U_prime = U  # U' = μ_U
        elif self.mode == 'exogenous':
            epsilon = self._sample_cauchy_noise()
            U_prime = U + self.b_noise * epsilon  # U' ~ Cauchy(μ_U, |b_noise|)
        elif self.mode == 'endogenous':
            U_prime = U  # U' = U ~ Cauchy(μ_U, γ_U)
        elif self.mode == 'standard':
            # 解析地融合噪声到尺度参数
            U_prime = U  # 但尺度参数会在ActionNetwork中调整
        elif self.mode == 'sampling':
            epsilon = self._sample_cauchy_noise()
            U_prime = U + self.b_noise * epsilon  # 位置扰动
        
        # 4. 行动决策（ActionNetwork接收U'）
        predictions = self.action_network(U_prime)
        
        if return_uncertainty:
            return predictions, self._estimate_uncertainty(U_prime)
        return predictions
```

---

## 6. 实验验证与基准测试

### 6.1 系统性基准测试设计

**验证目标**: 证明五模式系统的数学正确性、应用有效性和计算效率

```mermaid
graph TB
    subgraph Benchmark["五模式基准测试体系"]
        direction TB
        
        subgraph MathValidation["数学验证"]
            direction TB
            MV1["等价性验证<br/>Deterministic vs sklearn"]
            MV2["分布正确性<br/>U'分布计算"]
            MV3["损失函数统一性<br/>模式2-5一致性"]
        end
        
        subgraph PerformanceTest["性能测试"]
            direction TB
            PT1["回归基准<br/>4个真实数据集"]
            PT2["分类基准<br/>5个真实数据集"]
            PT3["可解释性对比<br/>不确定性量化"]
        end
        
        subgraph EfficiencyTest["效率测试"]
            direction TB
            ET1["计算复杂度<br/>时间与内存"]
            ET2["参数数量对比<br/>模型规模"]
            ET3["收敛性分析<br/>训练稳定性"]
        end
        
        subgraph ApplicationTest["应用测试"]
            direction TB
            AT1["因果发现能力<br/>合成因果数据"]
            AT2["鲁棒性测试<br/>噪声环境适应"]
            AT3["模式选择指导<br/>场景匹配验证"]
        end
    end
    
    MathValidation --> PerformanceTest --> EfficiencyTest --> ApplicationTest
    
    classDef mathStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef perfStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef effStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef appStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class MathValidation,MV1,MV2,MV3 mathStyle
    class PerformanceTest,PT1,PT2,PT3 perfStyle
    class EfficiencyTest,ET1,ET2,ET3 effStyle
    class ApplicationTest,AT1,AT2,AT3 appStyle
```

### 6.2 核心验证实验

#### 6.2.1 数学等价性验证

```python
def test_mathematical_equivalence():
    """验证Deterministic模式与sklearn的数学等价性"""
    
    # 数据准备
    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # sklearn基线
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        random_state=42,
        max_iter=500,
        alpha=0.0
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    
    # CausalEngine Deterministic模式
    causal_model = MLPCausalRegressor(
        mode='deterministic',
        hidden_layer_sizes=(64, 32),
        random_state=42
    )
    # 关键：配置等价性参数
    causal_model._setup_mathematical_equivalence()
    causal_model.fit(X_train, y_train)
    causal_pred = causal_model.predict(X_test)
    causal_r2 = r2_score(y_test, causal_pred)
    
    # 等价性检验
    r2_diff = abs(sklearn_r2 - causal_r2)
    pred_mse = mean_squared_error(sklearn_pred, causal_pred)
    
    print(f"sklearn R²: {sklearn_r2:.6f}")
    print(f"CausalEngine R²: {causal_r2:.6f}")
    print(f"R²差异: {r2_diff:.6f} (<0.001: ✓)")
    print(f"预测差异MSE: {pred_mse:.6f} (<0.001: ✓)")
    
    assert r2_diff < 0.001, f"等价性验证失败: R²差异 {r2_diff}"
    assert pred_mse < 0.001, f"等价性验证失败: 预测差异 {pred_mse}"
    
    return True

def test_distribution_correctness():
    """验证U'分布计算的数学正确性"""
    
    # 模拟参数
    mu_U = torch.tensor([1.0, -0.5, 2.0])
    gamma_U = torch.tensor([0.5, 1.0, 0.3])
    b_noise = torch.tensor(0.2)
    
    # 测试各模式的U'分布
    test_cases = {
        'deterministic': {
            'gamma_U': 0, 'b_noise': 0,
            'expected_loc': mu_U, 'expected_scale': torch.zeros_like(mu_U)
        },
        'exogenous': {
            'gamma_U': 0, 'b_noise': b_noise,
            'expected_loc': mu_U, 'expected_scale': torch.full_like(mu_U, abs(b_noise))
        },
        'endogenous': {
            'gamma_U': gamma_U, 'b_noise': 0,
            'expected_loc': mu_U, 'expected_scale': gamma_U
        },
        'standard': {
            'gamma_U': gamma_U, 'b_noise': b_noise,
            'expected_loc': mu_U, 'expected_scale': gamma_U + abs(b_noise)
        }
    }
    
    for mode, params in test_cases.items():
        loc_U_prime, scale_U_prime = compute_U_prime_distribution(
            mu_U, params['gamma_U'], params['b_noise'], mode
        )
        
        loc_close = torch.allclose(loc_U_prime, params['expected_loc'], atol=1e-6)
        scale_close = torch.allclose(scale_U_prime, params['expected_scale'], atol=1e-6)
        
        print(f"{mode} 模式 U' 分布: 位置参数✓={loc_close}, 尺度参数✓={scale_close}")
        
        assert loc_close and scale_close, f"{mode}模式 U'分布计算错误"
    
    return True

def test_loss_function_unity():
    """验证模式2-5使用相同的损失函数"""
    
    # 模拟数据
    loc_S = torch.randn(10, 5)
    scale_S = torch.abs(torch.randn(10, 5)) + 0.1
    targets = torch.randint(0, 5, (10,))
    
    # 计算模式2-5的损失
    losses = {}
    for mode in ['exogenous', 'endogenous', 'standard', 'sampling']:
        loss = compute_causal_loss(loc_S, scale_S, targets, mode)
        losses[mode] = loss
    
    # 验证损失函数完全相同
    base_loss = losses['exogenous']
    for mode, loss in losses.items():
        if mode != 'exogenous':
            assert torch.allclose(loss, base_loss, atol=1e-8), f"{mode}模式损失与基准不一致"
    
    print("✓ 模式2-5损失函数统一性验证通过")
    return True
```

#### 6.2.2 性能基准测试

```python
def benchmark_five_modes():
    """五模式系统性能基准测试"""
    
    # 真实数据集
    regression_datasets = [
        load_boston(), load_diabetes(), 
        load_california_housing(), make_regression(n_samples=1000, n_features=20)
    ]
    
    classification_datasets = [
        load_iris(), load_wine(), load_breast_cancer(),
        make_classification(n_samples=1000, n_features=20, n_classes=3)
    ]
    
    results = {'regression': {}, 'classification': {}}
    
    # 回归基准测试
    for i, (X, y) in enumerate(regression_datasets):
        print(f"\n=== 回归数据集 {i+1} ===")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dataset_results = {}
        for mode in ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']:
            model = MLPCausalRegressor(mode=mode, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            r2 = r2_score(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            
            dataset_results[mode] = {'r2': r2, 'rmse': rmse}
            print(f"{mode:12s}: R²={r2:.4f}, RMSE={rmse:.4f}")
        
        results['regression'][f'dataset_{i+1}'] = dataset_results
    
    # 分类基准测试 (类似结构)
    # ...
    
    return results
```

---

## 7. 总结与实践指南

### 7.1 应用场景与模式选择

#### 7.1.1 按应用需求选择

```mermaid
graph LR
    subgraph Scenarios["应用场景分类"]
        direction TB
        
        subgraph Development["🔧 开发与验证"]
            D1["基线对比 → Deterministic"]
            D2["算法验证 → Deterministic"]
            D3["功能调试 → Deterministic"]
        end
        
        subgraph Production["🏢 生产部署"]
            P1["通用应用 → Standard"]
            P2["高可解释性 → Endogenous"]
            P3["医疗金融 → Endogenous"]
        end
        
        subgraph Research["🔬 研究分析"]
            R1["因果发现 → Sampling"]
            R2["探索性分析 → Sampling"]
            R3["多样性生成 → Sampling"]
        end
        
        subgraph Environment["🌍 特殊环境"]
            E1["外生冲击 → Exogenous"]
            E2["噪声环境 → Exogenous"]
            E3["传感器误差 → Exogenous"]
        end
    end
    
    classDef devStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef prodStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef resStyle fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    classDef envStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    
    class Development,D1,D2,D3 devStyle
    class Production,P1,P2,P3 prodStyle
    class Research,R1,R2,R3 resStyle
    class Environment,E1,E2,E3 envStyle
```

#### 7.1.2 按数据特性选择

| 数据特性 | 推荐模式 | 数学原理 | 适用情况 |
|------------|----------|----------|----------|
| **完全确定性** | Deterministic | $U' = \mu_U$ | 无噪声、高质量数据 |
| **个体确定+外部噪声** | Exogenous | $U' \sim \text{Cauchy}(\mu_U, \|b_{noise}\|)$ | 传感器数据、市场波动 |
| **个体不确定性** | Endogenous | $U' \sim \text{Cauchy}(\mu_U, \gamma_U)$ | 认知差异、个性化 |
| **混合不确定性** | Standard | $U' \sim \text{Cauchy}(\mu_U, \gamma_U + \|b_{noise}\|)$ | 常见应用场景 |
| **探索性需求** | Sampling | $U' \sim \text{Cauchy}(\mu_U + b_{noise}\varepsilon, \gamma_U)$ | 创意任务、研究分析 |

### 7.2 实践开发流程

#### 7.2.1 渐进式开发路径

```mermaid
graph TD
    Start(["项目开始"]) --> Phase1["🎯 阶段1: 基线验证<br/>Deterministic Mode"]
    
    Phase1 --> Check1{"基础功能正常？"}
    Check1 -->|否| Debug1["调试修复"]
    Debug1 --> Phase1
    
    Check1 -->|是| Phase2["🌍 阶段2: 环境适应<br/>Exogenous/Endogenous"]
    
    Phase2 --> Check2{"适应性满足？"}
    Check2 -->|否| Tune2["模式调整"]
    Tune2 --> Phase2
    
    Check2 -->|是| Phase3["⚡ 阶段3: 性能优化<br/>Standard Mode"]
    
    Phase3 --> Check3{"性能指标达标？"}
    Check3 -->|否| Optimize3["参数优化"]
    Optimize3 --> Phase3
    
    Check3 -->|是| Phase4["🎲 阶段4: 探索扩展<br/>Sampling Mode"]
    
    Phase4 --> Deploy["🚀 生产部署"]
    
    classDef phaseStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef checkStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef actionStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    
    class Phase1,Phase2,Phase3,Phase4 phaseStyle
    class Check1,Check2,Check3 checkStyle
    class Debug1,Tune2,Optimize3 actionStyle
    class Start,Deploy actionStyle
```

#### 7.2.2 关键实践原则

1. **始终从 Deterministic 开始**: 确保算法正确性后再添加复杂性
2. **数学等价性验证**: 使用 sklearn 基线对比验证实现正确性
3. **损失函数统一**: 模式 2-5 必须使用相同的 Cauchy NLL 损失
4. **渐进式复杂化**: 逐步引入不确定性和噪声机制
5. **充分测试验证**: 每个模式都需要独立验证数学正确性

### 7.3 技术参考对照表

#### 7.3.1 完整数学定义

| 模式 | 参数设置 | $U'$ 分布 | 损失函数 | 实现特点 |
|------|----------|-----------|----------|----------|
| **Deterministic** | $\gamma_U=0, b_{noise}=0$ | $U' = \mu_U$ (确定性) | MSE/CrossEntropy | 等价sklearn，禁用AbductionNetwork |
| **Exogenous** | $\gamma_U=0, b_{noise} \neq 0$ | $U' \sim \text{Cauchy}(\mu_U, \|b_{noise}\|)$ | Cauchy NLL | 确定性个体，外生噪声采样 |
| **Endogenous** | $\gamma_U \neq 0, b_{noise}=0$ | $U' \sim \text{Cauchy}(\mu_U, \gamma_U)$ | Cauchy NLL | 纯内生因果，无外部噪声 |
| **Standard** | $\gamma_U \neq 0, b_{noise} \neq 0$ | $U' \sim \text{Cauchy}(\mu_U, \gamma_U + \|b_{noise}\|)$ | Cauchy NLL | 噪声融合到尺度，解析计算 |
| **Sampling** | $\gamma_U \neq 0, b_{noise} \neq 0$ | $U' \sim \text{Cauchy}(\mu_U + b_{noise}\varepsilon, \gamma_U)$ | Cauchy NLL | 噪声扰动位置，采样计算 |

#### 7.3.2 API 参数对照

```python
# 核心洞察：五模式本质上都是 ActionNetwork 的不同计算方式！
class ActionNetwork(nn.Module):
    def forward(self, loc_U, scale_U, mode='standard', temperature=1.0):
        """
        五模式的差异就在这里：ActionNetwork 如何处理输入的 (loc_U, scale_U)
        """
        # 步骤1: 根据模式计算 U' 的分布参数
        if mode == 'deterministic':
            # U' = μ_U (确定性)
            loc_U_final = loc_U
            scale_U_final = torch.zeros_like(scale_U)
        
        elif mode == 'exogenous':
            # U' ~ Cauchy(μ_U, |b_noise|)
            loc_U_final = loc_U
            scale_U_final = torch.full_like(scale_U, abs(self.b_noise))
        
        elif mode == 'endogenous':
            # U' ~ Cauchy(μ_U, γ_U)
            loc_U_final = loc_U
            scale_U_final = scale_U
        
        elif mode == 'standard':
            # U' ~ Cauchy(μ_U, γ_U + |b_noise|) - 解析融合
            loc_U_final = loc_U
            scale_U_final = scale_U + temperature * abs(self.b_noise)
        
        elif mode == 'sampling':
            # U' ~ Cauchy(μ_U + b_noise*ε, γ_U) - 位置扰动
            epsilon = torch.tan(torch.pi * (torch.rand_like(loc_U) - 0.5))
            loc_U_final = loc_U + temperature * self.b_noise * epsilon
            scale_U_final = scale_U
        
        # 步骤2: 线性变换 (ActionNetwork 的核心功能)
        # 利用柯西分布的线性稳定性：Y = WX + b
        loc_S = self.lm_head(loc_U_final)  # W * μ + b
        scale_S = scale_U_final @ torch.abs(self.lm_head.weight).T  # |W| * γ
        
        return loc_S, scale_S

# 总结：五模式的统一流程
def unified_causal_pipeline(X, mode='standard'):
    """
    核心认知：五模式本质上都是 ActionNetwork 的不同计算方式！
    
    统一流程：
    1. AbductionNetwork: X → (loc_U, scale_U)
    2. ActionNetwork(模式差异): (loc_U, scale_U) → (loc_S, scale_S)
    3. 损失计算: 统一的 Cauchy NLL (除了 Deterministic)
    """
    # 步骤1: 个体推断 (所有模式相同)
    loc_U, scale_U = abduction_network(X)
    
    # 步骤2: 行动决策 (模式差异的核心)
    loc_S, scale_S = action_network(loc_U, scale_U, mode=mode)
    
    # 步骤3: 损失计算 (简单二选一)
    if mode == 'deterministic':
        loss = mse_loss(loc_S, targets)  # 传统损失
    else:
        loss = cauchy_nll_loss(loc_S, scale_S, targets)  # 统一因果损失
    
    return predictions, loss
```

#### 重要认知

五种模式的本质差异就是 **ActionNetwork 的计算方式不同**！

- **AbductionNetwork**: 所有模式完全相同
- **ActionNetwork**: 模式差异的核心所在，如何从 (loc_U, scale_U) 计算 (loc_S, scale_S)
- **损失计算**: 只有 Deterministic vs 其他模式的区别

### 7.4 系统设计哲学

#### 7.4.1 参数空间完备性

五模式系统覆盖了 $(\gamma_U, b_{noise})$ 参数空间的所有有意义组合：

```mermaid
graph TB
    subgraph ParameterSpace["参数空间完备性 (γ_U, b_noise)"]
        direction TB
        
        subgraph Q1["γ_U = 0, b_noise = 0"]
            D["🎯 Deterministic<br/>等价sklearn MLP"]
        end
        
        subgraph Q2["γ_U = 0, b_noise ≠ 0"]
            E["🌍 Exogenous<br/>确定个体+外生噪声"]
        end
        
        subgraph Q3["γ_U ≠ 0, b_noise = 0"]
            EN["🧠 Endogenous<br/>纯内生因果推理"]
        end
        
        subgraph Q4["γ_U ≠ 0, b_noise ≠ 0"]
            direction LR
            S["⚡ Standard<br/>噪声→尺度"]
            SA["🎲 Sampling<br/>噪声→位置"]
        end
    end
    
    classDef deterministicStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef exogenousStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef endogenousStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef mixedStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class D deterministicStyle
    class E exogenousStyle
    class EN endogenousStyle
    class S,SA mixedStyle
```

#### 7.4.2 核心设计原则

1. **数学严谨性**: 每种模式都有明确的数学定义和理论基础
2. **实现统一性**: 所有模式共享相同的核心架构和统一公式
3. **损失函数一致性**: 模式2-5使用相同的Cauchy NLL损失
4. **参数空间完备性**: 覆盖所有有意义的$(\gamma_U, b_{noise})$组合
5. **应用场景互补性**: 每种模式都有明确的应用领域

#### 7.4.3 理论意义

- **轴1 ($\gamma_U$)**: 个体内在不确定性，体现认知差异和决策模糊性
- **轴2 ($b_{noise}$)**: 外部环境噪声，体现不可控的外生因素
- **交互效应**: Standard和Sampling模式在Q4象限中提供不同的混合策略
- **因果可解释性**: 每种模式都有明确的因果解释和现实意义

### 7.5 未来扩展与研究方向

...