# 初始化策略详解

本文档详细介绍 CausalQwen 的初始化策略及其数学保证。

> 返回主文档：[`../mathematical_foundations.md`](../mathematical_foundations.md)

## 设计原则

为了使 CausalQwen 能够无缝继承基座模型的强大语言能力，我们采用了一种**简单而精确**的初始化策略。其核心思想是：**在训练开始时，CausalQwen 的行为应与原始的 Qwen 完全一致**。设计原则如下：

1. **数学恒等**：新增模块在初始时表现为恒等映射或零映射
2. **知识保持**：完整继承 Qwen 的语言表征能力  
3. **渐进激活**：新功能在训练过程中逐步被"唤醒"

## 初始化步骤

### 步骤1：数值感知嵌入 → 保守初始化 (数值感知嵌入层)

数值感知嵌入层的初始化需要确保数值编码不会对原有的词元嵌入造成过大干扰。

- **`<NUM>` 词元嵌入处理**：因为我们将 `<NUM>` 设置成在 Qwen 词汇表中的第一个保留词元(Qwen2.5-0.5B有271个保留词元)，直接继承 Qwen 的 `<NUM>` 嵌入,无需额外初始化：$$\text{embed}(\text{<NUM>}) \leftarrow \text{embed}_{\text{Qwen}}(\text{<NUM>})$$

- **方向向量初始化**(后续可以考虑让他是可学习参数)：
$$\vec{e} \sim \mathcal{N}(0, \sigma_e^2 I), \quad \text{然后归一化: } \vec{e} \leftarrow \frac{\vec{e}}{\|\vec{e}\|}$$

其中 $\sigma_e$ 是小的标准差（如 $\sigma_e = 0.02$）。最终，位置 $i$ 的数值感知嵌入层的计算公式为：
$$e_i = \text{embed}(x_i) + \phi(v_i), \quad \text{where }  \phi(v) = \text{sign}(v) \cdot \ln(1 + |v|) \cdot \vec{e}$$

使用它作为输出，经过 Qwen 主干网络，得到高维的特征表征 $z_i$(形状: [B, S, H]), 作为后续归因推断网络的输入。

### 步骤2：归因推断网络 → 恒等映射

设定归因网络的权重和偏置，其位置网络和尺度网络的输入输出维度都分别是 [B, S, H] 和 [B, S, C] (默认 C=H), 使得：
$$\text{loc}_{U_i} = z_i, \quad \text{scale}_{U_i} = \gamma_i \text{ (大数值)}$$

**数学实现**：
- 初始化位置参数：$W_{\text{loc}} = I$（恒等矩阵），$b_{\text{loc}} = 0$
- 初始化尺度参数：$W_{\text{scale}} = 0$，$b_{\text{scale}} = \log(\gamma)$，其中 $\gamma$ 是大常数（如 $\gamma = 10$）

**效果**：因果表征 $U_i$ 的分布为宽泛分布 $U_i \sim \text{Cauchy}(z_i, \gamma_i)$，当 $\gamma_i$ 很大时，柯西分布近似**均匀先验**，归因网络在保持恒等映射的同时提供了高不确定性的表征。

**数学直觉**：大尺度的柯西分布 $\text{Cauchy}(\mu, \gamma)$ 当 $\gamma \gg 1$ 时，在较大范围内近似均匀分布，这为模型提供了"无知先验"——模型开始时对个体差异保持最大的不确定性。

### 步骤3：行动网络(分类) → 复制 Qwen 权重

直接将 Qwen 的词汇表预测头权重复制到分类行动网络：
$$\mathbf{W}_{\text{cls}} \leftarrow \mathbf{W}_{\text{Qwen\_lm\_head}}, \quad \mathbf{b}_{\text{cls}} = 0$$

**数学保证**：由于 $\text{loc}_{U_i} = z_i$，我们的分类 logits 与 Qwen 的原始 logits **完全相等**：
$$
S_{k,i} \sim \text{Cauchy}(\text{loc}_{S_{k,i}}, \text{scale}_{S_{k,i}}) \\
\text{loc}_{S_{k,i}} = \mathbf{W}_{\text{cls}}[k, :] \cdot z_i + \mathbf{b}_{\text{cls}}[k] = \mathbf{W}_{\text{Qwen}}[k, :] \cdot z_i = s_{k,i}^{\text{Qwen}} \\
\text{scale}_{S_{k,i}} = |\mathbf{W}_{\text{scale}}[k, :]| \cdot \gamma_i
$$

**关键结论**：在兼容传统采样模式下，CausalQwen 使用位置参数 $\text{loc}_{S_{k,i}}$ ( $\text{loc}_{S}$ 和 $\text{scale}_{S}$ 形状都是 [B, S, V_full]) 作为 logits 的 Softmax 概率分布与 Qwen 的输出**数学上完全一致**：
$$P_{\text{CausalQwen}}^{\text{softmax}}(y_i=k|\mathbf{x}) = P_{\text{Qwen}}(y_i=k|\mathbf{x})$$

这确保了 CausalQwen 在初始化时不仅行为类似 Qwen，而是**精确地复制了 Qwen 的语言建模能力**。

### 步骤4：行动网络(回归) → 常规初始化

将回归行动网络使用标准的小权重初始化，如 Xavier 或 Kaiming 初始化。

**数学效果**：由于 $\|W_{\text{reg}}\|$ 很小，结合大尺度的因果表征分布 $U_i \sim \text{Cauchy}(z_i, \gamma_i)$，位置 $i$ 的回归预测分布为：
$$Y_i \sim \text{Cauchy}(\mu_{\text{reg},i}, \gamma_{\text{reg},i}),  \\
\mu_{\text{reg},i} = W_{\text{reg}} \cdot z_i + b_{\text{reg}},  \gamma_{\text{reg},i} = |W_{\text{reg}}| \cdot \gamma_i$$

其中 $|W_{\text{reg}}|$ 是该向量每个元素都取绝对值，回归输出近似为以 $0$ 为中心的宽泛分布，提供了**无偏的回归先验** ($\mu_{\text{reg}}$ 和 $\gamma_{\text{reg}}$ 的张量形状都是 [B, S]) 。

### 步骤5：OvR 阈值 → 统一设置

在 OvR 分类中，阈值 $C_k$ 决定了柯西分布决策分数超过阈值的概率计算：

$$P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

得一个形状是 [B, S, V_full] 的概率张量 $\mathbf{P}$。

**阈值初始化**：所有类别使用相同的常数阈值
$$C_k = C_{\text{OvR}}, \quad \forall k \in \{0, 1, \ldots, V_{\text{full}}-1\}$$

其中 $C_{\text{OvR}}$ 是预设常数（如 100.0）, 后续可以考虑让它是可学习参数。

**数学效果**：
- $C_{\text{OvR}} = 0$: 初始概率接近 0.5，无明显偏好
- $C_{\text{OvR}} = 10$: 初始概率普遍较低，创造稀疏激活
- $C_{\text{OvR}} \geq 100$: 极度稀疏的初始概率分布

**推荐设置**：$C_{\text{OvR}} = 100.0$，这提供了良好的起点.

## 初始化效果总结

通过上述初始化步骤，CausalQwen 在训练开始时具有以下性质：

-   **因果表征**: 对于每个位置 $i$，因果表征 $U_i$ 服从宽泛的柯西分布 $U_i \sim \text{Cauchy}(z_i, \gamma_i)$，其中 $\gamma_i$ 是大数。
-   **分类决策**: 分类行动网络的输出与 Qwen 的原始输出完全一致，即 $\text{loc}_{S_{k,i}} = \mathbf{W}_{\text{Qwen}}[k, :] \cdot z_i$。
-   **回归决策**: 由于回归网络的标准初始化，回归预测的分布为 $Y_i \sim \text{Cauchy}(W_{\text{reg}} \cdot z_i + b_{\text{reg}},  |W_{\text{reg}}| \cdot \gamma_i)$，当 $\gamma_i$ 很大时，近似为以 0 为中心的宽泛分布。
-   **阈值设置**: 所有类别共享相同的初始阈值 $C_k = C_{\text{OvR}}$，这提供了一个适度稀疏且数值稳定的初始概率分布。
