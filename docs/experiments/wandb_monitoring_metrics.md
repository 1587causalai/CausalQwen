# Weights & Biases 监控指标详解 (序列到序列架构)

本文档详细解释了在 `CausalQwen2` 项目**序列到序列架构重构**后，通过 Weights & Biases (wandb) 实时监控的各项关键指标。每个指标都附有其定义、数学公式和监控目的。

> **重要提示**: 本文档反映的是V4架构（序列到序列模式）的指标计算逻辑，与之前的单一输出模式有本质区别。

---

## 架构重构对指标的根本性影响

### 关键变化概览

| 维度 | 旧架构 (单一输出) | 新架构 (序列到序列) |
|------|------------------|-------------------|
| **模型输出** | `[B, C]` | `[B, S, C]` |
| **因果表征** | 整个序列一个 U | 每个位置独立的 U_i |
| **预测范围** | 只关心<NUM>位置 | 每个位置都预测下一个token |
| **损失计算** | 基于单一预测 | 基于所有有效位置的平均 |
| **数学意义** | 序列级因果推理 | 位置独立的因果推理 |

### 核心数学框架

在新架构中，对于长度为 $S$ 的序列，每个位置 $i \in \{1,2,\ldots,S\}$ 都进行独立的推断-行动过程：

$$\forall i: \quad U_i | z_i \sim \text{Cauchy}(\mu_{U,i}, \gamma_{U,i})$$

其中 $z_i$ 是到位置 $i$ 为止的累积上下文特征。

---

## 1. 核心损失与性能指标 (Core Loss & Performance Metrics)

### 1.1. `total_loss` (总损失) - **新架构逻辑**

-   **定义**: 所有**有效序列位置**上的分类损失和门控回归损失的加权平均。
-   **公式**:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}}^{\text{seq}} + \lambda \cdot \mathcal{L}_{\text{reg\_gated}}^{\text{seq}}$$
    
    其中：
    $$\mathcal{L}_{\text{cls}}^{\text{seq}} = \frac{1}{|\mathcal{V}|} \sum_{(b,i) \in \mathcal{V}} \sum_{k=1}^{K} -\left[ y_{b,i,k} \log(p_{b,i,k}) + (1 - y_{b,i,k}) \log(1 - p_{b,i,k}) \right]$$
    
    $$\mathcal{L}_{\text{reg\_gated}}^{\text{seq}} = \frac{1}{|\mathcal{N}|} \sum_{(b,i) \in \mathcal{N}} P(\text{<NUM>}_{b,i}) \cdot \mathcal{L}_{\text{Cauchy}}(v_{b,i}, \mu_{reg,b,i}, \gamma_{reg,b,i})$$
    
    - $\mathcal{V} = \{(b,i) : \text{labels}[b,i] \neq -100\}$ 是所有有效位置的集合
    - $\mathcal{N} = \{(b,i) : \text{labels}[b,i] = \text{<NUM>}\}$ 是所有数值预测位置的集合

-   **关键变化**: 
    - **旧**: 基于单一预测的损失
    - **新**: 基于所有有效序列位置的平均损失，体现了真正的序列到序列学习

### 1.2. `cls_loss` (分类损失) - **序列化计算**

-   **定义**: 在所有有效序列位置上的 One-vs-Rest 分类损失的平均值。
-   **数学表达**: 
    $$\mathcal{L}_{\text{cls}}^{\text{seq}} = \frac{1}{|\mathcal{V}|} \sum_{(b,i) \in \mathcal{V}} \mathcal{L}_{\text{OvR}}(y_{b,i}, \{\mu_{S,k,b,i}, \gamma_{S,k,b,i}\}_{k=1}^K)$$
    
    其中每个位置 $(b,i)$ 的 OvR 概率为：
    $$p_{b,i,k} = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\mu_{S,k,b,i} - \theta}{\gamma_{S,k,b,i}}\right)$$

-   **物理意义**: 衡量模型在每个序列位置上预测下一个token的能力。这不再是"识别是否需要数值回归"，而是"在给定上下文下预测下一个token"。

### 1.3. `gated_reg_loss` (门控回归损失) - **位置感知门控**

-   **定义**: 仅在标签为 `<NUM>` 的位置计算的门控回归损失。
-   **公式**:
    $$\mathcal{L}_{\text{reg\_gated}}^{\text{seq}} = \frac{1}{|\mathcal{N}|} \sum_{(b,i) \in \mathcal{N}} P(\text{<NUM>}_{b,i}) \cdot \log\left(1 + \left(\frac{v_{b,i} - \mu_{reg,b,i}}{\gamma_{reg,b,i}}\right)^2\right)$$
    
    其中 $P(\text{<NUM>}_{b,i})$ 是位置 $(b,i)$ 预测 `<NUM>` token 的概率。

-   **关键洞察**: 每个位置都有自己的门控概率，模型学会在适当的位置"开启"数值预测模式。

### 1.4. `reg_mae` (回归平均绝对误差) - **位置过滤计算**

-   **定义**: 仅在标签为 `<NUM>` 且目标值非 NaN 的位置计算的平均绝对误差。
-   **公式**:
    $$\text{MAE}_{\text{reg}}^{\text{seq}} = \frac{1}{|\mathcal{N}_{\text{valid}}|} \sum_{(b,i) \in \mathcal{N}_{\text{valid}}} |v_{b,i} - \mu_{reg,b,i}|$$
    
    其中 $\mathcal{N}_{\text{valid}} = \{(b,i) : \text{labels}[b,i] = \text{<NUM>} \text{ and } \neg\text{isnan}(v_{b,i})\}$

-   **计算逻辑**: 
    ```python
    # 1. 筛选 <NUM> 位置
    num_mask = (batch_labels == self.tokenizer.num_token_id)
    # 2. 进一步筛选非 NaN 目标
    valid_targets = batch_target_values[num_mask]
    valid_targets = valid_targets[~torch.isnan(valid_targets)]
    # 3. 计算 MAE
    reg_mae = torch.abs(valid_preds - valid_targets).mean()
    ```

---

## 2. 个体因果表征指标 (Individual Causal Representation Metrics)

### 2.1. `units_mean_loc` - **序列平均的因果表征位置**

-   **定义**: 所有序列位置上因果表征分布位置参数的全局平均。
-   **公式**:
    $$\overline{\mu_U}^{\text{seq}} = \frac{1}{B \times S} \sum_{b=1}^B \sum_{i=1}^S \mu_{U,b,i}$$

-   **物理解释**: 反映了模型推断的"平均因果状态"。在序列到序列架构中，这代表所有位置的因果表征的中心趋势。

### 2.2. `units_mean_scale` - **序列平均的因果不确定性**

-   **定义**: 所有序列位置上因果表征分布尺度参数的全局平均。
-   **公式**:
    $$\overline{\gamma_U}^{\text{seq}} = \frac{1}{B \times S} \sum_{b=1}^B \sum_{i=1}^S \exp(\log \gamma_{U,b,i})$$

-   **监控意义**: 
    - **高值**: 模型对大部分位置的归因推断都不确定
    - **低值**: 模型对归因推断很有信心
    - **位置变化**: 理想情况下，困难位置（如数值预测前）应该有更高的不确定性

---

## 3. 序列化分类器诊断指标 (Sequence-level Classifier Diagnostics)

### 3.1. `ovr_prob_sum` - **位置平均的概率和**

-   **定义**: 所有有效序列位置上，OvR 概率和的平均值。
-   **公式**:
    $$\overline{\sum p_k}^{\text{seq}} = \frac{1}{|\mathcal{V}|} \sum_{(b,i) \in \mathcal{V}} \sum_{k=1}^{K} p_{b,i,k}$$

-   **序列化意义**: 不同序列位置可能有不同的"置信度模式"：
    - **句首位置**: 可能概率和较低（不确定接下来说什么）
    - **数值预测位置**: 可能概率和接近1（明确知道要输出<NUM>）
    - **句末位置**: 可能概率和较高（多个合理的结束方式）

### 3.2. `accuracy` - **序列化准确率**

-   **定义**: 所有有效序列位置上的平均预测准确率。
-   **公式**:
    $$\text{Accuracy}^{\text{seq}} = \frac{1}{|\mathcal{V}|} \sum_{(b,i) \in \mathcal{V}} \mathbb{I}(\hat{y}_{b,i} = y_{b,i})$$
    
    其中 $\hat{y}_{b,i} = \arg\max_k p_{b,i,k}$

-   **解释**: 这不再是"是否正确识别需要数值回归"，而是"在每个位置是否正确预测下一个token"。

### 3.3. `num_accuracy` - **数值位置专项准确率**

-   **定义**: 在所有标签为 `<NUM>` 的位置上，模型也正确预测为 `<NUM>` 的比例。
-   **公式**:
    $$\text{Num Accuracy}^{\text{seq}} = \frac{1}{|\mathcal{N}|} \sum_{(b,i) \in \mathcal{N}} \mathbb{I}(\hat{y}_{b,i} = \text{<NUM>})$$

-   **关键意义**: 这是**位置感知的门控性能**。模型需要学会在序列的特定位置（如"costs"之后）激活数值预测模式。

---

## 4. 新架构的监控策略建议

### 4.1. 健康训练的信号

1. **`total_loss`**: 应该平稳下降，不应出现剧烈波动
2. **`num_accuracy`**: 应该快速上升并保持高水平（>0.8）
3. **`reg_mae`**: 应该在`num_accuracy`提升后开始下降
4. **`units_mean_scale`**: 应该保持在合理范围内，不应崩溃或爆炸

### 4.2. 问题诊断指南

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `num_accuracy` 一直为0 | 门控机制失效 | 检查 `ovr_threshold` 设置 |
| `reg_mae` 不下降但 `num_accuracy` 很高 | 回归头学习困难 | 调整 `regression_weight` |
| `units_mean_scale` 趋于0 | 模型过于自信 | 检查正则化设置 |
| `ovr_prob_sum` 远离1 | OvR校准问题 | 调整决策阈值 |

### 4.3. 序列到序列特有的监控点

1. **位置依赖性**: 观察不同序列位置的学习速度差异
2. **上下文累积效应**: 长序列中后续位置的预测应该更准确
3. **门控位置特异性**: 数值预测应该在语义合理的位置被激活

---

## 5. 与旧架构的关键对比

| 指标 | 旧架构意义 | 新架构意义 |
|------|-----------|-----------|
| `accuracy` | 识别数值预测任务的准确率 | 所有位置的token预测准确率 |
| `num_accuracy` | 二元分类准确率 | 位置感知的门控准确率 |
| `reg_mae` | 条件回归性能 | 序列感知的回归性能 |
| `units_mean_*` | 全局因果状态 | 位置平均的因果表征 |

这种架构重构使得我们的模型真正实现了**位置独立的因果推理**，每个序列位置都进行独立的"推断-行动"过程，这是因果语言模型的核心理念的完美体现。 