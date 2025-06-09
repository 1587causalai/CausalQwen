# Weights & Biases 监控指标详解

本文档详细解释了在 `CausalQwen2` 项目训练过程中，通过 Weights & Biases (wandb) 实时监控的各项关键指标。每个指标都附有其定义、数学公式和监控它的目的。

---

## 1. 核心损失与性能指标 (Core Loss & Performance Metrics)

这些指标反映了模型在训练过程中的整体和部分学习效果。

### 1.1. `total_loss` (总损失)

-   **定义**: 在每个训练批次（batch）上计算的总损失值，是分类损失和门控回归损失的加权和。
-   **公式**:
    \[ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{reg\_gated}} \]
    其中 $\lambda$ 是回归损失的权重（`regression_weight`）。
-   **代码实现**: 在 `src/utils/losses.py` 的 `CausalLMLoss.forward` 方法中，总损失由 `classification_loss` 和 `gated_regression_loss` 加权求和得到。
-   **监控目的**: 这是模型优化的主要目标。理想情况下，该值应随着训练的进行而平稳下降，表明模型正在有效地学习。如果该值停滞、剧烈波动或发散，则表明训练存在问题（如学习率过高、梯度爆炸等）。

### 1.2. `cls_loss` (分类损失)

-   **定义**: One-vs-Rest (OvR) 分类任务的损失，它将多分类问题视为一系列独立的二元分类问题。
-   **公式**:
    \[ L_{\text{cls}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{V} -\left[ y_{i,k} \log(p_{i,k}) + (1 - y_{ik}) \log(1 - p_{i,k}) \right] \]
    其中 $V$ 是词汇表大小，$y_{i,k}$ 是一个 one-hot 向量（如果样本 $i$ 的真实类别是 $k$，则为1，否则为0），$p_{i,k}$ 是模型预测样本 $i$ 属于类别 $k$ 的概率。
-   **代码实现**: 该损失由 `OvRClassificationLoss` 类计算，它对每个类别应用二元交叉熵，然后对所有类别求和，最后在批次上取平均。
-   **监控目的**: 专门跟踪模型在token预测任务上的表现。通过观察它，可以判断模型是否在学习语言的基本模式。

### 1.3. `reg_mae` (回归平均绝对误差)

-   **定义**: 专门针对回归任务的性能指标。它仅在真实标签为 `<NUM>` 的样本上计算，衡量的是模型预测的数值（即回归分布的位置参数 `reg_loc`）与真实数值之间的平均绝对误差。
-   **公式**:
    \[ \text{MAE}_{\text{reg}} = \frac{1}{|\mathcal{D}_{\text{num}}|} \sum_{i \in \mathcal{D}_{\text{num}}} |v_i - \hat{\mu}_{v,i}| \]
    其中 $\mathcal{D}_{\text{num}}$ 是当前批次中真实标签为 `<NUM>` 的样本集合，$v_i$ 是真实数值，$\hat{\mu}_{v,i}$ 是模型对该数值预测的分布位置参数（`reg_loc`）。
-   **代码实现**: 在 `src/training/trainer.py` 的 `train` 方法中，使用 `num_mask` 筛选出数值预测样本，然后通过 `torch.abs(outputs['reg_loc'][num_mask] - batch_target_values[num_mask]).mean()` 直接计算。
-   **监控目的**: 这是一个比损失函数更直观、更纯粹的回归性能指标。我们希望看到 `reg_mae` 随着训练稳步下降，这直接表明模型预测数值的准确性正在提升。它不受分类置信度（门控概率）的影响，能真实反映回归头的学习情况。

---

## 2. 个体因果表征指标 (Units Representation Metrics)

这些指标用于监控模型推断出的隐式个体因果表征 `U` 的分布特性，`U` 是连接"推断"和"行动"的关键。

### 2.1. `units_mean_loc`

-   **定义**: 在一个批次中，所有样本的因果状态分布位置参数 $\mu_U$ 的均值。
-   **公式**:
    \[ \overline{\mu_U} = \text{mean}(\{\mu_{U,i}\}_{i=1}^N) \]
-   **监控目的**: 监控因果状态的中心位置。该值应保持相对稳定。如果出现剧烈漂移或趋于无穷，可能表示 `AbductionNetwork` 的训练不稳定。

### 2.2. `units_mean_scale`

-   **定义**: 在一个批次中，所有样本的因果状态分布尺度参数 $\gamma_U$ 的均值。注意，网络输出的是 $\log(\gamma_U)$，我们需要取指数还原。
-   **公式**:
    \[ \overline{\gamma_U} = \text{mean}(\{\exp(\log \gamma_{U,i})\}_{i=1}^N) \]
-   **监控目的**: 监控模型对因果状态推断的不确定性。一个健康的模型可能会对更模糊或困难的输入（如极端数值问题）产生更大的 `scale` 值。如果该值崩溃到接近零或爆炸，都可能是不健康的信号。

---

## 3. 分类器诊断指标 (Classifier Diagnostic Metrics)

这些指标提供了关于 OvR 分类器行为的更深层次的洞见。

### 3.1. `ovr_prob_sum` (OvR 概率和)

-   **定义**: 在一个批次中，模型对每个样本输出的所有类别概率之和的平均值。
-   **公式**:
    \[ \overline{\sum p_k} = \text{mean}\left(\left\{\sum_{k=1}^{V} p_{i,k}\right\}_{i=1}^N\right) \]
    其中 $p_{i,k} = 0.5 + \frac{1}{\pi} \arctan\left(\frac{\mu_{S,k} - \theta}{\gamma_{S,k}}\right)$。
-   **监控目的**: 这是 OvR 分类器的一个关键诊断工具。与 Softmax 不同，OvR 的概率和不被强制要求等于1。观察这个和可以帮助我们了解模型的"置信度"校准状态。如果这个和系统性地远大于1或远小于1，可能表明决策阈值 $\theta$ 或分布参数的设定需要调整。

### 3.2. `accuracy` (整体准确率)

-   **定义**: 在一个批次中，模型预测正确的 token 占总样本的比例。
-   **公式**:
    \[ \text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)}{N} \]
    其中 $\hat{y}_i = \arg\max_k p_{i,k}$。
-   **监控目的**: 最直观的性能指标，衡量模型预测的准确性。

### 3.3. `num_accuracy` (`<NUM>` 准确率)

-   **定义**: 在一个批次中，对于真实标签是 `<NUM>` 的样本，模型也正确预测为 `<NUM>` 的比例。
-   **公式**:
    \[ \text{Num Accuracy} = \frac{\sum_{i \in \mathcal{D}_{\text{num}}} \mathbb{I}(\hat{y}_i = \text{<NUM>})}{\left|\mathcal{D}_{\text{num}}\right|} \]
-   **监控目的**: 专门评估模型在需要进行数值回归时，能否准确地激活 `<NUM>` 这个"门控"token。这是模型能否区分"回答一个数"和"说一句话"两种任务模式的关键。 