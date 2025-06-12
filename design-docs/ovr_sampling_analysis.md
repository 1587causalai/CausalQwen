# OvR 模型下的采样机制深度分析

## 1. 问题背景

CausalQwen 使用 One-vs-Rest (OvR) 分类模型，这与传统的 softmax 多分类有本质区别。这种差异对 top-k 和 top-p 采样的实现产生了重要影响。

## 2. 数学基础对比

### 2.1 传统 Softmax 模型

在传统语言模型中，输出层使用 softmax：

$$p_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$$

关键性质：
- **自然归一化**：$\sum_{k=1}^K p_k = 1$
- **互斥性**：选择一个类别自动排除其他类别
- **相对性**：概率只依赖于 logits 的相对大小

### 2.2 OvR 柯西模型

CausalQwen 的 OvR 模型中，每个类别的概率独立计算：

$$P_{k,i} = P(S_{k,i} > C_k) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

其中 $S_{k,i} \sim \text{Cauchy}(\text{loc}_{S_{k,i}}, \text{scale}_{S_{k,i}})$

关键性质：
- **非归一化**：$\sum_{k=1}^K P_{k,i} \neq 1$（可能远大于或远小于 1）
- **独立性**：每个 $P_{k,i}$ 独立计算，类别间无直接竞争
- **绝对性**：概率依赖于分数与阈值的绝对关系

## 3. 采样机制的数学分析

### 3.1 为什么需要重新归一化？

在生成任务中，我们必须选择**恰好一个** token 作为下一个输出。这要求：

$$\sum_{k=1}^K \tilde{p}_k = 1 \quad \text{且} \quad \tilde{p}_k \geq 0$$

但 OvR 概率 $P_{k,i}$ 不满足归一化约束。

### 3.2 重新归一化的数学合理性

我们采用的重新归一化策略：

$$\tilde{p}_k = \frac{P_k}{\sum_{j=1}^K P_j}$$

这种方法的合理性：

1. **保持相对顺序**：如果 $P_i > P_j$，则 $\tilde{p}_i > \tilde{p}_j$
2. **保持零概率**：如果 $P_k = 0$，则 $\tilde{p}_k = 0$
3. **最大似然一致**：$\arg\max_k P_k = \arg\max_k \tilde{p}_k$

### 3.3 Top-k 采样的 OvR 实现

**算法**：
1. 计算所有 OvR 概率：$\{P_1, P_2, ..., P_K\}$
2. 选择最大的 $k$ 个：$\mathcal{K} = \text{top-k}(\{P_i\})$
3. 在子集上重新归一化：$\tilde{p}_i = P_i / \sum_{j \in \mathcal{K}} P_j$ for $i \in \mathcal{K}$
4. 从归一化分布采样：$i^* \sim \text{Categorical}(\{\tilde{p}_i : i \in \mathcal{K}\})$

### 3.4 Top-p 采样的 OvR 实现

**算法**：
1. 按 OvR 概率降序排序：$P_{(1)} \geq P_{(2)} \geq ... \geq P_{(K)}$
2. 找到最小的 $m$ 使得：$\sum_{i=1}^m P_{(i)} / \sum_{j=1}^K P_{(j)} \geq p$
3. 在前 $m$ 个类别上重新归一化并采样

## 4. 因果采样 vs 传统采样

### 4.1 因果采样（推荐）

**核心理念**：采样因果表征（原因），观察确定性结果。

```python
# 步骤 1：采样因果表征（无温度！）
u ~ Cauchy(μ, γ)  # U 是因果表征，不是概率分布

# 步骤 2：计算确定性分数
s_k = A_k · u + B_k  # 对所有 k

# 步骤 3：将分数转换为概率（可选方案）
# 方案 A：使用 softmax（简化，避免 OvR 复杂性）
p = softmax(s / T)  # T 是温度，应用于分数

# 方案 B：argmax（确定性）
next_token = argmax(s)

# 步骤 4：采样（如果使用概率）
next_token ~ Categorical(p)
```

**关键澄清**：
- 温度 **不应该** 应用于 U 的采样
- U 是因果表征，代表个体的内在属性
- 温度可以应用于将分数转换为概率的过程

### 4.2 传统采样（基于 OvR 概率）

```python
# 步骤 1：计算 OvR 概率
P_k = 0.5 + (1/π) * arctan((loc_k - C_k) / scale_k)

# 步骤 2：重新归一化（必须！）
P_normalized = P / sum(P)

# 步骤 3：温度调整（可选）
if T != 1.0:
    logits = log(P_normalized)
    P_normalized = softmax(logits / T)

# 步骤 4：应用 top-k/top-p 并采样
next_token ~ Categorical(filtered(P_normalized))
```

## 5. 理论洞察

### 5.1 为什么因果采样不需要温度？

因果表征 U 代表了个体的内在属性，是"本质"而非"表现"。温度参数用于控制随机性，应该应用于"选择"过程，而不是"本质"的定义。

正确的逻辑是：
1. 个体的本质（U）是确定的分布 Cauchy(μ, γ)
2. 基于本质的行动倾向（分数 s）是确定的
3. 最终的选择（采样）可以有随机性（通过温度控制）

### 5.2 OvR 概率的语义

- $P_k = 0.5$：类别 $k$ 与"其他所有"势均力敌
- $P_k > 0.5$：类别 $k$ 优于"其他所有"的聚合
- $P_k < 0.5$：类别 $k$ 劣于"其他所有"的聚合

### 5.3 温度的作用

在因果采样中，温度 $T$ 直接作用于柯西分布的尺度参数：

$$u \sim \text{Cauchy}(\mu, \gamma \cdot T)$$

- $T < 1$：降低随机性，使采样更确定
- $T > 1$：增加随机性，使采样更多样
- $T \to 0$：退化为确定性选择（最大 OvR 概率）

## 6. 实践建议

1. **优先使用因果采样**：它更符合模型的因果结构
2. **正确应用温度**：
   - 因果采样：温度应用于分数→概率转换
   - 传统采样：温度应用于归一化后的概率
3. **理解 OvR 语义**：不要期望原始概率和为 1
4. **简化实现**：在因果采样中使用 softmax 而非 OvR 概率

## 7. 代码示例

```python
def ovr_sampling_with_top_k(ovr_probs, k, temperature=1.0):
    """
    在 OvR 模型上实现 top-k 采样
    
    Args:
        ovr_probs: [batch_size, vocab_size] OvR 概率（未归一化）
        k: top-k 参数
        temperature: 温度参数（仅影响因果采样）
    
    Returns:
        sampled_indices: [batch_size] 采样的 token 索引
    """
    # 1. 获取 top-k
    top_k_probs, top_k_indices = torch.topk(ovr_probs, k, dim=-1)
    
    # 2. 重新归一化（关键步骤！）
    top_k_probs_normalized = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    
    # 3. 采样
    sampled_idx_in_top_k = torch.multinomial(top_k_probs_normalized, 1)
    
    # 4. 映射回原始索引
    sampled_indices = torch.gather(top_k_indices, -1, sampled_idx_in_top_k)
    
    return sampled_indices.squeeze(-1)
```

## 8. 总结

OvR 模型下的采样需要额外的重新归一化步骤，这是由于 OvR 概率的独立性导致的。虽然这增加了实现复杂度，但保持了模型的因果解释性和灵活性。理解这一点对于正确使用 CausalQwen 至关重要。
