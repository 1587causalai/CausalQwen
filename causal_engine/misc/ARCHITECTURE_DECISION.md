# CausalEngine 架构决策记录

## 决策：ActivationHead 作为独立模块

### 背景

在 CausalEngine v2.0 的模块化架构中，我们需要决定任务激活（ActivationHead）应该：
- A) 作为 ActionNetwork 的一部分
- B) 作为独立模块（当前设计）

### 决策：选择方案 B - 独立模块

### 理由

#### 1. 数学清晰性

整个因果推理流程在数学上是三个清晰的阶段：
- **归因推断**：X → U ~ Cauchy(μ_U, γ_U)  
- **行动决策**：U → S ~ Cauchy(loc_S, scale_S)
- **任务激活**：S → Y (通过 CDF 或线性变换)

其中 S（决策分布）是一个数学上有意义的中间表示：
- S 是抽象的"因果决策潜力"
- S 是任务无关的，只依赖于因果机制
- Y 是具体的任务输出，依赖于特定的激活函数

#### 2. 工程优势

**解耦性**：
- 因果核心（AbductionNetwork + ActionNetwork）与任务逻辑完全分离
- 可以独立测试和优化各个模块

**灵活性**：
- 同一个训练好的因果核心可以配置不同的激活头
- 支持多任务学习：共享因果表示，切换激活策略

**扩展性**：
- 添加新的激活模式（如离散有序）只需修改 ActivationHead
- 不影响核心的因果推理逻辑

#### 3. 使用场景

```python
# 场景1：多任务共享因果核心
causal_core = CausalEngine(hidden_size, vocab_size, apply_activation=False)
loc_S, scale_S = causal_core(hidden_states)[:2]

# 任务1：文本生成（分类）
text_head = ActivationHead(vocab_size, "classification")
text_output = text_head(loc_S, scale_S)

# 任务2：情感评分（离散有序）
sentiment_head = ActivationHead(1, "ordinal", ordinal_num_classes=5)
sentiment_score = sentiment_head(loc_S[:, :, :1], scale_S[:, :, :1])

# 场景2：迁移学习
# 预训练因果核心，然后为新任务添加特定的激活头
```

### 实现细节

目前支持三种激活模式：
1. **分类激活**：P(S_k > C_k) = 1/2 + (1/π)arctan((loc_S_k - C_k)/scale_S_k)
2. **回归激活**：y_k = a_k * loc_S_k + b_k
3. **离散有序激活**：P(Y=k) = P(C_k < S ≤ C_{k+1})

每个输出维度可以独立选择激活模式，实现最大的灵活性。

### 结论

保持 ActivationHead 作为独立模块是正确的架构选择，它既保持了数学的清晰性，
又提供了工程上的灵活性。这种设计使 CausalEngine 成为一个真正模块化的因果推理框架。 