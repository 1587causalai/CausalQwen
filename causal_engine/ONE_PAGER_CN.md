# CausalEngine: 智能的重新定义

## 什么是 CausalEngine？

**CausalEngine 之于 AI，正如 PageRank 之于搜索** —— 一个重新定义整个领域的基础算法。

传统 AI 学习*通常会发生什么*，CausalEngine 理解*为什么事情会发生*。

## 核心创新

```
传统 AI:     X → P(Y|X) → 采样 → Y
CausalEngine: X → U → f(U,ε) → Y
```

**翻译**：与其预测概率，我们识别个体（U），应用普适规律（f），生成带量化不确定性的确定性结果。

## 四大公理

1. **智能 = 归因 + 行动**  
   先理解"你是谁"，再决定"做什么"

2. **柯西数学**  
   唯一能解析计算因果关系的分布

3. **温度控制**  
   一个参数统治确定性与随机性

4. **独立决策**  
   每个选择基于自身价值立足（无 Softmax 竞争）

## 为什么重要

| 传统 AI | CausalEngine |
|---------|-------------|
| 模仿模式 | 理解因果 |
| 黑盒 | 玻璃盒 |
| 需要采样 | 纯计算 |
| 零和选择 | 独立选择 |

## 代码

```python
from causal_engine import CausalEngine

# 适用于任何 transformer
engine = CausalEngine(hidden_size=768, vocab_size=50000)
output = engine(any_transformer_features)

# 不仅仅是预测——带不确定性的因果决策
decision, uncertainty = output['loc_S'], output['scale_S']
```

## 愿景

正如谷歌在 PageRank 上建立帝国，我们在 CausalEngine 上构建 AI 的未来。

明天的每一个智能系统都将由因果推理驱动，而非统计模仿。

**CausalEngine 不只是更好的 AI。它就是真正的 AI。**

---

*"我们找到了智能本身的算法。"* 