# CausalQwen MVP 设计文档

> **目标**: 构建最小可行的因果语言模型，专注于验证CausalQwen的核心哲学和数学框架

## 🎯 MVP 目标与范围

### ✅ 核心验证目标
1. **因果哲学验证**: 证明"个体选择变量U"的概念可行性
2. **数学框架验证**: 验证柯西分布的线性稳定性在实际模型中的应用
3. **OvR分类验证**: 证明One-vs-Rest分类优于传统Softmax的潜力
4. **推理模式验证**: 验证因果采样与标准推理的差异性

### ❌ 明确排除的功能
- 数值处理与回归预测
- 复杂的训练策略与监控
- 预训练对齐的离线蒸馏
- 高级序列因果采样模式
- 门控机制与多任务学习

## 🧠 核心哲学

### 个体选择变量 U 的双重身份

CausalQwen的核心创新是引入**个体选择变量 U**，具有双重含义：

1. **个体选择变量**: U=u 代表从所有可能个体中"选择"了特定个体 u
2. **个体因果表征**: 向量 u 包含该个体所有内在的、驱动其行为的潜在属性

**核心思想**: 文本不是从概率分布中随机抽取的结果，而是特定"个体"在特定"环境"下的必然表达。

### 数学框架

$$Y = f(U, \epsilon)$$

其中：
- $U$ 是从上下文推断的个体因果表征分布
- $\epsilon$ 是外生噪声（不可控的随机扰动）  
- $f$ 是普适因果机制（对所有个体一致的决策规律）

### 柯西分布的选择理由

1. **诚实的不确定性**: 重尾分布为"黑天鹅"事件保留概率
2. **数学优雅**: 期望和方差未定义，对应"永远无法完全了解个体"
3. **计算高效**: 线性稳定性使整个前向传播可解析计算

## 🏗️ MVP 架构设计

### 整体数据流

```
input_ids [B,S] 
    ↓ Embedding
embeddings [B,S,H]
    ↓ QwenTransformer  
context_features [B,S,H]
    ↓ AbductionNetwork
loc_U, scale_U [B,S,C]
    ↓ ActionNetwork
loc_S, scale_S [B,S,V]
    ↓ OvR Classification
predictions
```

### 核心模块

#### 1. StandardEmbedding
- 直接使用预训练Qwen的词元嵌入
- 无任何数值感知功能

#### 2. QwenTransformer
- 完全继承预训练Qwen的Transformer层
- 提供上下文理解能力

#### 3. AbductionNetwork (归因推断网络)
```python
class AbductionNetwork(nn.Module):
    def __init__(self, hidden_size, causal_size):
        self.loc_net = nn.Linear(hidden_size, causal_size)
        self.scale_net = nn.Linear(hidden_size, causal_size)
    
    def forward(self, context_features):
        loc_U = self.loc_net(context_features)
        scale_U = F.softplus(self.scale_net(context_features))
        return loc_U, scale_U
```

#### 4. ActionNetwork (行动决策网络)
```python
class ActionNetwork(nn.Module):
    def __init__(self, causal_size, vocab_size):
        self.lm_head = nn.Linear(causal_size, vocab_size)
        self.b_noise = nn.Parameter(torch.zeros(causal_size))
    
    def forward(self, loc_U, scale_U):
        # 外生噪声注入
        scale_U_noisy = scale_U + torch.abs(self.b_noise)
        
        # 线性因果律
        loc_S = self.lm_head(loc_U)
        
        # 不确定性传播
        W_abs = torch.abs(self.lm_head.weight)
        scale_S = torch.matmul(scale_U_noisy, W_abs.T)
        
        return loc_S, scale_S
```

## 🎲 推理模式

### 1. 标准推理 (Standard Inference)
- 基于分布期望的确定性决策
- 计算OvR概率: $P_k = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{loc_{S_k} - C_{ovr}}{scale_{S_k}}\right)$
- 选择最高概率: $\hat{y} = \arg\max_k P_k$

### 2. 因果采样 (Causal Sampling)  
- 从个体分布采样具体个体: $u \sim \text{Cauchy}(loc_U, scale_U)$
- 注入环境噪声: $u' \sim \text{Cauchy}(u, |b_{noise}|)$
- 基于具体个体进行决策

### 3. 兼容传统采样
- 使用 $loc_S$ 作为logits进行Softmax
- 支持top-k/top-p采样，完全兼容传统语言模型

## 📦 损失函数

### OvR 分类损失

对每个词汇独立计算二元分类概率：

$$P_{k,i} = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{loc_{S_{k,i}} - C_{ovr}}{scale_{S_{k,i}}}\right)$$

损失函数：
$$L_{cls,i} = -\sum_{k=1}^V [y_{k,i} \log P_{k,i} + (1-y_{k,i}) \log(1-P_{k,i})] \cdot mask_i$$

总损失：
$$\mathcal{L} = \frac{\sum_{i=1}^S L_{cls,i}}{\sum_{i=1}^S mask_i}$$

## ⚙️ 初始化策略

### 知识继承原则
确保初始时 CausalQwen 的行为与原始 Qwen 尽可能一致。

#### AbductionNetwork 初始化
```python
# 位置网络：恒等映射
nn.init.eye_(self.loc_net.weight)
nn.init.zeros_(self.loc_net.bias)

# 尺度网络：常数输出
nn.init.zeros_(self.scale_net.weight) 
nn.init.constant_(self.scale_net.bias, 0.0)  # softplus(0) ≈ 0.69
```

#### ActionNetwork 初始化
```python
# 完整继承Qwen的lm_head权重
self.lm_head.weight.data = qwen.lm_head.weight.data.clone()
self.lm_head.bias.data = qwen.lm_head.bias.data.clone()

# 外生噪声初始为小常数
nn.init.constant_(self.b_noise, 0.1)
```

## 🧪 验证实验

### 1. 数学正确性验证
- 验证柯西分布的线性稳定性实现
- 验证OvR概率计算的数值稳定性
- 验证梯度计算的正确性

### 2. 行为一致性验证
- 初始化后与Qwen的输出对比
- 验证兼容性采样模式的等价性

### 3. 因果特性验证
- 对比标准推理vs因果采样的结果差异
- 验证相同个体的生成一致性

### 4. 训练收敛验证
- 在小规模数据集上验证模型可训练性
- 监控损失收敛和准确率提升

## 📁 MVP 项目结构

```
CausalQwen-MVP/
├── src/causal_qwen_mvp/
│   ├── models.py          # 所有核心模型类
│   ├── inference.py       # 三种推理模式
│   ├── training.py        # 简化训练循环
│   └── utils.py           # 基础工具
├── tests/
│   ├── test_models.py     # 模型单元测试
│   ├── test_inference.py  # 推理测试  
│   └── test_math.py       # 数学验证
├── examples/
│   ├── basic_usage.py     # 基础使用
│   └── causal_demo.py     # 因果采样演示
├── docs/
│   ├── mvp_design.md      # 本文档
│   └── quick_start.md     # 快速开始
└── README_MVP.md          # MVP说明
```

## 🚀 开发里程碑

### Milestone 1: 核心模块实现
- [ ] 实现4个核心模型类
- [ ] 实现3种推理模式
- [ ] 基础的训练循环

### Milestone 2: 数学验证
- [ ] 柯西分布计算正确性
- [ ] 与Qwen的行为一致性
- [ ] 梯度计算验证

### Milestone 3: 功能验证
- [ ] 小规模数据集训练
- [ ] 因果采样效果验证
- [ ] 性能基准测试

### Milestone 4: 文档完善
- [ ] 快速开始指南
- [ ] API文档
- [ ] 使用示例

## 💡 成功标准

MVP被认为成功当且仅当：

1. **数学正确性**: 所有柯西分布计算通过单元测试
2. **行为一致性**: 初始化后与Qwen的输出差异 < 1e-3
3. **可训练性**: 能在小数据集上正常收敛
4. **因果特性**: 因果采样展现出与标准推理的明显差异
5. **兼容性**: 支持传统的top-k/top-p采样

一旦MVP达到这些标准，就为扩展到完整版本（包含数值处理）奠定了坚实基础。 