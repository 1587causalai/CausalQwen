# CausalQwen MVP 设计文档

> **📋 文档用途**: MVP进度跟踪，任务管理用  
> **🎯 目标读者**: 项目负责人，用于监控开发进展和任务优先级  
> **📖 内容定位**: 当前阶段的具体任务、完成状态、阻塞问题的工作日志

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
├── src/causal_qwen_mvp/           # 🎯 核心MVP实现
│   ├── models.py                  # 所有核心模型类
│   ├── inference.py               # 三种推理模式  
│   ├── training.py                # 训练循环和验证
│   └── __init__.py                # 模块入口
├── scripts/
│   └── check_everything_works.py      # MVP框架测试
├── tests/                         # 单元测试 (待添加)
├── docs/
│   ├── mvp_design.md              # 本设计文档
│   └── implementation_plan.md     # 实现指南
├── archive/old_implementations/   # 🗂️ 已清理的旧代码
│   ├── models/                    # 旧的分散模型实现
│   ├── training/                  # 旧的复杂训练逻辑
│   ├── losses/                    # 旧的损失函数
│   ├── train.py                   # 旧的数值感知训练脚本
│   └── run_experiments.py         # 旧的复杂实验脚本
```

## ✅ 实现进度更新 (2025-06-16)

### Milestone 1: 核心模块实现 ✅ 已完成
- ✅ 实现4个核心模型类 (`CausalQwenMVPForCausalLM`, `AbductionNetwork`, `ActionNetwork`, `OvRClassifier`)
- ✅ 实现3种推理模式 (标准、因果采样、兼容传统)
- ✅ 基础的训练循环 (`CausalTrainer`)

### Milestone 2: 基础验证 ✅ 已完成  
- ✅ 框架架构正确性验证
- ✅ 前向传播和梯度计算验证
- ✅ 推理模式功能验证
- ✅ 序列生成验证

### 🎯 MVP测试结果 (2025-06-16)
```
🏆 CausalQwen MVP v0.1.0 测试结果: 6/6 通过

✅ 模型初始化      - 参数量: 944,464
✅ 前向传播        - 损失: 0.8361，输出维度正确
✅ 推理模式        - 三种模式全部正常工作
✅ 序列生成        - 自回归生成功能正常
✅ 梯度计算        - 反向传播和权重更新正常
✅ 推理验证器      - 验证工具正常

架构验证通过：
- 成功继承 Qwen2ForCausalLM
- 因果模块正确集成
- 三种推理模式输出正确
- 训练损失正常计算
```

### Milestone 3: 待完成
- [ ] 数学严格性完善 (完善Cauchy分布计算)
- [ ] 权重初始化优化 (从预训练Qwen复制权重)
- [ ] 小规模数据集训练验证
- [ ] 因果采样效果分析

### Milestone 4: 文档完善
- ✅ 设计文档完成
- ✅ 实现指南完成  
- [ ] 快速开始指南
- [ ] API文档

## 💡 成功标准

### ✅ 已达到的标准
1. **✅ 架构正确性**: 模型能够正确初始化并运行
2. **✅ 功能完整性**: 三种推理模式全部实现并验证通过
3. **✅ 梯度计算**: 前向和反向传播正常，支持训练
4. **✅ 代码整洁性**: 移除旧代码，保持项目结构清晰

### 🎯 待达到的标准
1. **数学严格性**: 完善Cauchy分布计算的数学实现
2. **行为一致性**: 初始化后与Qwen的输出差异 < 1e-3
3. **可训练性**: 能在小数据集上正常收敛
4. **因果特性**: 因果采样展现出与标准推理的明显差异
5. **兼容性**: 完全兼容HuggingFace生态

## 🎉 MVP阶段性成果

通过今天的工作，我们成功地：

1. **🏗️ 搭建了完整的MVP框架**，从模型定义到训练推理全套实现
2. **🔧 继承了Qwen2架构**，最大化复用现有基础设施 
3. **🎯 实现了三种推理模式**，体现了因果模型的核心特色
4. **✅ 通过了全部基础测试**，验证了架构的正确性
5. **🧹 清理了项目结构**，为后续开发奠定了清晰的基础

这个MVP为扩展到完整版本（包含严格的数学实现和数值处理）奠定了坚实的基础。接下来的工作重点是完善数学实现细节和进行更深入的验证。 