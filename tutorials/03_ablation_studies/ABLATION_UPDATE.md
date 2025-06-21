# 消融实验设计更新说明

## 重要更新 🔄

我们已经更新了消融实验的设计，使其更加科学和公平。

## 原始设计的问题

之前的设计创建了三个不同的网络：
1. 传统神经网络（完全不同的架构）
2. 简化的CausalEngine（仅输出loc）
3. 完整的CausalEngine（输出loc和scale）

这种设计存在的问题：
- 网络架构不同，无法准确评估因果机制的贡献
- 消融版本是"简化"的网络，而不是真正的消融实验

## 新设计的优势 ✨

**核心原则：使用完全相同的网络结构，仅在损失计算上有差异**

```python
# 新设计：同一个CausalEngine实例
engine = CausalEngine(...)

# 消融版本：忽略scale，仅使用loc计算损失
loss_ablation = CrossEntropy(loc, target)  # 分类
loss_ablation = MSE(loc, target)          # 回归

# 完整版本：使用完整的因果损失（利用loc和scale）
loss_full = engine.compute_causal_loss(loc, scale, target)
```

### 三层对比实验

1. **传统神经网络基准**
   - 标准的MLP架构
   - 分类：softmax + 交叉熵
   - 回归：MSE损失

2. **CausalEngine消融版本**
   - 使用完整的CausalEngine架构
   - 前向传播产生loc和scale
   - **但损失计算时仅使用loc**
   - 这样可以准确评估："如果忽略因果机制会怎样？"

3. **CausalEngine完整版本**
   - 使用完整的CausalEngine架构
   - 利用完整的因果损失（loc + scale）
   - 展示因果机制的完整威力

## 实现示例

```python
# 创建消融实验
engine, wrapper = create_ablation_experiment(
    input_dim=input_size,
    task_type='classification',
    num_classes=num_classes
)

trainer = AblationTrainer(engine, wrapper)

# 训练消融版本（仅loc损失）
metrics_ablation = trainer.train_step_ablation(inputs, targets)

# 训练完整版本（因果损失）
metrics_full = trainer.train_step_full(inputs, targets)
```

## 关键优势

1. **公平对比**：网络架构完全相同，唯一变量是损失函数
2. **科学严谨**：真正的消融实验，准确评估因果机制的贡献
3. **易于解释**：
   - 消融vs完整的差异 = 因果机制（scale）的贡献
   - 基准vs消融的差异 = 架构差异的影响

## 需要更新的文件

- [x] `tutorials/utils/ablation_networks.py` - 已更新
- [x] `tutorials/03_ablation_studies/comprehensive_comparison.py` - 已更新
- [x] `tutorials/01_classification/adult_income_prediction.py` - 已更新
- [ ] `tutorials/02_regression/bike_sharing_demand.py` - 需要更新
- [ ] 其他使用旧API的教程文件

## 预期结果

通过这种设计，我们期望看到：
1. **消融版本 ≈ 传统网络**：证明仅使用loc时，CausalEngine退化为普通网络
2. **完整版本 > 消融版本**：证明因果机制（scale）带来的性能提升
3. **定量评估**：准确量化因果推理的贡献度 