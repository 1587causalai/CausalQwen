# CausalQwen-0.5B 项目规划

## 代码架构设计

本项目采用模块化设计，清晰分离各个组件的职责，以便于后续扩展和维护。

### 核心模块结构

1. **FeatureNetwork（特征网络）**
   - 职责：将输入序列转换为高维特征表示
   - 实现：初始阶段使用模拟器（mocker）生成随机特征，后续可替换为真实的Qwen-0.5B主干
   - 接口：`forward(input_ids) -> feature_representation`

2. **AbductionNetwork（推断网络）**
   - 职责：从特征表示推断潜在因果状态的概率分布参数
   - 实现：线性层将特征映射到柯西分布的位置和尺度参数
   - 接口：`forward(features) -> (loc, scale)`

3. **ActionNetwork（行动网络）**
   - 职责：从因果状态生成分类和回归预测
   - 子组件：
     - 分类头：生成词元预测概率
     - 回归头：生成数值预测
   - 接口：`forward(causal_state) -> (classification_logits, regression_value)`

4. **CausalDistributions（因果分布）**
   - 职责：实现柯西分布相关操作
   - 功能：密度函数、累积分布函数、采样、重参数化等
   - 接口：`sample(loc, scale)`, `cdf(x, loc, scale)`, `pdf(x, loc, scale)`等

5. **LossFunction（损失函数）**
   - 职责：计算训练损失
   - 组件：
     - 分类损失：基于OvR的二元交叉熵
     - 回归损失：柯西负对数似然
     - 门控机制：根据`<NUM>`预测概率调整回归损失权重
   - 接口：`compute_loss(predictions, targets)`

### 数据流设计

```
输入序列 → 分词器 → FeatureNetwork → AbductionNetwork → 
                                      ↓
                                 因果状态分布
                                      ↓
                                 ActionNetwork → 分类预测 + 回归预测
```

### 训练与推理流程

1. **训练流程**
   - 前向传播：计算因果状态分布参数，解析计算输出分布
   - 损失计算：分类损失 + 门控回归损失
   - 反向传播：更新网络参数

2. **推理流程**
   - 确定性推理：使用分布参数的中位数作为预测
   - 随机推理：从分布中采样，模拟真实世界的随机性

## 模块职责详细说明

### 1. 模型组件 (`src/models/`)

- **feature_network.py**
  - `MockFeatureNetwork`: 模拟特征提取网络，生成随机特征
  - `QwenFeatureNetwork`: 封装Qwen-0.5B作为特征提取器（后续实现）

- **abduction_network.py**
  - `AbductionNetwork`: 推断因果状态分布参数的线性网络

- **action_network.py**
  - `ActionNetwork`: 包含分类头和回归头的行动网络
  - `ClassificationHead`: 实现OvR分类
  - `RegressionHead`: 实现数值回归

- **causal_lm.py**
  - `CausalLanguageModel`: 整合所有组件的完整模型

### 2. 工具函数 (`src/utils/`)

- **distributions.py**
  - 实现柯西分布相关函数
  - 实现重参数化采样

- **losses.py**
  - 实现OvR分类损失
  - 实现柯西回归损失
  - 实现门控机制

- **metrics.py**
  - 实现评估指标

### 3. 数据处理 (`src/data/`)

- **tokenizer.py**
  - 处理`<NUM>`词元
  - 数值编码与解码

- **dataset.py**
  - 实现数据集类
  - 数据预处理和批处理

- **synthetic.py**
  - 生成合成数据用于验证

### 4. 实验 (`experiments/`)

- **synthetic/**
  - 在合成数据上验证模型
  - 测试收敛性和性能

- **visualization/**
  - 可视化因果状态分布
  - 可视化决策边界

## 开发路线图

1. **阶段1：核心架构实现**
   - 实现所有模块的基础版本
   - 使用模拟器替代复杂组件

2. **阶段2：验证与测试**
   - 在合成数据上验证模型
   - 调整超参数和架构

3. **阶段3：集成真实模型**
   - 替换模拟器为真实的Qwen-0.5B
   - 实现完整的训练和推理流程

4. **阶段4：文档和示例**
   - 完善文档
   - 提供使用示例和教程

