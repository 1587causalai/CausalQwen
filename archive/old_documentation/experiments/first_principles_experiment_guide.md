# 第一性原理初始化实验指南

## 概述

本指南介绍如何使用更新后的 `src/run_experiments.py` 脚本进行基于第一性原理初始化策略的实验。

## 🎯 核心改进

### 第一性原理初始化策略
- **ActionNetwork偏置**: 所有偏置设为0.0（移除魔法数字8.0和50.0）
- **不确定性来源**: 纯粹通过归因推断网络 (AbductionNetwork) 的`scale_U`表达
- **数学一致性**: 完全符合柯西分布框架的数学性质
- **无偏见起点**: 模型从完全中性状态开始学习

### 新增实验类型
- **initialization**: 专门比较不同初始化策略的实验类型
- **可配置参数**: 支持调节初始不确定性水平

## 🚀 快速开始

### 基础实验
```bash
# 运行基础验证实验（推荐首次使用）
python src/run_experiments.py basic --epochs 5 --num_samples 1000

# 跳过训练，仅评估未训练模型
python src/run_experiments.py basic --no_train
```

### 完整实验
```bash
# 运行完整的多数据集评估
python src/run_experiments.py comprehensive --epochs 10 --num_samples 2000

# 使用WandB记录实验
python src/run_experiments.py comprehensive --use_wandb --epochs 10
```

### 对比实验
```bash
# 比较不同超参数设置
python src/run_experiments.py comparison --epochs 10 --num_samples 1500

# 测试不同的OvR阈值
python src/run_experiments.py comparison --ovr_threshold 50.0 --epochs 10
```

### 消融实验
```bash
# 验证核心架构组件的重要性
python src/run_experiments.py ablation --epochs 15 --num_samples 2000
```

### 初始化策略实验（新增）
```bash
# 比较不同初始化策略
python src/run_experiments.py initialization --epochs 10 --num_samples 1500

# 测试不同的初始不确定性水平
python src/run_experiments.py initialization --initial_scale_bias 1.0 --epochs 10
python src/run_experiments.py initialization --initial_scale_bias 3.0 --epochs 10
```

## 📊 参数说明

### 模型架构参数
- `--hidden_size`: 隐藏层大小（默认896，适配Qwen-0.5B）
- `--ovr_threshold`: OvR决策阈值（默认10.0）
- `--reg_loss_weight`: 回归损失权重（默认1.0）
- `--initial_scale_bias`: 初始scale偏置（默认2.3，对应exp(2.3)≈10的初始不确定性）

### 训练参数
- `--epochs`: 训练轮数（默认10）
- `--batch_size`: 批次大小（默认16）
- `--num_samples`: 合成训练样本数（默认1000）
- `--lr`: 学习率（默认1e-4）

### 实验控制参数
- `--no_train`: 跳过训练，仅评估
- `--use_wandb`: 使用Weights & Biases记录
- `--qwen_model_path`: Qwen模型路径（默认~/models/Qwen2.5-0.5B）
- `--results_base_dir`: 结果保存目录（默认results/）

## 🔬 实验类型详解

### 1. Basic实验
- **目的**: 快速验证模型基本功能
- **数据集**: 仅basic数据集
- **用途**: 调试、快速测试

### 2. Comprehensive实验
- **目的**: 全面评估模型性能
- **数据集**: 所有评估数据集
- **用途**: 完整性能评估

### 3. Comparison实验
- **目的**: 比较不同超参数设置
- **变量**: OvR阈值、回归损失权重等
- **用途**: 超参数优化

### 4. Ablation实验
- **目的**: 验证核心组件重要性
- **变量**: OvR分类器、柯西分布等
- **用途**: 架构验证

### 5. Initialization实验（新增）
- **目的**: 比较不同初始化策略
- **变量**: 初始不确定性水平
- **用途**: 初始化策略优化

## 📈 结果分析

### 输出文件
每次实验会生成以下文件：
- `results.json`: 详细数值结果
- `experiment_summary.md`: 实验总结报告
- `model_*.pth`: 训练后的模型权重
- `evaluation_outputs_*.pt`: 原始评估输出

### 关键指标
- **Classification F1**: 分类任务F1分数
- **Regression MAE**: 回归任务平均绝对误差
- **Regression PICP**: 回归预测区间覆盖概率

### 第一性原理效果观察
1. **初始状态**: 所有token概率接近均匀分布
2. **学习过程**: `P(<NUM>)`自然增长，回归损失逐渐激活
3. **收敛状态**: 分类和回归任务达到平衡

## 🎯 推荐实验流程

### 第一步：基础验证
```bash
# 验证环境和基本功能
python src/run_experiments.py basic --epochs 2 --num_samples 200 --no_train
```

### 第二步：初始化对比
```bash
# 测试不同初始不确定性水平
python src/run_experiments.py initialization --epochs 5 --num_samples 500
```

### 第三步：超参数调优
```bash
# 寻找最佳OvR阈值
python src/run_experiments.py comparison --epochs 10 --num_samples 1000
```

### 第四步：完整评估
```bash
# 最终性能评估
python src/run_experiments.py comprehensive --epochs 15 --num_samples 2000 --use_wandb
```

## 🔧 故障排除

### 常见问题
1. **内存不足**: 减少`--batch_size`或`--num_samples`
2. **Qwen模型未找到**: 检查`--qwen_model_path`路径
3. **WandB连接失败**: 检查网络连接或移除`--use_wandb`

### 调试技巧
1. 使用`--no_train`快速测试配置
2. 使用小的`--num_samples`进行快速迭代
3. 检查生成的`experiment_summary.md`了解结果

## 📚 相关文档

- [数学理论基础](../math/mathematical_foundations.md)
- [核心架构设计](../core-design.md)
- [因果表征变量U深度解读](../U_deep_dive.md)

## 🎉 预期效果

使用第一性原理初始化策略，您应该观察到：
1. **更稳定的训练过程**: 无人为偏见干扰
2. **更自然的学习节奏**: 门控机制自动调节
3. **更好的泛化能力**: 避免局部最优陷阱
4. **更强的数学一致性**: 完全符合柯西分布理论 