# 消融实验研究

本目录包含了CausalEngine的核心消融实验，旨在通过严谨的对比实验验证CausalEngine相比传统神经网络的优势。

## 🔬 消融实验设计理念

### 核心假设
**CausalEngine的消融假设**：当CausalEngine仅使用位置输出（loc）时，其性能应该与传统神经网络相当。这是因为：
- 位置输出本质上是确定性映射：`loc = g(X)`
- 传统神经网络也是确定性映射：`Y = f(X)`
- 两者在数学上等价

### 三层对比框架
```
1. 传统神经网络 (Baseline)
   └── MLP: X → Y (直接映射)

2. CausalEngine消融版本 (Ablated)
   └── 仅位置输出: X → U_loc → Y

3. 完整CausalEngine (Full)
   └── 因果推理: X → U~Cauchy(loc,scale) → Y
```

## 📊 实验文件说明

### 核心实验脚本

#### `comprehensive_comparison.py`
**综合消融实验的主程序**
- 🎯 **目标**: 在所有8个数据集上运行完整的三模型对比
- 🔧 **功能**: 
  - 自动化批量实验
  - 多轮实验统计
  - 统计显著性检验
  - 自动生成报告和可视化
- 🚀 **使用方法**:
  ```bash
  # 运行所有数据集
  python tutorials/03_ablation_studies/comprehensive_comparison.py --datasets all
  
  # 运行指定数据集
  python tutorials/03_ablation_studies/comprehensive_comparison.py --datasets adult bike_sharing
  
  # 指定输出目录和实验轮数
  python tutorials/03_ablation_studies/comprehensive_comparison.py \
    --output_dir results/my_experiment \
    --num_runs 5 \
    --device cuda
  ```

#### `causal_vs_traditional.py`
**理论对比和概念演示**
- 📚 **目标**: 深入解释因果推理vs传统模式识别的区别
- 🧠 **内容**:
  - 数学公式对比
  - 可视化演示
  - 实际案例分析
  - 哲学层面讨论

#### `loc_only_vs_full_engine.py`
**专门的消融对比实验**
- 🔍 **目标**: 验证消融假设的核心脚本
- ⚗️ **特色**:
  - 详细的位置vs完整引擎对比
  - 温度参数效应分析
  - 不确定性量化评估
  - 推理模式切换演示

## 🎯 关键评估指标

### 分类任务指标
- **Accuracy**: 整体准确率
- **Precision**: 精确率（减少假正例）
- **Recall**: 召回率（减少假负例）
- **F1-Score**: 精确率和召回率的调和平均
- **AUC-ROC**: 受试者工作特征曲线下面积

### 回归任务指标
- **R²**: 决定系数（解释方差比例）
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **Calibration**: 不确定性校准质量

### 统计显著性
- **t-test**: 配对t检验
- **Wilcoxon**: 非参数检验
- **Effect Size**: 效应量计算
- **Confidence Intervals**: 置信区间

## 🔍 预期实验结果

### 消融假设验证
```
传统神经网络 ≈ CausalEngine(消融版本)
差异应该 < 1-2% (在统计误差范围内)
```

### 因果推理优势
```
CausalEngine(完整版本) > CausalEngine(消融版本)
提升幅度取决于数据集的因果结构复杂性
```

### 推理模式比较
```
因果模式 (T=0): 最稳定，最可解释
不确定性模式 (T>0): 更好的校准，适合决策
采样模式 (T>0, sample): 探索性分析，鲁棒性测试
```

## 📈 结果解读指南

### 成功指标
1. **消融验证成功**: 传统网络 ≈ 消融版本 (差异<2%)
2. **因果优势明显**: 完整版本 > 消融版本 (提升>5%)
3. **统计显著性**: p < 0.05 且效应量 > 0.2
4. **多数据集稳定**: 在70%+数据集上表现一致

### 异常情况处理
1. **消融假设失败**: 检查实现是否正确，调试网络架构
2. **传统方法更优**: 分析数据特性，可能缺乏因果结构
3. **结果不稳定**: 增加实验轮数，检查随机种子设置
4. **计算资源不足**: 使用更小的数据集或简化模型

## 🛠️ 实验环境配置

### 硬件要求
- **CPU**: 8核心+ (推荐)
- **内存**: 16GB+ RAM
- **GPU**: 可选，CUDA兼容 (加速训练)
- **存储**: 10GB+ 空闲空间

### 软件依赖
```bash
# 核心依赖
torch>=1.10.0
numpy>=1.20.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# 统计分析
scipy>=1.7.0
statsmodels>=0.12.0

# 可选加速
numba>=0.54.0
```

### 运行配置
```python
# 推荐配置
config = {
    'batch_size': 64,           # 平衡速度和内存
    'num_epochs': 100,          # 充分训练
    'early_stopping': 15,       # 防止过拟合
    'num_runs': 3,              # 统计可靠性
    'device': 'auto',           # 自动选择GPU/CPU
    'random_seed': 42           # 可复现性
}
```

## 📊 输出文件说明

### 实验结果
```
results/
├── raw_results.json           # 原始实验数据
├── summary_results.json       # 汇总统计结果
├── comprehensive_report.md    # 详细文字报告
└── visualizations/
    ├── overall_performance.png    # 总体性能对比
    ├── dataset_comparisons/       # 各数据集详细图表
    ├── significance_matrix.png    # 统计显著性热图
    └── ablation_analysis.png     # 消融分析可视化
```

### 报告内容
1. **执行摘要**: 核心发现和结论
2. **方法论**: 实验设计和参数设置
3. **结果详情**: 每个数据集的完整结果
4. **统计分析**: 显著性检验和效应量
5. **讨论分析**: 结果解释和局限性
6. **附录**: 原始数据和技术细节

## 🚀 快速开始

### 5分钟体验
```bash
# 运行两个代表性数据集的快速实验
python comprehensive_comparison.py --datasets adult bike_sharing --num_runs 1

# 查看结果
cat results/comprehensive_ablation/comprehensive_report.md
```

### 完整评估
```bash
# 运行所有数据集的完整实验 (约2-4小时)
python comprehensive_comparison.py --datasets all --num_runs 3

# 生成额外分析
python loc_only_vs_full_engine.py
python causal_vs_traditional.py
```

## ❓ 常见问题

### Q: 为什么需要消融实验？
A: 消融实验是验证CausalEngine核心价值主张的关键。它证明了性能提升来自因果推理机制，而非简单的架构改进。

### Q: 如何判断实验是否成功？
A: 主要看两点：1) 消融假设验证成功（传统≈消融），2) 因果推理有明显优势（完整>消融）。

### Q: 实验失败怎么办？
A: 首先检查实现正确性，然后分析数据特性。某些数据集可能本身缺乏因果结构，传统方法表现更好是正常的。

### Q: 如何加速实验？
A: 可以减少数据集数量、降低训练轮数，或使用GPU加速。但要注意保持统计可靠性。

---

🎯 **目标**: 通过严谨的消融实验，为CausalEngine的革命性价值提供实证支持！