# 🌍 真实世界基准测试

本目录包含对 CausalEngine 在真实数据集上的综合基准测试，展示其在各种实际应用场景中的性能表现。

## 📊 基准测试脚本

### 🔍 分类任务基准测试
**文件**: `classification_benchmark.py`

测试 CausalEngine 在 4 个真实分类数据集上的性能：

1. **Adult Census Income** - 收入预测 (>50K / ≤50K)
2. **Bank Marketing** - 银行营销响应预测
3. **Credit Default** - 信用卡违约预测
4. **Mushroom** - 蘑菇安全分类 (可食用/有毒)

**使用方法**:
```bash
cd user_tutorials
python 04_real_world_examples/classification_benchmark.py
```

**评估指标**:
- Accuracy (准确率)
- Precision (精确率) 
- Recall (召回率)
- F1-Score (F1分数)

### 📈 回归任务基准测试
**文件**: `regression_benchmark.py`

测试 CausalEngine 在 4 个真实回归数据集上的性能：

1. **Bike Sharing** - 共享单车需求预测
2. **Wine Quality** - 葡萄酒质量评分预测
3. **Ames Housing** - 房价预测
4. **California Housing** - 加州房价中位数预测

**使用方法**:
```bash
cd user_tutorials
python 04_real_world_examples/regression_benchmark.py
```

**评估指标**:
- MAE (平均绝对误差)
- RMSE (均方根误差)
- MdAE (中位数绝对误差)
- MSE (均方误差)
- R² (决定系数)

## 🏆 对比基准模型

每个基准测试都将 CausalEngine 与以下 5 种传统机器学习方法进行对比：

1. **Random Forest** - 随机森林
2. **Gradient Boosting** - 梯度提升
3. **SVM** - 支持向量机
4. **Logistic/Linear Regression** - 逻辑/线性回归
5. **Neural Network** - 神经网络

## 📊 输出结果

### 控制台输出
- 数据集加载状态
- 训练进度显示
- 详细性能对比表格
- CausalEngine 性能分析
- 总体表现总结

### 文件输出
- **PNG 图表**: `user_tutorials/results/classification_benchmark.png` / `regression_benchmark.png`
- **CSV 数据**: `user_tutorials/results/classification_benchmark.csv` / `regression_benchmark.csv`

## 🎯 示例输出

### 分类结果示例
```
📊 Adult Census Income - 分类结果:
   模型                  | 准确率   | 精确率   | 召回率   | F1分数   | 训练时间
   -------------------- | -------- | -------- | -------- | -------- | --------
🏆 CausalEngine          | 0.8547   | 0.8423   | 0.8234   | 0.8327   | 12.3s
   Random Forest        | 0.8498   | 0.8456   | 0.8198   | 0.8325   | 3.2s
   Gradient Boosting    | 0.8521   | 0.8467   | 0.8156   | 0.8309   | 8.7s
```

### 回归结果示例
```
📊 Bike Sharing - 回归结果:
   模型                  | MAE      | RMSE     | MdAE     | R²       | 训练时间
   -------------------- | -------- | -------- | -------- | -------- | --------
🏆 CausalEngine          |   45.234 |   78.567 |   32.123 |   0.8745 | 15.6s
   Random Forest        |   48.567 |   82.345 |   35.678 |   0.8634 | 4.1s
   Linear Regression    |   52.234 |   89.456 |   41.234 |   0.8456 | 0.8s
```

## 🔍 性能分析

每次运行后会自动生成：

### 🎯 总体表现分析
- 胜率统计 (CausalEngine vs 最佳基准)
- 平均性能提升百分比
- 训练时间对比
- 性能等级评定

### 📈 详细对比
- 每个数据集的详细结果
- 相对改进程度
- 训练效率分析
- 可视化图表

## 💡 使用建议

### 运行环境
确保已安装必要依赖：
```bash
pip install torch scikit-learn matplotlib pandas seaborn numpy
```

### 数据准备
- 脚本会自动尝试加载真实数据集
- 如果真实数据不可用，会生成高质量的模拟数据
- Adult Census Income 需要 `data/adult_train.data` 和 `data/adult_test.test`
- Bike Sharing 需要 `data/hour.csv`

### 结果解读
- 🏆 标记表示在该数据集上的最佳模型
- 关注 F1-Score (分类) 和 R² (回归) 作为主要性能指标
- 训练时间仅供参考，实际时间取决于硬件配置

## 🚀 快速开始

```bash
# 进入用户教程目录
cd user_tutorials

# 运行分类基准测试
python 04_real_world_examples/classification_benchmark.py

# 运行回归基准测试  
python 04_real_world_examples/regression_benchmark.py

# 查看结果
ls results/
# classification_benchmark.png  regression_benchmark.png
# classification_benchmark.csv  regression_benchmark.csv
```

## 🔗 相关教程

- **基础入门**: `01_quick_start/first_example.py`
- **分类教程**: `02_classification/`
- **回归教程**: `03_regression/`
- **工具函数**: `utils/simple_models.py`

---

通过这些基准测试，您可以客观评估 CausalEngine 在真实场景中的表现，为您的项目选择提供数据支持！