# CausalEngine Sklearn接口使用教程

> **目标读者**: 数据科学家、机器学习工程师、研究人员  
> **前置知识**: 熟悉sklearn基础用法  
> **预计用时**: 30分钟

## 📖 教程概述

本教程将带您学习如何使用CausalEngine的sklearn风格接口进行机器学习任务。CausalEngine提供了两个核心估计器：`MLPCausalRegressor`和`MLPCausalClassifier`，它们可以直接替代sklearn的`MLPRegressor`和`MLPClassifier`，同时提供更强的噪声鲁棒性和因果推理能力。

### 🎯 核心优势

1. **sklearn完全兼容**: 无需修改现有代码，直接替换即可
2. **噪声鲁棒性**: 对标签噪声具有天然免疫力
3. **统一预测接口**: 从简单点估计到完整概率分布
4. **数学创新**: 基于Cauchy分布和OvR策略的因果推理

## 🚀 快速开始

### 安装和导入

```python
# 导入CausalEngine sklearn接口
from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier

# 导入sklearn工具（完全兼容）
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import make_regression, make_classification
```

### 基础回归示例

```python
import numpy as np

# 生成回归数据
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用CausalEngine回归器（用法与sklearn相同）
reg = MLPCausalRegressor(
    hidden_layer_sizes=(64, 32),    # MLP结构
    max_iter=500,                   # 训练轮数
    random_state=42                 # 随机种子
)

# 训练模型
reg.fit(X_train, y_train)

# 预测（sklearn兼容模式）
predictions = reg.predict(X_test)
r2 = r2_score(y_test, predictions)

print(f"R² Score: {r2:.4f}")
```

### 基础分类示例

```python
# 生成分类数据
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                          n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用CausalEngine分类器
clf = MLPCausalClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)

# 训练模型
clf.fit(X_train, y_train)

# 预测类别
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 预测概率
probabilities = clf.predict_proba(X_test)

print(f"Accuracy: {accuracy:.4f}")
print(f"Probability shape: {probabilities.shape}")
```

## 🎛️ 高级功能：多模式预测

CausalEngine的独特之处在于统一的`predict()`接口支持多种预测模式：

### 回归器的四种模式

```python
# 训练模型
reg = MLPCausalRegressor()
reg.fit(X_train, y_train)

# 1. Compatible模式：sklearn兼容，返回点估计
compatible_pred = reg.predict(X_test, mode='compatible')
print(f"Compatible: {compatible_pred[:3]}")

# 2. Standard模式：返回完整分布信息
standard_pred = reg.predict(X_test, mode='standard')
print(f"Standard keys: {list(standard_pred.keys())}")
print(f"Predictions: {standard_pred['predictions'][:3]}")

# 3. Causal模式：纯因果推理（无外生噪声）
causal_pred = reg.predict(X_test, mode='causal')

# 4. Sampling模式：探索性预测（增大不确定性）
sampling_pred = reg.predict(X_test, mode='sampling')
```

### 分类器的双概率策略

```python
# 训练模型
clf = MLPCausalClassifier()
clf.fit(X_train, y_train)

# Softmax兼容概率（严格归一化）
softmax_proba = clf.predict_proba(X_test, mode='compatible')
print(f"Softmax概率和: {softmax_proba[0].sum():.6f}")  # = 1.000000

# OvR原生概率（独立激活）
ovr_proba = clf.predict_proba(X_test, mode='standard')
print(f"OvR概率和: {ovr_proba[0].sum():.6f}")  # ≈ 1.0 但不严格

# 高级预测模式
advanced_pred = clf.predict(X_test, mode='standard')
print(f"Predicted classes: {advanced_pred['predictions'][:5]}")
print(f"Probability shape: {advanced_pred['probabilities'].shape}")
```

## 🛡️ 噪声鲁棒性演示

CausalEngine的核心优势是对标签噪声的天然鲁棒性：

### 分类任务噪声鲁棒性

```python
def add_label_noise(y, noise_level):
    """添加随机标签翻转噪声"""
    y_noisy = y.copy()
    n_noise = int(len(y) * noise_level)
    noise_indices = np.random.choice(len(y), n_noise, replace=False)
    
    for idx in noise_indices:
        available_labels = [l for l in np.unique(y) if l != y[idx]]
        y_noisy[idx] = np.random.choice(available_labels)
    
    return y_noisy

# 生成干净数据
X, y_clean = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size=0.2, random_state=42)

# 添加20%标签噪声
y_train_noisy = add_label_noise(y_train, noise_level=0.2)

# 对比传统方法 vs CausalEngine
from sklearn.neural_network import MLPClassifier

# 传统方法在噪声数据上训练
traditional_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
traditional_clf.fit(X_train, y_train_noisy)
traditional_acc = accuracy_score(y_test, traditional_clf.predict(X_test))

# CausalEngine在噪声数据上训练
causal_clf = MLPCausalClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
causal_clf.fit(X_train, y_train_noisy)
causal_acc = accuracy_score(y_test, causal_clf.predict(X_test))

print(f"传统方法准确率: {traditional_acc:.4f}")
print(f"CausalEngine准确率: {causal_acc:.4f}")
print(f"噪声鲁棒性优势: +{(causal_acc - traditional_acc)*100:.1f}%")
```

### 回归任务噪声鲁棒性

```python
def add_magnitude_noise(y, magnitude_factor=10):
    """添加数量级错误噪声"""
    y_noisy = y.copy()
    n_errors = int(0.1 * len(y))  # 10%的数据有错误
    error_indices = np.random.choice(len(y), n_errors, replace=False)
    y_noisy[error_indices] *= magnitude_factor  # 数量级错误
    return y_noisy

# 生成回归数据并添加噪声
X, y_clean = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size=0.2, random_state=42)

y_train_noisy = add_magnitude_noise(y_train, magnitude_factor=10)

# 对比实验
from sklearn.neural_network import MLPRegressor

traditional_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
traditional_reg.fit(X_train, y_train_noisy)
traditional_r2 = r2_score(y_test, traditional_reg.predict(X_test))

causal_reg = MLPCausalRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
causal_reg.fit(X_train, y_train_noisy)
causal_r2 = r2_score(y_test, causal_reg.predict(X_test))

print(f"传统方法R²: {traditional_r2:.4f}")
print(f"CausalEngine R²: {causal_r2:.4f}")
print(f"性能保持: {causal_r2/traditional_r2:.2f}x")
```

## 🔧 参数配置指南

### 回归器参数

```python
reg = MLPCausalRegressor(
    # MLP结构（与sklearn兼容）
    hidden_layer_sizes=(64, 32),       # 隐藏层结构
    
    # 训练参数
    max_iter=1000,                     # 最大迭代次数
    learning_rate=0.001,               # 学习率
    
    # CausalEngine特有参数
    default_mode='compatible',         # 默认预测模式
    causal_size=None,                  # 因果表征维度（默认自动）
    
    # 早停参数
    early_stopping=True,               # 启用早停
    validation_fraction=0.1,           # 验证集比例
    n_iter_no_change=20,              # 早停耐心值
    tol=1e-4,                         # 早停容忍度
    
    # 其他
    random_state=42,                   # 随机种子
    verbose=False                      # 训练日志
)
```

### 分类器参数

```python
clf = MLPCausalClassifier(
    # 基础参数（与回归器相同）
    hidden_layer_sizes=(64, 32),
    max_iter=1000,
    learning_rate=0.001,
    default_mode='compatible',
    causal_size=None,
    
    # 分类特有的早停参数
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-4,
    
    random_state=42,
    verbose=False
)
```

## 📊 性能监控

### 训练过程监控

```python
# 启用详细日志
reg = MLPCausalRegressor(verbose=True, max_iter=1000)
reg.fit(X_train, y_train)

# 查看损失曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(reg.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Cauchy NLL Loss')
plt.grid(True)
plt.show()

print(f"最终训练损失: {reg.loss_curve_[-1]:.4f}")
print(f"训练轮数: {len(reg.loss_curve_)}")
```

### 模型属性检查

```python
# 检查模型状态
print(f"是否已训练: {reg.is_fitted_}")
print(f"输入特征数: {reg.n_features_in_}")
print(f"输出维度: {reg.n_outputs_}")

# 特征重要性（简单实现）
feature_importance = reg.feature_importances_
print(f"特征重要性: {feature_importance}")

# 模型组件
print(f"隐藏层结构: {reg.hidden_layer_sizes}")
print(f"因果表征维度: {reg.causal_size}")
```

## 🔗 与sklearn生态集成

### 交叉验证

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 回归交叉验证
reg_scores = cross_val_score(
    MLPCausalRegressor(max_iter=200), 
    X, y, 
    cv=5, 
    scoring='r2'
)
print(f"回归CV R²: {reg_scores.mean():.4f} ± {reg_scores.std():.4f}")

# 分类交叉验证
clf_scores = cross_val_score(
    MLPCausalClassifier(max_iter=200), 
    X, y, 
    cv=StratifiedKFold(5), 
    scoring='accuracy'
)
print(f"分类CV准确率: {clf_scores.mean():.4f} ± {clf_scores.std():.4f}")
```

### 网格搜索

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (64, 32)],
    'learning_rate': [0.001, 0.01],
    'default_mode': ['compatible', 'standard']
}

# 网格搜索
grid_search = GridSearchCV(
    MLPCausalRegressor(max_iter=200),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.4f}")
```

### Pipeline集成

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 构建pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', MLPCausalRegressor(max_iter=200))
])

# 使用pipeline
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)

print(f"Pipeline R²: {score:.4f}")
```

## ⚠️ 注意事项和最佳实践

### 数据预处理

```python
# 推荐的数据预处理pipeline
from sklearn.preprocessing import StandardScaler

# 特征标准化（推荐）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CausalEngine对噪声鲁棒，但标准化仍有帮助
reg = MLPCausalRegressor()
reg.fit(X_train_scaled, y_train)
```

### 超参数建议

```python
# 根据数据规模调整网络结构
def get_recommended_structure(n_features, n_samples):
    """根据数据规模推荐网络结构"""
    if n_features <= 10:
        return (32,)
    elif n_features <= 50:
        return (64, 32)
    elif n_features <= 100:
        return (128, 64)
    else:
        return (256, 128, 64)

# 根据任务类型调整学习率
def get_recommended_lr(task_type, data_size):
    """根据任务类型推荐学习率"""
    base_lr = 0.001
    if task_type == 'classification':
        return base_lr * 0.5  # 分类任务稍低
    elif data_size < 1000:
        return base_lr * 2    # 小数据集稍高
    return base_lr
```

### 性能优化建议

1. **GPU加速**: CausalEngine自动使用可用的GPU
2. **早停策略**: 默认启用，可避免过拟合
3. **批量大小**: 自动处理，无需手动设置
4. **内存优化**: 大数据集时考虑分批训练

### 常见问题解决

**Q: 训练很慢怎么办？**
```python
# 减少网络大小或训练轮数
reg = MLPCausalRegressor(
    hidden_layer_sizes=(32,),  # 更小的网络
    max_iter=200,              # 更少的轮数
    early_stopping=True        # 确保早停启用
)
```

**Q: 如何处理不平衡数据？**
```python
# 分类任务：stratified采样已内置
# 特殊情况下可以使用class_weight（未来版本支持）
from sklearn.utils.class_weight import compute_class_weight

# 当前版本建议预处理数据平衡
```

**Q: 如何解释模型预测？**
```python
# 使用高级预测模式获取更多信息
advanced_pred = reg.predict(X_test, mode='standard')
distributions = advanced_pred['distributions']

# 检查不确定性
if 'scale_S' in distributions:
    uncertainty = distributions['scale_S']
    print(f"预测不确定性: {uncertainty.mean():.4f}")
```

## 🎓 总结

通过本教程，您学会了：

1. ✅ **基础使用**: 如何用CausalEngine替代sklearn估计器
2. ✅ **高级功能**: 多模式预测和双概率策略
3. ✅ **噪声鲁棒性**: 验证和利用抗噪声能力
4. ✅ **生态集成**: 与sklearn工具链无缝配合
5. ✅ **最佳实践**: 参数调优和性能优化

CausalEngine sklearn接口为您提供了一个强大而易用的机器学习工具，既保持了sklearn的简洁性，又引入了前沿的因果推理能力。开始在您的项目中使用吧！

## 📚 延伸阅读

- [CausalEngine数学基础文档](../causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md)
- [sklearn风格API设计文档](./sklearn_style_api_regressor_v1.md)
- [实验报告](./sklearn_interface_experiment_report.md)

---

*最后更新: 2024年6月*