# CausalEngine 快速测试指南

## 🚀 概述

这个工具包提供了简单灵活的端对端测试脚本，帮你快速验证CausalEngine的效果。主要包含：

- `quick_test_causal_engine.py` - 核心测试引擎
- `test_scenarios.py` - 预定义测试场景  
- 本指南 - 使用说明

## 📦 核心功能

### 支持的对比方法
- **sklearn**: MLPRegressor/MLPClassifier基线
- **pytorch**: 纯PyTorch神经网络基线
- **deterministic**: CausalEngine确定性模式 (等价sklearn)
- **standard**: CausalEngine标准因果模式

### 支持的评估指标
- **回归**: MAE, MdAE, RMSE, R²
- **分类**: Accuracy, Precision, Recall, F1

## 🛠️ 快速使用

### 1. 基础测试

```python
from quick_test_causal_engine import quick_regression_test, quick_classification_test

# 回归测试
quick_regression_test(
    n_samples=1000,           # 样本数
    n_features=10,            # 特征数
    hidden_layer_sizes=(64, 32),  # 网络结构
    gamma_init=10.0,          # γ_U初始化
    b_noise_init=0.1,         # 外生噪声初始化
    max_iter=800              # 训练轮数
)

# 分类测试
quick_classification_test(
    n_samples=1000,
    n_features=10,
    n_classes=3,              # 类别数
    hidden_layer_sizes=(64, 32),
    gamma_init=10.0,
    b_noise_init=0.1,
    ovr_threshold_init=0.0,   # OvR阈值初始化
    max_iter=800
)
```

### 2. 噪声鲁棒性测试

```python
# 回归 - 高斯标签噪声
quick_regression_test(
    n_samples=800,
    label_noise_ratio=0.2,    # 20%噪声
    label_noise_type='gaussian',
    gamma_init=15.0,          # 更大的初始尺度应对噪声
    b_noise_init=0.2
)

# 分类 - 标签翻转噪声
quick_classification_test(
    n_samples=800,
    n_classes=3,
    label_noise_ratio=0.15,   # 15%标签翻转
    label_noise_type='flip',
    gamma_init=15.0,
    b_noise_init=0.2
)
```

### 3. 使用预定义场景

```python
from test_scenarios import *

# 运行单个场景
scenario_clean_data()         # 干净数据基线测试
scenario_label_noise()       # 标签噪声鲁棒性测试
scenario_parameter_sensitivity()  # 参数敏感性分析
scenario_extreme_noise()     # 极端噪声环境测试
```

## 🎛️ 关键参数说明

### 数据生成参数
```python
n_samples=1000,              # 样本数量
n_features=10,               # 特征维度
noise=0.1,                   # 数据噪声水平 (回归)
n_classes=3,                 # 类别数 (分类)
class_sep=1.0,               # 类别分离度 (分类)
random_state=42,             # 随机种子
```

### 标签噪声参数
```python
label_noise_ratio=0.1,       # 噪声比例 (0.0-0.5)
label_noise_type='gaussian', # 'gaussian'(回归) 或 'flip'(分类)
```

### 网络结构参数
```python
hidden_layer_sizes=(64, 32), # 隐藏层结构
causal_size=None,            # 因果表征维度 (None=自动推断)
```

### CausalEngine核心参数
```python
gamma_init=10.0,             # γ_U初始化值 (建议5.0-20.0)
b_noise_init=0.1,            # 外生噪声初始值 (建议0.01-0.5)
ovr_threshold_init=0.0,      # OvR阈值初始化 (分类专用)
b_noise_trainable=True,      # 外生噪声是否可训练
```

### 训练参数
```python
max_iter=1000,               # 最大训练轮数
learning_rate=0.001,         # 学习率
early_stopping=True,         # 是否早停
```

## 📊 结果解读

### 期望的结果模式

#### 1. 干净数据环境
- **deterministic** ≈ **sklearn** (数学等价性验证)
- **pytorch** 通常表现最好 (更灵活的训练)
- **standard** 略低于deterministic (噪声建模的开销)

#### 2. 噪声数据环境
- **CausalEngine**(deterministic/standard) > **传统方法**(sklearn/pytorch)
- **standard** 可能优于 **deterministic** (噪声适应能力)
- 噪声越大，CausalEngine优势越明显

#### 3. 参数敏感性
- `gamma_init`太小(<1.0): 可能数值不稳定
- `gamma_init`太大(>50.0): 可能过度平滑
- `b_noise_init`适中(0.1-0.5): 平衡性能和鲁棒性

## 🔧 调试技巧

### 1. 性能异常时检查
```python
# 降低复杂度快速测试
quick_regression_test(
    n_samples=200,      # 小数据集
    n_features=5,       # 少特征
    hidden_layer_sizes=(32,),  # 简单网络
    max_iter=100,       # 短训练
    verbose=True        # 详细输出
)
```

### 2. 参数网格搜索
```python
# 手动网格搜索gamma_init
for gamma in [1.0, 5.0, 10.0, 20.0]:
    print(f"\\n🔧 Testing gamma_init={gamma}")
    results = quick_regression_test(
        gamma_init=gamma,
        verbose=False
    )
    # 分析results...
```

### 3. 噪声阈值测试
```python
# 寻找噪声破坏点
for noise_ratio in [0.0, 0.1, 0.2, 0.3, 0.5]:
    print(f"\\n🔊 Testing noise_ratio={noise_ratio}")
    results = quick_classification_test(
        label_noise_ratio=noise_ratio,
        verbose=False
    )
    # 观察性能下降趋势...
```

## 🎯 常见使用场景

### 算法开发者
```python
# 验证新初始化策略
quick_regression_test(gamma_init=5.0, b_noise_init=0.05)
quick_regression_test(gamma_init=15.0, b_noise_init=0.3)

# 测试网络结构影响
quick_regression_test(hidden_layer_sizes=(32,))
quick_regression_test(hidden_layer_sizes=(128, 64, 32))
```

### 应用研究者
```python
# 验证特定数据特征下的性能
quick_classification_test(
    n_features=20,      # 高维特征
    n_classes=10,       # 多分类
    class_sep=0.5,      # 困难分类
    label_noise_ratio=0.2  # 真实噪声环境
)
```

### 基准测试
```python
# 运行完整测试套件
from test_scenarios import *
scenario_clean_data()
scenario_label_noise() 
scenario_extreme_noise()
```

## ⚡ 性能优化建议

1. **小数据快速验证**: n_samples=200-500, max_iter=200-400
2. **大规模测试**: 使用GPU版本，增加batch_size
3. **并行测试**: 可以同时运行多个不同参数的测试
4. **结果缓存**: 保存关键实验结果避免重复计算

## 🐛 常见问题

### Q: CausalEngine性能异常低？
A: 检查gamma_init是否合理(5.0-20.0)，网络结构是否匹配数据复杂度

### Q: deterministic模式与sklearn差异大？
A: 正常现象，检查数学等价性验证脚本`demo_scientific_equivalence_validation.py`

### Q: 如何自定义数据集？
A: 修改`make_regression`/`make_classification`调用，或直接传入自己的X,y

### Q: 如何保存实验结果？
A: 使用tester.results访问详细结果，可以序列化保存

---

**💡 提示**: 这个工具设计为交互式探索，建议在Jupyter notebook中使用以便更好地可视化和分析结果！