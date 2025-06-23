# CausalEngine Sklearn-Style API 设计方案 V2

> **目标**: 将CausalEngine打包成类似sklearn神经网络模块那样易用的包，提供统一的API接口和智能默认配置，让用户能够轻松进行 `fit()`, `predict()`, `transform()` 等操作。

## 1. 灵感来源：sklearn神经网络模块分析

### 1.1 sklearn MLPRegressor/MLPClassifier 的成功之处

```python
# sklearn神经网络的经典用法
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 分类任务
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# 回归任务  
reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

### 1.2 sklearn设计的核心优势

- ✅ **统一API**: fit/predict/score 三步走
- ✅ **智能默认**: 大多数参数有合理默认值
- ✅ **任务分离**: Regressor vs Classifier 清晰分工
- ✅ **标准化集成**: 与preprocessing, model_selection无缝配合
- ✅ **丰富属性**: 训练后可查看权重、损失历史等
- ✅ **错误处理**: 友好的错误信息和警告

## 2. CausalEngine 当前架构分析

### 2.1 现有组件结构
```python
# 当前的使用方式（相对复杂）
from causal_engine import CausalEngine, AbductionNetwork, ActionNetwork

# 需要用户手动配置很多参数
abduction_net = AbductionNetwork(
    input_size=X.shape[1], 
    causal_size=64
)
action_net = ActionNetwork(
    causal_size=64,
    output_size=1
)

engine = CausalEngine(
    hidden_size=X.shape[1],
    vocab_size=1,
    causal_size=64
)

# 训练过程需要手动管理
engine.train()
for epoch in range(num_epochs):
    # 手动训练循环...
```

### 2.2 用户痛点
- 🚫 需要手动构建网络结构
- 🚫 需要了解内部架构细节 
- 🚫 缺乏统一的训练接口
- 🚫 参数配置复杂
- 🚫 没有标准的预测接口

## 3. 设计目标：理想的CausalEngine API

### 3.1 目标使用体验

```python
# 理想的使用方式 - 简单如sklearn
from causal_engine.sklearn import MLPCausalRegressor

# 回归任务 - 3行代码搞定
reg = MLPCausalRegressor()  # 智能默认配置
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# 高级用法 - 仍然简洁
reg = MLPCausalRegressor(
    hidden_layer_sizes=(64, 32),  # 网络结构（与sklearn兼容）
    max_iter=1000,          # 训练轮数
    inference_mode='standard', # 推理模式
    random_state=42         # 随机种子
)
```

### 3.2 与sklearn完全兼容

```python
# 与sklearn生态无缝集成
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 交叉验证
scores = cross_val_score(MLPCausalRegressor(), X, y, cv=5)

# 网格搜索
param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (64, 32)],
    'inference_mode': ['standard', 'causal']
}
grid_search = GridSearchCV(MLPCausalRegressor(), param_grid, cv=3)

# 管道集成
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('causal', MLPCausalRegressor())
])
```

## 4. 核心设计差异：MLPCausalRegressor vs MLPRegressor 🧮

### 4.1 设计哲学：仅替换输出层 ✨

**核心洞察**: MLPCausalRegressor 和 MLPRegressor 的**唯一区别**就是最后一个输出层！
- **MLPRegressor**: 线性输出层 `y = W·h + b`
- **MLPCausalRegressor**: CausalEngine输出层（归因→行动→激活）

这种设计的优雅之处：
- ✅ **最小化改动**: 保持sklearn的所有优秀特性
- ✅ **最大化收益**: 获得完整的因果推理能力
- ✅ **无缝替换**: 可以直接替代MLPRegressor使用

### 4.2 网络结构对比

```python
# 传统MLPRegressor架构
输入层 → 隐藏层们 → 线性输出层 → 确定性预测值
  X    →   MLPs   →  y = W·h + b  →    ŷ

# MLPCausalRegressor架构（仅最后一层不同！）  
输入层 → 隐藏层们 → CausalEngine → 分布输出 → 概率预测
  X    →   MLPs   → (归因+行动+激活) → S~Cauchy → P(Y)
```

**关键优势**: 
- 🚀 **训练效率**: 大部分网络结构完全相同，训练复杂度相当
- 🚀 **参数规模**: 仅CausalEngine部分增加少量参数
- 🚀 **收益巨大**: 从确定性预测升级到分布建模和因果推理

### 4.3 CausalEngine的独特Forward Pass

#### 第1阶段：归因推断 (Abduction)
```python
# 输入: 特征向量 h (来自前面的MLP层)
μ_U = loc_net(h)           # 个体中心位置
γ_U = softplus(scale_net(h))  # 个体群体多样性
# 输出: U ~ Cauchy(μ_U, γ_U) - 个体选择变量分布
```

#### 第2阶段：行动决策 (Action)
```python
# 外生噪声注入
U' = U + b_noise · ε  # ε ~ Cauchy(0,1)
# U' ~ Cauchy(μ_U, γ_U + |b_noise|)

# 线性因果变换
S = W_A @ U' + b_A
# S ~ Cauchy(loc_S, scale_S) - 决策得分分布
```

#### 第3阶段：回归激活 (Activation)
```python
# 回归激活：线性变换输出分布
Y ~ Cauchy(w_k·loc_S_k + b_k, |w_k|·scale_S_k)
# 预测：分布的期望值 E[Y] = μ_Y
```

### 4.4 损失函数的根本差异

#### 传统回归：均方误差
```python
# 预测确定值
y_pred = W @ h + b
loss = MSE(y_true, y_pred)
```

#### CausalEngine回归：柯西分布似然
```python
# 预测分布参数
Y ~ Cauchy(w_k·loc_S_k + b_k, |w_k|·scale_S_k)
# 柯西分布负对数似然损失
loss = log(π·γ_Y) + log(1 + ((y_true - μ_Y)/γ_Y)²)
```

## 5. API接口设计 - V1.0 专注回归

### 5.1 MLPCausalRegressor 核心接口

```python
class MLPCausalRegressor(BaseEstimator, RegressorMixin):
    """MLP因果回归器 - sklearn风格接口"""
    
    def __init__(self, 
                 hidden_layer_sizes=(64, 32),    # 网络结构（与sklearn兼容）
                 max_iter=1000,                  # 最大迭代次数
                 learning_rate=0.001,            # 学习率
                 inference_mode='standard',      # 推理模式
                 early_stopping=True,            # 早停
                 validation_fraction=0.1,        # 验证集比例
                 random_state=None,              # 随机种子
                 verbose=False):                 # 训练日志
        pass
    
    def fit(self, X, y, sample_weight=None):
        """训练模型"""
        # 1. 自动数据预处理和验证
        # 2. 自动构建MLP特征提取层
        # 3. 自动构建CausalEngine输出层
        # 4. 自动训练循环（含early stopping）
        return self
    
    def predict(self, X):
        """预测"""
        # 自动推理，返回分布期望值
        return predictions
    
    def score(self, X, y, sample_weight=None):
        """评分 (R²)"""
        return r2_score(y, self.predict(X))
    
    # sklearn标准属性
    @property
    def feature_importances_(self):
        """特征重要性"""
        pass
    
    @property
    def loss_curve_(self):
        """训练损失曲线"""
        pass
```

### 5.2 智能默认配置策略

```python
# 根据数据规模自动调整网络结构
def _auto_hidden_layer_sizes(n_features, n_samples):
    """根据特征数和样本数智能推荐网络结构"""
    if n_features <= 10:
        return (32,)
    elif n_features <= 50:
        return (64, 32)
    elif n_features <= 100:
        return (128, 64)
    else:
        return (256, 128, 64)

# 自动早停和学习率调整
AUTO_CONFIG = {
    'early_stopping': True,
    'patience': 20,
    'min_delta': 1e-4,
    'learning_rate_schedule': 'adaptive'
}
```

### 5.3 推理模式设计原则

**核心原则**: 训练和推理都默认使用 `standard` 模式，保持简洁统一。

```python
# 默认使用 - 适合99%的使用场景
reg = MLPCausalRegressor()  # inference_mode='standard'

# 实验对比 - 看看causal模式是否能带来提升
reg_causal = MLPCausalRegressor(inference_mode='causal')

# 性能对比
standard_score = cross_val_score(MLPCausalRegressor(), X, y, cv=5)
causal_score = cross_val_score(MLPCausalRegressor(inference_mode='causal'), X, y, cv=5)
```

**推理模式定位**:
- **`standard`**: 默认模式，训练和推理的标准选择，适合常规回归任务
- **`causal`**: 实验性选项，用于探索是否能在特定数据集上获得更好效果

## 6. 实现路线图

### 6.1 V1.0 当前版本：MLPCausalRegressor 核心实现
**重点**：专注回归任务，打造完整可用的sklearn风格接口

- [ ] 创建 `causal_engine.sklearn` 子模块
- [ ] 实现 `MLPCausalRegressor` 基础类
- [ ] 集成现有CausalEngine核心功能（AbductionNetwork + ActionNetwork + ActivationHead）
- [ ] 实现自动训练循环和标准sklearn接口
- [ ] 基础参数验证和错误处理
- [ ] 简单的使用示例和文档

### 6.2 V1.1 优化增强
- [ ] 实现自动网络结构推荐
- [ ] 添加early stopping和validation
- [ ] 提供causal推理模式作为standard模式的对比选项
- [ ] 完善错误处理和警告
- [ ] sklearn兼容性测试

### 6.3 V2.0 扩展功能
- [ ] 实现 `MLPCausalClassifier` 分类接口
- [ ] 添加特征重要性分析
- [ ] 训练过程可视化
- [ ] **实验性**: 探索训练阶段不同推理模式的效果差异

### 6.4 V3.0 生态集成
- [ ] 与pandas DataFrame深度集成
- [ ] 集成模型解释工具
- [ ] 性能优化和大规模数据支持

## 7. 使用场景对比

### 7.1 现在 vs 未来

| 场景 | 现在的方式 | 理想的方式 |
|------|-----------|-----------|
| **快速原型** | 20+行代码，需了解架构 | 3行代码，零配置 |
| **参数调优** | 手动试验各种组合 | GridSearchCV自动搜索 |
| **模型评估** | 手写评估代码 | cross_val_score一行搞定 |
| **生产部署** | 需要自己处理序列化 | joblib直接保存加载 |
| **特征分析** | 需要自己实现 | feature_importances_属性 |

### 7.2 竞争对比

```python
# XGBoost风格
import xgboost as xgb
reg = xgb.XGBRegressor()
reg.fit(X_train, y_train)

# LightGBM风格  
import lightgbm as lgb
reg = lgb.LGBMRegressor()
reg.fit(X_train, y_train)

# CausalEngine风格 (目标)
from causal_engine.sklearn import MLPCausalRegressor
reg = MLPCausalRegressor()  # 同样简洁，但是因果推理！
reg.fit(X_train, y_train)
```

## 8. 技术实现要点

### 8.1 关键挑战
1. **分布损失计算**: 实现柯西分布的高效似然计算
2. **参数映射**: sklearn参数 → CausalEngine内部参数
3. **训练循环**: 封装复杂的因果推理训练逻辑
4. **状态管理**: 模型训练状态的保存和恢复
5. **错误处理**: 友好的错误信息
6. **性能优化**: 保持解析计算的性能优势

### 8.2 架构设计 - V1.0 简化版本

```python
# V1.0 内部架构（专注回归）
causal_engine/sklearn/
├── __init__.py       # 导出MLPCausalRegressor
├── regressor.py      # MLPCausalRegressor核心实现
├── _base.py          # 基础工具函数和验证
└── _config.py        # 默认配置和自动推荐

# V2.0+ 扩展架构
causal_engine/sklearn/
├── __init__.py       # 导出所有接口
├── regressor.py      # MLPCausalRegressor实现
├── classifier.py     # MLPCausalClassifier实现（V2.0+）
├── _base.py          # 基础类和接口
├── _utils.py         # 工具函数
├── _validation.py    # 参数验证
└── _config.py        # 默认配置
```

## 9. V1.0 开发重点确认 ✅

### 9.1 当前版本明确目标
- **专注任务**: 回归任务 (`MLPCausalRegressor`)
- **核心功能**: 实现完整的sklearn风格接口
- **设计原则**: 简单、实用、可扩展

### 9.2 V1.0 最小可行产品（MVP）
```python
# V1.0 目标体验
from causal_engine.sklearn import MLPCausalRegressor

# 零配置使用
reg = MLPCausalRegressor()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# sklearn兼容性
from sklearn.model_selection import cross_val_score
scores = cross_val_score(MLPCausalRegressor(), X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### 9.3 后续版本规划
- **V1.1**: 优化和增强功能
- **V2.0**: 添加 `MLPCausalClassifier` 分类支持
- **V3.0**: 生态集成和高级功能

### 9.4 开发策略
- 🎯 **渐进式开发**: 先做好回归，再扩展分类
- 🎯 **用户驱动**: 基于实际使用反馈进行迭代
- 🎯 **质量优先**: 确保每个版本都是完整可用的

---

**💡 这个方案的价值**:
- 大幅降低使用门槛，让更多人能用上因果推理
- 与现有ML工作流无缝集成
- 保持CausalEngine的技术优势，包装成用户友好的接口
- 为CausalEngine的广泛采用奠定基础

**🎯 期待反馈**:
请提供您对这个设计方案的具体想法、需求和建议！