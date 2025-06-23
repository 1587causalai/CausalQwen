# CausalEngine Sklearn-Style API 设计方案

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



我们 MLPCausalRegressor 和 MLPRegressor 唯一的不同就是最后一个输出层？前者是一个最简单的 CausalEngine, 后者就是一个线性层！ 

## 2. CausalEngine 当前架构分析

### 2.1 现有组件结构
```python
# 当前的使用方式（相对复杂）
from causal_engine import CausalEngine, AbductionNetwork, ActionNetwork

# 需要用户手动配置很多参数
abduction_net = AbductionNetwork(
    input_dim=X.shape[1], 
    hidden_dim=64,
    output_dim=32
)
action_net = ActionNetwork(
    input_dim=32,
    hidden_dim=64, 
    output_dim=1
)

engine = CausalEngine(
    abduction_network=abduction_net,
    action_network=action_net,
    inference_mode='causal'
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
from causal_engine.sklearn import CausalRegressor, CausalClassifier

# 回归任务 - 3行代码搞定
reg = CausalRegressor()  # 智能默认配置
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# 分类任务 - 同样简单
clf = CausalClassifier() 
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# 高级用法 - 仍然简洁
reg = CausalRegressor(
    hidden_layers=(64, 32),  # 网络结构
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
scores = cross_val_score(CausalRegressor(), X, y, cv=5)

# 网格搜索
param_grid = {
    'hidden_layers': [(32,), (64,), (64, 32)],
    'inference_mode': ['standard', 'causal']
}
grid_search = GridSearchCV(CausalRegressor(), param_grid, cv=3)

# 管道集成
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('causal', CausalRegressor())
])
```

## 4. 核心设计方案

### 4.1 API接口设计

```python
class CausalRegressor(BaseEstimator, RegressorMixin):
    """因果回归器 - sklearn风格接口"""
    
    def __init__(self, 
                 hidden_layers=(64, 32),      # 网络结构
                 max_iter=1000,               # 最大迭代次数
                 learning_rate=0.001,         # 学习率
                 inference_mode='standard',   # 推理模式
                 early_stopping=True,         # 早停
                 validation_fraction=0.1,     # 验证集比例
                 random_state=None,           # 随机种子
                 verbose=False):              # 训练日志
        pass
    
    def fit(self, X, y, sample_weight=None):
        """训练模型"""
        # 自动数据预处理
        # 自动网络构建
        # 自动训练循环
        return self
    
    def predict(self, X):
        """预测"""
        # 自动推理
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

class CausalClassifier(BaseEstimator, ClassifierMixin):
    """因果分类器 - sklearn风格接口"""
    
    def predict_proba(self, X):
        """预测概率"""
        pass
    
    def predict_log_proba(self, X):
        """预测对数概率"""
        pass
```

### 4.2 智能默认配置策略

```python
# 根据数据规模自动调整网络结构
def _auto_hidden_layers(n_features, n_samples):
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

### 4.3 推理模式设计原则

**核心原则**: 训练和推理都默认使用 `standard` 模式，保持简洁统一。

```python
# 默认使用 - 适合99%的使用场景
reg = CausalRegressor()  # inference_mode='standard'

# 实验对比 - 看看causal模式是否能带来提升
reg_causal = CausalRegressor(inference_mode='causal')

# 性能对比
standard_score = cross_val_score(CausalRegressor(), X, y, cv=5)
causal_score = cross_val_score(CausalRegressor(inference_mode='causal'), X, y, cv=5)
```

**推理模式定位**:
- **`standard`**: 默认模式，训练和推理的标准选择，适合常规分类/回归任务
- **`causal`**: 实验性选项，用于探索是否能在特定数据集上获得更好效果
- **其他模式** (`sampling`, `stochastic`): 主要用于大模型集成和高级应用场景，sklearn接口中暂不暴露

这种设计确保：
- ✅ 用户无需理解复杂的推理模式选择
- ✅ 默认配置适合绝大多数场景
- ✅ 保留实验对比的灵活性
- ✅ 为未来大模型集成保留扩展空间

### 4.4 重要数学发现：训练阶段推理模式的可行性 🔬

**关键发现**: 基于柯西分布的线性稳定性，**训练阶段也可以使用不同的推理模式**！

#### 数学原理
```python
# 训练时的外生噪声注入机制
U'_i = U_i + b_noise · ε  # ε ~ Cauchy(0, 1)

# 三种模式的训练时表现：
# Standard: U' ~ Cauchy(μ, γ + T·|b_noise|)  # 噪声增加尺度参数
# Sampling: U' ~ Cauchy(μ + T·|b_noise|·ε, γ)  # 噪声扰动位置参数  
# Causal:   U' ~ Cauchy(μ, γ)                  # 纯因果生成(T=0)
```

#### 潜在优势
- **多样化训练**: 不同推理模式可能学到不同的特征表示
- **鲁棒性增强**: 训练时的噪声注入提高模型鲁棒性
- **模式专化**: 每种模式可能在特定数据类型上表现更好
- **解析高效**: 基于分布参数变换，无需复杂采样

#### 未来API设计潜力
```python
# 未来可能的高级API (暂不实现)
reg = CausalRegressor(
    train_inference_mode='sampling',  # 训练时用采样模式
    test_inference_mode='standard'    # 测试时用标准模式
)
```

**🎯 渐进式开发策略**:
- **V1.0**: 训练和推理使用统一模式 (`inference_mode` 参数)
- **V2.0**: 探索分离训练和测试推理模式的价值
- **V3.0**: 基于实验结果决定是否暴露高级API

## 5. 实现路线图

### 5.1 第一阶段：核心API构建
- [ ] 创建 `causal_engine.sklearn` 子模块
- [ ] 实现 `CausalRegressor` 基础类
- [ ] 实现 `CausalClassifier` 基础类
- [ ] 集成现有CausalEngine核心功能

### 5.2 第二阶段：智能化增强
- [ ] 实现自动网络结构推荐
- [ ] 实现自动超参数优化
- [ ] 提供causal推理模式作为standard模式的对比选项
- [ ] 添加训练过程可视化
- [ ] 完善错误处理和警告
- [ ] **实验性**: 探索训练阶段不同推理模式的效果差异

### 5.3 第三阶段：生态集成
- [ ] sklearn兼容性测试
- [ ] 与pandas DataFrame深度集成
- [ ] 添加特征重要性分析
- [ ] 集成模型解释工具

## 6. 使用场景对比

### 6.1 现在 vs 未来

| 场景 | 现在的方式 | 理想的方式 |
|------|-----------|-----------|
| **快速原型** | 20+行代码，需了解架构 | 3行代码，零配置 |
| **参数调优** | 手动试验各种组合 | GridSearchCV自动搜索 |
| **模型评估** | 手写评估代码 | cross_val_score一行搞定 |
| **生产部署** | 需要自己处理序列化 | joblib直接保存加载 |
| **特征分析** | 需要自己实现 | feature_importances_属性 |

### 6.2 竞争对比

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
from causal_engine.sklearn import CausalRegressor
reg = CausalRegressor()  # 同样简洁，但是因果推理！
reg.fit(X_train, y_train)
```

## 7. 技术实现要点

### 7.1 关键挑战
1. **参数映射**: sklearn参数 → CausalEngine内部参数
2. **训练循环**: 封装复杂的训练逻辑
3. **状态管理**: 模型训练状态的保存和恢复
4. **错误处理**: 友好的错误信息
5. **性能优化**: 保持原有性能优势

### 7.2 架构设计
```python
# 内部架构
sklearn_api/
├── base.py           # 基础类和接口
├── regressor.py      # CausalRegressor实现
├── classifier.py     # CausalClassifier实现  
├── utils.py          # 工具函数
├── validation.py     # 参数验证
└── _config.py        # 默认配置
```

## 8. 下一步讨论要点

1. **API细节确认**: 具体的参数名称和默认值
2. **训练策略**: 如何封装训练循环，保持灵活性
3. **推理模式**: 四种推理模式如何暴露给用户
4. **扩展性**: 如何保持高级用户的定制能力
5. **测试策略**: 如何确保与sklearn生态的兼容性

---

**💡 这个方案的价值**:
- 大幅降低使用门槛，让更多人能用上因果推理
- 与现有ML工作流无缝集成
- 保持CausalEngine的技术优势，包装成用户友好的接口
- 为CausalEngine的广泛采用奠定基础

**🎯 期待反馈**:
请提供您对这个设计方案的具体想法、需求和建议！