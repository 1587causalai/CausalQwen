# CausalEngine 实现规范

> **文档用途**：为 causal-sklearn 实现提供精确的数学规范和参数定义  
> **参考依据**：基于 `MATHEMATICAL_FOUNDATIONS_CN.md` 的权威理论框架  
> **验证标准**：所有实现必须严格遵循此规范

## 1. 三阶段架构实现规范

### 1.1 阶段1：归因推断（Abduction）

**输入**：观测特征 `X` 形状 `[batch_size, n_features]`

**网络结构**：
```python
class AbductionNetwork:
    def __init__(self, input_size, hidden_size):
        self.loc_net = MLP(input_size, hidden_size)     # 位置网络
        self.scale_net = MLP(input_size, hidden_size)   # 尺度网络
    
    def forward(self, X):
        μ_U = self.loc_net(X)                          # [batch_size, hidden_size]
        γ_U = softplus(self.scale_net(X))              # [batch_size, hidden_size]
        return μ_U, γ_U  # U ~ Cauchy(μ_U, γ_U)
```

**数学公式**：
- 位置参数：`μ_U = loc_net(X)`
- 尺度参数：`γ_U = softplus(scale_net(X)) = log(1 + exp(scale_net(X)))`
- 个体表征分布：`U ~ Cauchy(μ_U, γ_U)`

### 1.2 阶段2：行动决策（Action）

**输入**：个体表征分布 `U ~ Cauchy(μ_U, γ_U)`

**五种推理模式的数学定义**：

```python
def action_network(μ_U, γ_U, mode, b_noise):
    # 模式1: Deterministic - 确定性推理
    if mode == "deterministic":
        U_prime = μ_U
        return W_A @ U_prime + b_A
    
    # 模式2: Exogenous - 外生噪声推理  
    elif mode == "exogenous":
        # U' ~ Cauchy(μ_U, |b_noise|)
        μ_S = W_A @ μ_U + b_A
        γ_S = |W_A| @ |b_noise|
        return μ_S, γ_S
    
    # 模式3: Endogenous - 内生不确定性推理
    elif mode == "endogenous":
        # U' ~ Cauchy(μ_U, γ_U) 
        μ_S = W_A @ μ_U + b_A
        γ_S = |W_A| @ γ_U
        return μ_S, γ_S
    
    # 模式4: Standard - 混合推理（噪声→尺度）
    elif mode == "standard":
        # U' ~ Cauchy(μ_U, γ_U + |b_noise|)
        μ_S = W_A @ μ_U + b_A  
        γ_S = |W_A| @ (γ_U + |b_noise|)
        return μ_S, γ_S
    
    # 模式5: Sampling - 随机探索推理（噪声→位置）
    elif mode == "sampling":
        # U' ~ Cauchy(μ_U + b_noise*ε, γ_U)
        ε = sample_cauchy_noise()  # 仅此模式需要采样
        μ_S = W_A @ (μ_U + b_noise * ε) + b_A
        γ_S = |W_A| @ γ_U
        return μ_S, γ_S
```

**线性稳定性利用**：
- 柯西分布的线性变换：`W*U + b ~ Cauchy(W*μ_U + b, |W|*γ_U)`
- 加法稳定性：`Cauchy(μ₁,γ₁) + Cauchy(μ₂,γ₂) = Cauchy(μ₁+μ₂, γ₁+γ₂)`

### 1.3 阶段3：任务激活（Task Activation）

**输入**：决策得分分布 `S ~ Cauchy(μ_S, γ_S)`

**回归任务激活**：
```python
def regression_activation(μ_S, γ_S, mode):
    if mode == "deterministic":
        return μ_S  # 点预测
    else:
        return μ_S, γ_S  # 分布预测 [n_samples, output_dim, 2]
```

**分类任务激活（OvR策略）**：
```python
def classification_activation(μ_S, γ_S, threshold_C=0.0):
    # 柯西CDF计算：P(S_k > C_k)
    P_k = 0.5 + (1/π) * arctan((μ_S - threshold_C) / γ_S)
    return P_k  # [n_samples, n_classes]
```

## 2. 统一损失函数规范

### 2.1 确定性模式损失

```python
def deterministic_loss(y_true, y_pred, task_type):
    if task_type == "regression":
        # 标准MSE损失
        return mean((y_true - y_pred) ** 2)
    elif task_type == "classification":
        # 标准交叉熵损失
        return cross_entropy(y_true, softmax(y_pred))
```

### 2.2 因果模式损失

```python
def causal_loss(y_true, μ_S, γ_S, task_type):
    if task_type == "regression":
        # 柯西NLL损失
        normalized_residual = (y_true - μ_S) / γ_S
        loss = log(π) + log(γ_S) + log(1 + normalized_residual**2)
        return mean(loss)
    
    elif task_type == "classification":
        # OvR二元交叉熵损失
        P_k = 0.5 + (1/π) * arctan(μ_S / γ_S)  # 假设threshold_C=0
        # y_true: one-hot编码 [n_samples, n_classes]
        bce_loss = -(y_true * log(P_k) + (1 - y_true) * log(1 - P_k))
        return mean(sum(bce_loss, axis=1))
```

## 3. 关键参数规范

### 3.1 网络参数
- `hidden_layer_sizes`: 隐藏层架构，默认 `(100,)`
- `gamma_init`: AbductionNetwork尺度初始化，默认 `10.0` 
- `b_noise_init`: ActionNetwork外生噪声初始化，默认 `0.1`
- `b_noise_trainable`: 外生噪声是否可训练，默认 `True`

### 3.2 推理模式
- `mode`: 推理模式选择
  - `"deterministic"`: 与传统ML数学等价
  - `"exogenous"`: 纯外生噪声推理
  - `"endogenous"`: 纯内生不确定性推理  
  - `"standard"`: 混合推理（推荐）
  - `"sampling"`: 随机探索推理

### 3.3 分类专用参数  
- `ovr_threshold_init`: OvR分类阈值，默认 `0.0`
- 每个类别独立判断，无竞争性归一化

## 4. 实现验证要点

### 4.1 数学等价性验证
确定性模式下必须与标准MLP完全等价：
```python
# 验证回归等价性
causal_reg = MLPCausalRegressor(mode="deterministic")
sklearn_reg = MLPRegressor(hidden_layer_sizes=causal_reg.hidden_layer_sizes)
# 相同数据下预测结果应该等价（在适当的参数初始化下）

# 验证分类等价性  
causal_clf = MLPCausalClassifier(mode="deterministic")
sklearn_clf = MLPClassifier(hidden_layer_sizes=causal_clf.hidden_layer_sizes)
# 确定性模式应该表现类似于标准MLP
```

### 4.2 柯西分布性质验证
```python
# 验证线性稳定性
X ~ Cauchy(μ, γ)
aX + b ~ Cauchy(aμ + b, |a|γ)  # 必须严格满足

# 验证加法稳定性
X1 ~ Cauchy(μ1, γ1), X2 ~ Cauchy(μ2, γ2)
X1 + X2 ~ Cauchy(μ1 + μ2, γ1 + γ2)  # 必须严格满足
```

### 4.3 梯度计算验证
所有参数的梯度必须可计算且数值稳定：
- AbductionNetwork的 `loc_net` 和 `scale_net` 参数
- ActionNetwork的 `W_A` 和 `b_A` 参数  
- 外生噪声参数 `b_noise`（如果可训练）

## 5. sklearn 兼容性要求

### 5.1 必需接口
```python
class MLPCausalRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def predict_dist(self, X): ...  # CausalEngine特有
    def score(self, X, y, sample_weight=None): ...

class MLPCausalClassifier(BaseEstimator, ClassifierMixin):  
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def predict_dist(self, X): ...  # CausalEngine特有
    def score(self, X, y, sample_weight=None): ...
```

### 5.2 参数管理
```python
def get_params(self, deep=True): ...
def set_params(self, **params): ...
```

### 5.3 属性设置
```python
# 训练后必须设置的属性
self.n_features_in_ = X.shape[1]
self.classes_ = unique_labels(y)  # 仅分类器
self.n_iter_ = actual_iterations
```

---

**重要提醒**：
1. 所有实现必须严格遵循 `MATHEMATICAL_FOUNDATIONS_CN.md` 中的数学定义
2. 柯西分布的线性稳定性是整个算法的核心，必须正确实现
3. 五种推理模式的数学区别必须精确体现
4. 确定性模式与传统ML的等价性是验证算法正确性的基准