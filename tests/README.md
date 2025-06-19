# CausalQwen 测试套件

本目录包含 CausalQwen 的完整测试套件，使用 pytest 框架编写。

## 📁 测试文件结构

```
tests/
├── conftest.py              # 共享的测试配置和fixtures
├── pytest.ini               # pytest配置文件
├── test_math_framework.py   # 核心数学框架测试
├── test_compatibility.py    # Qwen接口兼容性测试
├── test_comparison.py       # 与原始Qwen对比测试（需要Qwen模型）
├── test_generation.py       # 生成功能测试
└── README.md               # 本文件
```

## 🚀 快速开始

### 安装依赖

```bash
# 确保已安装pytest
pip install pytest pytest-cov

# 安装项目依赖
pip install -r requirements.txt
```

### 运行所有测试

```bash
# 在项目根目录运行
pytest tests/

# 或者在tests目录内运行
cd tests
pytest
```

### 运行特定测试文件

```bash
# 测试核心数学框架
pytest tests/test_math_framework.py

# 测试兼容性
pytest tests/test_compatibility.py

# 测试生成功能
pytest tests/test_generation.py
```

### 运行特定测试类或方法

```bash
# 运行特定类
pytest tests/test_math_framework.py::TestCauchyMath

# 运行特定方法
pytest tests/test_math_framework.py::TestCauchyMath::test_cauchy_linear_stability_location
```

## 🏷️ 测试标记

我们使用标记来分类测试：

- `slow`: 运行较慢的测试
- `requires_qwen`: 需要Qwen预训练模型的测试

### 跳过需要Qwen模型的测试

```bash
# 排除需要Qwen模型的测试
cd tests
pytest -m "not requires_qwen"
```

### 只运行快速测试

```bash
cd tests
# 排除慢速测试
pytest -m "not slow"
```

## 📊 测试覆盖率

生成覆盖率报告：

```bash
# 生成终端报告
pytest --cov=causal_qwen_mvp --cov-report=term

# 生成HTML报告
pytest --cov=causal_qwen_mvp --cov-report=html

# 查看HTML报告
open htmlcov/index.html  # macOS
# 或
xdg-open htmlcov/index.html  # Linux
```

## 📐 数学原理验证

### 核心数学框架：Y = f(U, ε)

CausalQwen的核心数学框架是**因果分解**：

```
Y = f(U, ε)
```

其中：
- **U**: 个体选择变量，$U \sim \text{Cauchy}(\mu, \gamma)$
- **ε**: 外生噪声，$\varepsilon \sim \text{Cauchy}(0, 1)$  
- **f**: 普适因果机制（线性变换）

### 双模式数学定义

**双模式处理机制**：

#### 确定性模式 (do_sample=False)
```
U' = Cauchy(μ, γ + |b_noise|)
S = W · U' + b
```
**数学含义**: 噪声融合到尺度参数，增加决策不确定性，但保持个体因果表征中心不变。

#### 采样模式 (do_sample=True)  
```
ε ~ Cauchy(0, 1)
U' = Cauchy(μ + T·|b_noise|·ε, γ)
S = W · U' + b
```
**数学含义**: 噪声扰动位置参数，改变个体因果表征，产生不同决策路径。

### 柯西分布线性稳定性

**核心数学性质**：如果 $X \sim \text{Cauchy}(\mu, \gamma)$，则对于线性变换 $Y = aX + b$：

```
Y ~ Cauchy(aμ + b, |a|γ)
```

**位置参数变换**: $\mu_Y = a\mu_X + b$  
**尺度参数变换**: $\gamma_Y = |a|\gamma_X$

---

## 🧪 测试内容说明

### 1. 核心数学框架测试 (`test_math_framework.py`)

#### 🔬 TestCauchyMath: 柯西分布线性稳定性验证

**验证数学公式**：
```python
# test_cauchy_linear_stability_location
loc_output = CauchyMath.cauchy_linear_stable_loc(loc_input, weight, bias)
# 验证: loc_output = loc_input @ weight.T + bias
```

**数学验证**: $\mu_{out} = \mu_{in} \cdot W^T + b$

```python  
# test_cauchy_linear_stability_scale
scale_output = CauchyMath.cauchy_linear_stable_scale(scale_input, weight)
# 验证: scale_output = scale_input @ |weight|.T
```

**数学验证**: $\gamma_{out} = \gamma_{in} \cdot |W|^T$

#### ⚡ TestActionNetworkModes: 双模式数学验证

**确定性模式验证** (`test_v2_non_sampling_mode`):
```
输入: loc_U, scale_U, do_sample=False
期望输出:
  loc_S = W · loc_U + b
  scale_S = (scale_U + |b_noise|) @ |W|^T
```

**数学验证**: $S \sim \text{Cauchy}(W \cdot \mu_U + b, (\gamma_U + |b_{noise}|) \cdot |W|^T)$

**采样模式验证** (`test_v2_sampling_mode`):
```
输入: loc_U, scale_U, do_sample=True, temperature=T
期望过程:
  ε ~ Cauchy(0, 1)
  loc_U_noisy = loc_U + T·|b_noise|·ε
  loc_S = W · loc_U_noisy + b
  scale_S = scale_U @ |W|^T
```

**数学验证**: $S \sim \text{Cauchy}(W \cdot (\mu_U + T \cdot |b_{noise}| \cdot \varepsilon) + b, \gamma_U \cdot |W|^T)$

**模式差异验证** (`test_mode_differences`):
验证 $\text{loc\_S}_{det} \neq \text{loc\_S}_{samp}$，体现双模式的核心差异。

#### 🧠 TestAbductionNetwork: 个体推断数学验证

**前向传播验证** (`test_abduction_forward`):
```
输入: hidden_states [B, S, H]
输出: loc_U, scale_U [B, S, C]

数学验证:
  loc_U = Linear_loc(hidden_states)
  scale_U = softplus(Linear_scale(hidden_states))
```

**恒等初始化验证** (`test_abduction_initialization`):
当 `hidden_size == causal_size` 时，验证恒等映射：
```
loc_U ≈ hidden_states  (初始化时)
scale_U > 0  (确保为正)
```

#### 🔗 TestIntegration: 端到端数学流程验证

**完整因果推理链路**:
```
hidden_states → AbductionNetwork → (loc_U, scale_U)
                      ↓
(loc_U, scale_U) → ActionNetwork → (loc_S, scale_S)
```

**数学验证**: 完整的 $\text{hidden\_states} \xrightarrow{f_{abd}} U \xrightarrow{f_{act}} S$ 变换链路。

### 2. Qwen兼容性测试 (`test_compatibility.py`)

#### 🔄 TestQwenCompatibility: 接口数学一致性验证

**确定性生成数学验证** (`test_generate_interface_deterministic`):
```
对于相同输入 x，确定性模式应满足:
generate(x, do_sample=False) = generate(x, do_sample=False)
```

**数学保证**: $f(x, \text{do\_sample}=\text{False})$ 是确定性函数。

**采样生成多样性验证** (`test_do_sample_difference`):
```
对于相同输入 x，采样模式应满足:
P(generate(x, do_sample=True, seed=i) ≠ generate(x, do_sample=True, seed=j)) > 0
```

**数学验证**: 采样模式引入随机性，产生不同输出序列。

**温度参数数学效应** (`test_temperature_effect`):
```
对于 do_sample=True:
  - 低温度 T → 更确定的决策
  - 高温度 T → 更随机的决策
  
数学表达: 温度 T 控制噪声强度 T·|b_noise|·ε
```

#### 🧮 TestMathematicalPrinciples: 数学原理实证验证

**数学原理验证** (`test_v2_principles_validation`):
```
使用 InferenceValidator 验证:
1. position_difference = |loc_S_det - loc_S_samp| > ε
2. base_representations: loc_U, scale_U 的合理性
3. scale_U > 0 (尺度参数正性)
```

**数学验证**: 双模式在实际推理中的数学差异。

### 3. 与原始Qwen对比测试 (`test_comparison.py`)

⚠️ **需要Qwen预训练模型**

#### 🔄 TestWeightCopying: 权重继承数学验证

**Transformer权重一致性** (`test_transformer_weights_copied`):
```
验证: CausalQwen.model.state_dict() ≈ Qwen.model.state_dict()
数学保证: 特征提取 h = Transformer(x) 完全一致
```

**ActionNetwork权重继承** (`test_action_network_weights_copied`):
```
验证: CausalQwen.action_network.lm_head.weight = Qwen.lm_head.weight
数学保证: 词汇表映射 W 完全继承
```

#### 📊 TestLogitsConsistency: 数学输出一致性验证

**logits数学一致性** (`test_loc_s_vs_qwen_logits`):
```
对于确定性模式 (do_sample=False):
CausalQwen.loc_S ≈ Qwen.logits

数学验证: 
  loc_S = W · AbductionNetwork(h) + b ≈ W · h + b = Qwen.logits
  (当AbductionNetwork初始化为恒等映射时)
```

**关键数学等式**: $\text{loc\_S} = W \cdot h + b = \text{Qwen.logits}$

#### 🎯 TestGenerationComparison: 生成质量数学验证

**温度控制数学效应** (`test_temperature_control_in_real_generation`):
```
验证温度参数的数学作用:
- 低温度: 噪声项 T·|b_noise|·ε 较小，生成更确定
- 高温度: 噪声项 T·|b_noise|·ε 较大，生成更随机

数学关系: Var(output) ∝ T²
```

### 4. 生成功能测试 (`test_generation.py`)

#### 🚀 TestBasicGeneration: 基本生成数学验证

**序列长度数学约束** (`test_max_length_constraint`):
```
验证: |output| = |input| + max_new_tokens
数学约束: 生成长度受最大token数限制
```

**token有效性数学验证** (`test_token_validity`):
```
验证: ∀ token ∈ output, 0 ≤ token < vocab_size
数学约束: 所有生成token在词汇表范围内
```

#### 🎲 TestGenerationModes: 模式数学差异验证

**确定性模式数学验证** (`test_standard_mode`):
```
对于输入 x: f(x, do_sample=False) = f(x, do_sample=False)
数学保证: 确定性函数，无随机性
```

**采样模式数学验证** (`test_sampling_mode`):
```
对于输入 x: P(f(x, do_sample=True) ≠ f(x, do_sample=True)) > 0
数学保证: 随机函数，引入变异性
```

#### 🎯 TestSamplingStrategies: 采样策略数学验证

**注意**: CausalQwen的"采样"是ActionNetwork内部的噪声注入，不是传统的多项分布采样。

**内部噪声注入数学**:
```
CausalQwen采样 ≠ 传统top_k/top_p采样
CausalQwen: U' = μ + T·|b_noise|·ε
传统采样: token ~ Multinomial(softmax(logits/T))
```

**实现机制**: 在因果表征层面进行采样，而非最终概率分布采样。

## 🔧 自定义测试配置

### 修改pytest配置

编辑 `pytest.ini` 文件来自定义测试行为。

### 添加新的fixtures

在 `conftest.py` 中添加新的共享测试资源：

```python
@pytest.fixture
def my_custom_fixture():
    """自定义fixture"""
    return some_test_resource
```

### 添加新的测试标记

1. 在 `pytest.ini` 中注册标记：
```ini
markers =
    my_marker: 描述这个标记的作用
```

2. 在测试中使用：
```python
@pytest.mark.my_marker
def test_something():
    pass
```

## 📝 编写新测试

### 测试命名规范

- 测试文件: `test_*.py`
- 测试类: `Test*`
- 测试方法: `test_*`

### 测试结构示例

```python
class TestMyFeature:
    """测试某个功能"""
    
    def test_basic_functionality(self, test_model):
        """测试基本功能"""
        # Arrange
        input_data = ...
        
        # Act
        result = test_model.some_method(input_data)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
```

### 使用fixtures

```python
def test_with_fixtures(self, test_model, sample_input_ids, tolerance):
    """使用多个fixtures的测试"""
    output = test_model(sample_input_ids)
    torch.testing.assert_close(output, expected, **tolerance)
```

## 🐛 调试测试

### 查看详细输出

```bash
# 显示print语句
pytest -s

# 显示更详细的断言信息
pytest -vv

# 在第一个失败时停止
pytest -x

# 进入调试器
pytest --pdb
```

### 运行上次失败的测试

```bash
# 只运行上次失败的测试
pytest --lf

# 先运行失败的测试，然后运行其他
pytest --ff
```

## 📈 性能测试

```bash
# 显示最慢的10个测试
pytest --durations=10
```

## 🤝 贡献指南

1. 为新功能编写测试
2. 确保所有测试通过
3. 保持测试覆盖率在80%以上
4. 遵循现有的测试风格和命名规范

## ❓ 常见问题

### Q: 测试找不到causal_qwen_mvp模块？
A: 确保在项目根目录运行测试，或正确设置PYTHONPATH。

### Q: 需要Qwen模型的测试失败？
A: 确保Qwen模型路径正确（默认为`~/models/Qwen2.5-0.5B`），或使用`-m "not requires_qwen"`跳过这些测试。

### Q: 测试运行太慢？
A: 使用`-m "not slow"`跳过慢速测试，或使用`pytest -n auto`并行运行（需要安装pytest-xdist）。

---

## 📐 数学验证框架总结

### 核心数学不变量

我们的测试验证了以下关键数学不变量：

#### 1. 柯西分布线性稳定性
```
如果 X ~ Cauchy(μ, γ)，则 aX + b ~ Cauchy(aμ + b, |a|γ)
```
**测试验证**: `TestCauchyMath` 类确保这一数学性质在代码中正确实现。

#### 2. 双模式数学差异
```
确定性模式: S ~ Cauchy(W·μ + b, (γ + |b_noise|)·|W|^T)
采样模式:  S ~ Cauchy(W·(μ + T·|b_noise|·ε) + b, γ·|W|^T)
```
**测试验证**: `TestActionNetworkModes` 类验证噪声对位置vs尺度的差异化影响。

#### 3. 端到端因果链路
```
hidden_states → U ~ Cauchy(μ, γ) → S ~ Cauchy(W·μ + b, γ·|W|^T)
```
**测试验证**: `TestIntegration` 类确保完整因果推理链路的数学正确性。

#### 4. Qwen兼容性数学等价
```
当 AbductionNetwork ≈ Identity 时: CausalQwen.loc_S ≈ Qwen.logits
```
**测试验证**: `TestLogitsConsistency` 类验证在特定条件下的数学等价性。

### 测试数学覆盖率

- ✅ **柯西分布数学**: 线性稳定性、参数变换
- ✅ **噪声注入数学**: 位置vs尺度差异化处理  
- ✅ **温度控制数学**: $T \cdot |b_{noise}| \cdot \varepsilon$ 的作用
- ✅ **确定性vs随机性**: 数学函数性质验证
- ✅ **权重继承数学**: Transformer特征一致性
- ✅ **序列生成数学**: 长度约束、token有效性

**数学验证原则**: 每个测试都对应一个精确的数学公式或不变量，确保代码实现与数学理论完全一致。 