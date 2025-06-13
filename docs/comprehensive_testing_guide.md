# CausalQwen 数学实现全面测试指南

本文档提供一个全新的、更加严格的测试框架，专门用于验证 CausalQwen 中每个数学组件的正确实现。

## 🎯 核心测试原则

1. **数学精确性优先**：每个测试必须直接对应数学公式
2. **隔离测试**：每个组件独立测试，避免级联错误
3. **边界条件全覆盖**：特别关注零值、极值、特殊情况
4. **数值稳定性验证**：检查浮点运算的精度和稳定性
5. **维度一致性检查**：确保张量形状在整个流程中保持正确

## 📋 测试清单概览

### 第一部分：基础组件测试
- [ ] 数值编码函数 φ(v) 的完整验证
- [ ] 增强嵌入计算的精确性测试
- [ ] 词汇表扩展的正确性验证

### 第二部分：分布计算测试
- [ ] 柯西分布参数的有效性验证
- [ ] 线性变换的数学精确性测试
- [ ] CDF 计算的数值准确性

### 第三部分：网络组件测试
- [ ] 归因网络的初始化验证
- [ ] 行动网络的权重迁移测试
- [ ] OvR 阈值的影响分析

### 第四部分：损失函数测试
- [ ] 柯西负对数似然的精确计算
- [ ] 门控机制的边界情况
- [ ] 梯度流的正确性验证

### 第五部分：端到端集成测试
- [ ] 完整前向传播的数值验证
- [ ] 采样模式的一致性测试
- [ ] 批处理的正确性验证

## 🔬 详细测试规范

### 1. 数值编码函数 φ(v) 测试

#### 1.1 基础性质测试

**测试目标**：验证数值编码函数的核心数学性质

```python
import torch
import numpy as np

class TestNumericalEncodingFunction:
    def test_phi_zero_exact(self):
        """φ(0) 必须精确等于零向量"""
        # 测试单个零值
        v = 0.0
        direction = torch.randn(768)
        direction = direction / direction.norm()
        
        phi_v = compute_phi(v, direction)
        
        # 使用极小的容差
        assert torch.allclose(phi_v, torch.zeros_like(phi_v), atol=1e-10)
        assert phi_v.abs().max().item() == 0.0  # 精确为零
        
    def test_phi_sign_symmetry(self):
        """验证 φ(-v) = -φ(v)"""
        for v in [0.1, 1.0, 10.0, 100.0, 1e6]:
            direction = torch.randn(768)
            direction = direction / direction.norm()
            
            phi_pos = compute_phi(v, direction)
            phi_neg = compute_phi(-v, direction)
            
            # 符号对称性必须精确
            assert torch.allclose(phi_pos, -phi_neg, rtol=1e-7)
            
    def test_phi_logarithmic_growth(self):
        """验证对数增长特性"""
        direction = torch.randn(768)
        direction = direction / direction.norm()
        
        values = [1.0, 10.0, 100.0, 1000.0]
        norms = []
        
        for v in values:
            phi_v = compute_phi(v, direction)
            norms.append(phi_v.norm().item())
        
        # 验证增长率递减
        growth_rates = []
        for i in range(1, len(norms)):
            growth_rate = norms[i] / norms[i-1]
            growth_rates.append(growth_rate)
            
        # 增长率应该递减
        for i in range(1, len(growth_rates)):
            assert growth_rates[i] < growth_rates[i-1]
            
    def test_phi_numerical_stability(self):
        """测试极端值的数值稳定性"""
        direction = torch.randn(768)
        direction = direction / direction.norm()
        
        extreme_values = [
            1e-10,  # 极小值
            1e-5,
            1e10,   # 大值
            1e20,   # 极大值
        ]
        
        for v in extreme_values:
            phi_v = compute_phi(v, direction)
            
            # 不应该有 NaN 或 Inf
            assert not torch.isnan(phi_v).any()
            assert not torch.isinf(phi_v).any()
            
            # 验证公式：|φ(v)| = |ln(1 + |v|)|
            expected_norm = abs(np.log(1 + abs(v)))
            actual_norm = phi_v.norm().item()
            
            # 对于极端值允许稍大的相对误差
            rtol = 1e-5 if abs(v) < 1e10 else 1e-3
            assert np.isclose(actual_norm, expected_norm, rtol=rtol)
```

#### 1.2 批量计算测试

```python
def test_phi_batch_computation(self):
    """验证批量计算的正确性"""
    batch_size = 32
    seq_len = 128
    hidden_dim = 768
    
    # 创建批量数值数据
    numeric_values = torch.randn(batch_size, seq_len)
    # 添加特殊值
    numeric_values[0, 0] = 0.0  # 零值
    numeric_values[1, 1] = 1e10  # 大值
    numeric_values[2, 2] = -1e-5  # 小负值
    
    direction = torch.randn(hidden_dim)
    direction = direction / direction.norm()
    
    # 批量计算
    phi_batch = compute_phi_batch(numeric_values, direction)
    
    # 验证形状
    assert phi_batch.shape == (batch_size, seq_len, hidden_dim)
    
    # 验证特殊值
    assert torch.allclose(phi_batch[0, 0], torch.zeros(hidden_dim), atol=1e-10)
    
    # 逐个验证与单独计算的一致性
    for i in range(batch_size):
        for j in range(seq_len):
            v = numeric_values[i, j].item()
            phi_single = compute_phi(v, direction)
            assert torch.allclose(phi_batch[i, j], phi_single, rtol=1e-6)
```

### 2. 增强嵌入测试

#### 2.1 嵌入融合测试

```python
class TestEnhancedEmbedding:
    def test_embedding_fusion_non_numeric(self):
        """非数值位置的嵌入保持不变"""
        vocab_size = 151937  # Qwen词汇表大小
        hidden_dim = 768
        
        # 创建嵌入层
        embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        
        # 非数值词元
        input_ids = torch.randint(0, vocab_size-1, (4, 16))  # 避免<NUM>
        numeric_values = torch.zeros(4, 16)
        
        # 计算基础嵌入
        base_embeddings = embedding(input_ids)
        
        # 计算增强嵌入
        enhanced = compute_enhanced_embedding(
            input_ids, numeric_values, embedding
        )
        
        # 非数值位置应该完全相同
        assert torch.allclose(enhanced, base_embeddings, atol=1e-8)
        
    def test_embedding_fusion_numeric(self):
        """数值位置的嵌入正确增强"""
        num_token_id = 151936  # <NUM> token ID
        
        # 创建包含数值的输入
        input_ids = torch.tensor([[100, num_token_id, 200]])
        numeric_values = torch.tensor([[0.0, 99.9, 0.0]])
        
        # 获取增强嵌入
        enhanced = compute_enhanced_embedding(
            input_ids, numeric_values, embedding
        )
        
        # 验证数值位置被增强
        base_num_embedding = embedding(torch.tensor([num_token_id]))
        phi_value = compute_phi(99.9, direction)
        expected = base_num_embedding + phi_value
        
        assert torch.allclose(enhanced[0, 1], expected, rtol=1e-6)
```

#### 2.2 边界条件测试

```python
def test_embedding_edge_cases(self):
    """测试边界和特殊情况"""
    # 情况1：全是数值
    input_ids = torch.full((2, 8), num_token_id)
    numeric_values = torch.randn(2, 8) * 100
    
    enhanced = compute_enhanced_embedding(
        input_ids, numeric_values, embedding
    )
    
    # 每个位置都应该被增强
    base = embedding(input_ids)
    for i in range(2):
        for j in range(8):
            phi_v = compute_phi(numeric_values[i, j].item(), direction)
            expected = base[i, j] + phi_v
            assert torch.allclose(enhanced[i, j], expected, rtol=1e-6)
    
    # 情况2：数值为零的<NUM>位置
    input_ids = torch.tensor([[num_token_id]])
    numeric_values = torch.tensor([[0.0]])
    
    enhanced = compute_enhanced_embedding(
        input_ids, numeric_values, embedding
    )
    
    # 应该等于基础嵌入（因为φ(0) = 0）
    base = embedding(torch.tensor([num_token_id]))
    assert torch.allclose(enhanced[0, 0], base[0], atol=1e-8)
```

### 3. 柯西分布计算测试

#### 3.1 分布参数验证

```python
class TestCauchyDistribution:
    def test_cauchy_parameter_constraints(self):
        """验证柯西分布参数的有效性"""
        batch_size = 16
        seq_len = 32
        hidden_dim = 768
        
        # 创建随机特征
        features = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 通过归因网络
        loc_u, scale_u = abduction_network(features)
        
        # 验证形状
        assert loc_u.shape == (batch_size, seq_len, hidden_dim)
        assert scale_u.shape == (batch_size, seq_len, hidden_dim)
        
        # 验证 scale > 0
        assert (scale_u > 0).all()
        
        # 验证初始化后的大致范围
        if not model.is_trained:
            # 初始化时 scale 应该较大（约10）
            assert scale_u.mean().item() > 5.0
            assert scale_u.mean().item() < 20.0
            
    def test_cauchy_linear_transformation(self):
        """验证柯西分布的线性变换性质"""
        # U ~ Cauchy(μ, γ)
        loc_u = torch.tensor([2.0])
        scale_u = torch.tensor([3.0])
        
        # 线性变换 Y = aU + b
        a = 2.5
        b = -1.0
        
        # 计算变换后的参数
        loc_y = a * loc_u + b
        scale_y = abs(a) * scale_u
        
        # 验证公式正确性
        expected_loc = 2.5 * 2.0 - 1.0  # = 4.0
        expected_scale = 2.5 * 3.0  # = 7.5
        
        assert torch.isclose(loc_y, torch.tensor([expected_loc]))
        assert torch.isclose(scale_y, torch.tensor([expected_scale]))
```

#### 3.2 CDF 计算精度测试

```python
def test_cauchy_cdf_accuracy(self):
    """验证柯西CDF计算的数值精度"""
    # 标准柯西分布的已知值
    test_cases = [
        # (x, loc, scale, expected_cdf)
        (0.0, 0.0, 1.0, 0.5),  # 中位数
        (1.0, 0.0, 1.0, 0.75),  # 第三四分位数
        (-1.0, 0.0, 1.0, 0.25),  # 第一四分位数
    ]
    
    for x, loc, scale, expected in test_cases:
        # 计算 P(X > x)
        prob = compute_cauchy_survival(x, loc, scale)
        expected_survival = 1 - expected
        
        # 高精度验证
        assert abs(prob - expected_survival) < 1e-10
        
def test_cauchy_cdf_vectorized(self):
    """验证向量化CDF计算"""
    batch_size = 8
    vocab_size = 151937
    
    # 创建批量数据
    loc_s = torch.randn(batch_size, vocab_size)
    scale_s = torch.rand(batch_size, vocab_size) * 5 + 0.1
    thresholds = torch.randn(vocab_size)
    
    # 批量计算
    probs = compute_ovr_probabilities(loc_s, scale_s, thresholds)
    
    # 验证范围
    assert (probs >= 0).all()
    assert (probs <= 1).all()
    
    # 验证特殊情况
    # 当 loc >> threshold 时，概率应接近1
    loc_s[0, 0] = 1000.0
    scale_s[0, 0] = 1.0
    thresholds[0] = 0.0
    
    probs_special = compute_ovr_probabilities(loc_s, scale_s, thresholds)
    assert probs_special[0, 0] > 0.999
```

### 4. 网络初始化测试

#### 4.1 归因网络初始化验证

```python
class TestNetworkInitialization:
    def test_abduction_network_initialization(self):
        """验证归因网络的恒等映射初始化"""
        hidden_dim = 768
        
        # 创建归因网络
        abduction_net = AbductionNetwork(hidden_dim)
        
        # 测试输入
        z = torch.randn(4, 16, hidden_dim)
        
        # 获取输出
        loc_u, scale_u = abduction_net(z)
        
        # 验证位置参数的恒等映射
        if isinstance(abduction_net.loc_net, torch.nn.Linear):
            # 检查权重矩阵是否为恒等矩阵
            weight = abduction_net.loc_net.weight
            eye = torch.eye(hidden_dim)
            assert torch.allclose(weight, eye, atol=1e-6)
            
            # 检查偏置为零
            bias = abduction_net.loc_net.bias
            assert torch.allclose(bias, torch.zeros_like(bias), atol=1e-6)
        
        # 验证 loc_u ≈ z
        assert torch.allclose(loc_u, z, rtol=1e-6)
        
        # 验证 scale_u 是大值
        assert scale_u.mean().item() > 5.0
        
    def test_action_network_weight_transfer(self):
        """验证行动网络的权重迁移"""
        # 模拟Qwen的lm_head权重
        vocab_size = 151936
        hidden_dim = 768
        qwen_lm_head = torch.nn.Linear(hidden_dim, vocab_size)
        qwen_lm_head.weight.data = torch.randn(vocab_size, hidden_dim)
        qwen_lm_head.bias.data = torch.randn(vocab_size)
        
        # 创建行动网络
        action_net = ActionNetwork(hidden_dim, vocab_size + 1)
        
        # 执行权重迁移
        transfer_classification_weights(qwen_lm_head, action_net)
        
        # 验证权重复制
        # 前vocab_size个权重应该相同
        transferred_weight = action_net.cls_net.weight[:vocab_size]
        assert torch.allclose(transferred_weight, qwen_lm_head.weight)
        
        # <NUM> token的权重应该被正确初始化（不是零）
        num_weight = action_net.cls_net.weight[vocab_size]
        assert num_weight.norm() > 0
```

#### 4.2 OvR 阈值初始化测试

```python
def test_ovr_threshold_initialization(self):
    """验证OvR阈值的初始化策略"""
    vocab_size = 151937
    
    # 情况1：统一阈值
    thresholds = initialize_ovr_thresholds(vocab_size, uniform=True, value=100.0)
    assert thresholds.shape == (vocab_size,)
    assert (thresholds == 100.0).all()
    
    # 情况2：可学习参数
    thresholds = torch.nn.Parameter(torch.full((vocab_size,), 100.0))
    assert thresholds.requires_grad
    
    # 验证阈值对概率的影响
    loc = torch.zeros(vocab_size)
    scale = torch.ones(vocab_size)
    
    # 高阈值 -> 低概率
    probs_high = compute_ovr_probabilities(loc, scale, thresholds)
    assert probs_high.mean() < 0.1
    
    # 零阈值 -> 0.5概率
    zero_thresholds = torch.zeros(vocab_size)
    probs_zero = compute_ovr_probabilities(loc, scale, zero_thresholds)
    assert torch.allclose(probs_zero, torch.full_like(probs_zero, 0.5), atol=1e-6)
```

### 5. 损失函数测试

#### 5.1 柯西负对数似然测试

```python
class TestLossFunctions:
    def test_cauchy_nll_formula(self):
        """验证柯西负对数似然的精确公式"""
        # 测试案例
        test_cases = [
            # (y_true, loc, scale, expected_nll)
            (0.0, 0.0, 1.0, np.log(np.pi)),  # 最小损失情况
            (1.0, 0.0, 1.0, np.log(np.pi) + np.log(2)),  # 标准偏差
        ]
        
        for y_true, loc, scale, expected in test_cases:
            y_true_t = torch.tensor([y_true])
            loc_t = torch.tensor([loc])
            scale_t = torch.tensor([scale])
            
            # 计算NLL
            nll = compute_cauchy_nll(y_true_t, loc_t, scale_t)
            
            # 验证公式：log(π·scale) + log(1 + ((y-loc)/scale)²)
            z = (y_true - loc) / scale
            expected_computed = np.log(np.pi * scale) + np.log(1 + z**2)
            
            assert abs(nll.item() - expected) < 1e-10
            assert abs(nll.item() - expected_computed) < 1e-10
            
    def test_cauchy_nll_gradient(self):
        """验证柯西NLL的梯度正确性"""
        y_true = torch.tensor([2.0], requires_grad=False)
        loc = torch.tensor([1.0], requires_grad=True)
        scale = torch.tensor([0.5], requires_grad=True)
        
        # 计算损失
        nll = compute_cauchy_nll(y_true, loc, scale)
        nll.backward()
        
        # 手动计算梯度
        z = (y_true - loc) / scale
        expected_grad_loc = 2 * z / (scale * (1 + z**2))
        expected_grad_scale = 1/scale - 2*z**2 / (scale * (1 + z**2))
        
        # 验证梯度
        assert torch.isclose(loc.grad, expected_grad_loc, rtol=1e-5)
        assert torch.isclose(scale.grad, expected_grad_scale, rtol=1e-5)
```

#### 5.2 门控机制测试

```python
def test_gated_loss_mechanism(self):
    """测试门控损失的各种情况"""
    batch_size = 4
    seq_len = 8
    
    # 创建测试数据
    # 位置0,2,5是数值位置
    mask = torch.zeros(batch_size, seq_len)
    mask[:, [0, 2, 5]] = 1.0
    
    # NUM token的预测概率
    p_num = torch.rand(batch_size, seq_len)
    
    # 基础回归损失
    base_loss = torch.rand(batch_size, seq_len) + 0.1
    
    # 测试不同的alpha值
    for alpha in [0.0, 0.1, 0.5, 1.0]:
        gated_loss = compute_gated_regression_loss(
            base_loss, mask, p_num, alpha
        )
        
        # 验证形状
        assert gated_loss.shape == (batch_size, seq_len)
        
        # 非数值位置的损失应该为0
        assert (gated_loss[:, mask[0] == 0] == 0).all()
        
        # 数值位置的损失验证
        for i in [0, 2, 5]:
            if alpha == 1.0:
                # 完全忽略模型置信度
                assert torch.allclose(gated_loss[:, i], base_loss[:, i])
            elif alpha == 0.0:
                # 完全依赖模型置信度
                expected = base_loss[:, i] * p_num[:, i]
                assert torch.allclose(gated_loss[:, i], expected)
            else:
                # 混合情况
                gate = alpha + (1 - alpha) * p_num[:, i]
                expected = base_loss[:, i] * gate
                assert torch.allclose(gated_loss[:, i], expected)
```

### 6. 端到端数值验证

#### 6.1 完整前向传播测试

```python
class TestEndToEnd:
    def test_complete_forward_pass(self):
        """测试完整的前向传播数值正确性"""
        model = CausalQwen(config)
        model.eval()
        
        # 准备输入
        input_text = "价格是 <NUM> 元，涨幅 <NUM> %"
        input_ids = torch.tensor([[1234, 5678, num_token_id, 9012, 3456, num_token_id, 7890]])
        numeric_values = torch.tensor([[0.0, 0.0, 99.9, 0.0, 0.0, 3.5, 0.0]])
        
        with torch.no_grad():
            # 执行前向传播
            output = model(input_ids, numeric_values)
            
        # 验证输出结构
        assert 'loc_S' in output
        assert 'scale_S' in output
        assert 'loc_Y' in output
        assert 'scale_Y' in output
        
        # 验证维度
        batch_size, seq_len = input_ids.shape
        vocab_size = model.config.vocab_size
        
        assert output['loc_S'].shape == (batch_size, seq_len, vocab_size)
        assert output['scale_S'].shape == (batch_size, seq_len, vocab_size)
        assert output['loc_Y'].shape == (batch_size, seq_len)
        assert output['scale_Y'].shape == (batch_size, seq_len)
        
        # 验证数值范围
        assert (output['scale_S'] > 0).all()
        assert (output['scale_Y'] > 0).all()
        
    def test_gradient_flow(self):
        """验证梯度可以正确回传"""
        model = CausalQwen(config)
        
        # 准备数据
        input_ids = torch.randint(0, 151936, (2, 16))
        numeric_values = torch.zeros(2, 16)
        labels = torch.randint(0, 151936, (2, 16))
        
        # 添加一些数值位置
        input_ids[0, 5] = num_token_id
        numeric_values[0, 5] = 10.5
        labels[0, 5] = num_token_id
        
        # 前向传播
        output = model(input_ids, numeric_values)
        
        # 计算损失
        loss = compute_total_loss(output, labels, numeric_values)
        
        # 反向传播
        loss.backward()
        
        # 验证梯度存在
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
```

#### 6.2 数值稳定性压力测试

```python
def test_numerical_stability_stress(self):
    """压力测试数值稳定性"""
    model = CausalQwen(config)
    model.eval()
    
    # 极端输入
    extreme_cases = [
        # (description, numeric_values)
        ("极小值", torch.tensor([[1e-10, 1e-8, 1e-6]])),
        ("极大值", torch.tensor([[1e10, 1e12, 1e15]])),
        ("混合值", torch.tensor([[-1e10, 0.0, 1e10]])),
        ("高精度", torch.tensor([[3.141592653589793, 2.718281828459045, 1.414213562373095]])),
    ]
    
    for desc, values in extreme_cases:
        print(f"测试: {desc}")
        
        batch_size, seq_len = values.shape
        input_ids = torch.full((batch_size, seq_len), num_token_id)
        
        with torch.no_grad():
            output = model(input_ids, values)
        
        # 验证输出有效性
        for key in ['loc_S', 'scale_S', 'loc_Y', 'scale_Y']:
            tensor = output[key]
            assert not torch.isnan(tensor).any(), f"{key} 包含 NaN"
            assert not torch.isinf(tensor).any(), f"{key} 包含 Inf"
            
        # 验证概率计算
        probs = compute_ovr_probabilities(
            output['loc_S'], 
            output['scale_S'], 
            torch.zeros(vocab_size)
        )
        assert (probs >= 0).all()
        assert (probs <= 1).all()
```

### 7. 采样一致性测试

```python
class TestSamplingConsistency:
    def test_deterministic_vs_sampling_mode(self):
        """验证确定性模式与采样模式的一致性"""
        model = CausalQwen(config)
        model.eval()
        
        input_ids = torch.tensor([[1234, 5678, num_token_id]])
        numeric_values = torch.tensor([[0.0, 0.0, 50.0]])
        
        with torch.no_grad():
            output = model(input_ids, numeric_values)
        
        # 确定性预测
        det_cls = output['loc_S'].argmax(dim=-1)
        det_reg = output['loc_Y']
        
        # 采样多次
        n_samples = 10000
        sampled_cls_counts = torch.zeros_like(output['loc_S'])
        sampled_reg_sum = torch.zeros_like(output['loc_Y'])
        
        for _ in range(n_samples):
            # 因果采样
            u_sample = sample_cauchy(output['loc_U'], output['scale_U'])
            cls_scores = model.action_network.classify(u_sample)
            reg_values = model.action_network.regress(u_sample)
            
            # 累计
            cls_pred = cls_scores.argmax(dim=-1)
            for i in range(batch_size):
                for j in range(seq_len):
                    sampled_cls_counts[i, j, cls_pred[i, j]] += 1
            sampled_reg_sum += reg_values
        
        # 验证分类模式
        sampled_cls_mode = sampled_cls_counts.argmax(dim=-1)
        assert (sampled_cls_mode == det_cls).float().mean() > 0.9
        
        # 验证回归中位数
        sampled_reg_median = sampled_reg_sum / n_samples
        assert torch.allclose(sampled_reg_median, det_reg, rtol=0.1)
```

## 🔧 测试工具函数

### 辅助函数实现

```python
# test_utils.py

def compute_phi(v, direction):
    """计算单个数值的编码"""
    if v == 0:
        return torch.zeros_like(direction)
    return np.sign(v) * np.log(1 + abs(v)) * direction

def compute_cauchy_nll(y_true, loc, scale):
    """计算柯西分布的负对数似然"""
    z = (y_true - loc) / scale
    return torch.log(np.pi * scale) + torch.log(1 + z**2)

def compute_ovr_probabilities(loc_s, scale_s, thresholds):
    """计算OvR概率"""
    # P(S > C) = 1/2 + (1/π)arctan((loc-C)/scale)
    z = (loc_s - thresholds) / scale_s
    return 0.5 + torch.atan(z) / np.pi

def sample_cauchy(loc, scale):
    """从柯西分布采样"""
    u = torch.rand_like(loc)
    return loc + scale * torch.tan(np.pi * (u - 0.5))
```

## 📊 测试执行策略

### 分阶段测试

```bash
# 第一阶段：基础组件
pytest tests/test_numerical_encoding.py -v
pytest tests/test_enhanced_embedding.py -v

# 第二阶段：分布计算
pytest tests/test_cauchy_distribution.py -v
pytest tests/test_linear_transformation.py -v

# 第三阶段：网络组件
pytest tests/test_network_initialization.py -v
pytest tests/test_weight_transfer.py -v

# 第四阶段：损失函数
pytest tests/test_loss_functions.py -v
pytest tests/test_gated_mechanism.py -v

# 第五阶段：集成测试
pytest tests/test_end_to_end.py -v
pytest tests/test_numerical_stability.py -v
```

### 测试报告生成

```bash
# 生成详细的测试报告
pytest tests/ --html=report.html --self-contained-html

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## 🎯 关键验证点检查清单

- [ ] φ(0) 精确等于零向量（误差 < 1e-10）
- [ ] φ(-v) = -φ(v) 对所有测试值成立
- [ ] 增强嵌入在非数值位置精确等于基础嵌入
- [ ] 柯西分布的线性变换公式精确成立
- [ ] OvR概率严格在 [0,1] 区间
- [ ] 柯西NLL公式与手动计算完全一致
- [ ] 门控机制在α=0和α=1时的行为正确
- [ ] 梯度可以正确回传到所有参数
- [ ] 极端数值输入不产生NaN或Inf
- [ ] 批处理结果与逐个处理完全一致

## 🚨 常见实现错误

1. **数值编码的符号处理错误**
   ```python
   # ❌ 错误：忘记处理负数
   phi = torch.log(1 + v) * direction
   
   # ✅ 正确：包含符号
   phi = torch.sign(v) * torch.log(1 + torch.abs(v)) * direction
   ```

2. **柯西分布scale参数约束**
   ```python
   # ❌ 错误：scale可能为负
   scale = linear(features)
   
   # ✅ 正确：确保scale > 0
   scale = F.softplus(linear(features)) + eps
   ```

3. **OvR概率计算错误**
   ```python
   # ❌ 错误：使用sigmoid
   prob = torch.sigmoid((loc - threshold) / scale)
   
   # ✅ 正确：使用柯西CDF
   prob = 0.5 + torch.atan((loc - threshold) / scale) / np.pi
   ```

## 📝 测试日志模板

```python
# 每个测试应该记录
def test_something():
    """测试描述"""
    logger.info("="*50)
    logger.info(f"测试: {test_name}")
    logger.info(f"输入: {input_description}")
    
    # 执行测试
    result = perform_test()
    
    logger.info(f"期望: {expected}")
    logger.info(f"实际: {actual}")
    logger.info(f"误差: {error}")
    logger.info(f"结果: {'通过' if passed else '失败'}")
    logger.info("="*50)
```

## 🎯 下一步行动

1. **立即执行**：按照测试清单逐项验证每个数学组件
2. **记录问题**：发现的任何偏差都要详细记录
3. **修复验证**：修复后必须重新运行相关测试
4. **回归测试**：任何修改都要运行完整测试套件

记住：**没有通过测试的代码就是错误的代码！**
