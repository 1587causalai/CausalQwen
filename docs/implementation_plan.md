# CausalQwen 实现计划

> 本文档基于 `design-docs/causal_qwen_text_only.md` 的理论基础，制定详细的代码实现计划，确保理论创新能够准确、高效地转化为可工作的系统。

## 1. 实现目标与原则

### 1.1 核心目标

1. **理论忠实性**：确保代码实现完全符合因果推断理论，特别是个体选择变量 U 的双重身份
2. **计算高效性**：充分利用柯西分布的线性稳定性，实现无采样的解析前向传播
3. **工程优雅性**：模块化设计，清晰的接口定义，便于测试和扩展
4. **知识继承性**：最大程度复用预训练 Qwen 的能力

### 1.2 设计原则

- **理论驱动**：每个实现细节都有明确的理论依据
- **渐进式验证**：从核心模块到完整系统，逐步验证
- **可测试性**：每个模块独立可测，便于定位问题
- **可扩展性**：为未来的数值处理等功能预留接口

## 2. 模块实现计划

### 2.1 核心数学工具层 (`src/utils/cauchy_math.py`)

**目标**：实现柯西分布的所有数学操作，作为整个系统的数学基础。

```python
# 需要实现的核心函数
class CauchyMath:
    @staticmethod
    def linear_transform(loc, scale, weight, bias=None):
        """柯西分布的线性变换
        Y = w*X + b, where X ~ Cauchy(loc, scale)
        Returns: (new_loc, new_scale)
        """
        
    @staticmethod
    def add_cauchy(loc1, scale1, loc2, scale2):
        """两个柯西分布的和
        X1 ~ Cauchy(loc1, scale1), X2 ~ Cauchy(loc2, scale2)
        Returns: (sum_loc, sum_scale) for X1 + X2
        """
        
    @staticmethod
    def ovr_probability(loc, scale, threshold):
        """计算 P(X > threshold) where X ~ Cauchy(loc, scale)
        使用 arctan 的解析公式
        """
        
    @staticmethod
    def sample_cauchy(loc, scale, epsilon=None):
        """柯西分布采样（用于推理）
        如果提供 epsilon，使用重参数化；否则生成新的随机数
        """
```

**验证计划**：
- 单元测试：验证线性变换的数学正确性
- 数值稳定性测试：极端值情况下的表现
- 与 PyTorch 分布的对比验证

### 2.2 归因推断网络 (`src/models/abduction_network.py`)

**目标**：实现从证据推断个体子群体分布的核心模块。

```python
class AbductionNetwork(nn.Module):
    """从观察证据推断个体因果表征分布
    
    理论基础：
    - 输入：上下文特征 z (来自 Qwen)
    - 输出：个体子群体分布 U ~ Cauchy(μ, γ)
    - μ 表示群体的典型代表
    - γ 表示群体内部的多样性
    """
    
    def __init__(self, hidden_size: int, causal_size: int = None):
        super().__init__()
        causal_size = causal_size or hidden_size  # 默认 C = H
        
        # 双头网络架构
        self.loc_net = nn.Linear(hidden_size, causal_size)
        self.scale_net = nn.Linear(hidden_size, causal_size)
        
        # 精心设计的初始化
        self._init_weights()
    
    def _init_weights(self):
        """知识继承初始化策略
        目标：初始时直接使用 Qwen 特征作为个体表征
        """
        # loc_net: 恒等映射
        nn.init.eye_(self.loc_net.weight)
        nn.init.zeros_(self.loc_net.bias)
        
        # scale_net: 常数输出
        nn.init.zeros_(self.scale_net.weight)
        nn.init.constant_(self.scale_net.bias, 0.0)  # softplus(0) ≈ 0.693
```

**实现要点**：
- 位置网络输出无约束，表示任意个体中心
- 尺度网络使用 softplus 确保正值
- 初始化策略确保与 Qwen 行为一致

### 2.3 行动决策网络 (`src/models/action_network.py`)

**目标**：实现普适因果律，将个体表征映射为词汇决策。

```python
class ActionNetwork(nn.Module):
    """普适因果律：Y = f(U, ε)
    
    理论基础：
    - 输入：个体表征分布 U ~ Cauchy(loc_U, scale_U)
    - 噪声：外生扰动 ε ~ Cauchy(0, |b_noise|)
    - 输出：决策分布 S_k ~ Cauchy(loc_S_k, scale_S_k)
    """
    
    def __init__(self, causal_size: int, vocab_size: int):
        super().__init__()
        
        # 线性因果律（继承自 Qwen）
        self.lm_head = nn.Linear(causal_size, vocab_size, bias=True)
        
        # 外生噪声参数（可学习）
        self.b_noise = nn.Parameter(torch.zeros(causal_size))
        
        self._init_weights()
    
    def forward(self, loc_U: Tensor, scale_U: Tensor) -> Tuple[Tensor, Tensor]:
        """应用普适因果律
        
        步骤：
        1. 外生噪声注入：U' = U + ε
        2. 线性决策：S_k = W_k · U' + b_k
        3. 不确定性传播：利用柯西分布的线性稳定性
        """
        # Step 1: 噪声融合（利用柯西分布的加法稳定性）
        scale_U_noisy = scale_U + torch.abs(self.b_noise)
        
        # Step 2: 线性因果律
        loc_S = self.lm_head(loc_U)
        
        # Step 3: 不确定性传播
        W_abs = torch.abs(self.lm_head.weight)  # [V, C]
        scale_S = torch.matmul(scale_U_noisy, W_abs.T)  # [B, S, V]
        
        return loc_S, scale_S
```

**关键设计**：
- 外生噪声通过加法注入，保持柯西分布形式
- 权重绝对值用于尺度参数传播
- 继承 Qwen 的 lm_head 权重

### 2.4 OvR 分类层 (`src/models/ovr_classifier.py`)

**目标**：实现独立的二元判断机制，替代传统 Softmax。

```python
class OvrClassifier(nn.Module):
    """One-vs-Rest 分类器
    
    理论优势：
    - 每个词汇独立判断，无归一化约束
    - 自然支持不确定性表达
    - 可以所有词汇概率都低（高不确定性）
    """
    
    def __init__(self, vocab_size: int, threshold_init: float = 10.0):
        super().__init__()
        
        # 可学习的决策阈值
        self.thresholds = nn.Parameter(torch.ones(vocab_size) * threshold_init)
    
    def compute_probabilities(self, loc_S: Tensor, scale_S: Tensor) -> Tensor:
        """计算 P(S_k > threshold_k)
        
        使用柯西分布的 CDF 解析公式
        """
        # 标准化差值
        z = (loc_S - self.thresholds) / scale_S
        
        # 柯西 CDF: P(X > t) = 0.5 + arctan(z) / π
        probs = 0.5 + torch.atan(z) / math.pi
        
        return probs
```

### 2.5 因果语言模型主体 (`src/models/causal_qwen.py`)

**目标**：组装所有模块，实现完整的因果语言模型。

```python
class CausalQwen(nn.Module):
    """因果语言模型主体
    
    架构：
    1. 词元嵌入（继承 Qwen）
    2. 特征提取（Qwen Transformer）
    3. 归因推断（推断个体分布）
    4. 行动决策（应用因果律）
    5. OvR 分类（独立判断）
    """
    
    def __init__(self, qwen_model_path: str, config: CausalConfig = None):
        super().__init__()
        
        # 加载预训练 Qwen
        self.qwen = AutoModel.from_pretrained(qwen_model_path)
        hidden_size = self.qwen.config.hidden_size
        vocab_size = self.qwen.config.vocab_size
        
        # 因果模块
        self.abduction = AbductionNetwork(hidden_size)
        self.action = ActionNetwork(hidden_size, vocab_size)
        self.ovr = OvrClassifier(vocab_size)
        
        # 初始化：继承 Qwen 知识
        self._inherit_knowledge()
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor = None):
        """完整的因果推理流程"""
        # 1. 词元嵌入
        embeddings = self.qwen.embed_tokens(input_ids)
        
        # 2. 特征提取
        outputs = self.qwen.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        features = outputs.last_hidden_state  # [B, S, H]
        
        # 3. 归因推断：推断个体分布
        loc_U, scale_U = self.abduction(features)
        
        # 4. 行动决策：应用因果律
        loc_S, scale_S = self.action(loc_U, scale_U)
        
        # 5. OvR 概率
        probs = self.ovr.compute_probabilities(loc_S, scale_S)
        
        return CausalOutput(
            loc_S=loc_S,
            scale_S=scale_S,
            loc_U=loc_U,
            scale_U=scale_U,
            probs=probs
        )
```

## 3. 推理实现计划

### 3.1 标准推理 (`src/inference/standard.py`)

```python
def standard_inference(model: CausalQwen, input_ids: Tensor) -> Tensor:
    """基于期望的确定性推理
    
    直接使用分布参数，无需采样
    """
    with torch.no_grad():
        outputs = model(input_ids)
        
        # 选择最高 OvR 概率的词汇
        next_token = torch.argmax(outputs.probs[:, -1, :], dim=-1)
        
    return next_token
```

### 3.2 因果采样 (`src/inference/causal.py`)

```python
def causal_sampling(model: CausalQwen, input_ids: Tensor, 
                   shared_epsilon: Optional[Tensor] = None) -> Tensor:
    """因果采样：具现特定个体
    
    Args:
        shared_epsilon: 如果提供，使用共享的随机性（实现一致性生成）
    """
    with torch.no_grad():
        outputs = model(input_ids)
        
        # 从个体分布采样
        if shared_epsilon is None:
            epsilon = torch.rand_like(outputs.loc_U)
        else:
            epsilon = shared_epsilon
            
        # 重参数化采样
        u = outputs.loc_U + outputs.scale_U * torch.tan(math.pi * (epsilon - 0.5))
        
        # 应用因果律（需要重新前向传播 action network）
        loc_S_sampled = model.action.lm_head(u)
        
        # 计算采样后的 OvR 概率
        # 注意：这里 scale 变为 0（确定性个体）
        probs_sampled = model.ovr.compute_probabilities(
            loc_S_sampled, 
            torch.ones_like(loc_S_sampled) * 1e-6  # 小的 scale
        )
        
        next_token = torch.argmax(probs_sampled[:, -1, :], dim=-1)
        
    return next_token, epsilon  # 返回 epsilon 以支持共享
```

### 3.3 序列生成 (`src/inference/generation.py`)

```python
class CausalGenerator:
    """因果文本生成器
    
    支持多种生成模式：
    1. 标准推理（基于期望）
    2. 因果采样（具现个体）
    3. 共享个体采样（一致性生成）
    """
    
    def generate(self, model: CausalQwen, prompt: str, 
                max_length: int = 100,
                mode: str = "standard",
                shared_individual: bool = False) -> str:
        """生成文本
        
        Args:
            mode: "standard" | "causal" | "traditional"
            shared_individual: 是否在整个序列中共享个体
        """
        # 实现自回归生成循环
        pass
```

## 4. 训练实现计划

### 4.1 损失函数 (`src/losses/ovr_loss.py`)

```python
class OvrLoss(nn.Module):
    """OvR 分类损失
    
    对每个词汇计算独立的二元交叉熵
    """
    
    def forward(self, probs: Tensor, targets: Tensor, mask: Tensor = None):
        """
        Args:
            probs: [B, S, V] OvR 概率
            targets: [B, S] 目标词汇索引
            mask: [B, S] 有效位置掩码
        """
        # 创建 one-hot 目标
        target_one_hot = F.one_hot(targets, probs.size(-1)).float()
        
        # 二元交叉熵（数值稳定版本）
        bce = -(target_one_hot * torch.log(probs + 1e-10) + 
                (1 - target_one_hot) * torch.log(1 - probs + 1e-10))
        
        # 应用掩码
        if mask is not None:
            bce = bce * mask.unsqueeze(-1)
            loss = bce.sum() / mask.sum()
        else:
            loss = bce.mean()
            
        return loss
```

### 4.2 训练循环 (`src/training/trainer.py`)

```python
class CausalTrainer:
    """因果语言模型训练器"""
    
    def __init__(self, model: CausalQwen, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # 优化器：不同模块使用不同学习率
        self.optimizer = self._create_optimizer()
        
    def _create_optimizer(self):
        """分组优化策略"""
        return torch.optim.AdamW([
            # Qwen 参数：小学习率或冻结
            {'params': self.model.qwen.parameters(), 'lr': 1e-5},
            # 因果模块：正常学习率
            {'params': self.model.abduction.parameters(), 'lr': 1e-4},
            {'params': self.model.action.parameters(), 'lr': 1e-4},
            {'params': self.model.ovr.parameters(), 'lr': 1e-4},
        ])
```

## 5. 验证与测试计划

### 5.1 数学正确性验证

```python
# tests/test_cauchy_math.py
class TestCauchyMath:
    def test_linear_transform(self):
        """验证线性变换的数学正确性"""
        
    def test_addition_stability(self):
        """验证加法稳定性"""
        
    def test_ovr_probability(self):
        """验证 OvR 概率计算"""
```

### 5.2 初始化验证

```python
# tests/test_initialization.py
def test_qwen_compatibility():
    """验证初始化后与 Qwen 的行为一致性"""
    causal_model = CausalQwen("Qwen/Qwen-0.5B")
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-0.5B")
    
    # 比较 logits
    # 初始时 loc_S 应该与 Qwen logits 高度相似
```

### 5.3 因果特性验证

```python
# tests/test_causal_properties.py
def test_individual_consistency():
    """验证相同个体的生成一致性"""
    
def test_causal_vs_standard():
    """对比因果采样与标准推理的差异"""
```

## 6. 性能优化计划

### 6.1 计算优化

- **无采样前向传播**：充分利用解析计算
- **向量化操作**：避免逐位置循环
- **混合精度训练**：使用 AMP 加速

### 6.2 内存优化

- **梯度检查点**：对 Qwen Transformer 使用
- **参数冻结**：初期训练时冻结 Qwen 参数
- **批处理策略**：动态批大小

## 7. 监控与调试

### 7.1 训练监控指标

```python
# 需要监控的关键指标
metrics = {
    # 损失相关
    'train/loss': ovr_loss,
    'train/accuracy': token_accuracy,
    
    # 分布相关
    'dist/U_loc_mean': loc_U.mean(),
    'dist/U_scale_mean': scale_U.mean(),
    'dist/ovr_prob_entropy': -（probs * probs.log()).sum(-1).mean(),
    
    # 梯度相关
    'grad/abduction_norm': abduction_grad_norm,
    'grad/action_norm': action_grad_norm,
}
```

### 7.2 可视化工具

```python
# src/visualization/causal_viz.py
def visualize_individual_space(loc_U, scale_U):
    """可视化个体表征空间"""
    
def plot_ovr_probabilities(probs, targets):
    """可视化 OvR 概率分布"""
```

## 8. 实施时间表

### 第一阶段：核心实现（2周）
- [ ] Week 1: 数学工具层 + 核心模块
- [ ] Week 2: 推理模式 + 基础训练

### 第二阶段：验证优化（2周）
- [ ] Week 3: 完整测试套件 + 性能优化
- [ ] Week 4: 实验验证 + 文档完善

### 第三阶段：扩展功能（可选）
- [ ] 数值处理能力
- [ ] 高级采样模式
- [ ] 预训练对齐

## 9. 风险与缓解

### 9.1 技术风险

1. **数值稳定性**
   - 风险：极端值导致数值溢出
   - 缓解：使用 log-space 计算，梯度裁剪

2. **训练不收敛**
   - 风险：OvR 损失难以优化
   - 缓解：仔细的初始化，学习率调度

3. **性能瓶颈**
   - 风险：柯西计算开销大
   - 缓解：优化实现，使用 JIT 编译

### 9.2 理论风险

1. **因果假设不成立**
   - 风险：线性因果律过于简化
   - 缓解：保留扩展到非线性的接口

2. **柯西分布局限性**
   - 风险：重尾特性导致不稳定
   - 缓解：引入截断或其他稳定机制

## 10. 成功标准

实现被认为成功当且仅当：

1. **理论验证**：所有数学性质通过单元测试
2. **行为验证**：初始化后与 Qwen 输出差异 < 0.01
3. **训练验证**：在标准数据集上达到合理的困惑度
4. **因果验证**：展现出明确的个体一致性特征
5. **性能验证**：推理速度不低于原始 Qwen 的 50%

## 总结

本实现计划将 CausalQwen 的理论创新转化为具体的工程实践。通过模块化设计、渐进式验证和严格的测试，我们将构建一个既忠实于理论又高效实用的因果语言模型。这不仅是技术实现，更是对"语言生成本质"全新理解的具体体现。
