# CausalQwen 推理模块验证文档

> **📋 文档用途**: 推理模块验证，代码与数学对照用  
> **🎯 目标读者**: 项目负责人，用于验证推理实现的数学正确性  
> **📖 内容定位**: 代码实现与数学公式的逐一对照验证

## 1. 推理模块总览

### 1.1 代码结构
```python
# src/causal_qwen_mvp/inference.py
class CausalInferenceEngine:
    def inference(self, input_ids, mode='standard', **kwargs)
    def _standard_inference(self, input_ids, **kwargs)
    def _causal_sampling(self, input_ids, **kwargs)  
    def _compatible_sampling(self, input_ids, **kwargs)
```

### 1.2 三种推理模式对应的数学原理
- **标准推理**: 基于分布期望的确定性决策
- **因果采样**: 从个体分布采样具体个体后决策
- **兼容采样**: 传统Softmax采样（兼容性模式）

---

## 2. 模式一：标准推理 (Standard Inference)

### 2.1 数学定义
$$P_{k,i} = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

$$\hat{y}_i = \arg\max_k P_{k,i}$$

### 2.2 代码实现
```python
def _standard_inference(self, input_ids, **kwargs):
    """标准确定性推理：使用期望值计算"""
    with torch.no_grad():
        outputs = self.model(input_ids, **kwargs)
        # 计算OvR概率
        probs = self.model.ovr_classifier(outputs.loc_S, outputs.scale_S)
        # 选择概率最高的token
        next_token_ids = torch.argmax(probs, dim=-1)
        return next_token_ids[:, -1:]
```

### 2.3 验证要点
- [ ] **OvR概率计算**: `self.model.ovr_classifier()` 是否正确实现了arctan公式
- [ ] **最大值选择**: `torch.argmax(probs, dim=-1)` 对应数学上的argmax操作
- [ ] **输出形状**: 返回 `[B, 1]` 形状的下一词元预测

---

## 3. 模式二：因果采样 (Causal Sampling)

### 3.1 数学定义
第一步：采样个体表征
$$u_i \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i})$$

第二步：线性决策（确定性）
$$S_{k,i} = W_{\text{cls},k} \cdot u_i + b_{\text{cls},k}$$

第三步：选择最大值
$$\hat{y}_i = \arg\max_k S_{k,i}$$

### 3.2 代码实现
```python
def _causal_sampling(self, input_ids, **kwargs):
    """个体因果采样：从个体分布采样后决策"""
    with torch.no_grad():
        outputs = self.model(input_ids, **kwargs)
        
        # 第一步：从个体表征分布采样
        cauchy_U = Cauchy(outputs.loc_U, outputs.scale_U)
        u_sample = cauchy_U.sample()  
        
        # 外生噪声融合（当前实现未使用）
        scale_U_noisy = outputs.scale_U + torch.abs(self.model.action_network.b_noise)
        
        # 第二步：通过ActionNetwork的线性变换
        loc_S_sample = self.model.action_network.lm_head(u_sample)
        
        # 第三步：选择最大值对应的token
        next_token_ids = torch.argmax(loc_S_sample, dim=-1)
        return next_token_ids[:, -1:]
```

### 3.3 验证要点
- [ ] **柯西采样**: `Cauchy(outputs.loc_U, outputs.scale_U).sample()` 正确实现
- [ ] **线性变换**: `self.model.action_network.lm_head(u_sample)` 对应数学公式
- [ ] **噪声融合**: 当前代码计算了 `scale_U_noisy` 但未使用，需要确认是否符合预期
- [ ] **确定性决策**: 采样后的决策应该是确定性的

---

## 4. 模式三：兼容采样 (Compatible Sampling)

### 4.1 数学定义
$$P_{\text{softmax}}(y_i=k|x) = \frac{\exp(\text{loc}_{S_{k,i}} / T)}{\sum_{j=1}^{V} \exp(\text{loc}_{S_{j,i}} / T)}$$

其中 $T$ 是温度参数。

### 4.2 代码实现
```python
def _compatible_sampling(self, input_ids, top_k=50, top_p=0.9, temperature=1.0, **kwargs):
    """传统兼容采样：将位置参数作为logits采样"""
    with torch.no_grad():
        outputs = self.model(input_ids, **kwargs)
        logits = outputs.loc_S[:, -1, :] / temperature  # 温度缩放
        
        # 简化的top-k采样
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_indices = torch.multinomial(probs, 1)
            next_token_ids = top_k_indices.gather(-1, sampled_indices)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token_ids = torch.multinomial(probs, 1)
        
        return next_token_ids
```

### 4.3 验证要点
- [ ] **温度缩放**: `logits / temperature` 正确应用
- [ ] **Softmax计算**: `F.softmax(top_k_logits, dim=-1)` 实现归一化
- [ ] **Top-k选择**: `torch.topk(logits, top_k)` 正确提取前k个
- [ ] **多项式采样**: `torch.multinomial(probs, 1)` 按概率采样

---

## 5. 自回归生成验证

### 5.1 数学定义
自回归生成过程：
$$y_{t+1} = f(x_1, x_2, \ldots, x_t, y_1, y_2, \ldots, y_t)$$

### 5.2 代码实现
```python
def generate_step_by_step(self, input_ids, max_length=50, mode='standard', **kwargs):
    """自回归生成循环"""
    current_ids = input_ids.clone()
    
    for step in range(max_length):
        # 预测下一个token
        next_token = self.inference(current_ids, mode=mode, **kwargs)
        
        # 添加到序列中
        current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        # 检查停止条件
        if next_token[0, -1].item() in stop_tokens:
            break
            
        if current_ids.shape[-1] >= max_length:
            break
    
    return current_ids
```

### 5.3 验证要点
- [ ] **序列更新**: `torch.cat([current_ids, next_token], dim=-1)` 正确拼接
- [ ] **停止条件**: EOS token检查和最大长度限制
- [ ] **模式一致性**: 每步使用相同的推理模式

---

## 6. 关键数学问题待验证

### 6.1 噪声融合问题
**代码中的问题**：
```python
# 计算了但未使用
scale_U_noisy = outputs.scale_U + torch.abs(self.model.action_network.b_noise)
```

**期望的数学实现**：
$$U'_i \sim \text{Cauchy}(\text{loc}_{U_i}, \text{scale}_{U_i} + |b_{\text{noise}}|)$$

**验证要求**：
- [ ] 确认是否需要在因果采样中实际使用 `scale_U_noisy`
- [ ] 或者这种实现方式是否符合数学定义

### 6.2 OvR分类器实现
**需要验证**：
```python
probs = self.model.ovr_classifier(outputs.loc_S, outputs.scale_S)
```

**对应数学公式**：
$$P_{k,i} = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{\text{loc}_{S_{k,i}} - C_k}{\text{scale}_{S_{k,i}}}\right)$$

**验证要求**：
- [ ] 确认 `ovr_classifier` 中阈值参数 $C_k$ 的实现
- [ ] 验证arctan计算的数值稳定性

### 6.3 采样一致性
**验证目标**：
- [ ] 标准推理和因果采样在相同个体下是否产生一致结果
- [ ] 兼容采样是否真正兼容传统语言模型

---

## 7. 验证检查清单

### 7.1 数学正确性
- [ ] 柯西分布采样实现正确
- [ ] OvR概率计算公式正确
- [ ] 线性变换维度匹配
- [ ] 噪声融合数学一致

### 7.2 实现完整性  
- [ ] 三种推理模式都能正常工作
- [ ] 自回归生成循环正确
- [ ] 错误处理和边界情况

### 7.3 性能合理性
- [ ] 推理速度可接受
- [ ] 内存使用合理
- [ ] 支持批量处理

---

## 8. 待补充的实现

根据代码中的TODO注释，以下功能待完善：

1. **标准推理优化**: "实现更sophisticated的确定性推理"
2. **兼容采样集成**: "集成transformers库的采样函数"  
3. **生成控制**: "添加更多生成控制选项"
4. **批量优化**: "实现高效的批量生成"
5. **验证工具**: "实现更comprehensive的一致性检查"

这些需要在后续开发中逐步完善。 