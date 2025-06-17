# CausalQwen 测试脚本索引

> **🎯 最小可行版本**: 专注核心CausalQwen V2数学框架验证  
> **📅 更新**: 2024年6月17日  
> **🚀 精简状态**: 仅4个核心测试脚本

---

## 🔥 核心测试脚本（仅4个）

### 1. V2数学原理验证 ⭐
```bash
python scripts/causal_qwen_v2_validation_test.py
```
**用途**: 验证CausalQwen V2核心数学原理  
**核心验证**:
- ✅ 柯西分布线性稳定性 
- ✅ ActionNetwork双模式（位置vs尺度差异）
- ✅ do_sample参数控制噪声影响方式
- ✅ 温度参数选择性生效

### 2. Qwen兼容性验证 ⭐
```bash
python scripts/qwen_compatibility_test.py
```
**用途**: 验证与Qwen的完全兼容性  
**验证点**: 
- ✅ generate()接口完全兼容
- ✅ do_sample, temperature, top_k, top_p参数
- ✅ 批量生成功能

### 3. 端到端对比测试 ⭐
```bash
python scripts/end_to_end_comparison_test_v2.py
```
**用途**: CausalQwen vs 原始Qwen直接对比  
**需求**: 本地需有Qwen2.5-0.5B模型  
**关键验证**:
- 确定性模式：loc_S与Qwen logits一致性
- 采样模式多样性
- 数学框架正确性

### 4. 基本使用演示 ⭐
```bash
python scripts/simple_demo_v2.py
```
**用途**: 展示CausalQwen基本使用方法  
**特点**: 与Qwen完全相同的使用接口

---

## 🎯 推荐测试流程

### 🚀 快速验证（3分钟）
```bash
python scripts/causal_qwen_v2_validation_test.py
python scripts/simple_demo_v2.py
```

### 🔬 完整验证（10分钟）
```bash
# 所有核心测试
python scripts/causal_qwen_v2_validation_test.py
python scripts/qwen_compatibility_test.py  
python scripts/simple_demo_v2.py
python scripts/end_to_end_comparison_test_v2.py  # 需要Qwen模型
```

---

## 📋 测试结果预期

### ✅ 核心指标

- **位置参数差异**: > 2.0（do_sample模式差异显著）
- **柯西数学精度**: < 1e-6误差
- **Qwen兼容性**: 100%接口兼容
- **logits一致性**: loc_S与Qwen logits差异 < 1e-4

### 🎯 成功输出示例
```
🎯 V2核心创新验证：do_sample控制的位置vs尺度差异
✅ ActionNetwork统一框架：兼容Qwen的所有参数
✅ 温度参数选择性生效：仅在do_sample=True时影响噪声强度
✅ 柯西分布线性稳定性：严格的数学基础实现
✅ 完全Qwen兼容：generate()接口和所有采样参数
```

---

## 🧮 CausalQwen V2核心数学

### 位置vs尺度的精妙差异
```
do_sample=False: U' ~ Cauchy(μ, γ + |b_noise|)     # 噪声影响尺度参数  
do_sample=True:  U' ~ Cauchy(μ + T·|b_noise|·ε, γ) # 噪声影响位置参数
```

### 使用方式（与Qwen完全相同）
```python
from causal_qwen_mvp import CausalQwenMVPForCausalLM

# 确定性生成（噪声影响尺度）
output = model.generate(input_ids, do_sample=False)

# 采样生成（噪声影响位置） 
output = model.generate(input_ids, do_sample=True, temperature=0.8)
```

---

## 🎉 总结

**CausalQwen MVP**: 最小可行的因果语言模型
- **核心哲学**: 位置vs尺度的数学差异
- **完全兼容**: 与Qwen接口100%相同
- **精简验证**: 仅4个核心测试脚本
- **数学严谨**: 基于柯西分布线性稳定性

**完美平衡**: 革命性数学创新 + 零学习成本使用！