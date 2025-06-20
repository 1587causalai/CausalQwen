# CausalQwen 测试脚本索引

> **🎯 最小可行版本**: 专注核心CausalQwen V2数学框架验证  
> **📅 更新**: 2024年6月17日  
> **🚀 精简状态**: 仅4个核心测试脚本

---

## 🔥 核心测试脚本（仅4个）

### 1. 核心数学框架验证 ⭐
```bash
python scripts/test_core_math_framework.py
```
**用途**: 验证CausalQwen核心数学原理  
**核心验证**:
- ✅ 柯西分布线性稳定性 
- ✅ ActionNetwork双模式（位置vs尺度差异）
- ✅ do_sample参数控制噪声影响方式
- ✅ 温度参数选择性生效

### 2. Qwen接口兼容性验证 ⭐
```bash
python scripts/test_qwen_interface_compatibility.py
```
**用途**: 验证与Qwen的完全兼容性  
**验证点**: 
- ✅ generate()接口完全兼容
- ✅ do_sample, temperature, top_k, top_p参数
- ✅ 批量生成功能

### 3. 与原版Qwen对比测试 ⭐
```bash
python scripts/test_vs_original_qwen.py
```
**用途**: CausalQwen vs 原始Qwen直接对比  
**需求**: 本地需有Qwen2.5-0.5B模型  
**关键验证**:
- 确定性模式：loc_S与Qwen logits一致性
- 采样模式多样性
- 数学框架正确性

### 4. 基本使用演示 ⭐
```bash
python scripts/demo_basic_usage.py
```
**用途**: 展示CausalQwen基本使用方法  
**特点**: 与Qwen完全相同的使用接口

### 5. 模块化架构演示 🆕 (v2.0.3)
```bash
python scripts/demo_modular_architecture.py
```
**用途**: 展示CausalEngine v2.0.3的模块化架构  
**亮点**:
- ✨ 三大独立模块：AbductionNetwork → ActionNetwork → ActivationHead
- ✨ 独立网络架构：loc_net 和 scale_net 完全解耦
- ✨ 智能初始化：H=C时loc_net恒等初始化
- ✨ 梯度独立：位置和尺度参数分离优化
- ✨ 混合激活：同时支持分类和回归输出
- ✨ 多任务头：支持多模态应用

### 6. CauchyMath工具类测试 🆕 (v2.0.5)
```bash
python scripts/test_cauchy_math.py
```
**用途**: 验证CauchyMath工具类的完整数学函数库  
**核心验证**:
- ✅ 基本分布函数：PDF, CDF, 生存函数
- ✅ 分位函数与CDF互逆性
- ✅ 数值稳定性和极端值处理
- ✅ 采样一致性验证
- ✅ 可视化验证

### 7. CauchyMath演示 🆕 (v2.0.5)
```bash
python scripts/demo_cauchy_math.py
```
**用途**: 展示CauchyMath工具类的实际使用  
**功能**: 演示概率计算、分位函数、采样和线性稳定性

---

## 🎯 推荐测试流程

### 🚀 快速验证（3分钟）
```bash
python scripts/test_core_math_framework.py
python scripts/demo_basic_usage.py
```

### 🔬 完整验证（10分钟）
```bash
# 所有核心测试
python scripts/test_core_math_framework.py
python scripts/test_qwen_interface_compatibility.py  
python scripts/demo_basic_usage.py
python scripts/test_vs_original_qwen.py  # 需要Qwen模型
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

**完美平衡**: 突破性数学创新 + 零学习成本使用！