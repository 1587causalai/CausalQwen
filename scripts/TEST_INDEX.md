# CausalQwen 测试脚本索引

> **🎯 目的**: 快速找到和运行相关测试脚本  
> **📅 更新**: 2024年6月17日  
> **🚀 状态**: 所有脚本已更新为Qwen兼容接口

---

## 🔥 核心推荐测试（必运行）

### 1. Qwen兼容性验证
```bash
python scripts/qwen_compatibility_test.py
```
**用途**: 验证CausalQwen与Qwen的完全兼容性  
**验证点**: 
- ✅ generate()接口兼容性
- ✅ do_sample参数控制
- ✅ 温度和采样参数
- ✅ 批量生成功能

### 2. V2数学原理验证
```bash
python scripts/causal_qwen_v2_validation_test.py
```
**用途**: 深度验证V2数学原理的正确性  
**验证点**:
- ✅ 柯西分布线性稳定性
- ✅ ActionNetwork双模式
- ✅ 位置vs尺度差异
- ✅ 温度参数选择性生效

### 3. 简洁使用演示
```bash
python scripts/simple_demo_v2.py
```
**用途**: 展示CausalQwen的基本使用方法  
**特点**: 与Qwen完全相同的使用方式

---

## 📊 对比测试

### 4. 端到端对比测试（推荐有Qwen模型时运行）
```bash
python scripts/end_to_end_comparison_test_v2.py
```
**用途**: CausalQwen vs 原始Qwen的直接对比  
**需求**: 需要本地有Qwen2.5-0.5B模型  
**验证点**:
- 确定性模式一致性
- 采样模式多样性
- 温度参数效果

---

## 🧪 其他测试脚本

### 数学组件测试
```bash
# 验证柯西分布数学工具
python scripts/verify_cauchy_math.py

# 测试Action Network
python scripts/test_action_network.py

# 测试Abduction Network  
python scripts/test_abduction_network.py
```

### 功能测试
```bash
# 批量处理测试
python scripts/test_batch_processing.py

# 因果采样测试
python scripts/test_causal_sampling.py

# 确定性推理测试
python scripts/test_deterministic_inference.py
```

### 综合测试
```bash
# 全面组件测试（较长时间）
python scripts/comprehensive_component_test.py

# 简单快速测试
python scripts/simple_test.py
```

---

## 🎯 测试场景推荐

### 🚀 快速验证（5分钟内）
```bash
python scripts/qwen_compatibility_test.py
python scripts/simple_demo_v2.py
```

### 🔬 深度验证（10-15分钟）
```bash
python scripts/causal_qwen_v2_validation_test.py
python scripts/end_to_end_comparison_test_v2.py  # 需要Qwen模型
```

### 🧪 完整测试（30分钟+）
```bash
# 运行所有核心测试
python scripts/qwen_compatibility_test.py
python scripts/causal_qwen_v2_validation_test.py
python scripts/simple_demo_v2.py
python scripts/comprehensive_component_test.py
```

---

## 📋 测试结果预期

### ✅ 正常输出示例

```
🎉 所有测试通过！CausalQwen与Qwen完全兼容！
✅ 生成接口兼容性
✅ V2数学原理
✅ 温度参数效果  
✅ 批量生成
```

### 🔢 关键指标范围

- **位置参数差异**: > 1.0（显著差异）
- **do_sample差异**: 100%不同（5/5位置）
- **温度多样性**: ≥ 3/5种温度产生不同序列
- **数学精度**: < 1e-6误差

---

## 🛠️ 故障排除

### 常见问题

1. **ImportError**: 确保已安装依赖
   ```bash
   pip install torch transformers numpy
   ```

2. **模型路径错误**: 检查`~/models/Qwen2.5-0.5B`路径
   - 如无Qwen模型，跳过需要它的测试

3. **内存不足**: 使用较小的测试配置
   ```python
   # 在测试脚本中减小模型大小
   vocab_size=100, hidden_size=64
   ```

### 调试模式

在任何测试脚本开头添加：
```python
import torch
torch.manual_seed(42)  # 固定随机种子
```

---

## 📈 测试覆盖

### 核心功能覆盖率

| 功能模块 | 测试脚本 | 覆盖度 |
|---------|---------|--------|
| **Qwen兼容性** | qwen_compatibility_test.py | 100% |
| **V2数学原理** | causal_qwen_v2_validation_test.py | 100% |
| **生成接口** | simple_demo_v2.py | 95% |
| **对比验证** | end_to_end_comparison_test_v2.py | 90% |

### 参数覆盖

✅ `do_sample` (True/False)  
✅ `temperature` (0.1-2.0)  
✅ `top_k` (1-50)  
✅ `top_p` (0.1-1.0)  
✅ `max_new_tokens` (3-20)  

---

## 🎉 总结

**推荐运行顺序**：
1. `qwen_compatibility_test.py` - 验证基本兼容性
2. `simple_demo_v2.py` - 了解使用方法
3. `causal_qwen_v2_validation_test.py` - 深度数学验证
4. `end_to_end_comparison_test_v2.py` - 完整对比（可选）

所有测试都已更新为**与Qwen完全兼容的接口**，确保用户可以零学习成本使用CausalQwen！