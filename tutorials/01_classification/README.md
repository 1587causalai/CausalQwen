# 01_classification - 表格数据分类基准测试

## 📂 文件说明

### 🐍 核心脚本

| 文件 | 功能 | 难度 | 时间 |
|------|------|------|------|
| `tabular_quick_test.py` | 表格数据快速测试 | 🟢 基础 | ~2分钟 |
| `tabular_classification_benchmark.py` | 表格数据完整基准测试 | 🔥 进阶 | ~8分钟 |

### 📊 运行后生成的结果文件

运行脚本后会生成以下分析图片：

| 文件 | 说明 |
|------|------|
| `tabular_quick_test_results.png` | 快速测试对比结果 |
| `tabular_benchmark_results.png` | 基准测试详细分析 |

### 📝 文档

| 文件 | 说明 |
|------|------|
| `DEBUG_REPORT.md` | macOS兼容性问题调试记录 |

## 🚀 快速开始

### 方式一：使用教程启动器（推荐）
```bash
cd ../
python run_tutorial.py
# 选择 1 或 1b
```

### 方式二：直接运行
```bash
# 快速表格测试（推荐新手）
python tabular_quick_test.py

# 完整基准测试（深度分析）
python tabular_classification_benchmark.py
```

## 🎯 学习路径

1. **新手入门**：`tabular_quick_test.py`
2. **深度研究**：`tabular_classification_benchmark.py`

## 💡 CausalEngine 在表格数据上的性能

### ✅ 优势领域
- **二分类任务**：在不平衡数据上表现出色（~95%准确率）
  - Adult Income 预测：94.9%
  - Bank Marketing：94.5%
- **不确定性量化**：提供可靠的预测置信度
- **外生噪声建模**：适合复杂数据分布

### ⚠️ 待改进领域
- **多分类任务**：相比传统方法仍有差距
  - Wine Quality：41.6% vs 最佳66.4%
  - Cover Type：66.0% vs 最佳79.4%

### 📊 基准测试结果总结
| 数据集 | CausalEngine | 最佳传统方法 | 排名 |
|--------|-------------|-------------|------|
| Adult Income | 94.9% | 97.4% | 2/5 ⭐ |
| Bank Marketing | 94.5% | 95.5% | 2/5 ⭐ |
| Wine Quality | 41.6% | 66.4% | 5/5 |
| Cover Type | 66.0% | 79.4% | 5/5 |

## 🔧 技术要求

- Python 3.8+
- PyTorch, scikit-learn, matplotlib
- macOS/Linux/Windows 兼容（已修复 XGBoost 兼容性问题）

## 📈 适用场景建议

- ✅ **推荐使用**：二分类、不平衡数据、需要不确定性量化
- ⚠️ **谨慎使用**：多分类、追求极致准确率的场景

---
📅 最后更新：2024-06-21
🐛 问题报告：请查看 `DEBUG_REPORT.md` 