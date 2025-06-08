# 实验脚本使用说明

## 概述

本项目提供了两个实验脚本来生成 `docs/experiments/qwen_finetuning_report.md` 格式的实验报告：

1. **完整实验脚本** (`run_qwen_experiment.py`)：运行真实的模型训练和评估
2. **演示脚本** (`demo_report_generation.py`)：使用模拟数据快速生成报告格式

## 使用方法

### 方法一：完整实验（推荐用于真实实验）

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整实验（需要提供Qwen模型路径）
python run_qwen_experiment.py --qwen_model_path /path/to/Qwen2.5-0.5B

# 或使用默认路径（模拟模式）
python run_qwen_experiment.py
```

**输出**：
- `experiment_results/experiment_YYYYMMDD_HHMMSS/`
  - `qwen_finetuning_report.md` - 实验报告
  - `experiment_results.json` - 详细数据
  - `finetuned_model.pth` - 微调后的模型

### 方法二：快速演示（用于验证报告格式）

```bash
# 快速生成演示报告
python demo_report_generation.py
```

**输出**：
- `demo_experiment_results/`
  - `qwen_finetuning_report.md` - 演示报告
  - `experiment_results.json` - 模拟数据

## 脚本特点

### `run_qwen_experiment.py`
- 完整的实验流程：基线评估 → 模型训练 → 微调后评估
- 自动保存模型权重和实验数据
- 生成详细的性能对比报告
- 支持自定义实验参数

### `demo_report_generation.py`
- 使用预设的理想实验结果
- 快速验证报告格式和内容
- 不需要实际训练，适合演示和测试

## 报告格式

生成的报告包含：
- 实验目标和方法说明
- 详细的性能对比表格
- 关键发现和结论
- 技术细节和配置信息
- 自动时间戳和实验ID

## 自定义配置

可以通过修改脚本中的 `config` 字典来调整：
- 模型参数（词汇表大小、隐藏层维度等）
- 训练参数（学习率、训练轮数等）
- 数据集大小和批次大小

## 注意事项

1. 完整实验需要安装PyTorch等依赖包
2. 演示脚本可以在任何Python环境中运行
3. 报告使用Markdown格式，可以直接在GitHub等平台查看
4. 实验结果会自动保存，避免数据丢失

