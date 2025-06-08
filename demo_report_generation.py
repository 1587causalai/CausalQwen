"""
快速演示实验报告生成

这个脚本使用模拟数据快速演示实验报告的生成过程，
不需要实际训练模型，适合快速验证报告格式。
"""

import json
import os
from datetime import datetime
import random


def generate_mock_experiment_results():
    """生成模拟的实验结果数据"""
    
    # 模拟基线结果（微调前）
    baseline_results = {
        'basic': {
            'accuracy': 0.0,
            'mean_rank': 82046,
            'mean_prob': 0.001,
            'rmse': float('inf'),
            'total_samples': 200,
            'correct_samples': 0
        },
        'qa': {
            'accuracy': 0.0,
            'mean_rank': 91343,
            'mean_prob': 0.0008,
            'rmse': float('inf'),
            'total_samples': 200,
            'correct_samples': 0
        },
        'extreme': {
            'accuracy': 0.0,
            'mean_rank': 81144,
            'mean_prob': 0.0012,
            'rmse': float('inf'),
            'total_samples': 200,
            'correct_samples': 0
        },
        'boundary': {
            'accuracy': 0.0,
            'mean_rank': 81314,
            'mean_prob': 0.0009,
            'rmse': float('inf'),
            'total_samples': 200,
            'correct_samples': 0
        }
    }
    
    # 模拟微调后结果
    finetuned_results = {
        'basic': {
            'accuracy': 1.0,
            'mean_rank': 1.0,
            'mean_prob': 0.95,
            'rmse': 42.1,
            'total_samples': 200,
            'correct_samples': 200
        },
        'qa': {
            'accuracy': 1.0,
            'mean_rank': 1.0,
            'mean_prob': 0.93,
            'rmse': 48.1,
            'total_samples': 200,
            'correct_samples': 200
        },
        'extreme': {
            'accuracy': 1.0,
            'mean_rank': 1.0,
            'mean_prob': 0.91,
            'rmse': 50724.6,
            'total_samples': 200,
            'correct_samples': 200
        },
        'boundary': {
            'accuracy': 1.0,
            'mean_rank': 1.0,
            'mean_prob': 0.96,
            'rmse': 6.53,
            'total_samples': 200,
            'correct_samples': 200
        }
    }
    
    # 模拟训练历史
    training_history = []
    for epoch in range(20):
        training_history.append({
            'epoch': epoch,
            'total_loss': 5.0 * (0.8 ** epoch) + random.uniform(-0.1, 0.1),
            'cls_loss': 3.0 * (0.8 ** epoch) + random.uniform(-0.05, 0.05),
            'reg_loss': 2.0 * (0.8 ** epoch) + random.uniform(-0.05, 0.05)
        })
    
    return {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'vocab_size': 1000,
            'embed_dim': 256,
            'hidden_dim': 512,
            'causal_dim': 64,
            'batch_size': 16,
            'epochs': 20,
            'learning_rate': 0.001,
            'train_size': 1000,
            'eval_size': 200
        },
        'baseline_results': baseline_results,
        'finetuned_results': finetuned_results,
        'training_history': training_history,
        'model_path': 'mock_model.pth'
    }


def generate_report(experiment_results, output_path):
    """生成实验报告"""
    
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    dataset_types = ['basic', 'qa', 'extreme', 'boundary']
    
    report_content = f"""# 实验报告：Qwen2.5-0.5B 微调前后性能对比

**文档作者**: 自动生成  
**创建时间**: {current_time}

本次实验的核心目标是：**量化评估我们提出的因果语言模型框架在微调（Fine-tuning）真实大语言模型（LLM）Qwen2.5-0.5B 后，对其处理和预测文本中数值的能力带来的提升。**

我们将通过对比模型在微调**前**和微调**后**在特定任务上的性能差异，来验证我们方法的有效性。

## 1. 实验目标

### 1.1 基础模型
- **模型**: Qwen2.5-0.5B
- **来源**: 从 Hugging Face Hub 下载的官方预训练权重。

### 1.2 评估数据集

我们使用 `src/data/evaluation_data.py` 生成的一系列标准化的测试数据集，覆盖不同场景：

- **基础数值文本**: 包含简单陈述句中的数字。
- **问答（QA）** 格式: 以问答形式呈现的数字。
- **极端数值**: 包含非常大或非常小的数字。
- **边界值**: 测试模型对边界情况（如0, 1）的处理。

## 2. 实验设计与方法

整个实验，从数据评估、模型微调、到结果保存，都可以通过一个自动化脚本一键完成。

要完整复现本报告中的所有结果，请在项目根目录下执行以下命令：

```bash
python run_qwen_experiment.py --qwen_model_path /path/to/your/Qwen2.5-0.5B
```

**重要提示**:
- 请务必将 `/path/to/your/Qwen2.5-0.5B` 替换为您在本地存放 Qwen2.5-0.5B 模型权重的**实际路径**。
- 脚本的其他参数（如 `epochs`, `batch_size` 等）已设置为与本次实验一致的默认值，直接运行即可复现。

### 2.1 实验流程

执行上述脚本后，程序将自动完成以下所有步骤：

1. **加载预训练模型**：加载未经任何修改的 Qwen2.5-0.5B 模型。
2. **训练前评估 (Baseline)**：在评估数据集上对**原始**模型进行全面评估，记录其在`<NUM>`词元预测上的准确率、排名等指标，作为后续对比的基线。
3. **模型微调**: 使用因果语言模型框架对 Qwen 模型进行微调，使其学习我们的因果框架。
4. **训练后评估**: 加载**微调后**的模型，在完全相同的评估数据集上再次进行评估。
5. **保存结果**: 所有评估结果（JSON文件）和微调后的模型权重（.pth文件）将被保存在 `experiment_results/` 目录下的一个以实验时间戳命名的文件夹中。

### 2.2 评估指标

- **分类准确率 (Accuracy)**: 模型成功将最高概率赋予`<NUM>`词元的比例。
- **`<NUM>`词元平均排名 (Mean Rank)**: 在整个词汇表（超过15万个词元）中，`<NUM>`词元的概率排名。排名越靠前越好。
- **`<NUM>`词元平均概率 (Mean Probability)**: 模型赋予`<NUM>`词元的平均概率。
- **回归误差 (Regression Error)**: 对于成功预测`<NUM>`的样本，模型预测的数值与真实数值之间的差距（例如，使用均方根误差RMSE）。

### 2.3 预期结果

我们预期观察到以下现象：

- **微调前**:
  - 分类准确率接近于0。
  - `<NUM>`词元的平均排名非常靠后。
  - 模型的行为基本等同于一个不知道`<NUM>`概念的通用语言模型。

- **微调后**:
  - 分类准确率显著提升。
  - `<NUM>`词元的平均排名大幅提前，理想情况下接近前10。
  - 回归误差显著降低，证明模型不仅知道**何时**预测数字，也知道预测**什么**数字。

## 3. 实验结果

我们的实验取得了完全符合预期的、非常理想的结果。数据显示，经过我们的因果框架微调后，Qwen 模型在处理和预测数值方面的能力得到了根本性的提升。

下面的表格直观地展示了模型在微调前后的巨大差异：

| Dataset | Baseline Accuracy | Finetuned Accuracy | Baseline Mean Rank | Finetuned Mean Rank | RMSE (Finetuned) |
|---------|-------------------|--------------------|--------------------|---------------------|-------------------|"""

    # 添加实验结果数据
    baseline_results = experiment_results['baseline_results']
    finetuned_results = experiment_results['finetuned_results']
    
    for dataset_type in dataset_types:
        baseline = baseline_results[dataset_type]
        finetuned = finetuned_results[dataset_type]
        
        dataset_name = {
            'basic': 'Basic',
            'qa': 'Question Answering', 
            'extreme': 'Extreme Values',
            'boundary': 'Boundary Values'
        }[dataset_type]
        
        report_content += f"""
| {dataset_name} | {baseline['accuracy']:.1%} | {finetuned['accuracy']:.1%} | {baseline['mean_rank']:.0f} | {finetuned['mean_rank']:.0f} | {finetuned['rmse']:.1f} |"""

    report_content += f"""

### 3.1 关键发现

- **准确率 (Accuracy)**: 从 0% 跃升至 100%，表明模型现在能够可靠地识别出需要生成数值的上下文。
- **平均排名 (Mean Rank)**: `<NUM>` 词元的排名从词汇表的后50%（约8-9万名）一举冲到第1名，说明模型对预测数字这件事有了极高的置信度。
- **回归误差 (RMSE)**: RMSE衡量了模型预测数值的精确度。可以看到，在大部分数据集上，模型的预测值与真实值相当接近。在"极端数值"集上误差较大，这符合预期，因为该数据集包含了数量级跨度极大的数字，是未来值得进一步优化的方向。

### 3.2 可视化分析

这些结果清晰地显示，在所有类型的评估数据集上，未经微调的基线模型（Baseline）的准确率均为0，而经过微调后（Finetuned）的模型准确率达到了完美的100%。

平均排名的变化更加震撼地展示了模型的改变。基线模型的`<NUM>`词元排名高达数万，与随机猜测无异。而微调后的模型，其排名稳定在1.0，实现了"指哪打哪"的精确性。

## 4. 结论

本次实验成功地验证了我们提出的因果语言模型框架的有效性。通过在预训练的 Qwen2.5-0.5B 模型上进行微调，我们：

1. 成功地让模型学会了在适当的上下文中，将`<NUM>`词元作为最高优先级的预测。
2. 实现了对文本中数值的精确回归预测。

这证明我们的方法为解决大语言模型在处理精确数值上的固有弱点，提供了一条行之有效的路径。

## 5. 技术细节

### 5.1 模型配置

- 词汇表大小: {experiment_results['config']['vocab_size']}
- 嵌入维度: {experiment_results['config']['embed_dim']}
- 隐藏层维度: {experiment_results['config']['hidden_dim']}
- 因果状态维度: {experiment_results['config']['causal_dim']}
- 学习率: {experiment_results['config']['learning_rate']}

### 5.2 训练配置

- 训练轮数: {experiment_results['config']['epochs']}
- 批次大小: {experiment_results['config']['batch_size']}
- 训练集大小: {experiment_results['config']['train_size']}
- 评估集大小: {experiment_results['config']['eval_size']}

---

**文档更新时间**: {current_time}  
**实验ID**: {experiment_results['timestamp']}

---"""

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return output_path


def main():
    """主函数"""
    print("=" * 60)
    print("快速演示：实验报告生成")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = "demo_experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成模拟实验结果
    print("生成模拟实验数据...")
    experiment_results = generate_mock_experiment_results()
    
    # 保存实验结果
    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    print(f"实验结果已保存: {results_path}")
    
    # 生成报告
    print("生成实验报告...")
    report_path = os.path.join(output_dir, "qwen_finetuning_report.md")
    generate_report(experiment_results, report_path)
    
    print("=" * 60)
    print("演示完成！")
    print(f"实验报告: {report_path}")
    print("=" * 60)
    
    # 显示报告的前几行
    print("\n报告预览:")
    print("-" * 40)
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:15]):
            print(line.rstrip())
        if len(lines) > 15:
            print("...")
    print("-" * 40)


if __name__ == "__main__":
    main()

