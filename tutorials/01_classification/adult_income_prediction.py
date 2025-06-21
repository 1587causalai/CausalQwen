"""
Adult Income 数据集分类教程 (带消融实验)
演示如何使用CausalEngine进行收入预测，并与传统方法对比

这是最重要的分类演示之一，包含完整的消融实验！
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from tutorials.utils.data_loaders import load_dataset
from tutorials.utils.baseline_networks import BaselineMLPClassifier, BaselineTrainer
from tutorials.utils.ablation_networks import create_ablation_experiment, AblationTrainer
from tutorials.utils.evaluation_metrics import (
    calculate_classification_metrics, compare_model_performance,
    plot_confusion_matrix, generate_evaluation_report
)


def explore_adult_dataset():
    """
    探索Adult Income数据集
    """
    print("📊 Adult Income 数据集探索")
    print("=" * 50)
    
    # 加载数据
    data_dict = load_dataset('adult', batch_size=64)
    
    print(f"\n📈 数据集基本信息:")
    print(f"   数据集名称: {data_dict['name']}")
    print(f"   任务类型: {data_dict['task_type']}")
    print(f"   输入特征数: {data_dict['input_size']}")
    print(f"   输出类别数: {data_dict['num_classes']}")
    print(f"   训练样本: {data_dict['train_size']}")
    print(f"   验证样本: {data_dict['val_size']}")
    print(f"   测试样本: {data_dict['test_size']}")
    
    # 显示特征信息
    print(f"\n🔍 特征列表:")
    for i, feature in enumerate(data_dict['feature_names']):
        print(f"   {i+1:2d}. {feature}")
    
    # 分析目标变量分布
    y_train = data_dict['y_train']
    unique, counts = np.unique(y_train, return_counts=True)
    
    print(f"\n📊 目标变量分布:")
    for label, count in zip(unique, counts):
        label_name = "<=50K" if label == 0 else ">50K"
        percentage = count / len(y_train) * 100
        print(f"   {label_name}: {count} 样本 ({percentage:.1f}%)")
    
    return data_dict


def run_ablation_experiment(data_dict):
    """
    运行完整的消融实验
    使用新的消融设计：同一个网络，仅损失函数不同
    """
    print("\n🔬 Adult Income 消融实验")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = data_dict['input_size']
    num_classes = data_dict['num_classes']
    
    print(f"使用设备: {device}")
    print(f"输入维度: {input_size}")
    print(f"类别数: {num_classes}")
    
    results = {}
    
    # 1. 训练传统神经网络基准
    print(f"\n🏗️  第1步: 训练传统神经网络基准")
    start_time = time.time()
    
    baseline_model = BaselineMLPClassifier(
        input_dim=input_size,
        num_classes=num_classes,
        hidden_dims=[512, 256],
        dropout=0.1
    )
    
    baseline_trainer = BaselineTrainer(baseline_model, device=device)
    
    baseline_trainer.train_classification(
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        num_epochs=50
    )
    
    baseline_time = time.time() - start_time
    
    # 评估基准模型
    baseline_metrics = evaluate_baseline_model(
        baseline_model, data_dict['test_loader'], device, "传统神经网络"
    )
    baseline_metrics['training_time'] = baseline_time
    results['baseline'] = baseline_metrics
    
    print(f"✅ 传统神经网络训练完成 ({baseline_time:.2f}s)")
    
    # 2. 创建CausalEngine（用于消融和完整版本）
    print(f"\n⚡ 创建CausalEngine...")
    
    # 消融版本 - 使用相同网络但仅loc损失
    engine_ablation, wrapper_ablation = create_ablation_experiment(
        input_dim=input_size,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        task_type='classification',
        num_classes=num_classes,
        dropout=0.1,
        device=device
    )
    
    trainer_ablation = AblationTrainer(engine_ablation, wrapper_ablation)
    
    # 3. 训练CausalEngine消融版本（仅使用loc损失）
    print(f"\n⚗️  第2步: 训练CausalEngine消融版本 (仅使用loc损失)")
    start_time = time.time()
    
    # 准备输入转换函数 - 真实CausalEngine API
    def prepare_causal_inputs(batch_x):
        # CausalEngine期望hidden_states，不需要input_ids
        return {
            'values': batch_x  # 保留values字段供ablation wrapper使用
        }
    
    # 训练消融版本
    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss = 0
        train_acc = 0
        num_batches = 0
        
        for batch_x, batch_y in data_dict['train_loader']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            inputs = prepare_causal_inputs(batch_x)
            hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
            metrics = trainer_ablation.train_step_ablation(hidden_states, batch_y)
            
            train_loss += metrics['loss']
            train_acc += metrics['accuracy']
            num_batches += 1
        
        # 验证
        val_loss = 0
        val_acc = 0
        val_batches = 0
        
        for batch_x, batch_y in data_dict['val_loader']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            inputs = prepare_causal_inputs(batch_x)
            hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
            metrics = trainer_ablation.eval_step(hidden_states, batch_y, use_ablation=True)
            
            val_loss += metrics['loss']
            val_acc += metrics['accuracy']
            val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ablation_state = engine_ablation.state_dict()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: train_acc={train_acc/num_batches:.4f}, val_acc={val_acc/val_batches:.4f}")
    
    # 恢复最佳模型
    engine_ablation.load_state_dict(best_ablation_state)
    
    ablation_time = time.time() - start_time
    
    # 评估消融模型
    ablation_metrics = evaluate_causal_model(
        engine_ablation, wrapper_ablation, data_dict['test_loader'], 
        device, "CausalEngine(消融)", use_ablation=True, prepare_fn=prepare_causal_inputs
    )
    ablation_metrics['training_time'] = ablation_time
    results['ablation'] = ablation_metrics
    
    print(f"✅ CausalEngine消融版本训练完成 ({ablation_time:.2f}s)")
    
    # 4. 训练完整CausalEngine（使用完整因果损失）
    print(f"\n🌟 第3步: 训练完整CausalEngine (使用完整因果损失)")
    start_time = time.time()
    
    # 创建新的CausalEngine实例用于完整版本
    engine_full, wrapper_full = create_ablation_experiment(
        input_dim=input_size,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        task_type='classification',
        num_classes=num_classes,
        dropout=0.1,
        device=device
    )
    
    trainer_full = AblationTrainer(engine_full, wrapper_full)
    
    # 训练完整版本
    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss = 0
        train_acc = 0
        num_batches = 0
        
        for batch_x, batch_y in data_dict['train_loader']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            inputs = prepare_causal_inputs(batch_x)
            hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
            metrics = trainer_full.train_step_full(hidden_states, batch_y)
            
            train_loss += metrics['loss']
            train_acc += metrics['accuracy']
            num_batches += 1
        
        # 验证
        val_loss = 0
        val_acc = 0
        val_batches = 0
        
        for batch_x, batch_y in data_dict['val_loader']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            inputs = prepare_causal_inputs(batch_x)
            hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
            metrics = trainer_full.eval_step(hidden_states, batch_y, use_ablation=False)
            
            val_loss += metrics['loss']
            val_acc += metrics['accuracy']
            val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_full_state = engine_full.state_dict()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: train_acc={train_acc/num_batches:.4f}, val_acc={val_acc/val_batches:.4f}")
    
    # 恢复最佳模型
    engine_full.load_state_dict(best_full_state)
    
    full_time = time.time() - start_time
    
    # 评估完整模型
    full_metrics = evaluate_causal_model(
        engine_full, wrapper_full, data_dict['test_loader'], 
        device, "CausalEngine(完整)", use_ablation=False, prepare_fn=prepare_causal_inputs
    )
    full_metrics['training_time'] = full_time
    results['full_causal'] = full_metrics
    
    print(f"✅ 完整CausalEngine训练完成 ({full_time:.2f}s)")
    
    return results


def evaluate_baseline_model(model, test_loader, device, model_name):
    """
    评估传统基准模型
    """
    print(f"   评估 {model_name}...")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=-1)
            probs = torch.softmax(outputs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    metrics = calculate_classification_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_proba=np.array(all_probs)
    )
    
    print(f"     准确率: {metrics['accuracy']:.4f}")
    print(f"     F1分数: {metrics['f1_score']:.4f}")
    
    return metrics


def evaluate_causal_model(engine, wrapper, test_loader, device, model_name, use_ablation, prepare_fn):
    """
    评估CausalEngine模型
    """
    print(f"   评估 {model_name}...")
    
    engine.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            inputs = prepare_fn(batch_x)
            
            # 使用真实CausalEngine API  
            hidden_states = inputs.get('values')
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            
            outputs = engine(
                hidden_states=hidden_states,
                do_sample=False,
                temperature=1.0,
                return_dict=True,
                apply_activation=not use_ablation  # 消融版本不用激活头，完整版本用
            )
            
            if use_ablation:
                # 消融版本：使用loc进行预测
                loc = outputs['loc_S'][:, -1, :]  # [batch_size, vocab_size]
                logits = loc[:, :wrapper.num_classes]
                preds = torch.argmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
            else:
                # 完整版本：使用激活头输出
                final_output = outputs['output'][:, -1, :]  # [batch_size, output_dim]
                preds = torch.argmax(final_output, dim=-1)
                probs = torch.softmax(final_output, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    metrics = calculate_classification_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_proba=np.array(all_probs)
    )
    
    print(f"     准确率: {metrics['accuracy']:.4f}")
    print(f"     F1分数: {metrics['f1_score']:.4f}")
    
    return metrics


def analyze_results(results):
    """
    分析和可视化实验结果
    """
    print("\n📊 结果分析")
    print("=" * 50)
    
    # 1. 性能对比表格
    print("\n📋 性能对比表格:")
    print("   模型                      | 准确率    | 精确率    | 召回率    | F1分数    | AUC-ROC   | 训练时间")
    print("   ------------------------- | --------- | --------- | --------- | --------- | --------- | ---------")
    
    for model_name, metrics in results.items():
        display_name = {
            'baseline': '传统神经网络',
            'ablation': 'CausalEngine(消融)',
            'full_causal': 'CausalEngine(完整)'
        }.get(model_name, model_name)
        
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        auc = metrics.get('auc_roc', 0) if metrics.get('auc_roc') is not None else 0
        time_val = metrics.get('training_time', 0)
        
        print(f"   {display_name:25} | {acc:.4f}    | {prec:.4f}    | {rec:.4f}    | {f1:.4f}    | {auc:.4f}    | {time_val:.1f}s")
    
    # 2. 消融实验验证
    print("\n🔬 消融实验验证:")
    
    baseline_acc = results['baseline']['accuracy']
    ablation_acc = results['ablation']['accuracy']
    full_acc = results['full_causal']['accuracy']
    
    acc_diff = abs(baseline_acc - ablation_acc)
    print(f"   传统神经网络准确率: {baseline_acc:.4f}")
    print(f"   CausalEngine(消融)准确率: {ablation_acc:.4f}")
    print(f"   CausalEngine(完整)准确率: {full_acc:.4f}")
    print(f"   消融vs基准差异: {acc_diff:.4f}")
    
    if acc_diff < 0.01:  # 差异小于1%
        print("   ✅ 消融假设验证成功：仅使用位置输出时性能接近传统网络")
    else:
        print("   ⚠️  注意：消融版本与传统网络存在差异，可能由于架构差异")
    
    # 3. 性能提升分析
    print("\n📈 性能提升分析:")
    improvement = ((full_acc - baseline_acc) / baseline_acc) * 100
    print(f"   完整CausalEngine相对基准提升: {improvement:+.2f}%")
    
    causal_gain = ((full_acc - ablation_acc) / ablation_acc) * 100
    print(f"   因果机制带来的提升: {causal_gain:+.2f}%")
    
    # 4. 可视化结果
    visualize_results(results)


def visualize_results(results):
    """
    可视化实验结果
    """
    print("\n📊 生成可视化图表...")
    
    # Setup plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    models = list(results.keys())
    model_names = ['Traditional NN', 'CausalEngine\n(Ablated)', 'CausalEngine\n(Full)']
    accuracies = [results[m]['accuracy'] for m in models]
    
    bars = axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'gold'])
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylim(min(accuracies) - 0.05, max(accuracies) + 0.02)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{acc:.4f}', ha='center', va='bottom')
    
    # 2. 多指标对比
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        axes[0, 1].bar(x + i*width, values, width, label=metric_names[i], alpha=0.8)
    
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Multi-metric Performance Comparison')
    axes[0, 1].set_xticks(x + width * 1.5)
    axes[0, 1].set_xticklabels(model_names, rotation=0)
    axes[0, 1].legend()
    
    # 3. 训练时间对比
    training_times = [results[m].get('training_time', 0) for m in models]
    
    bars = axes[1, 0].bar(model_names, training_times, color=['lightsteelblue', 'lightcoral', 'lightsalmon'])
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time Comparison')
    
    # 添加数值标签
    for bar, time_val in zip(bars, training_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{time_val:.1f}s', ha='center', va='bottom')
    
    # 4. 消融实验分析
    categories = ['Baseline', 'Loc Only\n(Ablation)', 'Loc + Scale\n(Full)']
    values = [
        results['baseline']['accuracy'],
        results['ablation']['accuracy'],
        results['full_causal']['accuracy']
    ]
    
    axes[1, 1].plot(categories, values, 'o-', linewidth=2, markersize=10, color='darkgreen')
    axes[1, 1].fill_between(range(len(categories)), values, alpha=0.3, color='lightgreen')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Ablation Study: Impact of Causal Mechanism')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 标注关键点
    for i, (cat, val) in enumerate(zip(categories, values)):
        axes[1, 1].annotate(f'{val:.4f}', (i, val), textcoords="offset points", 
                           xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('tutorials/01_classification/adult_income_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   图表已保存: tutorials/01_classification/adult_income_results.png")


def main():
    """
    主函数：运行完整的Adult Income预测教程
    """
    print("🎯 Adult Income 预测 - CausalEngine消融实验教程")
    print("本教程演示如何使用CausalEngine进行收入预测，并通过消融实验验证其优势")
    print("=" * 80)
    
    # 1. 数据探索
    data_dict = explore_adult_dataset()
    
    # 2. 运行消融实验
    results = run_ablation_experiment(data_dict)
    
    # 3. 分析结果
    analyze_results(results)
    
    # 4. 总结
    print("\n🎉 Adult Income 消融实验完成！")
    print("\n🔍 关键发现:")
    
    baseline_acc = results['baseline']['accuracy']
    ablation_acc = results['ablation']['accuracy']
    full_acc = results['full_causal']['accuracy']
    
    print(f"   1. 传统神经网络基准: {baseline_acc:.4f}")
    print(f"   2. CausalEngine仅loc: {ablation_acc:.4f} (消融版本)")
    print(f"   3. CausalEngine完整: {full_acc:.4f} (loc + scale)")
    
    if full_acc > baseline_acc:
        improvement = ((full_acc - baseline_acc) / baseline_acc) * 100
        print(f"\n   ✅ CausalEngine在Adult数据集上优于传统方法 ({improvement:+.2f}%提升)")
        
        causal_gain = ((full_acc - ablation_acc) / ablation_acc) * 100
        print(f"   ✅ 因果机制(scale)贡献了 {causal_gain:+.2f}% 的性能提升")
    else:
        print(f"\n   📊 在此数据集上需要进一步调优")
    
    print(f"\n📚 下一步学习:")
    print(f"   1. 尝试其他分类数据集")
    print(f"   2. 了解回归任务：tutorials/02_regression/")
    print(f"   3. 运行完整评估：python tutorials/03_ablation_studies/comprehensive_comparison.py")


if __name__ == "__main__":
    main() 