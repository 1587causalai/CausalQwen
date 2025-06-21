"""
Bike Sharing 需求预测教程 (带消融实验)
演示如何使用CausalEngine进行共享单车需求预测，并与传统方法对比

这是最重要的回归演示之一，包含完整的消融实验！
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from tutorials.utils.data_loaders import load_dataset
from tutorials.utils.baseline_networks import create_baseline_regressor, BaselineTrainer
from tutorials.utils.ablation_networks import (
    create_ablated_regressor, create_full_causal_regressor
)
from tutorials.utils.evaluation_metrics import (
    calculate_regression_metrics, compare_model_performance,
    plot_regression_diagnostics, generate_evaluation_report
)


def explore_bike_sharing_dataset():
    """
    探索Bike Sharing数据集
    """
    print("🚲 Bike Sharing 数据集探索")
    print("=" * 50)
    
    # 加载数据
    data_dict = load_dataset('bike_sharing', batch_size=64)
    
    print(f"\n📈 数据集基本信息:")
    print(f"   数据集名称: {data_dict['name']}")
    print(f"   任务类型: {data_dict['task_type']}")
    print(f"   输入特征数: {data_dict['input_size']}")
    print(f"   输出维度: {data_dict['output_size']}")
    print(f"   训练样本: {data_dict['train_size']}")
    print(f"   验证样本: {data_dict['val_size']}")
    print(f"   测试样本: {data_dict['test_size']}")
    
    # 显示特征信息
    print(f"\n🔍 特征列表:")
    for i, feature in enumerate(data_dict['feature_names']):
        print(f"   {i+1:2d}. {feature}")
    
    # 分析目标变量分布
    y_train = data_dict['y_train']
    
    print(f"\n📊 目标变量统计:")
    print(f"   最小值: {y_train.min():.2f}")
    print(f"   最大值: {y_train.max():.2f}")
    print(f"   均值: {y_train.mean():.2f}")
    print(f"   标准差: {y_train.std():.2f}")
    print(f"   中位数: {np.median(y_train):.2f}")
    
    # 分析特征统计
    X_train = data_dict['X_train']
    print(f"\n🔍 特征统计摘要:")
    print(f"   特征维度: {X_train.shape}")
    print(f"   特征均值范围: [{X_train.mean(axis=0).min():.3f}, {X_train.mean(axis=0).max():.3f}]")
    print(f"   特征标准差范围: [{X_train.std(axis=0).min():.3f}, {X_train.std(axis=0).max():.3f}]")
    
    # 可视化数据分布
    visualize_data_exploration(data_dict)
    
    return data_dict


def visualize_data_exploration(data_dict):
    """
    可视化数据探索结果
    """
    # Setup plotting style
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    y_train = data_dict['y_train']
    X_train = data_dict['X_train']
    
    # 1. Target variable distribution
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Bike Demand')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Target Variable Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Target variable Q-Q plot (normality test)
    stats.probplot(y_train, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Target Variable Normality Test')
    
    # 3. Target variable box plot
    axes[0, 2].boxplot(y_train, vert=True)
    axes[0, 2].set_ylabel('Bike Demand')
    axes[0, 2].set_title('Target Variable Box Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 特征相关性热图（选择前10个特征）
    n_features_to_show = min(10, X_train.shape[1])
    feature_subset = X_train[:, :n_features_to_show]
    feature_names_subset = data_dict['feature_names'][:n_features_to_show]
    
    # 添加目标变量
    data_for_corr = np.column_stack([feature_subset, y_train])
    feature_names_with_target = feature_names_subset + ['target']
    
    corr_matrix = np.corrcoef(data_for_corr.T)
    
    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(feature_names_with_target)))
    axes[1, 0].set_yticks(range(len(feature_names_with_target)))
    axes[1, 0].set_xticklabels(feature_names_with_target, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(feature_names_with_target)
    axes[1, 0].set_title('Feature Correlation Matrix')
    
    # Add correlation coefficient text
    for i in range(len(feature_names_with_target)):
        for j in range(len(feature_names_with_target)):
            text = axes[1, 0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=axes[1, 0])
    
    # 5. Feature distribution examples (select important features)
    for idx in range(min(2, X_train.shape[1])):
        axes[1, 1].hist(X_train[:, idx], bins=30, alpha=0.6, 
                       label=f'{data_dict["feature_names"][idx]}', density=True)
    
    axes[1, 1].set_xlabel('Feature Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Feature Distribution Examples')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Feature vs target variable scatter plot
    if X_train.shape[1] > 0:
        axes[1, 2].scatter(X_train[:, 0], y_train, alpha=0.5, s=10)
        axes[1, 2].set_xlabel(f'{data_dict["feature_names"][0]}')
        axes[1, 2].set_ylabel('Bike Demand')
        axes[1, 2].set_title('Feature vs Target Variable Relationship')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(X_train[:, 0], y_train, 1)
        p = np.poly1d(z)
        axes[1, 2].plot(X_train[:, 0], p(X_train[:, 0]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('tutorials/02_regression/bike_sharing_exploration.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking


def run_ablation_experiment(data_dict):
    """
    运行完整的消融实验
    """
    print("\n🔬 Bike Sharing 消融实验")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = data_dict['input_size']
    output_size = data_dict['output_size']
    
    print(f"使用设备: {device}")
    print(f"输入维度: {input_size}")
    print(f"输出维度: {output_size}")
    
    results = {}
    
    # 1. 训练传统神经网络基准
    print(f"\n🏗️  第1步: 训练传统神经网络基准")
    start_time = time.time()
    
    baseline_model = create_baseline_regressor(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.2
    )
    
    baseline_trainer = BaselineTrainer(
        model=baseline_model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    baseline_trainer.train_regression(
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        num_epochs=100,
        early_stopping_patience=15
    )
    
    baseline_time = time.time() - start_time
    
    # 评估基准模型
    baseline_metrics = evaluate_model(
        baseline_model, data_dict['test_loader'], device, "Traditional NN"
    )
    baseline_metrics['training_time'] = baseline_time
    results['baseline'] = baseline_metrics
    
    print(f"✅ 传统神经网络训练完成 ({baseline_time:.2f}s)")
    
    # 2. 训练CausalEngine消融版本
    print(f"\n⚗️  第2步: 训练CausalEngine消融版本 (仅位置输出)")
    start_time = time.time()
    
    ablated_model = create_ablated_regressor(
        input_size=input_size,
        output_size=output_size,
        causal_size=128
    )
    
    ablated_trainer = BaselineTrainer(
        model=ablated_model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    ablated_trainer.train_regression(
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        num_epochs=100,
        early_stopping_patience=15
    )
    
    ablated_time = time.time() - start_time
    
    # 评估消融模型
    ablated_metrics = evaluate_model(
        ablated_model, data_dict['test_loader'], device, "CausalEngine(Ablated)"
    )
    ablated_metrics['training_time'] = ablated_time
    results['ablated'] = ablated_metrics
    
    print(f"✅ CausalEngine消融版本训练完成 ({ablated_time:.2f}s)")
    
    # 3. 训练完整CausalEngine
    print(f"\n🌟 第3步: 训练完整CausalEngine (位置+尺度)")
    start_time = time.time()
    
    full_model = create_full_causal_regressor(
        input_size=input_size,
        output_size=output_size,
        causal_size=128
    )
    
    full_trainer = BaselineTrainer(
        model=full_model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    full_trainer.train_regression(
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        num_epochs=100,
        early_stopping_patience=15
    )
    
    full_time = time.time() - start_time
    
    # 评估完整模型 (多种推理模式)
    for mode_name, (temp, do_sample) in [
        ("CausalEngine(Causal)", (0, False)),
        ("CausalEngine(Standard)", (1.0, False)),
        ("CausalEngine(Sampling)", (0.8, True))
    ]:
        full_metrics = evaluate_model(
            full_model, data_dict['test_loader'], device, mode_name,
            temperature=temp, do_sample=do_sample
        )
        full_metrics['training_time'] = full_time
        # 使用简化的key名
        key_mapping = {
            'CausalEngine(Causal)': 'causal',
            'CausalEngine(Standard)': 'standard', 
            'CausalEngine(Sampling)': 'sampling'
        }
        results[key_mapping.get(mode_name, mode_name.lower())] = full_metrics
    
    print(f"✅ 完整CausalEngine训练完成 ({full_time:.2f}s)")
    
    return results


def evaluate_model(model, test_loader, device, model_name, temperature=1.0, do_sample=False):
    """
    评估单个模型的性能
    """
    print(f"   评估 {model_name}...")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 获取预测
            if hasattr(model, 'predict') and hasattr(model, 'causal_encoder'):
                # 真正的CausalEngine模型（有causal_encoder属性）
                preds = model.predict(batch_x, temperature=temperature, do_sample=do_sample)
            else:
                # 传统模型或基准模型（忽略temperature和do_sample参数）
                if hasattr(model, 'predict'):
                    preds = model.predict(batch_x)
                else:
                    preds = model(batch_x)
                    
            # 修复维度不匹配：如果输出是[batch_size, 1]，flatten到[batch_size]
            if preds.dim() > 1 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())
    
    # 计算指标
    metrics = calculate_regression_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds)
    )
    
    print(f"     R²: {metrics['r2']:.4f}")
    print(f"     RMSE: {metrics['rmse']:.4f}")
    print(f"     MAE: {metrics['mae']:.4f}")
    
    return metrics


def analyze_results(results, data_dict):
    """
    分析和可视化实验结果
    """
    print("\n📊 结果分析")
    print("=" * 50)
    
    # 1. 性能对比表格
    print("\n📋 性能对比表格:")
    print("   模型                      | R²        | MAE       | RMSE      | MAPE      | 训练时间")
    print("   ------------------------- | --------- | --------- | --------- | --------- | ---------")
    
    for model_name, metrics in results.items():
        display_name = {
            'baseline': 'Traditional NN',
            'ablated': 'CausalEngine(Ablated)',
            'causal': 'CausalEngine(Causal)',
            'standard': 'CausalEngine(Standard)',
            'sampling': 'CausalEngine(Sampling)'
        }.get(model_name, model_name)
        
        r2 = metrics.get('r2', 0)
        mae = metrics.get('mae', 0)
        rmse = metrics.get('rmse', 0)
        mape = metrics.get('mape', 0)
        time_val = metrics.get('training_time', 0)
        
        print(f"   {display_name:25} | {r2:.4f}    | {mae:.4f}    | {rmse:.4f}    | {mape:.2f}%   | {time_val:.1f}s")
    
    # 2. 性能提升分析
    print("\n📈 性能提升分析:")
    
    baseline_r2 = results['baseline']['r2']
    
    for model_name, metrics in results.items():
        if model_name != 'baseline':
            model_r2 = metrics['r2']
            improvement = ((model_r2 - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
            
            display_name = {
                'ablated': 'CausalEngine(消融)',
                '因果': 'CausalEngine(因果)',
                '不确定性': 'CausalEngine(不确定性)',
                '采样': 'CausalEngine(采样)'
            }.get(model_name, model_name)
            
            print(f"   {display_name}: {improvement:+.2f}% 相对于基准 (R²)")
    
    # 3. 消融实验验证
    print("\n🔬 消融实验验证:")
    
    baseline_r2 = results['baseline']['r2']
    ablated_r2 = results['ablated']['r2']
    
    r2_diff = abs(baseline_r2 - ablated_r2)
    print(f"   传统神经网络R²: {baseline_r2:.4f}")
    print(f"   CausalEngine(消融)R²: {ablated_r2:.4f}")
    print(f"   差异: {r2_diff:.4f}")
    
    if r2_diff < 0.05:  # 差异小于5%
        print("   ✅ 消融假设验证成功：仅使用位置输出时性能接近传统网络")
    else:
        print("   ⚠️  消融假设需要进一步验证：存在较大性能差异")
    
    # 4. 预测质量分析
    analyze_prediction_quality(results, data_dict)
    
    # 5. 可视化结果
    visualize_results(results)
    
    # 6. 生成详细报告
    comparison = compare_model_performance(results, 'regression')
    report = generate_evaluation_report(
        results, 'regression', 'Bike Sharing',
        'tutorials/02_regression/bike_sharing_report.md'
    )
    
    print(f"\n📄 详细报告已生成: tutorials/02_regression/bike_sharing_report.md")


def analyze_prediction_quality(results, data_dict):
    """
    分析预测质量
    """
    print("\n🎯 预测质量分析:")
    
    # 获取最佳模型的预测结果进行详细分析
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_metrics = results[best_model_name]
    
    print(f"   最佳模型: {best_model_name} (R² = {best_metrics['r2']:.4f})")
    
    # 残差分析
    print(f"   残差分析:")
    print(f"     残差均值: {best_metrics.get('residual_mean', 0):.4f}")
    print(f"     残差标准差: {best_metrics.get('residual_std', 0):.4f}")
    print(f"     残差偏度: {best_metrics.get('residual_skewness', 0):.4f}")
    print(f"     残差峰度: {best_metrics.get('residual_kurtosis', 0):.4f}")
    
    # 预测区间分析（如果有不确定性信息）
    if 'prediction_coverage_95' in best_metrics:
        print(f"   预测区间分析:")
        print(f"     95%预测区间覆盖率: {best_metrics['prediction_coverage_95']:.2%}")
        print(f"     平均预测区间宽度: {best_metrics['mean_prediction_interval_width']:.4f}")
        print(f"     校准误差: {best_metrics['calibration_error']:.4f}")


def visualize_results(results):
    """
    可视化实验结果
    """
    print("\n📊 生成可视化图表...")
    
    # Setup plotting style
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. R² comparison
    models = list(results.keys())
    model_names = [
        {'baseline': 'Traditional NN', 'ablated': 'CausalEngine(Ablated)', 
         '因果': 'CausalEngine(Causal)', '不确定性': 'CausalEngine(Uncertainty)', 
         '采样': 'CausalEngine(Sampling)'}.get(m, m) for m in models
    ]
    r2_scores = [results[m]['r2'] for m in models]
    
    bars = axes[0, 0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'gold', 'coral', 'plum'])
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Bike Sharing Regression R² Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, r2 in zip(bars, r2_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{r2:.4f}', ha='center', va='bottom')
    
    # 2. Multi-metric comparison
    metrics = ['mae', 'rmse', 'mape']
    metric_names = ['MAE', 'RMSE', 'MAPE(%)']
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        # MAPE may need scaling for display
        if metric == 'mape':
            values = [v/10 for v in values]  # Scale for display
        
        axes[0, 1].bar(x + i*width, values, width, label=metric_names[i], alpha=0.8)
    
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].set_title('Multi-metric Performance Comparison')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].legend()
    
    # 3. Training time comparison
    training_times = [results[m].get('training_time', 0) for m in models]
    
    bars = axes[1, 0].bar(model_names, training_times, color='lightsteelblue')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time_val in zip(bars, training_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{time_val:.1f}s', ha='center', va='bottom')
    
    # 4. Relative improvement percentage (based on R²)
    baseline_r2 = results['baseline']['r2']
    improvements = []
    improved_models = []
    
    for model_name in models[1:]:  # Skip baseline
        model_r2 = results[model_name]['r2']
        improvement = ((model_r2 - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
        improvements.append(improvement)
        improved_models.append(model_names[models.index(model_name)])
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = axes[1, 1].bar(improved_models, improvements, color=colors, alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_ylabel('Relative Improvement (%)')
    axes[1, 1].set_title('R² Improvement Relative to Baseline Model')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + (0.5 if imp > 0 else -1.0),
                       f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('tutorials/02_regression/bike_sharing_results.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    
    print("   图表已保存: tutorials/02_regression/bike_sharing_results.png")


def main():
    """
    主函数：运行完整的Bike Sharing需求预测教程
    """
    print("🚲 Bike Sharing 需求预测 - CausalEngine消融实验教程")
    print("本教程演示如何使用CausalEngine进行共享单车需求预测，并通过消融实验验证其优势")
    print("=" * 85)
    
    # 1. 数据探索
    data_dict = explore_bike_sharing_dataset()
    
    # 2. 运行消融实验
    results = run_ablation_experiment(data_dict)
    
    # 3. 分析结果
    analyze_results(results, data_dict)
    
    # 4. 总结
    print("\n🎉 Bike Sharing 消融实验完成！")
    print("\n🔍 关键发现:")
    
    baseline_r2 = results['baseline']['r2']
    best_causal = max([
        results.get('因果', {}).get('r2', 0),
        results.get('不确定性', {}).get('r2', 0),
        results.get('采样', {}).get('r2', 0)
    ])
    
    if best_causal > baseline_r2:
        improvement = ((best_causal - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
        print(f"   ✅ CausalEngine在Bike Sharing数据集上优于传统方法 ({improvement:.2f}%提升)")
    else:
        print(f"   📊 传统方法在此数据集上表现更好，需要进一步分析")
    
    print(f"\n🧠 因果推理的优势:")
    print(f"   • 温度参数控制：可调节确定性vs不确定性")
    print(f"   • 多模式推理：支持因果/不确定性/采样推理")
    print(f"   • 可解释性：每个预测可追溯到个体特征U + 因果法则f")
    print(f"   • 泛化能力：基于因果关系而非统计相关性")
    
    print(f"\n📚 下一步学习:")
    print(f"   1. 尝试其他回归数据集：tutorials/02_regression/")
    print(f"   2. 了解分类任务：tutorials/01_classification/")
    print(f"   3. 运行完整评估：tutorials/03_ablation_studies/comprehensive_comparison.py")
    print(f"   4. 深入理论学习：causal_engine/MATHEMATICAL_FOUNDATIONS.md")


if __name__ == "__main__":
    main()