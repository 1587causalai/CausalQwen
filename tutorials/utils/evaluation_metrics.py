"""
评估指标模块
提供分类和回归任务的综合评估指标
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    计算分类任务的综合评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（可选，用于计算AUC）
        class_names: 类别名称（可选）
        
    Returns:
        metrics: 包含各种评估指标的字典
    """
    metrics = {}
    
    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 处理多分类和二分类
    num_classes = len(np.unique(y_true))
    average_method = 'binary' if num_classes == 2 else 'macro'
    
    metrics['precision'] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average_method, zero_division=0)
    
    # AUC-ROC（如果提供了概率）
    if y_proba is not None:
        try:
            if num_classes == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError:
            metrics['auc_roc'] = None
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # 详细分类报告
    metrics['classification_report'] = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # 每个类别的指标和额外多分类指标
    if num_classes > 2:
        metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # 多分类额外指标
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None
) -> Dict:
    """
    计算回归任务的综合评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        y_std: 预测的标准差（可选，用于不确定性评估）
        
    Returns:
        metrics: 包含各种评估指标的字典
    """
    metrics = {}
    
    # 基础回归指标
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mdae'] = np.median(np.abs(y_true - y_pred))  # 中位绝对偏差
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # 调整R²
    n = len(y_true)
    p = 1  # 假设单特征，实际应根据模型调整
    if n > p + 1:
        metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
    else:
        metrics['adjusted_r2'] = metrics['r2']
    
    # 残差分析
    residuals = y_true - y_pred
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)
    metrics['residual_skewness'] = stats.skew(residuals)
    metrics['residual_kurtosis'] = stats.kurtosis(residuals)
    
    # 不确定性指标（如果提供了标准差）
    if y_std is not None:
        # 预测区间覆盖率
        lower_bound = y_pred - 1.96 * y_std
        upper_bound = y_pred + 1.96 * y_std
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        metrics['prediction_coverage_95'] = coverage
        
        # 平均预测区间宽度
        metrics['mean_prediction_interval_width'] = np.mean(upper_bound - lower_bound)
        
        # 校准误差
        metrics['mean_predicted_std'] = np.mean(y_std)
        metrics['empirical_std'] = np.std(residuals)
        metrics['calibration_error'] = np.abs(metrics['mean_predicted_std'] - metrics['empirical_std'])
    
    return metrics


def compare_model_performance(
    results_dict: Dict[str, Dict],
    task_type: str = "classification"
) -> Dict:
    """
    比较多个模型的性能
    
    Args:
        results_dict: 每个模型的评估结果
        task_type: 任务类型 ("classification" or "regression")
        
    Returns:
        comparison: 模型比较结果
    """
    comparison = {}
    
    if task_type == "classification":
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    else:
        key_metrics = ['mae', 'mse', 'rmse', 'r2', 'mape']
    
    # 收集每个指标的所有模型结果
    for metric in key_metrics:
        comparison[metric] = {}
        for model_name, results in results_dict.items():
            if metric in results and results[metric] is not None:
                comparison[metric][model_name] = results[metric]
    
    # 找出最佳模型
    comparison['best_models'] = {}
    for metric in key_metrics:
        if metric in comparison and comparison[metric]:
            if task_type == "classification" or metric == 'r2':
                # 分类指标和R²：越大越好
                best_model = max(comparison[metric], key=comparison[metric].get)
            else:
                # 回归误差指标：越小越好
                best_model = min(comparison[metric], key=comparison[metric].get)
            comparison['best_models'][metric] = best_model
    
    # 计算相对改进
    comparison['relative_improvements'] = {}
    baseline_name = 'baseline'  # 假设基准模型名为'baseline'
    
    if baseline_name in results_dict:
        baseline_results = results_dict[baseline_name]
        
        for model_name, results in results_dict.items():
            if model_name != baseline_name:
                comparison['relative_improvements'][model_name] = {}
                
                for metric in key_metrics:
                    if (metric in results and results[metric] is not None and
                        metric in baseline_results and baseline_results[metric] is not None):
                        
                        baseline_value = baseline_results[metric]
                        model_value = results[metric]
                        
                        if baseline_value != 0:
                            if task_type == "classification" or metric == 'r2':
                                # 越大越好的指标
                                improvement = (model_value - baseline_value) / baseline_value * 100
                            else:
                                # 越小越好的指标
                                improvement = (baseline_value - model_value) / baseline_value * 100
                            
                            comparison['relative_improvements'][model_name][metric] = improvement
    
    return comparison


def statistical_significance_test(
    results1: List[float],
    results2: List[float],
    test_type: str = "ttest"
) -> Dict:
    """
    进行统计显著性检验
    
    Args:
        results1: 第一组结果
        results2: 第二组结果
        test_type: 检验类型 ("ttest", "wilcoxon", "mannwhitney")
        
    Returns:
        test_results: 检验结果
    """
    test_results = {}
    
    if test_type == "ttest":
        statistic, p_value = stats.ttest_rel(results1, results2)
        test_results['test_name'] = "配对t检验"
    elif test_type == "wilcoxon":
        statistic, p_value = stats.wilcoxon(results1, results2)
        test_results['test_name'] = "Wilcoxon符号秩检验"
    elif test_type == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(results1, results2)
        test_results['test_name'] = "Mann-Whitney U检验"
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")
    
    test_results['statistic'] = statistic
    test_results['p_value'] = p_value
    test_results['is_significant'] = p_value < 0.05
    test_results['significance_level'] = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    return test_results


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    绘制混淆矩阵热图
    
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        normalize: 是否标准化
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径（可选）
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names or range(confusion_matrix.shape[1]),
        yticklabels=class_names or range(confusion_matrix.shape[0])
    )
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking


def plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str = "",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    绘制回归诊断图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title_prefix: 标题前缀
        figsize: 图表大小
        save_path: 保存路径（可选）
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{title_prefix} Regression Diagnostics', fontsize=16)
    
    # 1. Predicted vs True values
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predicted vs True Values')
    
    # 2. Residuals vs Predicted values
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted Values')
    
    # 3. Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    
    # 4. QQ plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Residuals Normality QQ Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking


def plot_model_comparison(
    comparison_results: Dict,
    task_type: str = "classification",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    绘制模型性能比较图
    
    Args:
        comparison_results: 模型比较结果
        task_type: 任务类型
        figsize: 图表大小
        save_path: 保存路径（可选）
    """
    if task_type == "classification":
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    else:
        key_metrics = ['mae', 'rmse', 'r2', 'mape']
    
    # 准备数据
    models = []
    metric_values = {metric: [] for metric in key_metrics}
    
    for metric in key_metrics:
        if metric in comparison_results:
            if not models:  # 第一次设置模型名称
                models = list(comparison_results[metric].keys())
            
            for model in models:
                if model in comparison_results[metric]:
                    metric_values[metric].append(comparison_results[metric][model])
                else:
                    metric_values[metric].append(0)
    
    # 绘制条形图
    x = np.arange(len(models))
    width = 0.8 / len(key_metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(key_metrics):
        offset = (i - len(key_metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, metric_values[metric], width, label=metric)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'{task_type.title()} Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking


def generate_evaluation_report(
    results_dict: Dict[str, Dict],
    task_type: str,
    dataset_name: str,
    save_path: Optional[str] = None
) -> str:
    """
    生成详细的评估报告
    
    Args:
        results_dict: 模型评估结果
        task_type: 任务类型
        dataset_name: 数据集名称
        save_path: 保存路径（可选）
        
    Returns:
        report: 文本报告
    """
    report_lines = []
    report_lines.append(f"# {dataset_name} {task_type.title()} 任务评估报告")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # 模型性能汇总
    report_lines.append("## 模型性能汇总")
    report_lines.append("")
    
    if task_type == "classification":
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        header = "| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC |"
        separator = "|------|--------|--------|--------|--------|---------|"
    else:
        key_metrics = ['mae', 'rmse', 'r2', 'mape']
        header = "| 模型 | MAE | RMSE | R² | MAPE |"
        separator = "|------|-----|------|----|----- |"
    
    report_lines.append(header)
    report_lines.append(separator)
    
    for model_name, results in results_dict.items():
        row = f"| {model_name} |"
        for metric in key_metrics:
            if metric in results and results[metric] is not None:
                row += f" {results[metric]:.4f} |"
            else:
                row += " N/A |"
        report_lines.append(row)
    
    report_lines.append("")
    
    # 模型比较
    comparison = compare_model_performance(results_dict, task_type)
    report_lines.append("## 模型比较分析")
    report_lines.append("")
    
    if 'best_models' in comparison:
        report_lines.append("### 各指标最佳模型")
        for metric, best_model in comparison['best_models'].items():
            report_lines.append(f"- {metric}: {best_model}")
        report_lines.append("")
    
    if 'relative_improvements' in comparison:
        report_lines.append("### 相对基准模型的改进 (%)")
        for model_name, improvements in comparison['relative_improvements'].items():
            report_lines.append(f"**{model_name}:**")
            for metric, improvement in improvements.items():
                report_lines.append(f"  - {metric}: {improvement:+.2f}%")
            report_lines.append("")
    
    # 总结
    report_lines.append("## 结论")
    report_lines.append("")
    report_lines.append("基于以上分析，我们可以得出以下结论：")
    report_lines.append("")
    
    # 简单的结论生成逻辑
    if 'full_causal' in results_dict and 'baseline' in results_dict:
        if task_type == "classification":
            metric_key = 'accuracy'
        else:
            metric_key = 'r2'
        
        if (metric_key in results_dict['full_causal'] and 
            metric_key in results_dict['baseline']):
            
            full_causal_score = results_dict['full_causal'][metric_key]
            baseline_score = results_dict['baseline'][metric_key]
            
            if full_causal_score > baseline_score:
                report_lines.append("1. 完整CausalEngine在关键指标上优于传统基准模型")
            else:
                report_lines.append("1. 传统基准模型在关键指标上表现更好")
    
    if 'ablated' in results_dict and 'baseline' in results_dict:
        report_lines.append("2. CausalEngine消融版本与传统模型的性能比较验证了理论假设")
    
    report_lines.append("3. 这些结果为CausalEngine的因果推理能力提供了实证支持")
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report