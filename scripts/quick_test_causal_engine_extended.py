#!/usr/bin/env python3
"""
CausalEngine 扩展噪声鲁棒性分析脚本

🎯 目标：全面分析各种算法在不同噪声水平下的鲁棒性表现
🔬 核心：0%-100%噪声级别下的算法性能对比

主要特性：
- 回归算法9种：sklearn MLP, PyTorch MLP, CausalEngine(4种模式), Huber, Pinball, Cauchy
- 分类算法8种：sklearn MLP, PyTorch MLP, sklearn OvR, PyTorch Shared OvR, CausalEngine(4种模式)  
- 噪声级别：0%, 10%, 20%, ..., 100% (11个级别)
- 完整指标：回归(MAE, MdAE, RMSE, R²) + 分类(Accuracy, Precision, Recall, F1)
- 折线图可视化：清晰展示算法鲁棒性对比

使用方法：
1. 直接运行：python scripts/quick_test_causal_engine_extended.py
2. 调整参数：修改下方的 EXTENDED_CONFIG
3. 单独测试：调用 test_regression_noise_robustness() 或 test_classification_noise_robustness()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import warnings
from tqdm import tqdm
import pandas as pd

# 设置matplotlib后端，避免弹出窗口
plt.switch_backend('Agg')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有回归器和分类器
from causal_sklearn.regressor import (
    MLPCausalRegressor, MLPPytorchRegressor, 
    MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
)
from causal_sklearn.classifier import (
    MLPCausalClassifier, MLPPytorchClassifier, 
    MLPSklearnOvRClassifier, MLPPytorchSharedOvRClassifier
)
from causal_sklearn.data_processing import inject_shuffle_noise

warnings.filterwarnings('ignore')

# =============================================================================
# 配置部分 - 在这里修改实验参数
# =============================================================================

EXTENDED_CONFIG = {
    # 噪声级别设置
    'noise_levels': np.linspace(0, 1, 11),  # 0%, 10%, 20%, ..., 100%
    
    # 数据生成参数
    'n_samples': 3000,      # 增加样本量，提高稳定性
    'n_features': 10,       # 适中特征数
    'random_state': 42,     # 固定随机种子
    'test_size': 0.2,       # 测试集比例
    
    # 回归任务参数
    'regression_noise': 1.0,        # 回归数据本身的噪声
    'n_classes': 3,                 # 分类任务类别数
    'class_sep': 1.0,               # 分类任务类别分离度
    
    # 网络结构（所有算法统一）- 优化稳定性
    'hidden_layers': (64, 32),      # 保持网络结构
    'max_iter': 2000,               # 增加最大迭代次数
    'learning_rate': 0.001,         # 降低学习率提高稳定性
    'patience': 50,                 # 增加早停耐心
    'tol': 1e-4,                    # 收敛容忍度
    
    # 算法开关（可以关闭某些算法加快测试）
    'test_regression': True,
    'test_classification': True,
    
    # 稳定性改进参数
    'n_runs': 3,                     # 多次运行次数
    'base_random_seed': 42,          # 基础随机种子
    
    # 输出控制
    'output_dir': 'results/extended_robustness',
    'save_plots': True,
    'save_data': True,
    'verbose': True,
    'figure_dpi': 300
}

# =============================================================================
# 工具函数
# =============================================================================

def _ensure_output_dir(output_dir):
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

def _get_output_path(output_dir, filename):
    """获取输出文件路径"""
    return os.path.join(output_dir, filename)

# =============================================================================
# 回归鲁棒性测试
# =============================================================================

def test_regression_noise_robustness(config):
    """测试回归算法的噪声鲁棒性"""
    print("\n" + "="*80)
    print("🔬 回归算法噪声鲁棒性测试")
    print("="*80)
    
    noise_levels = config['noise_levels']
    results = {}
    
    # 定义所有回归算法
    algorithms = {
        # 'sklearn_mlp': ('sklearn MLP', None),
        'pytorch_mlp': ('PyTorch MLP', None),
        'causal_deterministic': ('CausalEngine (deterministic)', 'deterministic'),
        # 'causal_exogenous': ('CausalEngine (exogenous)', 'exogenous'),
        # 'causal_endogenous': ('CausalEngine (endogenous)', 'endogenous'),
        'causal_standard': ('CausalEngine (standard)', 'standard'),
        'huber': ('Huber Regressor', None),
        'pinball': ('Pinball Regressor', None),
        'cauchy': ('Cauchy Regressor', None)
    }
    
    # 初始化结果字典
    for algo_key, (algo_name, _) in algorithms.items():
        results[algo_key] = {
            'name': algo_name,
            'noise_levels': [],
            'mae': [], 'mdae': [], 'rmse': [], 'r2': []
        }
    
    # 生成基础数据（只生成一次）
    print(f"📊 生成回归数据: {config['n_samples']}样本, {config['n_features']}特征")
    X, y = make_regression(
        n_samples=config['n_samples'],
        n_features=config['n_features'],
        noise=config['regression_noise'],
        random_state=config['random_state']
    )
    
    # 分割数据
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_clean_scaled = scaler_y.fit_transform(y_train_clean.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 在不同噪声级别下测试
    for noise_level in tqdm(noise_levels, desc="噪声级别"):
        print(f"\n📊 测试噪声级别: {noise_level:.1%}")
        
        # 对训练标签注入噪声 - 使用专门的inject_shuffle_noise函数
        if noise_level > 0:
            # 使用项目标准的噪声注入方法
            y_train_noisy, noise_indices = inject_shuffle_noise(
                y_train_clean_scaled,
                noise_ratio=noise_level,
                random_state=42
            )
        else:
            y_train_noisy = y_train_clean_scaled.copy()
        
        # 测试每个算法
        for algo_key, (algo_name, causal_mode) in algorithms.items():
            try:
                if config['verbose']:
                    print(f"  🔧 训练 {algo_name}...")
                
                # 创建和训练模型
                if algo_key == 'sklearn_mlp':
                    model = MLPRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate_init=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=config['patience']
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'pytorch_mlp':
                    model = MLPPytorchRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key.startswith('causal_'):
                    model = MLPCausalRegressor(
                        perception_hidden_layers=config['hidden_layers'],
                        mode=causal_mode,
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'huber':
                    model = MLPHuberRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'pinball':
                    model = MLPPinballRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'cauchy':
                    model = MLPCauchyRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                
                # 在测试集上评估
                y_pred_scaled = model.predict(X_test_scaled)
                
                # 转换回原始尺度评估
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                # 计算指标
                mae = mean_absolute_error(y_test, y_pred)
                mdae = median_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # 存储结果
                results[algo_key]['noise_levels'].append(noise_level)
                results[algo_key]['mae'].append(mae)
                results[algo_key]['mdae'].append(mdae)
                results[algo_key]['rmse'].append(rmse)
                results[algo_key]['r2'].append(r2)
                
                if config['verbose']:
                    print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"    ❌ {algo_name} 训练失败: {str(e)}")
                # 添加NaN值保持数组长度一致
                results[algo_key]['noise_levels'].append(noise_level)
                results[algo_key]['mae'].append(np.nan)
                results[algo_key]['mdae'].append(np.nan)
                results[algo_key]['rmse'].append(np.nan)
                results[algo_key]['r2'].append(np.nan)
    
    return results

# =============================================================================
# 分类鲁棒性测试
# =============================================================================

def test_classification_noise_robustness(config):
    """测试分类算法的噪声鲁棒性"""
    print("\n" + "="*80)
    print("🎯 分类算法噪声鲁棒性测试")
    print("="*80)
    
    noise_levels = config['noise_levels']
    results = {}
    
    # 定义所有分类算法
    algorithms = {
        # 'sklearn_mlp': ('sklearn MLP', None),
        'pytorch_mlp': ('PyTorch MLP', None),
        'sklearn_ovr': ('sklearn OvR MLP', None),
        'pytorch_shared_ovr': ('PyTorch Shared OvR', None),
        'causal_deterministic': ('CausalEngine (deterministic)', 'deterministic'),
        # 'causal_exogenous': ('CausalEngine (exogenous)', 'exogenous'),
        # 'causal_endogenous': ('CausalEngine (endogenous)', 'endogenous'),
        'causal_standard': ('CausalEngine (standard)', 'standard')
    }
    
    # 初始化结果字典
    for algo_key, (algo_name, _) in algorithms.items():
        results[algo_key] = {
            'name': algo_name,
            'noise_levels': [],
            'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
    
    # 生成基础数据
    print(f"📊 生成分类数据: {config['n_samples']}样本, {config['n_features']}特征, {config['n_classes']}类别")
    X, y = make_classification(
        n_samples=config['n_samples'],
        n_features=config['n_features'],
        n_classes=config['n_classes'],
        n_clusters_per_class=1,
        class_sep=config['class_sep'],
        random_state=config['random_state']
    )
    
    # 分割数据
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # 标准化特征
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 在不同噪声级别下测试
    for noise_level in tqdm(noise_levels, desc="噪声级别"):
        print(f"\n📊 测试噪声级别: {noise_level:.1%}")
        
        # 对训练标签注入噪声 - 使用专门的inject_shuffle_noise函数
        if noise_level > 0:
            # 使用项目标准的噪声注入方法
            y_train_noisy, noise_indices = inject_shuffle_noise(
                y_train_clean,
                noise_ratio=noise_level,
                random_state=42
            )
        else:
            y_train_noisy = y_train_clean.copy()
        
        # 测试每个算法
        for algo_key, (algo_name, causal_mode) in algorithms.items():
            try:
                if config['verbose']:
                    print(f"  🔧 训练 {algo_name}...")
                
                # 创建和训练模型
                if algo_key == 'sklearn_mlp':
                    model = MLPClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate_init=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=config['patience']
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'pytorch_mlp':
                    model = MLPPytorchClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'sklearn_ovr':
                    model = MLPSklearnOvRClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate_init=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=config['patience'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'pytorch_shared_ovr':
                    model = MLPPytorchSharedOvRClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key.startswith('causal_'):
                    model = MLPCausalClassifier(
                        perception_hidden_layers=config['hidden_layers'],
                        mode=causal_mode,
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=True,
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                
                # 在测试集上评估
                y_pred = model.predict(X_test_scaled)
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # 存储结果
                results[algo_key]['noise_levels'].append(noise_level)
                results[algo_key]['accuracy'].append(accuracy)
                results[algo_key]['precision'].append(precision)
                results[algo_key]['recall'].append(recall)
                results[algo_key]['f1'].append(f1)
                
                if config['verbose']:
                    print(f"    Acc: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"    ❌ {algo_name} 训练失败: {str(e)}")
                # 添加NaN值保持数组长度一致
                results[algo_key]['noise_levels'].append(noise_level)
                results[algo_key]['accuracy'].append(np.nan)
                results[algo_key]['precision'].append(np.nan)
                results[algo_key]['recall'].append(np.nan)
                results[algo_key]['f1'].append(np.nan)
    
    return results

# =============================================================================
# 可视化函数
# =============================================================================

def create_robustness_plots(regression_results, classification_results, config):
    """创建鲁棒性分析折线图"""
    if not config.get('save_plots', False):
        return
    
    _ensure_output_dir(config['output_dir'])
    
    print("\n📊 创建鲁棒性分析图表")
    print("-" * 50)
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12种不同颜色
    
    # 创建回归鲁棒性图表
    if regression_results and config.get('test_regression', True):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        n_runs = config.get('n_runs', 1)
        title_suffix = f' (Averaged over {n_runs} runs)' if n_runs > 1 else ''
        fig.suptitle(f'Regression Algorithms Noise Robustness Analysis{title_suffix}', fontsize=16, fontweight='bold')
        
        metrics = ['mae', 'mdae', 'rmse', 'r2']
        metric_names = ['MAE', 'MdAE', 'RMSE', 'R²']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            color_idx = 0
            for algo_key, data in regression_results.items():
                if data[metric]:  # 确保有数据
                    noise_levels = np.array(data['noise_levels']) * 100  # 转换为百分比
                    values = np.array(data[metric])
                    
                    # 过滤NaN值
                    valid_mask = ~np.isnan(values)
                    if valid_mask.any():
                        # 判断是否为因果算法
                        is_causal = algo_key.startswith('causal_')
                        linestyle = '-' if is_causal else '--'  # 因果算法实线，其他虚线
                        
                        # 只绘制平均值线条，不显示误差条
                        ax.plot(noise_levels[valid_mask], values[valid_mask], 
                               marker='o', linewidth=2, markersize=4, linestyle=linestyle,
                               label=data['name'], color=colors[color_idx])
                        color_idx += 1
            
            ax.set_xlabel('Noise Level (%)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Noise Level')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 对R²使用特殊的y轴范围
            if metric == 'r2':
                ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        regression_path = _get_output_path(config['output_dir'], 'regression_robustness.png')
        plt.savefig(regression_path, dpi=config['figure_dpi'], bbox_inches='tight')
        print(f"📊 回归鲁棒性图表已保存为 {regression_path}")
        plt.close()
    
    # 创建分类鲁棒性图表
    if classification_results and config.get('test_classification', True):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        n_runs = config.get('n_runs', 1)
        title_suffix = f' (Averaged over {n_runs} runs)' if n_runs > 1 else ''
        fig.suptitle(f'Classification Algorithms Noise Robustness Analysis{title_suffix}', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            color_idx = 0
            for algo_key, data in classification_results.items():
                if data[metric]:  # 确保有数据
                    noise_levels = np.array(data['noise_levels']) * 100  # 转换为百分比
                    values = np.array(data[metric])
                    
                    # 过滤NaN值
                    valid_mask = ~np.isnan(values)
                    if valid_mask.any():
                        # 判断是否为因果算法
                        is_causal = algo_key.startswith('causal_')
                        linestyle = '-' if is_causal else '--'  # 因果算法实线，其他虚线
                        
                        # 只绘制平均值线条，不显示误差条
                        ax.plot(noise_levels[valid_mask], values[valid_mask], 
                               marker='o', linewidth=2, markersize=4, linestyle=linestyle,
                               label=data['name'], color=colors[color_idx])
                        color_idx += 1
            
            ax.set_xlabel('Noise Level (%)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Noise Level')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        classification_path = _get_output_path(config['output_dir'], 'classification_robustness.png')
        plt.savefig(classification_path, dpi=config['figure_dpi'], bbox_inches='tight')
        print(f"📊 分类鲁棒性图表已保存为 {classification_path}")
        plt.close()

# =============================================================================
# 主函数
# =============================================================================

def run_single_robustness_analysis(config, run_idx=0):
    """运行单次鲁棒性分析"""
    if config['verbose']:
        print(f"\n🔄 第 {run_idx + 1}/{config['n_runs']} 次运行 (随机种子: {config['random_state']})")
    
    regression_results = None
    classification_results = None
    
    # 回归鲁棒性测试
    if config.get('test_regression', True):
        regression_results = test_regression_noise_robustness(config)
    
    # 分类鲁棒性测试
    if config.get('test_classification', True):
        classification_results = test_classification_noise_robustness(config)
    
    return regression_results, classification_results

def aggregate_results(all_regression_results, all_classification_results):
    """聚合多次运行的结果"""
    aggregated_regression = {}
    aggregated_classification = {}
    
    # 聚合回归结果
    if all_regression_results and len(all_regression_results) > 0:
        first_reg_result = all_regression_results[0]
        if first_reg_result:
            for algo_key in first_reg_result.keys():
                aggregated_regression[algo_key] = {
                    'name': first_reg_result[algo_key]['name'],
                    'noise_levels': first_reg_result[algo_key]['noise_levels'],
                    'mae': [], 'mdae': [], 'rmse': [], 'r2': [],
                    'mae_std': [], 'mdae_std': [], 'rmse_std': [], 'r2_std': []
                }
                
                # 收集所有运行的结果
                metrics = ['mae', 'mdae', 'rmse', 'r2']
                for metric in metrics:
                    all_values = []
                    for run_result in all_regression_results:
                        if run_result and algo_key in run_result:
                            all_values.append(run_result[algo_key][metric])
                    
                    if all_values:
                        # 计算每个噪声级别的均值和标准差
                        all_values = np.array(all_values)  # shape: (n_runs, n_noise_levels)
                        means = np.nanmean(all_values, axis=0)
                        stds = np.nanstd(all_values, axis=0)
                        
                        aggregated_regression[algo_key][metric] = means.tolist()
                        aggregated_regression[algo_key][f'{metric}_std'] = stds.tolist()
    
    # 聚合分类结果
    if all_classification_results and len(all_classification_results) > 0:
        first_cls_result = all_classification_results[0]
        if first_cls_result:
            for algo_key in first_cls_result.keys():
                aggregated_classification[algo_key] = {
                    'name': first_cls_result[algo_key]['name'],
                    'noise_levels': first_cls_result[algo_key]['noise_levels'],
                    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                    'accuracy_std': [], 'precision_std': [], 'recall_std': [], 'f1_std': []
                }
                
                # 收集所有运行的结果
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                for metric in metrics:
                    all_values = []
                    for run_result in all_classification_results:
                        if run_result and algo_key in run_result:
                            all_values.append(run_result[algo_key][metric])
                    
                    if all_values:
                        # 计算每个噪声级别的均值和标准差
                        all_values = np.array(all_values)  # shape: (n_runs, n_noise_levels)
                        means = np.nanmean(all_values, axis=0)
                        stds = np.nanstd(all_values, axis=0)
                        
                        aggregated_classification[algo_key][metric] = means.tolist()
                        aggregated_classification[algo_key][f'{metric}_std'] = stds.tolist()
    
    return aggregated_regression, aggregated_classification

def run_robustness_analysis(config=None):
    """运行完整的多次鲁棒性分析"""
    if config is None:
        config = EXTENDED_CONFIG
    
    print("🚀 CausalEngine 扩展噪声鲁棒性分析 (多次运行版本)")
    print("=" * 70)
    print(f"噪声级别: {config['noise_levels'][0]:.0%} - {config['noise_levels'][-1]:.0%} ({len(config['noise_levels'])}个级别)")
    print(f"数据规模: {config['n_samples']}样本, {config['n_features']}特征")
    print(f"运行次数: {config['n_runs']}次 (随机种子: {config['base_random_seed']} - {config['base_random_seed'] + config['n_runs'] - 1})")
    print(f"学习率: {config['learning_rate']}, 最大迭代: {config['max_iter']}, 早停耐心: {config['patience']}")
    
    all_regression_results = []
    all_classification_results = []
    
    # 多次运行
    for run_idx in range(config['n_runs']):
        # 为每次运行设置不同的随机种子
        run_config = config.copy()
        run_config['random_state'] = config['base_random_seed'] + run_idx
        
        regression_result, classification_result = run_single_robustness_analysis(run_config, run_idx)
        
        all_regression_results.append(regression_result)
        all_classification_results.append(classification_result)
    
    # 聚合结果
    print(f"\n📊 聚合 {config['n_runs']} 次运行的结果...")
    aggregated_regression, aggregated_classification = aggregate_results(
        all_regression_results, all_classification_results
    )
    
    # 创建可视化（使用聚合后的结果）
    create_robustness_plots(aggregated_regression, aggregated_classification, config)
    
    # 保存结果数据
    if config.get('save_data', True):
        _ensure_output_dir(config['output_dir'])
        
        if aggregated_regression:
            reg_data_path = _get_output_path(config['output_dir'], 'regression_results_aggregated.npy')
            np.save(reg_data_path, aggregated_regression)
            print(f"📊 聚合回归结果已保存为 {reg_data_path}")
        
        if aggregated_classification:
            cls_data_path = _get_output_path(config['output_dir'], 'classification_results_aggregated.npy')
            np.save(cls_data_path, aggregated_classification)
            print(f"📊 聚合分类结果已保存为 {cls_data_path}")
    
    print(f"\n✅ 多次运行鲁棒性分析完成!")
    print(f"📊 结果保存在: {config['output_dir']}")
    print(f"🎯 稳定性提升: 通过 {config['n_runs']} 次运行取平均，降低随机波动")
    
    return aggregated_regression, aggregated_classification

# =============================================================================
# 入口点
# =============================================================================

if __name__ == '__main__':
    # 运行完整分析
    regression_results, classification_results = run_robustness_analysis()