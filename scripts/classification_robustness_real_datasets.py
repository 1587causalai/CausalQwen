#!/usr/bin/env python3
"""
分类算法真实数据集噪声鲁棒性测试脚本

🎯 目标：在真实数据集上测试分类算法的噪声鲁棒性表现
🔬 核心：使用sklearn内置分类数据集，测试0%-100%噪声级别下的算法性能

主要特性：
- 真实数据集：Wine, Breast Cancer, Digits, Iris等sklearn内置数据集
- 分类算法8种：sklearn MLP, PyTorch MLP, sklearn OvR, PyTorch Shared OvR, CausalEngine(4种模式)
- 噪声级别：0%, 10%, 20%, ..., 100% (11个级别)
- 完整指标：Accuracy, Precision, Recall, F1
- 折线图可视化：清晰展示算法在真实数据上的鲁棒性对比

使用方法：
1. 直接运行：python scripts/classification_robustness_real_datasets.py
2. 调整参数：修改下方的 REAL_DATASETS_CONFIG
3. 选择数据集：在配置中指定 'dataset' 参数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_iris
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

# 导入分类器
from causal_sklearn.classifier import (
    MLPCausalClassifier, MLPPytorchClassifier, 
    MLPSklearnOvRClassifier, MLPPytorchSharedOvRClassifier
)
from causal_sklearn.data_processing import inject_shuffle_noise

warnings.filterwarnings('ignore')

# =============================================================================
# 配置部分 - 在这里修改实验参数
# =============================================================================

REAL_DATASETS_CONFIG = {
    # 数据集选择 - 可选择的真实数据集
    'dataset': 'breast_cancer',  # 'wine', 'breast_cancer', 'digits', 'iris'
    
    # 噪声级别设置
    'noise_levels': np.linspace(0, 1, 11),  # 0%, 10%, 20%, ..., 100%
    
    # 数据参数
    'random_state': 42,     # 固定随机种子
    'test_size': 0.2,       # 测试集比例
    
    # 网络结构（所有算法统一）- 优化稳定性配置
    'hidden_layers': (128, 64, 64),      # 保持网络结构
    'max_iter': 3000,               # 进一步增加最大迭代次数
    'learning_rate': 0.001,        # 进一步降低学习率提高稳定性
    'patience': 100,                # 早停耐心（统一参数，适用于所有算法）
    'tol': 1e-4,                    # 更严格的收敛容忍度
    'batch_size': None,             # 批处理大小 (None=全批次, 数字=小批次)
    
    # 稳定性改进参数
    'n_runs': 5,                     # 增加到5次运行
    'base_random_seed': 42,          # 基础随机种子
    
    # 额外稳定性参数
    'validation_fraction': 0.2,     # 验证集比例（早停用）
    'early_stopping': True,          # 确保早停开启
    
    # 输出控制
    'output_dir': 'results/classification_real_datasets',
    'save_plots': True,
    'save_data': True,
    'verbose': True,
    'figure_dpi': 300
}

# 可用的真实数据集
REAL_DATASETS = {
    'wine': {
        'name': 'Wine Dataset',
        'loader': load_wine,
        'description': '178 samples, 13 features, 3 classes'
    },
    'breast_cancer': {
        'name': 'Breast Cancer Dataset', 
        'loader': load_breast_cancer,
        'description': '569 samples, 30 features, 2 classes'
    },
    'digits': {
        'name': 'Digits Dataset',
        'loader': load_digits,
        'description': '1797 samples, 64 features, 10 classes'
    },
    'iris': {
        'name': 'Iris Dataset',
        'loader': load_iris,
        'description': '150 samples, 4 features, 3 classes'
    }
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

def load_real_dataset(dataset_name):
    """加载真实数据集"""
    if dataset_name not in REAL_DATASETS:
        raise ValueError(f"不支持的数据集: {dataset_name}. 可选择: {list(REAL_DATASETS.keys())}")
    
    dataset_info = REAL_DATASETS[dataset_name]
    data = dataset_info['loader']()
    
    print(f"📊 加载真实数据集: {dataset_info['name']}")
    print(f"📋 数据集描述: {dataset_info['description']}")
    print(f"📐 实际数据形状: {data.data.shape}, 类别数: {len(np.unique(data.target))}")
    
    return data.data, data.target

# =============================================================================
# 分类鲁棒性测试
# =============================================================================

def test_classification_noise_robustness_real_data(config):
    """测试分类算法在真实数据集上的噪声鲁棒性"""
    print("\n" + "="*80)
    print("🎯 分类算法真实数据集噪声鲁棒性测试")
    print("="*80)
    
    noise_levels = config['noise_levels']
    results = {}
    
    # 定义所有分类算法 - 简化为与参考脚本一致
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
    
    # 加载真实数据集
    X, y = load_real_dataset(config['dataset'])
    
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
        
        # 对训练标签注入噪声
        if noise_level > 0:
            y_train_noisy, noise_indices = inject_shuffle_noise(
                y_train_clean,
                noise_ratio=noise_level,
                random_state=config['random_state']
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
                        batch_size=config['batch_size'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        n_iter_no_change=config['patience'],
                        tol=config['tol']
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'pytorch_mlp':
                    model = MLPPytorchClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        batch_size=config['batch_size'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'sklearn_ovr':
                    model = MLPSklearnOvRClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate_init=config['learning_rate'],
                        batch_size=config['batch_size'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        n_iter_no_change=config['patience'],
                        tol=config['tol'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'pytorch_shared_ovr':
                    model = MLPPytorchSharedOvRClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        batch_size=config['batch_size'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key.startswith('causal_'):
                    model = MLPCausalClassifier(
                        perception_hidden_layers=config['hidden_layers'],
                        mode=causal_mode,
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        batch_size=config['batch_size'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
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

def create_classification_robustness_plots(results, config):
    """创建分类鲁棒性分析折线图"""
    if not config.get('save_plots', False):
        return
    
    _ensure_output_dir(config['output_dir'])
    
    print("\n📊 创建分类鲁棒性分析图表")
    print("-" * 50)
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12种不同颜色
    
    # 创建分类鲁棒性图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    dataset_name = REAL_DATASETS[config['dataset']]['name']
    n_runs = config.get('n_runs', 1)
    title_suffix = f' (Averaged over {n_runs} runs)' if n_runs > 1 else ''
    fig.suptitle(f'Classification Algorithms Noise Robustness on {dataset_name}{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        color_idx = 0
        for algo_key, data in results.items():
            if data[metric]:  # 确保有数据
                noise_levels = np.array(data['noise_levels']) * 100  # 转换为百分比
                values = np.array(data[metric])
                
                # 过滤NaN值
                valid_mask = ~np.isnan(values)
                if valid_mask.any():
                    # 判断是否为因果算法
                    is_causal = algo_key.startswith('causal_')
                    linestyle = '-' if is_causal else '--'  # 因果算法实线，其他虚线
                    
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
    
    # 生成文件名
    dataset_key = config['dataset']
    plot_filename = f'classification_robustness_{dataset_key}.png'
    plot_path = _get_output_path(config['output_dir'], plot_filename)
    
    plt.savefig(plot_path, dpi=config['figure_dpi'], bbox_inches='tight')
    print(f"📊 分类鲁棒性图表已保存为 {plot_path}")
    plt.close()

# =============================================================================
# 主函数
# =============================================================================

def run_single_classification_robustness_analysis(config, run_idx=0):
    """运行单次分类鲁棒性分析"""
    if config['verbose']:
        print(f"\n🔄 第 {run_idx + 1}/{config['n_runs']} 次运行 (随机种子: {config['random_state']})")
    
    # 运行分类鲁棒性测试
    results = test_classification_noise_robustness_real_data(config)
    
    return results

def aggregate_classification_results(all_results):
    """聚合多次运行的分类结果"""
    aggregated_results = {}
    
    if all_results and len(all_results) > 0:
        first_result = all_results[0]
        if first_result:
            for algo_key in first_result.keys():
                aggregated_results[algo_key] = {
                    'name': first_result[algo_key]['name'],
                    'noise_levels': first_result[algo_key]['noise_levels'],
                    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                    'accuracy_std': [], 'precision_std': [], 'recall_std': [], 'f1_std': []
                }
                
                # 收集所有运行的结果
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                for metric in metrics:
                    all_values = []
                    for run_result in all_results:
                        if run_result and algo_key in run_result:
                            all_values.append(run_result[algo_key][metric])
                    
                    if all_values:
                        # 计算每个噪声级别的均值和标准差
                        all_values = np.array(all_values)  # shape: (n_runs, n_noise_levels)
                        means = np.nanmean(all_values, axis=0)
                        stds = np.nanstd(all_values, axis=0)
                        
                        aggregated_results[algo_key][metric] = means.tolist()
                        aggregated_results[algo_key][f'{metric}_std'] = stds.tolist()
    
    return aggregated_results

def run_classification_robustness_analysis(config=None):
    """运行完整的多次分类鲁棒性分析"""
    if config is None:
        config = REAL_DATASETS_CONFIG
    
    print("🚀 分类算法真实数据集噪声鲁棒性分析 (稳定性优化版本)")
    print("=" * 70)
    print(f"数据集: {REAL_DATASETS[config['dataset']]['name']}")
    print(f"噪声级别: {config['noise_levels'][0]:.0%} - {config['noise_levels'][-1]:.0%} ({len(config['noise_levels'])}个级别)")
    print(f"运行次数: {config['n_runs']}次 (随机种子: {config['base_random_seed']} - {config['base_random_seed'] + config['n_runs'] - 1})")
    print(f"稳定性配置: 学习率={config['learning_rate']}, 迭代={config['max_iter']}, 耐心={config['patience']}, 容忍度={config['tol']}")
    print(f"验证集比例: {config['validation_fraction']}, 早停耐心: {config['n_iter_no_change']}")
    
    all_results = []
    
    # 多次运行
    for run_idx in range(config['n_runs']):
        # 为每次运行设置不同的随机种子
        run_config = config.copy()
        run_config['random_state'] = config['base_random_seed'] + run_idx
        
        result = run_single_classification_robustness_analysis(run_config, run_idx)
        all_results.append(result)
    
    # 聚合结果
    print(f"\n📊 聚合 {config['n_runs']} 次运行的结果...")
    aggregated_results = aggregate_classification_results(all_results)
    
    # 创建可视化（使用聚合后的结果）
    create_classification_robustness_plots(aggregated_results, config)
    
    # 保存结果数据
    if config.get('save_data', True):
        _ensure_output_dir(config['output_dir'])
        
        dataset_key = config['dataset']
        data_filename = f'classification_results_{dataset_key}_aggregated.npy'
        data_path = _get_output_path(config['output_dir'], data_filename)
        
        np.save(data_path, aggregated_results)
        print(f"📊 聚合分类结果已保存为 {data_path}")
    
    print(f"\n✅ 多次运行分类鲁棒性分析完成!")
    print(f"📊 结果保存在: {config['output_dir']}")
    print(f"🎯 数据集: {REAL_DATASETS[config['dataset']]['name']}")
    print(f"🎯 稳定性提升: 通过 {config['n_runs']} 次运行取平均，降低随机波动")
    
    return aggregated_results

# =============================================================================
# 批量测试多个数据集
# =============================================================================

def run_all_datasets_analysis():
    """运行所有数据集的分类鲁棒性分析（多次运行版本）"""
    print("🚀 批量测试所有真实数据集 (多次运行版本)")
    print("=" * 70)
    
    all_results = {}
    
    for dataset_name in REAL_DATASETS.keys():
        print(f"\n🔄 开始测试数据集: {dataset_name}")
        
        # 创建特定数据集的配置
        dataset_config = REAL_DATASETS_CONFIG.copy()
        dataset_config['dataset'] = dataset_name
        
        try:
            results = run_classification_robustness_analysis(dataset_config)
            all_results[dataset_name] = results
            print(f"✅ 数据集 {dataset_name} 测试完成")
        except Exception as e:
            print(f"❌ 数据集 {dataset_name} 测试失败: {str(e)}")
    
    print(f"\n🎉 所有数据集测试完成!")
    print(f"📊 总计测试了 {len(all_results)} 个数据集")
    print(f"🎯 每个数据集运行了 {REAL_DATASETS_CONFIG['n_runs']} 次取平均")
    
    return all_results

# =============================================================================
# 入口点
# =============================================================================

if __name__ == '__main__':
    # 可以选择运行单个数据集或所有数据集
    
    # 运行单个数据集分析
    results = run_classification_robustness_analysis()
    
    # 取消注释下面的行来运行所有数据集的分析
    # all_results = run_all_datasets_analysis()