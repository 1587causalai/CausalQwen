#!/usr/bin/env python3
"""
回归算法真实数据集噪声鲁棒性分析脚本

🎯 目标：在真实数据集上分析各种回归算法的噪声鲁棒性表现
🔬 核心：使用sklearn内置真实数据集，测试0%-100%噪声级别下的算法性能对比

主要特性：
- 回归算法6种：PyTorch MLP, CausalEngine(2种模式), Huber, Pinball, Cauchy
- 真实数据集：California Housing, Diabetes, Boston Housing等
- 噪声级别：0%, 10%, 20%, ..., 100% (11个级别)
- 完整指标：MAE, MdAE, RMSE, R²
- 折线图可视化：清晰展示算法在真实数据上的鲁棒性对比

使用方法：
1. 直接运行：python scripts/regression_robustness_real_datasets.py
2. 调整参数：修改下方的 CONFIG
3. 选择数据集：在配置中修改 'dataset_name'
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.datasets import (
    fetch_california_housing, 
    load_diabetes, 
    load_linnerud,
    fetch_openml
)
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

# 导入所有回归器
from causal_sklearn.regressor import (
    MLPCausalRegressor, MLPPytorchRegressor, 
    MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
)
from causal_sklearn.data_processing import (
    inject_shuffle_noise, 
    load_extended_regression_dataset,
    list_available_regression_datasets,
    EXTENDED_REGRESSION_DATASETS
)

warnings.filterwarnings('ignore')

# =============================================================================
# 配置部分 - 在这里修改实验参数
# =============================================================================

CONFIG = {
    # 数据集选择 - 支持扩展数据集
    'dataset_name': 'california_housing',  # 支持所有EXTENDED_REGRESSION_DATASETS中的数据集
    'use_extended_datasets': True,  # 是否使用扩展数据集加载器
    
    # 噪声级别设置
    'noise_levels': np.linspace(0, 1, 11),  # 0%, 10%, 20%, ..., 100%
    
    # 数据分割参数
    'test_size': 0.2,       # 测试集比例
    'random_state': 42,     # 固定随机种子
    
    # 网络结构（所有算法统一）- 优化参数
    'hidden_layers': (128, 128, 64),    # 增大网络结构
    'max_iter': 3000,               # 最大迭代次数
    'learning_rate': 0.003,          # 提高学习率
    'patience': 100,                 # 增加早停耐心
    'tol': 1e-4,                    # 收敛容忍度
    
    # 验证集参数
    'validation_fraction': 0.2,    # 验证集比例（早停用）
    'early_stopping': True,         # 开启早停
    'n_iter_no_change': 100,         # sklearn算法早停耐心
    
    # 多次运行参数
    'n_runs': 3,                    # 运行次数（1=单次，>1=多次平均）
    'base_random_seed': 42,         # 多次运行的基础随机种子
    
    # 输出控制
    'output_dir': 'results/regression_real_datasets',
    'save_plots': True,
    'save_data': True,
    'verbose': True,
    'figure_dpi': 300
}

# =============================================================================
# 数据集加载函数
# =============================================================================

def load_real_dataset(dataset_name, random_state=42, use_extended=True):
    """
    加载真实数据集 - 支持扩展数据集
    
    Args:
        dataset_name: 数据集名称
        random_state: 随机种子
        use_extended: 是否使用扩展数据集加载器
    """
    if use_extended and dataset_name in EXTENDED_REGRESSION_DATASETS:
        # 使用扩展数据集加载器
        return load_extended_regression_dataset(
            dataset_name=dataset_name,
            random_state=random_state,
            return_info=True,
            handle_missing='auto',
            standardize_features=False
        )
    
    # 传统数据集加载方式（向后兼容）
    print(f"📊 加载真实数据集: {dataset_name}")
    
    if dataset_name == 'california_housing':
        data = fetch_california_housing()
        X, y = data.data, data.target
        dataset_info = {
            'name': 'California Housing',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'description': 'California housing prices dataset'
        }
    
    elif dataset_name == 'diabetes':
        data = load_diabetes()
        X, y = data.data, data.target
        dataset_info = {
            'name': 'Diabetes',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'description': 'Diabetes dataset'
        }
    
    elif dataset_name == 'linnerud':
        data = load_linnerud()
        X, y = data.data, data.target
        # Linnerud是多输出，我们只取第一个输出
        if y.ndim > 1:
            y = y[:, 0]
        dataset_info = {
            'name': 'Linnerud (Physical Exercise)',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'description': 'Linnerud physical exercise dataset (first target)'
        }
    
    elif dataset_name == 'boston':
        try:
            # 尝试从OpenML加载Boston Housing数据集
            data = fetch_openml(name='boston', version=1, as_frame=False)
            X, y = data.data, data.target
            dataset_info = {
                'name': 'Boston Housing (OpenML)',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'description': 'Boston housing prices dataset from OpenML'
            }
        except Exception as e:
            print(f"❌ 无法加载Boston数据集: {e}")
            print("📊 改用California Housing数据集")
            return load_real_dataset('california_housing', random_state)
    
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")
    
    print(f"✅ 数据集加载成功: {dataset_info['name']}")
    print(f"   样本数: {dataset_info['n_samples']}, 特征数: {dataset_info['n_features']}")
    
    return X, y, dataset_info

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

def test_regression_robustness_real_data(config):
    """在真实数据集上测试回归算法的噪声鲁棒性"""
    print("\n" + "="*80)
    print("🔬 真实数据集回归算法噪声鲁棒性测试")
    print("="*80)
    
    # 加载真实数据集
    X, y, dataset_info = load_real_dataset(
        config['dataset_name'], 
        config['random_state'],
        config.get('use_extended_datasets', True)
    )
    
    noise_levels = config['noise_levels']
    results = {}
    
    # 定义所有回归算法
    algorithms = {
        'pytorch_mlp': ('PyTorch MLP', None),
        'causal_deterministic': ('CausalEngine (deterministic)', 'deterministic'),
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
    
    print(f"📊 数据集信息: {dataset_info['name']}")
    print(f"   训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")
    print(f"   特征维度: {X_train.shape[1]}")
    
    # 在不同噪声级别下测试
    for noise_level in tqdm(noise_levels, desc="噪声级别"):
        print(f"\n📊 测试噪声级别: {noise_level:.1%}")
        
        # 对训练标签注入噪声
        if noise_level > 0:
            y_train_noisy, noise_indices = inject_shuffle_noise(
                y_train_clean_scaled,
                noise_ratio=noise_level,
                random_state=config['random_state']
            )
        else:
            y_train_noisy = y_train_clean_scaled.copy()
        
        # 测试每个算法
        for algo_key, (algo_name, causal_mode) in algorithms.items():
            try:
                if config['verbose']:
                    print(f"  🔧 训练 {algo_name}...")
                
                # 创建和训练模型
                if algo_key == 'pytorch_mlp':
                    model = MLPPytorchRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
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
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'huber':
                    model = MLPHuberRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'pinball':
                    model = MLPPinballRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                elif algo_key == 'cauchy':
                    model = MLPCauchyRegressor(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
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
                    print(f"    MAE: {mae:.4f}, MdAE: {mdae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"    ❌ {algo_name} 训练失败: {str(e)}")
                # 添加NaN值保持数组长度一致
                results[algo_key]['noise_levels'].append(noise_level)
                results[algo_key]['mae'].append(np.nan)
                results[algo_key]['mdae'].append(np.nan)
                results[algo_key]['rmse'].append(np.nan)
                results[algo_key]['r2'].append(np.nan)
    
    return results, dataset_info

# =============================================================================
# 可视化函数
# =============================================================================

def create_robustness_plots(results, dataset_info, config):
    """创建回归鲁棒性分析折线图"""
    if not config.get('save_plots', False):
        return
    
    _ensure_output_dir(config['output_dir'])
    
    print("\n📊 创建鲁棒性分析图表")
    print("-" * 50)
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    # 创建回归鲁棒性图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    n_runs = config.get('n_runs', 1)
    title_suffix = f' (Averaged over {n_runs} runs)' if n_runs > 1 else ''
    fig.suptitle(f'Regression Algorithms Noise Robustness Analysis\n({dataset_info["name"]} Dataset){title_suffix}', 
                fontsize=16, fontweight='bold')
    
    metrics = ['mae', 'mdae', 'rmse', 'r2']
    metric_names = ['MAE', 'MdAE', 'RMSE', 'R²']
    
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
        
        # 对R²使用特殊的y轴范围
        if metric == 'r2':
            ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    n_runs = config.get('n_runs', 1)
    suffix = f"_avg_{n_runs}runs" if n_runs > 1 else ""
    plot_path = _get_output_path(config['output_dir'], 
                                f'regression_robustness_{config["dataset_name"]}{suffix}.png')
    plt.savefig(plot_path, dpi=config['figure_dpi'], bbox_inches='tight')
    print(f"📊 回归鲁棒性图表已保存为 {plot_path}")
    plt.close()

# =============================================================================
# 多次运行聚合函数
# =============================================================================

def aggregate_multiple_runs(all_results):
    """聚合多次运行的结果"""
    if not all_results or len(all_results) == 0:
        return {}
    
    aggregated_results = {}
    first_result = all_results[0]
    
    for algo_key in first_result.keys():
        aggregated_results[algo_key] = {
            'name': first_result[algo_key]['name'],
            'noise_levels': first_result[algo_key]['noise_levels'],
            'mae': [], 'mdae': [], 'rmse': [], 'r2': [],
            'mae_std': [], 'mdae_std': [], 'rmse_std': [], 'r2_std': []
        }
        
        # 收集所有运行的结果
        metrics = ['mae', 'mdae', 'rmse', 'r2']
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

def run_single_experiment(config, run_idx=0):
    """运行单次实验"""
    if config['verbose'] and config['n_runs'] > 1:
        print(f"\n🔄 第 {run_idx + 1}/{config['n_runs']} 次运行 (随机种子: {config['random_state']})")
    
    results, dataset_info = test_regression_robustness_real_data(config)
    return results, dataset_info

# =============================================================================
# 主函数
# =============================================================================

def show_available_datasets():
    """显示所有可用的回归数据集"""
    print("\n🎯 可用的回归数据集:")
    print("=" * 60)
    list_available_regression_datasets()

def run_regression_robustness_analysis(config=None):
    """运行完整的回归鲁棒性分析（支持多次运行）"""
    if config is None:
        config = CONFIG
    
    n_runs = config.get('n_runs', 1)
    title_suffix = f" ({n_runs}次运行平均)" if n_runs > 1 else ""
    
    print("🚀 回归算法真实数据集噪声鲁棒性分析" + title_suffix)
    print("=" * 60)
    print(f"数据集: {config['dataset_name']}")
    
    # 显示是否使用扩展数据集
    if config.get('use_extended_datasets', True):
        print(f"🔧 使用扩展数据集加载器 (支持 {len(EXTENDED_REGRESSION_DATASETS)} 个数据集)")
        
        # 检查数据集是否在扩展列表中
        if config['dataset_name'] not in EXTENDED_REGRESSION_DATASETS:
            print(f"⚠️  警告: 数据集 '{config['dataset_name']}' 不在扩展数据集中，将使用传统加载方式")
            show_available_datasets()
    else:
        print("🔧 使用传统数据集加载器")
    
    print(f"噪声级别: {config['noise_levels'][0]:.0%} - {config['noise_levels'][-1]:.0%} ({len(config['noise_levels'])}个级别)")
    print(f"学习率: {config['learning_rate']}, 最大迭代: {config['max_iter']}, 耐心: {config['patience']}")
    print(f"验证集比例: {config['validation_fraction']}, 早停: {config['early_stopping']}")
    if n_runs > 1:
        print(f"运行次数: {n_runs}次 (随机种子: {config['base_random_seed']} - {config['base_random_seed'] + n_runs - 1})")
    
    all_results = []
    dataset_info = None
    
    # 多次运行实验
    for run_idx in range(n_runs):
        # 为每次运行设置不同的随机种子
        run_config = config.copy()
        if n_runs > 1:
            run_config['random_state'] = config['base_random_seed'] + run_idx
        
        results, dataset_info = run_single_experiment(run_config, run_idx)
        all_results.append(results)
    
    # 聚合结果（如果是多次运行）
    if n_runs > 1:
        print(f"\n📊 聚合 {n_runs} 次运行的结果...")
        final_results = aggregate_multiple_runs(all_results)
    else:
        final_results = all_results[0]
    
    # 创建可视化
    create_robustness_plots(final_results, dataset_info, config)
    
    # 保存结果数据
    if config.get('save_data', True):
        _ensure_output_dir(config['output_dir'])
        
        suffix = f"_avg_{n_runs}runs" if n_runs > 1 else ""
        results_data = {
            'results': final_results,
            'dataset_info': dataset_info,
            'config': config,
            'n_runs': n_runs
        }
        
        data_path = _get_output_path(config['output_dir'], 
                                   f'regression_results_{config["dataset_name"]}{suffix}.npy')
        np.save(data_path, results_data)
        print(f"📊 回归结果数据已保存为 {data_path}")
    
    print(f"\n✅ 回归鲁棒性分析完成!")
    print(f"📊 结果保存在: {config['output_dir']}")
    print(f"🎯 数据集: {dataset_info['name']} ({dataset_info['n_samples']} samples, {dataset_info['n_features']} features)")
    if n_runs > 1:
        print(f"🎯 稳定性提升: 通过 {n_runs} 次运行取平均，降低随机波动")
    
    return final_results, dataset_info

# =============================================================================
# 入口点
# =============================================================================

if __name__ == '__main__':
    # 显示可用数据集
    show_available_datasets()
    
    # 运行完整分析
    results, dataset_info = run_regression_robustness_analysis()