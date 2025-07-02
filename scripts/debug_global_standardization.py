#!/usr/bin/env python3
"""
全局标准化策略验证脚本
===========================================

实施绝对公平的竞技场：
1. 全局标准化：对 X 和 y 都进行标准化
2. 统一输入：所有模型接收完全标准化的数据
3. 统一评估：所有预测结果都转换回原始尺度进行评估

这样可以确保：
- 所有模型在相同的抽象空间中学习
- 稳健回归器不能利用未缩放数据的优势
- CausalEngine 在困难环境下展示其真正能力
"""

import numpy as np
import warnings
import os
import sys
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_sklearn.regressor import (
    MLPCausalRegressor, MLPPytorchRegressor, 
    MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
)
from causal_sklearn.utils import causal_split

warnings.filterwarnings('ignore')

class GlobalStandardizationConfig:
    """全局标准化实验配置"""
    # 数据配置
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    ANOMALY_RATIO = 0.25
    
    # 模型超参数
    LEARNING_RATE = 0.01
    ALPHA_CAUSAL = 0.0
    ALPHA_PYTORCH = 0.0001
    BATCH_SIZE = 200
    HIDDEN_SIZES = (128, 64, 32)
    MAX_EPOCHS = 3000
    PATIENCE = 50
    TOL = 1e-4
    
    # CausalEngine专属参数
    GAMMA_INIT = 1.0
    B_NOISE_INIT = 1.0
    B_NOISE_TRAINABLE = True
    
    # 要测试的模型
    MODELS_TO_TEST = {
        'pytorch_mlp': True,
        'causal_standard': True,
        'causal_deterministic': True,
        'mlp_huber': True,
        'mlp_pinball': True,
        'mlp_cauchy': True,
    }

def load_and_prepare_global_standardized_data(config):
    """加载数据并实施全局标准化策略"""
    print("📊 1. 加载数据并实施全局标准化策略")
    print("=" * 60)
    
    # 加载数据
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    print(f"   - 数据集加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 数据分割和异常注入
    X_train_full, X_test, y_train_full, y_test = causal_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        anomaly_ratio=config.ANOMALY_RATIO,
        anomaly_type='regression',
        anomaly_strategy='shuffle'
    )
    
    # 从训练集中分割出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    print(f"   - 数据分割完成: 训练{X_train.shape[0]} + 验证{X_val.shape[0]} + 测试{X_test.shape[0]}")
    print(f"   - 异常注入: {config.ANOMALY_RATIO:.0%} 标签噪声已添加到训练集")
    
    # 🎯 实施全局标准化策略
    print(f"\n   🎯 实施全局标准化策略 - 绝对公平的竞技场:")
    
    # 1. 特征标准化
    print(f"   - 对特征 X 进行标准化（基于训练集统计）...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 2. 目标标准化（关键！）
    print(f"   - 对目标 y 进行标准化（基于训练集统计）...")
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"   - ✅ 标准化完成:")
    print(f"     * X: 均值≈0, 标准差≈1")
    print(f"     * y: 均值≈{y_train_scaled.mean():.3f}, 标准差≈{y_train_scaled.std():.3f}")
    print(f"   - ✅ 所有模型将在相同的标准化空间中竞争")
    print(f"   - ✅ 所有预测结果将转换回原始尺度进行公平评估")
    
    return {
        # 原始数据（用于最终评估）
        'y_test_original': y_test,
        
        # 标准化数据（用于模型训练）
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled, 
        'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'y_val_scaled': y_val_scaled,
        'y_test_scaled': y_test_scaled,
        
        # 标准化器（用于逆变换）
        'scaler_y': scaler_y,
        
        # 统计信息
        'data_stats': {
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': X_train.shape[1]
        }
    }

def train_and_evaluate_model(model_name, model_class, model_params, data, config):
    """在全局标准化环境中训练和评估单个模型"""
    print(f"\n   🔧 训练 {model_name}...")
    start_time = time.time()
    
    # 组合训练集和验证集（标准化版本）
    X_train_val_scaled = np.concatenate([data['X_train_scaled'], data['X_val_scaled']])
    y_train_val_scaled = np.concatenate([data['y_train_scaled'], data['y_val_scaled']])
    
    # 在标准化空间中训练
    model = model_class(**model_params)
    model.fit(X_train_val_scaled, y_train_val_scaled)
    
    # 在标准化空间中预测
    y_pred_scaled = model.predict(data['X_test_scaled'])
    
    # 🎯 关键步骤：转换回原始尺度进行评估
    y_pred_original = data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 在原始尺度下计算性能指标
    training_time = time.time() - start_time
    
    result = {
        'MAE': mean_absolute_error(data['y_test_original'], y_pred_original),
        'MdAE': median_absolute_error(data['y_test_original'], y_pred_original),
        'RMSE': np.sqrt(mean_squared_error(data['y_test_original'], y_pred_original)),
        'R²': r2_score(data['y_test_original'], y_pred_original),
        'Time': training_time
    }
    
    print(f"     ✅ 完成 (用时: {training_time:.2f}s)")
    print(f"        性能: MAE={result['MAE']:.4f}, R²={result['R²']:.4f}")
    
    return result

def run_global_standardization_experiment(config):
    """运行全局标准化实验"""
    print("\n🚀 2. 在全局标准化环境中训练所有模型")
    print("=" * 60)
    print("   🎯 所有模型接收相同的标准化数据")
    print("   🎯 所有预测结果转换回原始尺度评估")
    
    # 加载标准化数据
    data = load_and_prepare_global_standardized_data(config)
    
    results = {}
    
    # 通用参数
    common_params = {
        'max_iter': config.MAX_EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'early_stopping': True,
        'validation_fraction': config.VAL_SIZE,
        'n_iter_no_change': config.PATIENCE,
        'tol': config.TOL,
        'batch_size': config.BATCH_SIZE,
        'random_state': config.RANDOM_STATE,
        'verbose': False
    }
    
    # PyTorch MLP
    if config.MODELS_TO_TEST.get('pytorch_mlp'):
        pytorch_params = {
            **common_params,
            'hidden_layer_sizes': config.HIDDEN_SIZES,
            'alpha': config.ALPHA_PYTORCH,
        }
        results['PyTorch MLP'] = train_and_evaluate_model(
            'PyTorch MLP', MLPPytorchRegressor, pytorch_params, data, config
        )
    
    # CausalEngine modes
    causal_base_params = {
        **common_params,
        'perception_hidden_layers': config.HIDDEN_SIZES,
        'alpha': config.ALPHA_CAUSAL,
        'gamma_init': config.GAMMA_INIT,
        'b_noise_init': config.B_NOISE_INIT,
        'b_noise_trainable': config.B_NOISE_TRAINABLE,
    }
    
    if config.MODELS_TO_TEST.get('causal_standard'):
        causal_params = {**causal_base_params, 'mode': 'standard'}
        results['CausalEngine (standard)'] = train_and_evaluate_model(
            'CausalEngine (standard)', MLPCausalRegressor, causal_params, data, config
        )
    
    if config.MODELS_TO_TEST.get('causal_deterministic'):
        causal_params = {**causal_base_params, 'mode': 'deterministic'}
        results['CausalEngine (deterministic)'] = train_and_evaluate_model(
            'CausalEngine (deterministic)', MLPCausalRegressor, causal_params, data, config
        )
    
    # 稳健回归器
    robust_base_params = {
        **common_params,
        'hidden_layer_sizes': config.HIDDEN_SIZES,
        'alpha': config.ALPHA_PYTORCH,
    }
    
    if config.MODELS_TO_TEST.get('mlp_huber'):
        results['MLP Huber'] = train_and_evaluate_model(
            'MLP Huber', MLPHuberRegressor, robust_base_params, data, config
        )
    
    if config.MODELS_TO_TEST.get('mlp_pinball'):
        results['MLP Pinball'] = train_and_evaluate_model(
            'MLP Pinball', MLPPinballRegressor, robust_base_params, data, config
        )
    
    if config.MODELS_TO_TEST.get('mlp_cauchy'):
        results['MLP Cauchy'] = train_and_evaluate_model(
            'MLP Cauchy', MLPCauchyRegressor, robust_base_params, data, config
        )
    
    return results, data

def print_global_standardization_results(results, data_stats, config):
    """打印全局标准化实验结果"""
    print("\n\n" + "="*80)
    print("🔬 3. 全局标准化实验结果分析")
    print("="*80)
    
    print("\n--- 实验配置 ---")
    print(f"策略: 全局标准化 (X + y)")
    print(f"学习率: {config.LEARNING_RATE}, 异常比例: {config.ANOMALY_RATIO}")
    print(f"隐藏层: {config.HIDDEN_SIZES}")
    print(f"数据: 训练{data_stats['n_train']} + 验证{data_stats['n_val']} + 测试{data_stats['n_test']}")
    print("-" * 20)
    
    # 性能对比表格
    header = f"| {'Model':<25} | {'MAE':<8} | {'MdAE':<8} | {'RMSE':<8} | {'R²':<8} | {'Time(s)':<8} |"
    separator = "-" * len(header)
    
    print("\n" + separator)
    print(header)
    print(separator)
    
    # 按性能排序（MdAE）
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MdAE'])
    
    for model_name, metrics in sorted_results:
        print(f"| {model_name:<25} | {metrics['MAE']:.4f} | {metrics['MdAE']:.4f} | "
              f"{metrics['RMSE']:.4f} | {metrics['R²']:.4f} | {metrics['Time']:.2f} |")
    
    print(separator)
    
    # 性能分析
    print(f"\n💡 全局标准化环境下的性能分析:")
    
    best_model = sorted_results[0]
    worst_model = sorted_results[-1]
    
    print(f"   🥇 最佳性能: {best_model[0]} (MdAE: {best_model[1]['MdAE']:.4f})")
    print(f"   📊 最差性能: {worst_model[0]} (MdAE: {worst_model[1]['MdAE']:.4f})")
    
    # 计算性能差距
    performance_gap = (worst_model[1]['MdAE'] - best_model[1]['MdAE']) / best_model[1]['MdAE'] * 100
    print(f"   📈 性能差距: {performance_gap:.1f}%")
    
    # CausalEngine 分析
    causal_models = [name for name in results.keys() if 'CausalEngine' in name]
    if causal_models:
        print(f"\n🧠 CausalEngine 在全局标准化环境下的表现:")
        for model_name in causal_models:
            model_rank = next(i for i, (name, _) in enumerate(sorted_results, 1) if name == model_name)
            print(f"   - {model_name}: 第{model_rank}名 (MdAE: {results[model_name]['MdAE']:.4f})")
    
    print(f"\n🎯 关键洞察:")
    print(f"   - 所有模型在完全相同的标准化空间中竞争")
    print(f"   - 稳健回归器无法利用未缩放数据的优势")
    print(f"   - 结果反映各算法在抽象空间中的真实能力")
    print(f"   - CausalEngine 的因果学习能力得到公平验证")

def main():
    """主函数"""
    print("🔬 全局标准化策略验证实验")
    print("=" * 80)
    print("目标: 建立绝对公平的竞技场，验证 CausalEngine 的真实能力")
    print("策略: 全局标准化 X + y，统一评估尺度")
    print("=" * 80)
    
    config = GlobalStandardizationConfig()
    
    # 显示要测试的方法
    enabled_methods = [k for k, v in config.MODELS_TO_TEST.items() if v]
    print(f"\n📊 将在全局标准化环境中测试 {len(enabled_methods)} 种方法:")
    for i, method in enumerate(enabled_methods, 1):
        print(f"   {i}. {method}")
    
    # 运行实验
    results, data = run_global_standardization_experiment(config)
    
    # 分析结果
    print_global_standardization_results(results, data['data_stats'], config)
    
    print(f"\n🎉 全局标准化实验完成！")
    print(f"💡 这些结果反映了各模型在真正困难环境下的表现")
    print(f"🧠 CausalEngine 的优势（如果存在）现在得到了公平验证")

if __name__ == "__main__":
    main()