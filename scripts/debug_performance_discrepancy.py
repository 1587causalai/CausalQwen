#!/usr/bin/env python3
"""
性能差异调试脚本
==================================================================

**目的**: 系统性对比每个回归算法的 "Legacy" (基于BaselineBenchmark) 和 
"Sklearn-Style" 两种实现的性能差异，并找出导致差异的根本原因。

**背景**: `examples` 目录中存在两套功能相同的教程脚本，但它们的性能表现
可能不一致。本脚本提供了全面的对比分析，涵盖所有主要回归算法。

**支持的算法对比**:
- MLPPytorchRegressor vs BaselineBenchmark('pytorch_mlp')
- MLPCausalRegressor vs BaselineBenchmark('standard'/'deterministic')
- MLPHuberRegressor vs BaselineBenchmark('mlp_huber')
- MLPPinballRegressor vs BaselineBenchmark('mlp_pinball_median')
- MLPCauchyRegressor vs BaselineBenchmark('mlp_cauchy')

**方法**:
1.  **受控实验**: 确保两种实现在完全相同的数据和超参数下运行
2.  **中央配置**: 使用 `ExperimentConfig` 类统一管理所有关键参数
3.  **并行对比**: 分别运行 Legacy 和 Sklearn-Style 实现
4.  **差异分析**: 生成详细的性能对比表格，并计算相对差异百分比
5.  **自动总结**: 自动识别显著性能差异(>5%)并提供分析建议

**如何使用**:
1.  直接运行: `python scripts/debug_performance_discrepancy.py`
2.  选择性测试: 在 `ExperimentConfig.MODELS_TO_TEST` 中启用/禁用特定方法
3.  参数调优: 修改 `ExperimentConfig` 中的超参数来测试不同假设
4.  结果分析: 查看输出表格中的 "Diff %" 列来识别问题算法

**示例输出解读**:
- Diff % = -3.2%: Sklearn-Style 比 Legacy 好 3.2%
- Diff % = +8.5%: Legacy 比 Sklearn-Style 好 8.5% (需要调查)
- Diff % < 5%: 两种实现基本一致
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

# 导入两种实现的核心模块
from causal_sklearn.benchmarks import BaselineBenchmark
from causal_sklearn.regressor import (
    MLPCausalRegressor, MLPPytorchRegressor, 
    MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
)
from causal_sklearn.data_processing import inject_shuffle_noise

warnings.filterwarnings('ignore')

# --- 实验配置 ---
class ExperimentConfig:
    """
    中央实验配置。修改这里的参数来测试不同的假设。
    """
    # 🎯 数据配置
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    ANOMALY_RATIO = 0.25  # 统一使用旧脚本的25%噪声比例进行对比

    # 🧠 模型超参数 (与最新统一配置保持一致)
    # ----------------------------------------------------------------------
    # 统一使用最新的参数配置，确保与tutorial脚本完全一致
    LEARNING_RATE = 0.01

    # 统一所有alpha参数为0.0 (与最新的参数统一保持一致)
    ALPHA_CAUSAL = 0.0
    ALPHA_PYTORCH = 0.0  # 更新为0.0
    
    # 批处理大小: 统一使用None (全批次训练)
    BATCH_SIZE = None  # 全批次训练，与最新配置一致

    # 其他通用参数 (与最新统一配置保持一致)
    HIDDEN_SIZES = (128, 64, 32)
    MAX_EPOCHS = 3000
    PATIENCE = 200  # 更新为200
    TOL = 1e-4
    
    # CausalEngine专属参数
    GAMMA_INIT = 1.0
    B_NOISE_INIT = 1.0
    B_NOISE_TRAINABLE = True
    
    # 💡 要测试不同假设，可以修改上面的值。例如:
    # LEARNING_RATE = 0.001
    # ANOMALY_RATIO = 0.3
    # ----------------------------------------------------------------------

    # 🔬 要对比的模型 (每个都有Legacy vs Sklearn-Style两种实现)
    MODELS_TO_TEST = {
        'pytorch_mlp': True,           # MLPPytorchRegressor vs BaselineBenchmark('pytorch_mlp')
        'causal_standard': True,       # MLPCausalRegressor vs BaselineBenchmark('standard')
        'causal_deterministic': True,  # MLPCausalRegressor vs BaselineBenchmark('deterministic')  
        'mlp_huber': True,            # MLPHuberRegressor vs BaselineBenchmark('mlp_huber')
        'mlp_pinball': True,          # MLPPinballRegressor vs BaselineBenchmark('mlp_pinball_median')
        'mlp_cauchy': True,           # MLPCauchyRegressor vs BaselineBenchmark('mlp_cauchy')
    }
    
    # 🔍 分析增强配置
    DETAILED_ANALYSIS = True          # 是否进行详细分析
    SAVE_TRAINING_LOGS = False        # 是否保存训练日志(用于深度分析)
    COMPARE_TRAINING_CURVES = False   # 是否比较训练曲线


def load_and_prepare_data(config: ExperimentConfig):
    """
    加载和准备数据，实施全局标准化策略
    
    核心理念：建立绝对公平的竞技场
    1. 全局标准化：对训练集的 X 和 y 都进行标准化
    2. 统一输入：所有模型接收完全标准化的数据
    3. 统一评估：所有预测结果都转换回原始尺度进行评估
    """
    print("📊 1. 加载和准备数据...")
    
    # 加载数据
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    print(f"   - 数据集加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 使用标准的train_test_split进行数据分割
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # 保存原始干净数据（用于Legacy对比）
    X_train_full_original = X_train_full.copy()
    y_train_full_original = y_train_full.copy()
    X_test_original = X_test.copy()
    y_test_original = y_test.copy()
    
    # 对训练集标签进行异常注入
    if config.ANOMALY_RATIO > 0:
        y_train_full_noisy, noise_indices = inject_shuffle_noise(
            y_train_full,
            noise_ratio=config.ANOMALY_RATIO,
            random_state=config.RANDOM_STATE
        )
        y_train_full = y_train_full_noisy
        print(f"   - 异常注入完成: {config.ANOMALY_RATIO:.0%} ({len(noise_indices)}/{len(y_train_full)} 样本受影响)")
    else:
        print(f"   - 无异常注入: 纯净环境")
    
    # 从训练集中分割出验证集（含异常的数据）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # 同样从原始干净数据中分割出验证集（用于Legacy对比）
    X_train_original, X_val_original, y_train_original, y_val_original = train_test_split(
        X_train_full_original, y_train_full_original,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    print(f"   - 数据分割完成。")
    print(f"     - 训练集: {X_train.shape[0]}")
    print(f"     - 验证集: {X_val.shape[0]}")
    print(f"     - 测试集: {X_test.shape[0]}")
    
    # 🎯 关键改进：全局标准化策略
    print(f"\n   🎯 实施全局标准化策略（绝对公平的竞技场）:")
    
    # 1. 特征标准化 - 基于训练集拟合
    print(f"   - 对特征 X 进行标准化...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 2. 目标标准化 - 基于训练集拟合（关键！）
    print(f"   - 对目标 y 进行标准化...")
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"   - ✅ 所有模型将接收完全标准化的数据")
    print(f"   - ✅ 所有预测结果将转换回原始尺度进行评估")
    
    return {
        # 含异常的数据（用于Sklearn-Style实现）
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        
        # 原始干净数据（用于Legacy实现对比）
        'X_train_original': X_train_original, 'X_val_original': X_val_original, 'X_test_original': X_test_original,
        'y_train_original': y_train_original, 'y_val_original': y_val_original, 'y_test_original': y_test_original,
        
        # 标准化数据（用于Sklearn-Style模型训练）
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled, 
        'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'y_val_scaled': y_val_scaled,
        'y_test_scaled': y_test_scaled,
        
        # 标准化器（用于逆变换）
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        
        # 完整数据
        'X_full': X,
        'y_full': y
    }

def run_legacy_benchmark(config: ExperimentConfig, data: dict):
    """
    使用 BaselineBenchmark (旧版实现) 运行实验
    
    🎯 关键改进：BaselineBenchmark 现在接收全局标准化的数据
    确保与 Sklearn-Style 实现在完全相同的数据环境下竞争
    """
    print("\n🚀 2a. 运行 Legacy 实现 (BaselineBenchmark)...")
    print("   🎯 使用全局标准化数据确保公平竞争")
    
    benchmark = BaselineBenchmark()
    
    # 确定要运行的基准方法
    baseline_methods = []
    if config.MODELS_TO_TEST.get('pytorch_mlp'):
        baseline_methods.append('pytorch_mlp')
    if config.MODELS_TO_TEST.get('mlp_huber'):
        baseline_methods.append('mlp_huber')
    if config.MODELS_TO_TEST.get('mlp_pinball'):
        baseline_methods.append('mlp_pinball_median')  # BaselineBenchmark中的方法名
    if config.MODELS_TO_TEST.get('mlp_cauchy'):
        baseline_methods.append('mlp_cauchy')
        
    # 确定要运行的因果模式
    causal_modes = []
    if config.MODELS_TO_TEST.get('causal_standard'):
        causal_modes.append('standard')
    if config.MODELS_TO_TEST.get('causal_deterministic'):
        causal_modes.append('deterministic')

    # 🎯 关键修正：使用原始干净数据让BaselineBenchmark自己处理
    # BaselineBenchmark内部会自动进行数据分割和噪声注入
    X_combined = data['X_full']
    y_combined = data['y_full']
    
    print(f"   - 传递原始完整数据集: X({X_combined.shape}), y({y_combined.shape})")
    print(f"   - BaselineBenchmark将自动处理数据分割和噪声注入")
    
    # 🎯 关键修复：使用baseline_config机制正确传递PyTorch MLP参数
    baseline_config = {
        'method_params': {
            'pytorch_mlp': {
                'hidden_layer_sizes': config.HIDDEN_SIZES,
                'max_iter': config.MAX_EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'early_stopping': True,
                'validation_fraction': config.VAL_SIZE,
                'n_iter_no_change': config.PATIENCE,
                'tol': config.TOL,
                'alpha': config.ALPHA_PYTORCH,
                'batch_size': config.BATCH_SIZE,
                'random_state': config.RANDOM_STATE,
                'verbose': False
            },
            'mlp_huber': {
                'hidden_layer_sizes': config.HIDDEN_SIZES,
                'max_iter': config.MAX_EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'early_stopping': True,
                'validation_fraction': config.VAL_SIZE,
                'n_iter_no_change': config.PATIENCE,
                'tol': config.TOL,
                'alpha': config.ALPHA_PYTORCH,
                'batch_size': config.BATCH_SIZE,
                'random_state': config.RANDOM_STATE
            },
            'mlp_pinball_median': {
                'hidden_layer_sizes': config.HIDDEN_SIZES,
                'max_iter': config.MAX_EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'early_stopping': True,
                'validation_fraction': config.VAL_SIZE,
                'n_iter_no_change': config.PATIENCE,
                'tol': config.TOL,
                'alpha': config.ALPHA_PYTORCH,
                'batch_size': config.BATCH_SIZE,
                'random_state': config.RANDOM_STATE
            },
            'mlp_cauchy': {
                'hidden_layer_sizes': config.HIDDEN_SIZES,
                'max_iter': config.MAX_EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'early_stopping': True,
                'validation_fraction': config.VAL_SIZE,
                'n_iter_no_change': config.PATIENCE,
                'tol': config.TOL,
                'alpha': config.ALPHA_PYTORCH,
                'batch_size': config.BATCH_SIZE,
                'random_state': config.RANDOM_STATE
            }
        }
    }
    
    results = benchmark.compare_models(
        X=X_combined,
        y=y_combined,
        task_type='regression',
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        anomaly_ratio=config.ANOMALY_RATIO,
        verbose=False,
        
        # 基准方法配置
        baseline_methods=baseline_methods,
        baseline_config=baseline_config,  # 🎯 关键添加：传递完整参数
        
        # CausalEngine配置
        causal_modes=causal_modes,
        
        # CausalEngine专属参数
        hidden_sizes=config.HIDDEN_SIZES,
        max_epochs=config.MAX_EPOCHS,
        lr=config.LEARNING_RATE,
        patience=config.PATIENCE,
        tol=config.TOL,
        gamma_init=config.GAMMA_INIT,
        b_noise_init=config.B_NOISE_INIT,
        b_noise_trainable=config.B_NOISE_TRAINABLE
    )
    
    # 🎯 关键改进：BaselineBenchmark现在自动处理全局标准化和逆变换
    print("   - BaselineBenchmark将自动处理标准化和逆变换")
    print(f"   - 返回的结果键: {list(results.keys())}")
    print(f"   - 使用了baseline_config机制来传递完整参数")
    
    # 调试信息：检查是否包含pytorch_mlp的结果
    if 'pytorch_mlp' not in results and 'pytorch_mlp' in baseline_methods:
        print(f"   ⚠️ 警告: 请求了pytorch_mlp但结果中没有pytorch_mlp键")
        print(f"   - 请求的baseline_methods: {baseline_methods}")
        print(f"   - 实际返回的键: {list(results.keys())}")
    
    print("   - Legacy 实现运行完成。")
    return results

def run_sklearn_benchmark(config: ExperimentConfig, data: dict):
    """
    使用 sklearn-style learners (新版实现) 运行实验
    
    🎯 关键改进：使用全局标准化数据，确保与 Legacy 实现完全公平竞争
    """
    print("\n🚀 2b. 运行 Sklearn-Style 实现...")
    print("   🎯 使用全局标准化数据确保公平竞争")

    results = {}
    
    # 🎯 关键修复：使用与简单测试完全相同的数据处理策略
    # 模拟BaselineBenchmark内部的数据处理流程来确保一致性
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from causal_sklearn.data_processing import inject_shuffle_noise
    
    # 重新加载原始数据（确保数据处理完全一致）
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # 与BaselineBenchmark相同的数据分割
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # 噪声注入（在原始尺度）
    if config.ANOMALY_RATIO > 0:
        y_train_noisy, _ = inject_shuffle_noise(
            y_train_full, noise_ratio=config.ANOMALY_RATIO, random_state=config.RANDOM_STATE
        )
    else:
        y_train_noisy = y_train_full.copy()
    
    # 验证集分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_noisy, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
    )
    
    # 标准化处理
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    
    # 组合训练+验证数据
    X_train_val_scaled = np.concatenate([X_train_scaled, X_val_scaled])
    y_train_val_scaled = np.concatenate([y_train_scaled, y_val_scaled])
    
    print(f"   - 简单测试风格数据处理: X_train_val({X_train_val_scaled.shape}), y_train_val({y_train_val_scaled.shape})")
    print(f"   - 使用与compare_huber_vs_pytorch_mlp.py完全相同的数据处理策略")

    # 通用训练函数
    def train_and_evaluate(model_name, model_class, model_params, result_key):
        print(f"   - 正在训练 {model_name}...")
        start_time = time.time()
        
        model = model_class(**model_params)
        # 🎯 关键改进：在标准化空间中训练
        model.fit(X_train_val_scaled, y_train_val_scaled)
        
        # 🎯 关键改进：在标准化空间中预测
        y_pred_scaled = model.predict(X_test_scaled)
        
        # 🎯 关键改进：将预测结果转换回原始尺度进行评估
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 记录训练时间和模型信息(用于详细分析)
        
        # 在原始尺度下计算性能指标
        results[result_key] = {
            'test': {
                'MAE': mean_absolute_error(y_test, y_pred_original),
                'MdAE': median_absolute_error(y_test, y_pred_original),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_original)),
                'R²': r2_score(y_test, y_pred_original)
            },
            'time': time.time() - start_time,
            'model_info': {
                'model_class': model.__class__.__name__,
                'n_features': X_train_val_scaled.shape[1],
                'n_samples': X_train_val_scaled.shape[0]
            }
        }
        print(f"     ...完成 (用时: {results[result_key]['time']:.2f}s)")

    # 通用参数（所有方法共用）
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

    # --- 训练和评估 PyTorch MLP ---
    if config.MODELS_TO_TEST.get('pytorch_mlp'):
        pytorch_params = {
            **common_params,
            'hidden_layer_sizes': config.HIDDEN_SIZES,
            'alpha': config.ALPHA_PYTORCH,
        }
        train_and_evaluate('PyTorch MLP', MLPPytorchRegressor, pytorch_params, 'pytorch_mlp')

    # --- 训练和评估 CausalEngine modes ---
    # 🎯 关键修复：MLPCausalRegressor使用sklearn风格参数名称
    causal_base_params = {
        # 使用sklearn风格的参数名称，与MLPCausalRegressor接口一致
        'perception_hidden_layers': config.HIDDEN_SIZES,
        'max_iter': config.MAX_EPOCHS,        # MLPCausalRegressor使用max_iter
        'learning_rate': config.LEARNING_RATE, # MLPCausalRegressor使用learning_rate  
        'n_iter_no_change': config.PATIENCE,   # MLPCausalRegressor使用n_iter_no_change
        'tol': config.TOL,
        'alpha': config.ALPHA_CAUSAL,
        'gamma_init': config.GAMMA_INIT,
        'b_noise_init': config.B_NOISE_INIT,
        'b_noise_trainable': config.B_NOISE_TRAINABLE,
        'early_stopping': True,
        'validation_fraction': config.VAL_SIZE,
        'random_state': config.RANDOM_STATE,
        'verbose': False
    }
    
    if config.MODELS_TO_TEST.get('causal_standard'):
        causal_params = {**causal_base_params, 'mode': 'standard'}
        train_and_evaluate('CausalEngine (standard)', MLPCausalRegressor, causal_params, 'standard')
    
    if config.MODELS_TO_TEST.get('causal_deterministic'):
        causal_params = {**causal_base_params, 'mode': 'deterministic'}
        train_and_evaluate('CausalEngine (deterministic)', MLPCausalRegressor, causal_params, 'deterministic')

    # --- 训练和评估稳健回归器 ---
    robust_base_params = {
        **common_params,
        'hidden_layer_sizes': config.HIDDEN_SIZES,
        'alpha': config.ALPHA_PYTORCH,  # 稳健回归器使用与PyTorch MLP相同的alpha
    }

    if config.MODELS_TO_TEST.get('mlp_huber'):
        train_and_evaluate('MLP Huber', MLPHuberRegressor, robust_base_params, 'mlp_huber')

    if config.MODELS_TO_TEST.get('mlp_pinball'):
        train_and_evaluate('MLP Pinball', MLPPinballRegressor, robust_base_params, 'mlp_pinball')

    if config.MODELS_TO_TEST.get('mlp_cauchy'):
        train_and_evaluate('MLP Cauchy', MLPCauchyRegressor, robust_base_params, 'mlp_cauchy')
    
    print("   - Sklearn-Style 实现运行完成。")
    print(f"   - 返回的结果键: {list(results.keys())}")
    return results

def perform_detailed_analysis(legacy_results, sklearn_results, config):
    """执行详细的性能差异分析"""
    print("\n📈 总结分析:")
    
    # 修正的模型映射：(config_key, legacy_key, sklearn_key, display_name)
    # 注意：legacy_key 使用 BaselineBenchmark 实际返回的结果键名
    models_map = [
        ('pytorch_mlp', 'pytorch_mlp', 'pytorch_mlp', 'PyTorch MLP'),  # 修正: BaselineBenchmark实际返回'pytorch_mlp'
        ('causal_standard', 'standard', 'standard', 'Causal (standard)'),
        ('causal_deterministic', 'deterministic', 'deterministic', 'Causal (deterministic)'),
        ('mlp_huber', 'mlp_huber', 'mlp_huber', 'MLP Huber'),
        ('mlp_pinball', 'mlp_pinball_median', 'mlp_pinball', 'MLP Pinball'),  # 修正: BaselineBenchmark返回'mlp_pinball_median'
        ('mlp_cauchy', 'mlp_cauchy', 'mlp_cauchy', 'MLP Cauchy'),
    ]
    
    significant_diffs = []
    all_diffs = []
    
    for config_key, legacy_key, sklearn_key, display_name in models_map:
        if not config.MODELS_TO_TEST.get(config_key):
            continue
            
        # 修正了legacy_key的检查逻辑
        actual_legacy_key = legacy_key
        if config_key == 'pytorch_mlp' and legacy_key not in legacy_results:
            # 检查可能的键名变体
            for possible_key in ['pytorch', 'pytorch_mlp']:
                if possible_key in legacy_results:
                    actual_legacy_key = possible_key
                    break
        
        if actual_legacy_key in legacy_results and sklearn_key in sklearn_results:
            legacy_result = legacy_results[actual_legacy_key]['test']
            sklearn_result = sklearn_results[sklearn_key]['test']
            
            # 计算所有指标的差异
            metrics_diff = {}
            for metric in ['MAE', 'MdAE', 'RMSE', 'R²']:
                if metric in legacy_result and metric in sklearn_result:
                    legacy_val = legacy_result[metric]
                    sklearn_val = sklearn_result[metric]
                    
                    if metric == 'R²':
                        # R²越高越好，计算方向相反
                        diff_pct = ((sklearn_val - legacy_val) / abs(legacy_val)) * 100
                    else:
                        # MAE、MdAE、RMSE越低越好
                        diff_pct = ((sklearn_val - legacy_val) / legacy_val) * 100
                    
                    metrics_diff[metric] = diff_pct
            
            # 以MdAE为主要指标判断显著性差异
            main_diff = metrics_diff.get('MdAE', 0)
            all_diffs.append((display_name, main_diff, metrics_diff))
            
            if abs(main_diff) > 5.0:  # 差异超过5%
                significant_diffs.append((display_name, main_diff, metrics_diff))
    
    # 打印总体分析
    print(f"\n📊 分析了 {len(all_diffs)} 个算法的性能对比")
    
    if significant_diffs:
        print(f"   ⚠️ 发现 {len(significant_diffs)} 个方法存在显著性能差异 (>5%):")
        for name, main_diff, metrics_diff in significant_diffs:
            direction = "Legacy更好" if main_diff > 0 else "Sklearn-Style更好"
            print(f"\n      🔍 {name}: MdAE差异 {main_diff:+.2f}% ({direction})")
            
            # 打印详细指标对比
            if config.DETAILED_ANALYSIS:
                print(f"         详细指标差异:")
                for metric, diff in metrics_diff.items():
                    direction_detail = "Legacy更好" if diff > 0 else "Sklearn-Style更好"
                    print(f"           - {metric}: {diff:+.2f}% ({direction_detail})")
        
        print("\n   💡 建议检查这些方法的参数配置或实现细节")
        print("   🔧 可能的原因包括:")
        print("      - 训练过程中的随机性")
        print("      - 数据预处理策略差异")
        print("      - 早停策略的微妙差异")
        print("      - 优化器或学习率调度差异")
        
    else:
        print("   ✅ 所有方法的性能差异都在可接受范围内 (<5%)")
        print("   💡 两种实现基本一致，性能差异可能来自:")
        print("      - 训练过程中的正常随机波动")
        print("      - 数值精度差异")
        print("      - 模型初始化的微小差异")
    
    # 打印最大和最小差异
    if all_diffs:
        max_diff = max(all_diffs, key=lambda x: abs(x[1]))
        min_diff = min(all_diffs, key=lambda x: abs(x[1]))
        
        print(f"\n📏 差异范围:")
        print(f"   - 最大差异: {max_diff[0]} ({max_diff[1]:+.2f}%)")
        print(f"   - 最小差异: {min_diff[0]} ({min_diff[1]:+.2f}%)")
        
        # 计算平均差异
        avg_abs_diff = np.mean([abs(diff[1]) for diff in all_diffs])
        print(f"   - 平均绝对差异: {avg_abs_diff:.2f}%")


def print_comparison_table(legacy_results, sklearn_results, config):
    """打印最终的性能对比表格"""
    print("\n\n" + "="*80)
    print("🔬 3. 性能对比分析")
    print("="*80)
    
    print("\n--- 实验配置 ---")
    print(f"学习率: {config.LEARNING_RATE}, 异常比例: {config.ANOMALY_RATIO}, "
          f"批处理大小: {config.BATCH_SIZE}")
    print(f"Causal Alpha: {config.ALPHA_CAUSAL}, Pytorch Alpha: {config.ALPHA_PYTORCH}")
    print(f"隐藏层: {config.HIDDEN_SIZES}")
    print("-" * 20)

    header = f"| {'Model':<22} | {'Implementation':<16} | {'MAE':<8} | {'MdAE':<8} | {'RMSE':<8} | {'R²':<8} | {'Diff %':<8} |"
    separator = "-" * len(header)
    
    print("\n" + separator)
    print(header)
    print(separator)

    # 模型映射：(config_key, legacy_key, sklearn_key, display_name)
    # 注意：legacy_key 使用 BaselineBenchmark 实际返回的结果键名
    models_map = [
        ('pytorch_mlp', 'pytorch_mlp', 'pytorch_mlp', 'PyTorch MLP'),  # 修正: BaselineBenchmark实际返回'pytorch_mlp'
        ('causal_standard', 'standard', 'standard', 'Causal (standard)'),
        ('causal_deterministic', 'deterministic', 'deterministic', 'Causal (deterministic)'),
        ('mlp_huber', 'mlp_huber', 'mlp_huber', 'MLP Huber'),
        ('mlp_pinball', 'mlp_pinball_median', 'mlp_pinball', 'MLP Pinball'),  # 修正: BaselineBenchmark返回'mlp_pinball_median'
        ('mlp_cauchy', 'mlp_cauchy', 'mlp_cauchy', 'MLP Cauchy'),
    ]

    for config_key, legacy_key, sklearn_key, display_name in models_map:
        if not config.MODELS_TO_TEST.get(config_key):
            continue

        legacy_result = None
        sklearn_result = None

        # Legacy results
        if legacy_key in legacy_results:
            legacy_result = legacy_results[legacy_key]['test']
            print(f"| {display_name:<22} | {'Legacy':<16} | {legacy_result['MAE']:.4f} | {legacy_result['MdAE']:.4f} | {legacy_result['RMSE']:.4f} | {legacy_result['R²']:.4f} | {'':<8} |")
        else:
            # 检查是否缺少Legacy结果
            print(f"| {display_name:<22} | {'Legacy':<16} | {'MISSING':<8} | {'MISSING':<8} | {'MISSING':<8} | {'MISSING':<8} | {'':<8} |")

        # Sklearn results
        if sklearn_key in sklearn_results:
            sklearn_result = sklearn_results[sklearn_key]['test']
            
            # 计算差异百分比 (以MdAE为主要指标)
            diff_pct = ""
            if legacy_result and sklearn_result:
                mdae_diff = ((sklearn_result['MdAE'] - legacy_result['MdAE']) / legacy_result['MdAE']) * 100
                diff_pct = f"{mdae_diff:+.2f}%"
            elif sklearn_result and not legacy_result:
                diff_pct = "N/A"
            
            print(f"| {display_name:<22} | {'Sklearn-Style':<16} | {sklearn_result['MAE']:.4f} | {sklearn_result['MdAE']:.4f} | {sklearn_result['RMSE']:.4f} | {sklearn_result['R²']:.4f} | {diff_pct:<8} |")
        
        if legacy_result or sklearn_result:
            print(separator)
    
    # 打印差异分析
    print("\n💡 差异分析:")
    print("   - Diff % 表示 Sklearn-Style 相对于 Legacy 在 MdAE 指标上的相对差异")
    print("   - 负值表示 Sklearn-Style 性能更好，正值表示 Legacy 性能更好")
    print("   - 如果差异很小(<5%)，说明两种实现基本一致")

def main():
    """主函数"""
    print("🔍 性能差异调试脚本")
    print("="*60)
    print("目标: 系统性对比每个回归算法的 Legacy vs Sklearn-Style 实现")
    print("方法: 在相同数据和参数下运行两种实现，并计算性能差异")
    print()
    
    config = ExperimentConfig()
    
    # 显示要测试的方法
    enabled_methods = [k for k, v in config.MODELS_TO_TEST.items() if v]
    print(f"📊 将测试以下 {len(enabled_methods)} 种方法:")
    for i, method in enumerate(enabled_methods, 1):
        print(f"   {i}. {method}")
    print()
    
    # 1. 加载和准备数据
    data = load_and_prepare_data(config)
    
    # 2a. 运行旧版实现
    legacy_results = run_legacy_benchmark(config, data)
    
    # 2b. 运行新版实现
    sklearn_results = run_sklearn_benchmark(config, data)

    # 3. 打印对比结果
    print_comparison_table(legacy_results, sklearn_results, config)
    
    # 4. 总结分析
    perform_detailed_analysis(legacy_results, sklearn_results, config)
    
    print("\n🎉 调试脚本运行完毕！")
    print("💡 使用建议:")
    print("   - 如需调整测试参数，请修改 ExperimentConfig 类中的配置")
    print("   - 如需测试特定方法，请在 MODELS_TO_TEST 中启用/禁用相应选项")
    print("   - 如需详细分析，请将 DETAILED_ANALYSIS 设为 True")
    print("   - 如需较大差异，请检查训练随机性或参数配置")


if __name__ == "__main__":
    main()
