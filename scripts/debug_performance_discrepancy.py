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

    # 🧠 模型超参数 (关键！在这里对齐新旧实现)
    # ----------------------------------------------------------------------
    # 假设1: 尝试复现旧脚本的 "更好" 性能
    # 旧脚本的学习率是 0.01，而新脚本是 0.001。这是最大的嫌疑。
    LEARNING_RATE = 0.01

    # 正则化: 新脚本中 CausalEngine 的 alpha 明确为 0，Pytorch MLP 为 0.0001
    # 我们在这里保持一致，并假设旧脚本的默认行为与此类似。
    ALPHA_CAUSAL = 0.0
    ALPHA_PYTORCH = 0.0001
    
    # 批处理大小: 新的 sklearn-style regressor 默认为 'auto' (即200)
    # 我们假设旧的 Benchmark 也是类似行为。可以修改为 `None` 来测试全量批次。
    BATCH_SIZE = 200 # 使用 'auto' 的默认值
    # BATCH_SIZE = None # 设为None来强制使用全量批次

    # 其他通用参数 (在两个脚本中基本一致)
    HIDDEN_SIZES = (128, 64, 32)
    MAX_EPOCHS = 3000
    PATIENCE = 50
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

    # 🎯 关键改进：传递原始干净数据，让BaselineBenchmark自己处理异常注入
    # 这确保了与Sklearn-Style实现完全相同的异常注入和数据处理流程
    X_combined_original = np.concatenate([data['X_train_original'], data['X_val_original'], data['X_test_original']])
    y_combined_original = np.concatenate([data['y_train_original'], data['y_val_original'], data['y_test_original']])
    
    print(f"   - 传递原始干净数据: X({X_combined_original.shape}), y({y_combined_original.shape})")
    print(f"   - 让BaselineBenchmark自己处理异常注入，确保完全一致")
    
    results = benchmark.compare_models(
        X=X_combined_original,  # 🎯 原始干净特征
        y=y_combined_original,  # 🎯 原始干净目标
        task_type='regression',
        baseline_methods=baseline_methods,
        causal_modes=causal_modes,
        global_standardization=True,  # 🎯 启用全局标准化以匹配Sklearn-Style实现
        # 数据参数 - 使用相同的分割比例和异常注入比例
        test_size=len(data['X_test_original']) / len(X_combined_original),
        val_size=len(data['X_val_original']) / (len(data['X_train_original']) + len(data['X_val_original'])),
        anomaly_ratio=config.ANOMALY_RATIO,  # 🎯 关键：让BaselineBenchmark执行相同的异常注入
        anomaly_strategy='shuffle',  # 使用新的简化策略
        random_state=config.RANDOM_STATE,
        # 统一模型参数
        hidden_sizes=config.HIDDEN_SIZES,
        max_epochs=config.MAX_EPOCHS,
        lr=config.LEARNING_RATE,
        patience=config.PATIENCE,
        tol=config.TOL,
        alpha=config.ALPHA_PYTORCH, # for pytorch_mlp
        batch_size=config.BATCH_SIZE,
        # CausalEngine 参数
        gamma_init=config.GAMMA_INIT,
        b_noise_init=config.B_NOISE_INIT,
        b_noise_trainable=config.B_NOISE_TRAINABLE,
        causal_alpha=config.ALPHA_CAUSAL,
        verbose=False # 保持输出整洁
    )
    
    # 🎯 关键改进：BaselineBenchmark现在自动处理全局标准化和逆变换
    print("   - BaselineBenchmark将自动处理标准化和逆变换")
    
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
    
    # 🎯 关键改进：使用原始数据但手动实施全局标准化，确保与 Legacy 实现完全一致
    # 组合训练集和验证集（含异常的原始数据）
    X_train_val_original = np.concatenate([data['X_train'], data['X_val']])
    y_train_val_original = np.concatenate([data['y_train'], data['y_val']])
    X_test_original = data['X_test']
    y_test_original = data['y_test']
    
    # 手动实施全局标准化策略（与Legacy保持一致）
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_train_val_scaled = scaler_X.fit_transform(X_train_val_original)
    X_test_scaled = scaler_X.transform(X_test_original)
    
    scaler_y = StandardScaler()
    y_train_val_scaled = scaler_y.fit_transform(y_train_val_original.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_original.reshape(-1, 1)).flatten()
    
    print(f"   - 手动全局标准化: X_train_val({X_train_val_scaled.shape}), y_train_val({y_train_val_scaled.shape})")
    print(f"   - 确保与Legacy实现使用完全相同的标准化策略")

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
        
        # 在原始尺度下计算性能指标
        results[result_key] = {
            'test': {
                'MAE': mean_absolute_error(y_test_original, y_pred_original),
                'MdAE': median_absolute_error(y_test_original, y_pred_original),
                'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
                'R²': r2_score(y_test_original, y_pred_original)
            },
            'time': time.time() - start_time
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
    return results

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
        ('pytorch_mlp', 'pytorch', 'pytorch_mlp', 'PyTorch MLP'),
        ('causal_standard', 'standard', 'standard', 'Causal (standard)'),
        ('causal_deterministic', 'deterministic', 'deterministic', 'Causal (deterministic)'),
        ('mlp_huber', 'mlp_huber', 'mlp_huber', 'MLP Huber'),
        ('mlp_pinball', 'mlp_pinball_median', 'mlp_pinball', 'MLP Pinball'),
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

        # Sklearn results
        if sklearn_key in sklearn_results:
            sklearn_result = sklearn_results[sklearn_key]['test']
            
            # 计算差异百分比 (以MdAE为主要指标)
            diff_pct = ""
            if legacy_result and sklearn_result:
                mdae_diff = ((sklearn_result['MdAE'] - legacy_result['MdAE']) / legacy_result['MdAE']) * 100
                diff_pct = f"{mdae_diff:+.2f}%"
            
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
    print("\n📈 总结分析:")
    significant_diffs = []
    for config_key, legacy_key, sklearn_key, display_name in [
        ('pytorch_mlp', 'PyTorch MLP', 'pytorch_mlp', 'PyTorch MLP'),
        ('causal_standard', 'standard', 'standard', 'Causal (standard)'),
        ('causal_deterministic', 'deterministic', 'deterministic', 'Causal (deterministic)'),
        ('mlp_huber', 'MLP Huber', 'mlp_huber', 'MLP Huber'),
        ('mlp_pinball', 'MLP Pinball Median', 'mlp_pinball', 'MLP Pinball'),
        ('mlp_cauchy', 'MLP Cauchy', 'mlp_cauchy', 'MLP Cauchy'),
    ]:
        if (config.MODELS_TO_TEST.get(config_key) and 
            legacy_key in legacy_results and sklearn_key in sklearn_results):
            
            legacy_mdae = legacy_results[legacy_key]['test']['MdAE']
            sklearn_mdae = sklearn_results[sklearn_key]['test']['MdAE']
            diff_pct = ((sklearn_mdae - legacy_mdae) / legacy_mdae) * 100
            
            if abs(diff_pct) > 5.0:  # 差异超过5%
                significant_diffs.append((display_name, diff_pct))
    
    if significant_diffs:
        print(f"   ⚠️ 发现 {len(significant_diffs)} 个方法存在显著性能差异 (>5%):")
        for name, diff in significant_diffs:
            direction = "Sklearn-Style更差" if diff > 0 else "Sklearn-Style更好"
            print(f"      - {name}: {diff:+.2f}% ({direction})")
        print("   💡 建议检查这些方法的参数配置或实现细节")
    else:
        print("   ✅ 所有方法的性能差异都在可接受范围内 (<5%)")
        print("   💡 两种实现基本一致，性能差异可能来自其他因素")
    
    print("\n🎉 调试脚本运行完毕！")
    print("💡 如需调整测试参数，请修改 ExperimentConfig 类中的配置")
    print("🔧 如需测试特定方法，请在 MODELS_TO_TEST 中启用/禁用相应选项")


if __name__ == "__main__":
    main()
