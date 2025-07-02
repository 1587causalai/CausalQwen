#!/usr/bin/env python3
"""
CausalEngine 快速测试脚本 - 全局标准化版本

🎯 核心理念：绝对公平的竞技场
- 全局标准化：对 X 和 y 都进行标准化
- 统一输入：所有模型接收完全标准化的数据  
- 统一评估：所有预测结果都转换回原始尺度进行评估

这确保了：
- 所有模型在相同的抽象空间中学习
- 稳健回归器不能利用未缩放数据的优势
- CausalEngine 在困难环境下展示其真正能力

使用说明:
1. 修改 CONFIG 部分的参数
2. 运行 python scripts/quick_test_causal_engine.py
3. 根据需要启用/禁用特定的模型比较
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import warnings

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的CausalEngine实现
from causal_sklearn.regressor import MLPCausalRegressor, MLPPytorchRegressor
from causal_sklearn.classifier import MLPCausalClassifier, MLPPytorchClassifier
from causal_sklearn.data_processing import inject_shuffle_noise

warnings.filterwarnings('ignore')

# =============================================================================
# 配置部分 - 在这里修改实验参数
# =============================================================================

REGRESSION_CONFIG = {
    # 数据生成
    'n_samples': 4000,  # 更大规模
    'n_features': 12,
    'noise': 1.0,
    'random_state': 42,
    'test_size': 0.2,  # 测试集比例
    'anomaly_ratio': 0.4,  # 40%异常数据，匹配其他脚本
    
    # 网络结构
    'perception_hidden_layers': (128, 64, 32),  # 统一网络结构
    'abduction_hidden_layers': (),
    'repre_size': None,
    'causal_size': None,
    
    # CausalEngine参数
    'gamma_init': 1.0,
    'b_noise_init': 1.0,
    'b_noise_trainable': True,
    'alpha': 0.0001, # 添加L2正则化，与sklearn默认一致
    
    # 训练参数
    'max_iter': 3000,  # 统一最大迭代次数
    'learning_rate': 0.01,  # 降低学习率，更接近sklearn默认
    'patience': 50,  # 减少patience，更接近sklearn默认
    'tol': 1e-4,  # 更接近sklearn默认tolerance
    'validation_fraction': 0.2,
    'batch_size': None,  # 统一使用全量训练(full-batch)
    
    # 测试控制
    'test_sklearn': True,
    'test_pytorch': True,
    'test_causal_deterministic': True,
    'test_causal_standard': True,
    'verbose': True
}

CLASSIFICATION_CONFIG = {
    # 数据生成 - 与sklearn更相似的设置
    'n_samples': 4000,  # 减少样本量，更像sklearn经典测试
    'n_features': 10,   # 减少特征数
    'n_classes': 3,
    'class_sep': 1.0,   # 提高类别分离度
    'random_state': 42,
    'test_size': 0.2,   # 测试集比例
    'label_noise_ratio': 0.4,  # 统一标签噪声水平
    
    # 网络结构 - 更简单的网络
    'perception_hidden_layers': (128, 64, 32),  # 统一网络结构
    'abduction_hidden_layers': (),
    'repre_size': None,
    'causal_size': None,
    
    # CausalEngine参数
    'gamma_init': 1.0,
    'b_noise_init': 1.0,
    'b_noise_trainable': True,
    'ovr_threshold': 2.0,
    'alpha': 0.0,  # 匹配sklearn默认L2正则化
    
    # 训练参数 - 更接近sklearn默认值
    'max_iter': 3000,   # 减少最大迭代次数
    'learning_rate': 0.01,  # 使用sklearn默认学习率
    'patience': 10,     # 使用sklearn默认patience
    'tol': 1e-4,        # 匹配sklearn默认tolerance
    'validation_fraction': 0.2,  # 使用sklearn默认验证集比例
    'batch_size': None,  # 统一使用全量训练(full-batch)
    
    # 测试控制
    'test_sklearn': True,
    'test_pytorch': True,
    'test_causal_deterministic': True,
    'test_causal_standard': True,
    'verbose': True
}

# =============================================================================
# 数据生成函数
# =============================================================================

def generate_regression_data(config):
    """生成回归测试数据 - 全局标准化版本"""
    print(f"📊 生成回归数据: {config['n_samples']}样本, {config['n_features']}特征, 噪声={config['noise']}")
    
    # 生成基础数据
    X, y = make_regression(
        n_samples=config['n_samples'], 
        n_features=config['n_features'],
        noise=config['noise'], 
        random_state=config['random_state']
    )
    
    # 进行数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    # 对训练集标签进行异常注入
    if config['anomaly_ratio'] > 0:
        y_train_noisy, noise_indices = inject_shuffle_noise(
            y_train, 
            noise_ratio=config['anomaly_ratio'],
            random_state=config['random_state']
        )
        y_train = y_train_noisy
        print(f"   异常注入: {config['anomaly_ratio']:.1%} ({len(noise_indices)}/{len(y_train)} 样本受影响)")
    else:
        print(f"   无异常注入: 纯净环境")
    
    print(f"   训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    # 🎯 全局标准化策略
    print(f"   🎯 实施全局标准化策略:")
    
    # 特征标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 目标标准化（关键！）
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"      - X 和 y 都已标准化")
    print(f"      - 所有模型将在标准化空间中竞争")
    
    data = {
        # 原始数据（用于最终评估）
        'X_train_original': X_train, 'X_test_original': X_test,
        'y_train_original': y_train, 'y_test_original': y_test,
        
        # 标准化数据（用于模型训练）
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_scaled, 'y_test': y_test_scaled,
        
        # 标准化器（用于逆变换）
        'scaler_X': scaler_X, 'scaler_y': scaler_y
    }
    
    return data

def generate_classification_data(config):
    """生成分类测试数据 - 全局标准化版本"""
    print(f"📊 生成分类数据: {config['n_samples']}样本, {config['n_features']}特征, {config['n_classes']}类别")
    
    n_informative = min(config['n_features'], max(2, config['n_features'] // 2))
    
    # 生成基础数据
    X, y = make_classification(
        n_samples=config['n_samples'], 
        n_features=config['n_features'], 
        n_classes=config['n_classes'],
        n_informative=n_informative, 
        n_redundant=0, 
        n_clusters_per_class=1,
        class_sep=config['class_sep'], 
        random_state=config['random_state']
    )
    
    # 进行数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        stratify=y
    )
    
    # 对训练集标签进行异常注入
    if config['label_noise_ratio'] > 0:
        y_train_noisy, noise_indices = inject_shuffle_noise(
            y_train, 
            noise_ratio=config['label_noise_ratio'],
            random_state=config['random_state']
        )
        y_train = y_train_noisy
        print(f"   标签噪声: {config['label_noise_ratio']:.1%} ({len(noise_indices)}/{len(y_train)} 样本受影响)")
    else:
        print(f"   无标签噪声: 纯净环境")
    
    print(f"   训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    # 🎯 全局标准化策略 - 仅对特征进行标准化（分类任务y不需要标准化）
    print(f"   🎯 实施全局标准化策略:")
    
    # 特征标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"      - X 已标准化，y 保持原始（分类标签）")
    print(f"      - 所有模型将在标准化特征空间中竞争")
    
    data = {
        # 原始数据（用于参考）
        'X_train_original': X_train, 'X_test_original': X_test,
        'y_train_original': y_train, 'y_test_original': y_test,
        
        # 标准化数据（用于模型训练）
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,  # 分类标签不标准化
        
        # 标准化器
        'scaler_X': scaler_X
    }
    
    return data

# =============================================================================
# 模型训练函数
# =============================================================================

def train_sklearn_regressor(data, config):
    """训练sklearn回归器"""
    print("🔧 训练 sklearn MLPRegressor...")
    
    model = MLPRegressor(
        hidden_layer_sizes=config['perception_hidden_layers'],
        max_iter=config['max_iter'],
        learning_rate_init=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        tol=config['tol'],
        random_state=config['random_state'],
        alpha=config['alpha'],
        batch_size=len(data['X_train']) if config['batch_size'] is None else config['batch_size']
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    if config['verbose']:
        print(f"   训练完成: {model.n_iter_} epochs")
    
    return model

def train_sklearn_classifier(data, config):
    """训练sklearn分类器"""
    print("🔧 训练 sklearn MLPClassifier...")
    
    model = MLPClassifier(
        hidden_layer_sizes=config['perception_hidden_layers'],
        max_iter=config['max_iter'],
        learning_rate_init=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        tol=config['tol'],
        random_state=config['random_state'],
        alpha=config['alpha'],
        batch_size=len(data['X_train']) if config['batch_size'] is None else config['batch_size']
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    if config['verbose']:
        print(f"   训练完成: {model.n_iter_} epochs")
    
    return model

def train_pytorch_regressor(data, config):
    """训练PyTorch回归器"""
    print("🔧 训练 PyTorch MLPRegressor...")
    
    model = MLPPytorchRegressor(
        hidden_layer_sizes=config['perception_hidden_layers'],
        max_iter=config['max_iter'],
        learning_rate=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        tol=config['tol'],
        random_state=config['random_state'],
        verbose=config['verbose'],
        alpha=config['alpha'],
        batch_size=len(data['X_train']) if config['batch_size'] is None else config['batch_size']
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    if config['verbose']:
        n_iter = model.n_iter_
        print(f"   训练完成: {n_iter} epochs")
    
    return model

def train_pytorch_classifier(data, config):
    """训练PyTorch分类器"""
    print("🔧 训练 PyTorch MLPClassifier...")
    
    model = MLPPytorchClassifier(
        hidden_layer_sizes=config['perception_hidden_layers'],
        max_iter=config['max_iter'],
        learning_rate=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        tol=config['tol'],
        random_state=config['random_state'],
        verbose=config['verbose'],
        alpha=config['alpha'],
        batch_size=len(data['X_train']) if config['batch_size'] is None else config['batch_size']
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    if config['verbose']:
        n_iter = model.n_iter_
        print(f"   训练完成: {n_iter} epochs")
    
    return model

def train_causal_regressor(data, config, mode='standard'):
    """训练因果回归器"""
    print(f"🔧 训练 CausalRegressor ({mode})...")
    
    model = MLPCausalRegressor(
        repre_size=config['repre_size'],
        causal_size=config['causal_size'],
        perception_hidden_layers=config['perception_hidden_layers'],
        abduction_hidden_layers=config['abduction_hidden_layers'],
        mode=mode,
        gamma_init=config['gamma_init'],
        b_noise_init=config['b_noise_init'],
        b_noise_trainable=config['b_noise_trainable'],
        max_iter=config['max_iter'],
        learning_rate=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        tol=config['tol'],
        random_state=config['random_state'],
        verbose=config['verbose'],
        alpha=config['alpha'],
        batch_size=len(data['X_train']) if config['batch_size'] is None else config['batch_size']
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    if config['verbose']:
        n_iter = model.n_iter_
        print(f"   训练完成: {n_iter} epochs")
    
    return model

def train_causal_classifier(data, config, mode='standard'):
    """训练因果分类器"""
    print(f"🔧 训练 CausalClassifier ({mode})...")
    
    model = MLPCausalClassifier(
        repre_size=config['repre_size'],
        causal_size=config['causal_size'],
        perception_hidden_layers=config['perception_hidden_layers'],
        abduction_hidden_layers=config['abduction_hidden_layers'],
        mode=mode,
        gamma_init=config['gamma_init'],
        b_noise_init=config['b_noise_init'],
        b_noise_trainable=config['b_noise_trainable'],
        ovr_threshold=config['ovr_threshold'],
        max_iter=config['max_iter'],
        learning_rate=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        tol=config['tol'],
        random_state=config['random_state'],
        verbose=config['verbose'],
        alpha=config['alpha'],
        batch_size=len(data['X_train']) if config['batch_size'] is None else config['batch_size']
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    if config['verbose']:
        n_iter = model.n_iter_
        print(f"   训练完成: {n_iter} epochs")
    
    return model

# =============================================================================
# 评估函数
# =============================================================================

def evaluate_regression(y_true, y_pred):
    """回归评估指标"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MdAE': median_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred)
    }

def evaluate_classification(y_true, y_pred, n_classes):
    """分类评估指标"""
    avg_method = 'binary' if n_classes == 2 else 'macro'
    return {
        'Acc': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=avg_method, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=avg_method, zero_division=0),
        'F1': f1_score(y_true, y_pred, average=avg_method, zero_division=0)
    }

def predict_and_evaluate_regression(model, data, model_name, config):
    """回归模型预测和评估 - 全局标准化版本"""
    # 在标准化空间中预测
    test_pred_scaled = model.predict(data['X_test'])
    
    # 🎯 关键：将预测结果转换回原始尺度进行评估
    test_pred_original = data['scaler_y'].inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    
    # 验证集：重新分割来评估
    X_train_pt, X_val_pt, y_train_pt, y_val_pt = train_test_split(
        data['X_train'], data['y_train'],
        test_size=config['validation_fraction'],
        random_state=config['random_state']
    )
    val_pred_scaled = model.predict(X_val_pt)
    
    # 将验证集预测结果也转换回原始尺度
    val_pred_original = data['scaler_y'].inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    y_val_original = data['scaler_y'].inverse_transform(y_val_pt.reshape(-1, 1)).flatten()
    
    # 在原始尺度下评估性能
    results = {
        'test': evaluate_regression(data['y_test_original'], test_pred_original),
        'val': evaluate_regression(y_val_original, val_pred_original)
    }
    
    return results

def predict_and_evaluate_classification(model, data, model_name, config):
    """分类模型预测和评估 - 全局标准化版本"""
    n_classes = len(np.unique(data['y_train']))
    
    # 在标准化特征空间中预测（分类标签无需转换）
    test_pred = model.predict(data['X_test'])
    
    # 验证集：重新分割来评估
    X_train_pt, X_val_pt, y_train_pt, y_val_pt = train_test_split(
        data['X_train'], data['y_train'],
        test_size=config['validation_fraction'],
        random_state=config['random_state'],
        stratify=data['y_train']
    )
    val_pred = model.predict(X_val_pt)
    
    # 分类任务：标签无需转换，直接在原始标签空间评估
    results = {
        'test': evaluate_classification(data['y_test'], test_pred, n_classes),
        'val': evaluate_classification(y_val_pt, val_pred, n_classes)
    }
    
    return results

# =============================================================================
# 结果显示函数
# =============================================================================

def print_regression_results(results):
    """打印回归结果"""
    print("\n📊 回归结果对比:")
    print("=" * 120)
    print(f"{'方法':<20} {'验证集':<50} {'测试集':<50}")
    print(f"{'':20} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 120)
    
    for method, metrics in results.items():
        val_m = metrics['val']
        test_m = metrics['test']
        print(f"{method:<20} {val_m['MAE']:<10.4f} {val_m['MdAE']:<10.4f} {val_m['RMSE']:<10.4f} {val_m['R²']:<10.4f} "
              f"{test_m['MAE']:<10.4f} {test_m['MdAE']:<10.4f} {test_m['RMSE']:<10.4f} {test_m['R²']:<10.4f}")
    
    print("=" * 120)

def print_classification_results(results, n_classes):
    """打印分类结果"""
    print(f"\n📊 {n_classes}分类结果对比:")
    print("=" * 120)
    print(f"{'方法':<20} {'验证集':<50} {'测试集':<50}")
    print(f"{'':20} {'Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 120)
    
    for method, metrics in results.items():
        val_m = metrics['val']
        test_m = metrics['test']
        print(f"{method:<20} {val_m['Acc']:<10.4f} {val_m['Precision']:<10.4f} {val_m['Recall']:<10.4f} {val_m['F1']:<10.4f} "
              f"{test_m['Acc']:<10.4f} {test_m['Precision']:<10.4f} {test_m['Recall']:<10.4f} {test_m['F1']:<10.4f}")
    
    print("=" * 120)

# =============================================================================
# 主测试函数
# =============================================================================

def test_regression(config=None):
    """回归任务测试"""
    if config is None:
        config = REGRESSION_CONFIG
    
    print("\n🔬 回归任务测试")
    print("=" * 80)
    print_config_summary(config, 'regression')
    
    # 1. 生成数据
    data = generate_regression_data(config)
    results = {}
    
    # 2. 训练各种模型
    if config['test_sklearn']:
        sklearn_model = train_sklearn_regressor(data, config)
        results['sklearn'] = predict_and_evaluate_regression(sklearn_model, data, 'sklearn', config)
    
    if config['test_pytorch']:
        pytorch_model = train_pytorch_regressor(data, config)
        results['pytorch'] = predict_and_evaluate_regression(pytorch_model, data, 'causal', config)
    
    if config['test_causal_deterministic']:
        causal_det = train_causal_regressor(data, config, 'deterministic')
        results['deterministic'] = predict_and_evaluate_regression(causal_det, data, 'causal', config)
    
    if config['test_causal_standard']:
        causal_std = train_causal_regressor(data, config, 'standard')
        results['standard'] = predict_and_evaluate_regression(causal_std, data, 'causal', config)
    
    # 3. 显示结果
    if config['verbose']:
        print_regression_results(results)
    
    return results

def test_classification(config=None):
    """分类任务测试"""
    if config is None:
        config = CLASSIFICATION_CONFIG
    
    print("\n🎯 分类任务测试")
    print("=" * 80)
    print_config_summary(config, 'classification')
    
    # 1. 生成数据
    data = generate_classification_data(config)
    results = {}
    
    # 2. 训练各种模型
    if config['test_sklearn']:
        sklearn_model = train_sklearn_classifier(data, config)
        results['sklearn'] = predict_and_evaluate_classification(sklearn_model, data, 'sklearn', config)
    
    if config['test_pytorch']:
        pytorch_model = train_pytorch_classifier(data, config)
        results['pytorch'] = predict_and_evaluate_classification(pytorch_model, data, 'causal', config)
    
    if config['test_causal_deterministic']:
        causal_det = train_causal_classifier(data, config, 'deterministic')
        results['deterministic'] = predict_and_evaluate_classification(causal_det, data, 'causal', config)
    
    if config['test_causal_standard']:
        causal_std = train_causal_classifier(data, config, 'standard')
        results['standard'] = predict_and_evaluate_classification(causal_std, data, 'causal', config)
    
    # 3. 显示结果
    if config['verbose']:
        n_classes = len(np.unique(data['y_train']))
        print_classification_results(results, n_classes)
    
    return results

def print_config_summary(config, task_type):
    """打印配置摘要"""
    if task_type == 'regression':
        print(f"数据: {config['n_samples']}样本, {config['n_features']}特征, 噪声={config['noise']}")
        print(f"异常: {config['anomaly_ratio']:.1%} 异常数据注入")
    else:
        print(f"数据: {config['n_samples']}样本, {config['n_features']}特征, {config['n_classes']}类别")
        print(f"噪声: {config['label_noise_ratio']:.1%} 标签噪声, 分离度={config['class_sep']}")
    
    print(f"网络: {config['perception_hidden_layers']}")
    print(f"训练: {config['max_iter']} epochs, lr={config['learning_rate']}, patience={config['patience']}")
    print(f"测试: sklearn={config['test_sklearn']}, pytorch={config['test_pytorch']}, "
          f"deterministic={config['test_causal_deterministic']}, standard={config['test_causal_standard']}")
    print()

# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序 - 运行所有测试"""
    print("🚀 CausalEngine 快速测试脚本 - 全局标准化版")
    print("=" * 60)
    
    # 运行回归测试
    regression_results = test_regression()
    
    # 运行分类测试  
    classification_results = test_classification()
    
    print(f"\n✅ 测试完成!")
    print("💡 修改脚本顶部的 CONFIG 部分来调整实验参数")

def quick_regression_test():
    """快速回归测试 - 用于调试"""
    quick_config = REGRESSION_CONFIG.copy()
    quick_config.update({
        'n_samples': 1000,
        'max_iter': 500,
        'test_pytorch': False,  # 跳过pytorch基线以节省时间
        'verbose': True
    })
    return test_regression(quick_config)

def quick_classification_test():
    """快速分类测试 - 用于调试"""
    quick_config = CLASSIFICATION_CONFIG.copy()
    quick_config.update({
        'n_samples': 1500,
        'max_iter': 500,
        'test_pytorch': False,  # 跳过pytorch基线以节省时间
        'verbose': True
    })
    return test_classification(quick_config)

if __name__ == "__main__":
    # 你可以选择运行以下任一函数:
    main()                        # 完整测试
    # quick_regression_test()     # 快速回归测试
    # quick_classification_test() # 快速分类测试