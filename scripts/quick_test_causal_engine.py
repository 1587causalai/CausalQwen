#!/usr/bin/env python3
"""
🚀 CausalEngine 快速测试脚本
========================

╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    CausalEngine 快速测试流程架构图                                                  ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                    ║
║  📊 数据处理管道 (Data Processing Pipeline)                                                                         ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  1. 合成数据生成 → 2. 数据分割 → 3. 标准化(无泄露) → 4. 噪声注入(仅训练集) → 5. 模型训练                          │  ║
║  │                                                                                                                │  ║
║  │  Regression:         Train/Test     X & Y         30% Label     8种方法                                        │  ║
║  │  4000 samples    →   Split      →   Scaling   →   Noise     →   快速对比                                        │  ║
║  │  12 features         (80/20)       (基于干净数据)   (训练集Only)   Performance                                    │  ║
║  │                                     📌无数据泄露    📌测试集保持纯净                                             │  ║
║  │                                                                                                                │  ║
║  │  Classification:                                                                                               │  ║
║  │  4000 samples, 10 features, 3 classes → 同样流程 → 30% Label Noise → 8种方法对比                              │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                    ║
║  🔧 统一参数传递机制 (Unified Parameter Passing)                                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  REGRESSION_CONFIG / CLASSIFICATION_CONFIG                                                                    │  ║
║  │  ├─ 🧠 统一神经网络配置                                                                                          │  ║
║  │  │   ├─ perception_hidden_layers = (128, 64, 32)  # 所有神经网络使用相同架构                                    │  ║
║  │  │   ├─ max_iter = 3000                          # 统一最大训练轮数                                            │  ║
║  │  │   ├─ learning_rate = 0.01                     # 统一学习率                                                  │  ║
║  │  │   ├─ patience = 50                            # 统一早停patience                                            │  ║
║  │  │   └─ batch_size = None                        # 统一批处理大小                                              │  ║
║  │  │                                                                                                            │  ║
║  │  ├─ 🎯 CausalEngine特定参数                                                                                     │  ║
║  │  │   ├─ gamma_init = 1.0, b_noise_init = 1.0     # CausalEngine初始化参数                                      │  ║
║  │  │   ├─ modes = ['deterministic', 'exogenous', 'endogenous', 'standard']                                     │  ║
║  │  │   └─ alpha = 0.0001                           # L2正则化                                                    │  ║
║  │  │                                                                                                            │  ║
║  │  └─ 📊 实验控制参数                                                                                              │  ║
║  │      ├─ anomaly_ratio = 0.3 (回归)              # 30%异常/噪声数据                                             │  ║
║  │      ├─ label_noise_ratio = 0.3 (分类)         # 30%标签噪声                                                  │  ║
║  │      ├─ test_size = 0.2                         # 测试集比例                                                   │  ║
║  │      └─ random_state = 42                       # 随机种子                                                    │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                    ║
║  🔄 8种方法对比架构 (8-Method Comparison Framework)                                                                 ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  传统神经网络 (4种)                            因果推理 (4种)                                                    │  ║
║  │  ├─ sklearn MLP                              ├─ CausalEngine (deterministic)                                  │  ║
║  │  ├─ PyTorch MLP                              ├─ CausalEngine (exogenous)                                      │  ║
║  │  ├─ sklearn OvR MLP (分类)                   ├─ CausalEngine (endogenous)                                     │  ║
║  │  └─ PyTorch Shared OvR (分类)                └─ CausalEngine (standard)                                       │  ║
║  │                                                                                                                │  ║
║  │  💡 关键设计亮点:                                                                                                │  ║
║  │  • 快速验证设计: 合成数据 + 中等规模，快速验证CausalEngine基本性能                                               │  ║
║  │  • 科学实验设计: 先基于干净数据标准化，再在训练集注入30%噪声，测试集保持纯净                                      │  ║
║  │  • 无数据泄露: 标准化器只在干净训练数据上fit，避免噪声污染统计量                                                  │  ║
║  │  • 统一数据标准化策略: 所有方法在标准化空间训练，确保实验公平性                                                     │  ║
║  │  • 参数公平性: 所有神经网络使用统一配置，确保公平比较                                                             │  ║
║  │  • 评估一致性: 所有方法在原始尺度下评估，便于结果解释                                                             │  ║
║  │  • 双任务支持: 回归和分类任务并行测试，全面验证CausalEngine能力                                                   │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                    ║
║  📈 输出分析 (Analysis & Visualization)                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  1. 数据分析图        2. 回归性能对比      3. 分类性能对比      4. 快速测试报告                                   │  ║
║  │     (特征分布)           (MAE, RMSE, R²)    (Acc, F1, Precision)  (性能排名)                                    │  ║
║  │                                                                                                                │  ║
║  │  📊 评估指标:                                                                                                   │  ║
║  │  • 回归: MAE, MdAE, RMSE, R² (原始尺度下统一评估)                                                              │  ║
║  │  • 分类: Accuracy, Precision, Recall, F1 (原始标签空间评估)                                                    │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

🚀 目标：快速验证 CausalEngine 的基本性能
🎯 核心：统一标准化策略，确保公平比较

主要特性：
- 支持回归和分类任务
- 统一数据预处理：所有方法使用相同的标准化策略
- 快速对比：sklearn MLP, PyTorch MLP, CausalEngine (4种模式)
- 科学评估：训练时抗噪声，评估时在原始尺度上统一比较
- 双任务验证：同时测试回归和分类性能

使用方法：
1. 直接运行：python scripts/quick_test_causal_engine.py
2. 调整参数：修改下方的 REGRESSION_CONFIG 和 CLASSIFICATION_CONFIG
3. 快速测试：使用 quick_regression_test() 或 quick_classification_test()

数据流程：
合成数据生成 → 训练/测试分割 → 标准化(无泄露) → 噪声注入(仅训练集) → 模型训练 → 原始尺度评估
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

# 设置matplotlib后端，避免弹出窗口
plt.switch_backend('Agg')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的CausalEngine实现
from causal_sklearn.regressor import MLPCausalRegressor, MLPPytorchRegressor
from causal_sklearn.classifier import MLPCausalClassifier, MLPPytorchClassifier, MLPSklearnOvRClassifier, MLPPytorchSharedOvRClassifier
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
    'anomaly_ratio': 0.3,  # 40%异常数据，匹配其他脚本
    
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
    'test_sklearn_ovr': True,
    'test_pytorch_shared_ovr': True,
    'test_causal_deterministic': True,
    'test_causal_exogenous': True,
    'test_causal_endogenous': True,
    'test_causal_standard': True,
    'verbose': True,
    
    # 可视化控制
    'save_plots': True,
    'output_dir': 'results/quick_test_results',
    'figure_dpi': 300
}

CLASSIFICATION_CONFIG = {
    # 数据生成 - 与sklearn更相似的设置
    'n_samples': 4000,  # 减少样本量，更像sklearn经典测试
    'n_features': 10,   # 减少特征数
    'n_classes': 3,
    'class_sep': 1.0,   # 提高类别分离度
    'random_state': 42,
    'test_size': 0.2,   # 测试集比例
    'label_noise_ratio': 0.3,  # 统一标签噪声水平
    
    # 网络结构 - 更简单的网络
    'perception_hidden_layers': (128, 64, 32),  # 统一网络结构
    'abduction_hidden_layers': (),
    'repre_size': None,
    'causal_size': None,
    
    # CausalEngine参数
    'gamma_init': 1.0,
    'b_noise_init': 1.0,
    'b_noise_trainable': True,
    'ovr_threshold': 0.0,
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
    'test_sklearn_ovr': True,
    'test_pytorch_shared_ovr': True,
    'test_causal_deterministic': True,
    'test_causal_exogenous': True,
    'test_causal_endogenous': True,
    'test_causal_standard': True,
    'verbose': True,
    
    # 可视化控制
    'save_plots': True,
    'output_dir': 'results/quick_test_results',
    'figure_dpi': 300
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
    
    print(f"   训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    # 🎯 全局标准化策略 (先标准化，后注入噪声)
    print(f"   🎯 实施全局标准化策略 (先标准化，后注入噪声):")
    
    # 特征标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 目标标准化（关键！）- 在干净数据上拟合
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    print(f"      - X 和 y 都已在干净数据上完成标准化")

    # 对 *标准化后* 的训练集标签进行异常注入
    if config['anomaly_ratio'] > 0:
        y_train_scaled_noisy, noise_indices = inject_shuffle_noise(
            y_train_scaled, 
            noise_ratio=config['anomaly_ratio'],
            random_state=config['random_state']
        )
        y_train_for_training = y_train_scaled_noisy
        print(f"   异常注入: 在标准化空间中对 {config['anomaly_ratio']:.1%} 的标签注入噪声")
        print(f"             ({len(noise_indices)}/{len(y_train)} 样本受影响)")
    else:
        y_train_for_training = y_train_scaled
        print(f"   无异常注入: 纯净环境")
    
    print(f"      - 所有模型将在含噪标准化空间中竞争")
    
    data = {
        # 原始数据（用于最终评估）
        'X_train_original': X_train, 'X_test_original': X_test,
        'y_train_original': y_train, 'y_test_original': y_test,
        
        # 标准化数据（用于模型训练）
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_for_training,
        'y_test': y_test_scaled,
        
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
    
    print(f"   训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    # 🎯 无泄露标准化策略：先标准化特征，后注入标签噪声
    print(f"   🎯 实施无泄露标准化策略（先标准化，后注入噪声）:")
    
    # 特征标准化 - 在干净数据上fit
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    print(f"      - X 已在干净数据上完成标准化")
    
    # 对训练集标签进行异常注入（在标准化后）
    if config['label_noise_ratio'] > 0:
        y_train_noisy, noise_indices = inject_shuffle_noise(
            y_train, 
            noise_ratio=config['label_noise_ratio'],
            random_state=config['random_state']
        )
        y_train_for_training = y_train_noisy
        print(f"   标签噪声: {config['label_noise_ratio']:.1%} ({len(noise_indices)}/{len(y_train)} 样本受影响)")
    else:
        y_train_for_training = y_train
        print(f"   无标签噪声: 纯净环境")
    
    print(f"      - 分类标签不标准化，所有模型在标准化特征空间中竞争")
    
    data = {
        # 原始数据（用于参考）
        'X_train_original': X_train, 'X_test_original': X_test,
        'y_train_original': y_train, 'y_test_original': y_test,
        
        # 标准化数据（用于模型训练）
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_for_training, 'y_test': y_test,  # 分类标签不标准化
        
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

def train_sklearn_ovr_classifier(data, config):
    """训练sklearn OvR分类器"""
    print("🔧 训练 sklearn OvR MLPClassifier...")
    
    model = MLPSklearnOvRClassifier(
        hidden_layer_sizes=config['perception_hidden_layers'],
        max_iter=config['max_iter'],
        learning_rate_init=config['learning_rate'],
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
        print(f"   训练完成: 平均 {n_iter} epochs")
    
    return model

def train_pytorch_shared_ovr_classifier(data, config):
    """训练PyTorch共享OvR分类器"""
    print("🔧 训练 PyTorch Shared OvR MLPClassifier...")
    
    model = MLPPytorchSharedOvRClassifier(
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
    """回归模型预测和评估 - 统一逆变换策略"""
    # 在标准化空间中预测测试集
    test_pred_scaled = model.predict(data['X_test'])
    
    # 将预测结果转换回原始尺度进行评估
    test_pred_original = data['scaler_y'].inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    
    # 验证集：从原始干净数据重新分割（确保验证评估的科学性）
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        data['X_train_original'], data['y_train_original'],
        test_size=config['validation_fraction'],
        random_state=config['random_state']
    )
    
    # 对验证集应用相同的标准化
    X_val_scaled = data['scaler_X'].transform(X_val_orig)
    val_pred_scaled = model.predict(X_val_scaled)
    
    # 将验证集预测结果转换回原始尺度
    val_pred_original = data['scaler_y'].inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    
    # 在原始尺度下评估性能（测试集和验证集都用干净数据评估）
    results = {
        'test': evaluate_regression(data['y_test_original'], test_pred_original),
        'val': evaluate_regression(y_val_orig, val_pred_original)
    }
    
    return results

def predict_and_evaluate_classification(model, data, model_name, config):
    """分类模型预测和评估 - 统一评估策略"""
    n_classes = len(np.unique(data['y_train_original']))
    
    # 在标准化特征空间中预测测试集（分类标签无需转换）
    test_pred = model.predict(data['X_test'])
    
    # 验证集：从原始干净数据重新分割（确保验证评估的科学性）
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        data['X_train_original'], data['y_train_original'],
        test_size=config['validation_fraction'],
        random_state=config['random_state'],
        stratify=data['y_train_original']
    )
    
    # 对验证集应用相同的标准化
    X_val_scaled = data['scaler_X'].transform(X_val_orig)
    val_pred = model.predict(X_val_scaled)
    
    # 分类任务：在原始标签空间评估（测试集和验证集都用干净数据评估）
    results = {
        'test': evaluate_classification(data['y_test'], test_pred, n_classes),
        'val': evaluate_classification(y_val_orig, val_pred, n_classes)
    }
    
    return results

# =============================================================================
# 可视化和结果显示函数
# =============================================================================

def _ensure_output_dir(output_dir):
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}/")

def _get_output_path(output_dir, filename):
    """获取输出文件的完整路径"""
    return os.path.join(output_dir, filename)

def create_regression_visualization(results, config):
    """创建回归任务可视化图表"""
    if not config.get('save_plots', False):
        return
        
    _ensure_output_dir(config['output_dir'])
    
    print("\n📊 创建回归可视化图表")
    print("-" * 30)
    
    # 准备数据 - 确保方法顺序
    method_order = ['sklearn', 'pytorch', 'sklearn_ovr', 'pytorch_shared_ovr', 'deterministic', 'exogenous', 'endogenous', 'standard']
    methods = [method for method in method_order if method in results]
    method_labels = {
        'sklearn': 'sklearn MLP',
        'pytorch': 'PyTorch MLP',
        'sklearn_ovr': 'sklearn OvR MLP',
        'pytorch_shared_ovr': 'PyTorch Shared OvR',
        'deterministic': 'CausalEngine (deterministic)',
        'exogenous': 'CausalEngine (exogenous)',
        'endogenous': 'CausalEngine (endogenous)',
        'standard': 'CausalEngine (standard)'
    }
    
    metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
    
    # 创建子图 - 调整尺寸以容纳更多方法
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle(f'CausalEngine Quick Regression Test Results ({config["anomaly_ratio"]:.0%} Label Noise)', 
                 fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    # 定义颜色方案 - 扩展支持8种方法
    colors = {
        'sklearn': '#1f77b4',           # 蓝色
        'pytorch': '#ff7f0e',           # 橙色
        'sklearn_ovr': '#2ca02c',       # 绿色 (新增)
        'pytorch_shared_ovr': '#d62728', # 红色 (新增)
        'deterministic': '#9467bd',     # 紫色
        'exogenous': '#8c564b',         # 棕色
        'endogenous': '#e377c2',        # 粉色
        'standard': '#7f7f7f'           # 灰色
    }
    
    for i, metric in enumerate(metrics):
        values = [results[method]['test'][metric] for method in methods]
        labels = [method_labels[method] for method in methods]
        bar_colors = [colors[method] for method in methods]
        
        bars = axes[i].bar(labels, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[i].set_title(f'{metric} (Test Set)', fontweight='bold', fontsize=14)
        axes[i].set_ylabel(metric, fontsize=12)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if metric == 'R²':
                label_text = f'{value:.4f}'
            else:
                label_text = f'{value:.3f}'
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       label_text, ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 高亮最佳结果
        if metric == 'R²':
            best_idx = values.index(max(values))
        else:
            best_idx = values.index(min(values))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        axes[i].tick_params(axis='x', rotation=45, labelsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = _get_output_path(config['output_dir'], 'regression_performance.png')
    plt.savefig(output_path, dpi=config['figure_dpi'], bbox_inches='tight')
    print(f"📊 回归性能图表已保存为 {output_path}")
    plt.close()

def create_classification_visualization(results, config, n_classes):
    """创建分类任务可视化图表"""
    if not config.get('save_plots', False):
        return
        
    _ensure_output_dir(config['output_dir'])
    
    print("\n📊 创建分类可视化图表")
    print("-" * 30)
    
    # 准备数据 - 确保方法顺序（支持8种方法）
    method_order = ['sklearn', 'pytorch', 'sklearn_ovr', 'pytorch_shared_ovr', 'deterministic', 'exogenous', 'endogenous', 'standard']
    methods = [method for method in method_order if method in results]
    method_labels = {
        'sklearn': 'sklearn MLP',
        'pytorch': 'PyTorch MLP',
        'sklearn_ovr': 'sklearn OvR MLP',
        'pytorch_shared_ovr': 'PyTorch Shared OvR',
        'deterministic': 'CausalEngine (deterministic)',
        'exogenous': 'CausalEngine (exogenous)',
        'endogenous': 'CausalEngine (endogenous)',
        'standard': 'CausalEngine (standard)'
    }
    
    metrics = ['Acc', 'Precision', 'Recall', 'F1']
    
    # 创建子图（扩展支持8种方法）
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle(f'CausalEngine Quick {n_classes}-Class Classification Results ({config["label_noise_ratio"]:.0%} Label Noise)', 
                 fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    # 定义颜色方案 - 支持8种方法
    colors = {
        'sklearn': '#1f77b4',       # 蓝色
        'pytorch': '#ff7f0e',       # 橙色
        'sklearn_ovr': '#17becf',   # 青色
        'pytorch_shared_ovr': '#e377c2',  # 粉色
        'deterministic': '#d62728', # 红色
        'exogenous': '#2ca02c',     # 绿色 
        'endogenous': '#9467bd',    # 紫色
        'standard': '#8c564b'       # 棕色
    }
    
    for i, metric in enumerate(metrics):
        values = [results[method]['test'][metric] for method in methods]
        labels = [method_labels[method] for method in methods]
        bar_colors = [colors[method] for method in methods]
        
        bars = axes[i].bar(labels, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[i].set_title(f'{metric} (Test Set)', fontweight='bold', fontsize=14)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].set_ylim(0, 1.1)  # 分类指标都在0-1之间
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 高亮最佳结果
        best_idx = values.index(max(values))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        axes[i].tick_params(axis='x', rotation=45, labelsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = _get_output_path(config['output_dir'], 'classification_performance.png')
    plt.savefig(output_path, dpi=config['figure_dpi'], bbox_inches='tight')
    print(f"📊 分类性能图表已保存为 {output_path}")
    plt.close()

def create_data_analysis_visualization(data, config, task_type):
    """创建数据分析可视化图表"""
    if not config.get('save_plots', False):
        return
        
    _ensure_output_dir(config['output_dir'])
    
    print(f"\n📊 创建{task_type}数据分析图表")
    print("-" * 30)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if task_type == '回归':
        fig.suptitle(f'Regression Data Analysis (Features: {data["X_train"].shape[1]}, Samples: {len(data["X_train"]) + len(data["X_test"])})', 
                     fontsize=16, fontweight='bold')
        
        # 1. 目标变量分布 (原始尺度)
        y_all_original = np.concatenate([data['y_train_original'], data['y_test_original']])
        axes[0, 0].hist(y_all_original, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Target Distribution (Original Scale)')
        axes[0, 0].set_xlabel('Target Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(y_all_original.mean(), color='red', linestyle='--', 
                          label=f'Mean: {y_all_original.mean():.3f}')
        axes[0, 0].legend()
        
        # 2. 噪声注入影响对比（标准化尺度）
        axes[0, 1].hist(data['y_train_original'], bins=30, alpha=0.7, label='Clean Train', color='lightblue')
        
        # 将标准化空间的带噪声标签转换回原始尺度进行可视化
        if 'scaler_y' in data:
            y_train_noisy_original = data['scaler_y'].inverse_transform(data['y_train'].reshape(-1, 1)).flatten()
            axes[0, 1].hist(y_train_noisy_original, bins=30, alpha=0.7, label='With Noise', color='lightcoral')
        else:
            # 如果没有scaler_y，直接使用带噪声的标签
            axes[0, 1].hist(data['y_train'], bins=30, alpha=0.7, label='With Noise', color='lightcoral')
        
        axes[0, 1].set_title('Impact of Label Noise (Original Scale)')
        axes[0, 1].set_xlabel('Target Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
    else:  # 分类
        fig.suptitle(f'Classification Data Analysis (Features: {data["X_train"].shape[1]}, Classes: {len(np.unique(data["y_train"]))})', 
                     fontsize=16, fontweight='bold')
        
        # 1. 类别分布
        unique_classes, counts = np.unique(data['y_train_original'], return_counts=True)
        axes[0, 0].bar(unique_classes, counts, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Class Distribution (Training Set)')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # 2. 训练集噪声影响
        clean_counts = np.unique(data['y_train_original'], return_counts=True)[1]
        noisy_counts = np.unique(data['y_train'], return_counts=True)[1]
        
        x = np.arange(len(unique_classes))
        width = 0.35
        axes[0, 1].bar(x - width/2, clean_counts, width, label='Original', alpha=0.7, color='lightblue')
        axes[0, 1].bar(x + width/2, noisy_counts, width, label='With Noise', alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Impact of Label Noise')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(unique_classes)
        axes[0, 1].legend()
    
    # 3. 特征分布 (标准化后)
    n_features = min(data['X_train'].shape[1], 5)  # 最多显示5个特征
    for i in range(n_features):
        axes[1, 0].hist(data['X_train'][:, i], bins=30, alpha=0.5, 
                       label=f'Feature {i+1}', density=True)
    axes[1, 0].set_title(f'Feature Distributions (Standardized, Top {n_features})')
    axes[1, 0].set_xlabel('Standardized Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # 4. 数据集规模分析
    train_size = len(data['X_train'])
    test_size = len(data['X_test'])
    
    axes[1, 1].pie([train_size, test_size], labels=['Training', 'Test'], 
                  autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    axes[1, 1].set_title(f'Data Split\n(Total: {train_size + test_size} samples)')
    
    plt.tight_layout()
    
    task_name = 'regression' if task_type == '回归' else 'classification'
    output_path = _get_output_path(config['output_dir'], f'{task_name}_data_analysis.png')
    plt.savefig(output_path, dpi=config['figure_dpi'], bbox_inches='tight')
    print(f"📊 {task_type}数据分析图表已保存为 {output_path}")
    plt.close()

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
    
    # 2. 数据分析可视化
    if config.get('save_plots', False):
        create_data_analysis_visualization(data, config, '回归')
    
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
    
    if config['test_causal_exogenous']:
        causal_exo = train_causal_regressor(data, config, 'exogenous')
        results['exogenous'] = predict_and_evaluate_regression(causal_exo, data, 'causal', config)
    
    if config['test_causal_endogenous']:
        causal_endo = train_causal_regressor(data, config, 'endogenous')
        results['endogenous'] = predict_and_evaluate_regression(causal_endo, data, 'causal', config)
    
    if config['test_causal_standard']:
        causal_std = train_causal_regressor(data, config, 'standard')
        results['standard'] = predict_and_evaluate_regression(causal_std, data, 'causal', config)
    
    # 3. 显示结果
    if config['verbose']:
        print_regression_results(results)
    
    # 4. 可视化结果
    if config.get('save_plots', False):
        create_regression_visualization(results, config)
    
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
    
    # 2. 数据分析可视化
    if config.get('save_plots', False):
        create_data_analysis_visualization(data, config, '分类')
    
    results = {}
    
    # 2. 训练各种模型
    if config['test_sklearn']:
        sklearn_model = train_sklearn_classifier(data, config)
        results['sklearn'] = predict_and_evaluate_classification(sklearn_model, data, 'sklearn', config)
    
    if config['test_pytorch']:
        pytorch_model = train_pytorch_classifier(data, config)
        results['pytorch'] = predict_and_evaluate_classification(pytorch_model, data, 'pytorch', config)
    
    if config['test_sklearn_ovr']:
        sklearn_ovr_model = train_sklearn_ovr_classifier(data, config)
        results['sklearn_ovr'] = predict_and_evaluate_classification(sklearn_ovr_model, data, 'sklearn_ovr', config)
    
    if config['test_pytorch_shared_ovr']:
        pytorch_shared_ovr_model = train_pytorch_shared_ovr_classifier(data, config)
        results['pytorch_shared_ovr'] = predict_and_evaluate_classification(pytorch_shared_ovr_model, data, 'pytorch_shared_ovr', config)
    
    if config['test_causal_deterministic']:
        causal_det = train_causal_classifier(data, config, 'deterministic')
        results['deterministic'] = predict_and_evaluate_classification(causal_det, data, 'causal', config)
    
    if config['test_causal_exogenous']:
        causal_exo = train_causal_classifier(data, config, 'exogenous')
        results['exogenous'] = predict_and_evaluate_classification(causal_exo, data, 'causal', config)
    
    if config['test_causal_endogenous']:
        causal_endo = train_causal_classifier(data, config, 'endogenous')
        results['endogenous'] = predict_and_evaluate_classification(causal_endo, data, 'causal', config)
    
    if config['test_causal_standard']:
        causal_std = train_causal_classifier(data, config, 'standard')
        results['standard'] = predict_and_evaluate_classification(causal_std, data, 'causal', config)
    
    # 3. 显示结果
    if config['verbose']:
        n_classes = len(np.unique(data['y_train']))
        print_classification_results(results, n_classes)
    
    # 4. 可视化结果
    if config.get('save_plots', False):
        n_classes = len(np.unique(data['y_train']))
        create_classification_visualization(results, config, n_classes)
    
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
          f"deterministic={config['test_causal_deterministic']}, exogenous={config['test_causal_exogenous']}, "
          f"endogenous={config['test_causal_endogenous']}, standard={config['test_causal_standard']}")
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
    
    # 显示生成的文件信息
    if REGRESSION_CONFIG.get('save_plots', False):
        print(f"\n📊 生成的可视化文件:")
        print(f"   - {REGRESSION_CONFIG['output_dir']}/regression_data_analysis.png")
        print(f"   - {REGRESSION_CONFIG['output_dir']}/regression_performance.png")
        print(f"   - {REGRESSION_CONFIG['output_dir']}/classification_data_analysis.png")
        print(f"   - {REGRESSION_CONFIG['output_dir']}/classification_performance.png")
    
    print("💡 修改脚本顶部的 CONFIG 部分来调整实验参数")

def quick_regression_test():
    """快速回归测试 - 用于调试"""
    quick_config = REGRESSION_CONFIG.copy()
    quick_config.update({
        'n_samples': 1000,
        'max_iter': 500,
        'test_pytorch': False,  # 跳过pytorch基线以节省时间
        'test_causal_exogenous': False,  # 跳过部分模式以节省时间
        'test_causal_endogenous': False,
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
        'test_causal_exogenous': False,  # 跳过部分模式以节省时间
        'test_causal_endogenous': False,
        'verbose': True
    })
    return test_classification(quick_config)

if __name__ == "__main__":
    # 你可以选择运行以下任一函数:
    main()                        # 完整测试
    # quick_regression_test()     # 快速回归测试
    # quick_classification_test() # 快速分类测试