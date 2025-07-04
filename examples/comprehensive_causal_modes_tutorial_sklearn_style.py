#!/usr/bin/env python3
"""
🏠 全面CausalEngine模式教程：加州房价预测 - Sklearn-Style版本
================================================================

╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    Sklearn-Style CausalEngine 实验流程架构图                                        ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                    ║
║  📊 数据处理管道 (Data Processing Pipeline)                                                                         ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  1. 原始数据加载 → 2. 数据分割 → 3. 标准化(仅训练集fit) → 4. 噪声注入(仅训练集) → 5. 模型训练                      │  ║
║  │                                                                                                                │  ║
║  │  California Housing    Train/Test     X & Y         40% Label     13种方法                                      │  ║
║  │  (20,640 samples)  →   Split      →   Scaling   →   Noise     →   训练对比                                       │  ║
║  │  8 features            (80/20)       (基于干净数据)   (训练集Only)   Performance                                  │  ║
║  │                                      📌无数据泄露    📌测试集保持纯净                                             │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                    ║
║  🔧 统一参数传递机制 (Unified Parameter Passing)                                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  SklearnStyleTutorialConfig                                                                                    │  ║
║  │  ├─ 🧠 统一神经网络配置                                                                                          │  ║
║  │  │   ├─ NN_HIDDEN_SIZES = (128, 64, 32)     # 所有神经网络使用相同架构                                          │  ║
║  │  │   ├─ NN_MAX_EPOCHS = 3000                # 统一最大训练轮数                                                  │  ║
║  │  │   ├─ NN_LEARNING_RATE = 0.01             # 统一学习率                                                      │  ║
║  │  │   ├─ NN_PATIENCE = 200                   # 统一早停patience                                                │  ║
║  │  │   └─ NN_BATCH_SIZE = 200                 # 统一批处理大小                                                   │  ║
║  │  │                                                                                                            │  ║
║  │  ├─ 🎯 方法特定参数                                                                                              │  ║
║  │  │   ├─ CausalEngine: gamma_init, b_noise_init, modes=['deterministic', 'exogenous', 'endogenous', 'standard'] │  ║
║  │  │   ├─ Robust MLP: delta (Huber), quantile (Pinball), Cauchy loss                                           │  ║
║  │  │   └─ Tree Methods: n_estimators, max_depth, learning_rate                                                  │  ║
║  │  │                                                                                                            │  ║
║  │  └─ 📊 实验控制参数                                                                                              │  ║
║  │      ├─ ANOMALY_RATIO = 0.4                 # 40%标签噪声                                                      │  ║
║  │      ├─ TEST_SIZE = 0.2                     # 测试集比例                                                       │  ║
║  │      └─ RANDOM_STATE = 42                   # 随机种子                                                        │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                    ║
║  🔄 13种方法对比架构 (13-Method Comparison Framework)                                                               ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  传统神经网络 (2种)          稳健回归 (3种)              树模型 (4种)              因果推理 (4种)                    │  ║
║  │  ├─ sklearn MLP             ├─ Huber MLP              ├─ Random Forest          ├─ deterministic              │  ║
║  │  └─ PyTorch MLP             ├─ Pinball MLP            ├─ XGBoost                ├─ exogenous                  │  ║
║  │                             └─ Cauchy MLP             ├─ LightGBM               ├─ endogenous                 │  ║
║  │                                                       └─ CatBoost               └─ standard                   │  ║
║  │                                                                                                                │  ║
║  │  💡 关键设计亮点:                                                                                                │  ║
║  │  • 科学实验设计: 先基于干净数据标准化，再在训练集注入40%噪声，测试集保持纯净                                      │  ║
║  │  • 无数据泄露: 标准化器只在干净训练数据上fit，避免噪声污染统计量                                                  │  ║
║  │  • 统一数据标准化策略: 所有方法在标准化空间训练，确保实验公平性                                                     │  ║
║  │  • 参数公平性: 所有神经网络使用统一配置，确保公平比较                                                             │  ║
║  │  • 评估一致性: 所有方法在原始尺度下评估，便于结果解释                                                             │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                    ║
║  📈 输出分析 (Analysis & Visualization)                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                                                                │  ║
║  │  1. 数据探索分析图    2. 标准版性能对比    3. 扩展版性能对比    4. CausalEngine专项对比                             │  ║
║  │     (特征分布)           (9种核心方法)       (13种全部方法)       (4种模式详细)                                    │  ║
║  │                                                                                                                │  ║
║  │  📊 评估指标: MAE, MdAE, RMSE, R² (所有方法在原始房价尺度下统一评估)                                               │  ║
║  │                                                                                                                │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

这个教程演示所有CausalEngine推理模式在真实世界回归任务中的性能表现，使用sklearn-style实现。

数据集：加州房价数据集（California Housing Dataset）
- 20,640个样本
- 8个特征（房屋年龄、收入、人口等）
- 目标：预测房价中位数

我们将比较所有方法：
**标准版比较图（9种核心方法）：**
1. sklearn MLPRegressor（传统神经网络）
2. PyTorch MLP（传统深度学习）
3. Random Forest（随机森林）
4. XGBoost（梯度提升）
5. LightGBM（轻量梯度提升）
6. CatBoost（强力梯度提升）
7. CausalEngine - exogenous（外生噪声主导）
8. CausalEngine - endogenous（内生不确定性主导）
9. CausalEngine - standard（内生+外生混合）

**扩展版比较图（包含所有13种方法）：**
- 上述9种核心方法 + 4种额外方法：
10. CausalEngine - deterministic（确定性推理）
11. MLP Huber（Huber损失稳健回归）
12. MLP Pinball Median（中位数回归）
13. MLP Cauchy（Cauchy损失稳健回归）

关键亮点：
- 4种CausalEngine推理模式的全面对比
- 9种强力传统机器学习方法（包含2种神经网络+3种梯度提升+1种随机森林+3种稳健回归）
- 真实世界数据的鲁棒性测试
- 因果推理vs传统方法的性能差异分析
- 标准版(9种核心)与扩展版(13种全部)双重可视化
- 使用sklearn-style regressor实现，与Legacy版本形成对比

实验设计说明
==================================================================
本脚本使用sklearn-style实现专注于全面评估CausalEngine的4种推理模式，
旨在揭示不同因果推理策略在真实回归任务上的性能特点和适用场景。

核心实验：全模式性能对比 (在40%标签噪声下)
--------------------------------------------------
- **目标**: 比较所有4种CausalEngine模式和9种传统方法的预测性能（标准版9种核心方法，扩展版13种总方法）
- **设置**: 40%标签噪声，模拟真实世界数据质量挑战
- **对比模型**: 
  - 传统方法（核心6种）: sklearn MLP, PyTorch MLP, Random Forest, XGBoost, LightGBM, CatBoost
  - 稳健回归（额外3种）: Huber MLP, Pinball MLP, Cauchy MLP
  - CausalEngine（4种模式）: deterministic, exogenous, endogenous, standard
- **分析重点**: 
  - 哪种因果推理模式表现最优？
  - 不同模式的性能特点和差异
  - 因果推理相对传统方法的优势
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import warnings
import os
import sys
import time

# 设置matplotlib后端为非交互式，避免弹出窗口
plt.switch_backend('Agg')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入sklearn-style实现
from causal_sklearn.regressor import (
    MLPCausalRegressor, MLPPytorchRegressor, 
    MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
)
from causal_sklearn.data_processing import inject_shuffle_noise
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


class SklearnStyleTutorialConfig:
    """
    Sklearn-Style教程配置类 - 测试所有CausalEngine模式
    
    🔧 在这里修改参数来自定义实验设置！
    """
    
    # 🎯 数据分割参数
    TEST_SIZE = 0.2          # 测试集比例
    VAL_SIZE = 0.2           # 验证集比例 (相对于训练集)
    RANDOM_STATE = 42        # 随机种子
    
    # 🧠 统一神经网络配置 - 所有神经网络方法使用相同参数确保公平比较
    # =========================================================================
    # 🔧 在这里修改所有神经网络方法的共同参数！
    NN_HIDDEN_SIZES = (128, 64, 32)                  # 神经网络隐藏层结构
    NN_MAX_EPOCHS = 3000                         # 最大训练轮数
    NN_LEARNING_RATE = 0.01                      # 学习率
    NN_PATIENCE = 50                            # 早停patience
    NN_TOLERANCE = 1e-4                          # 早停tolerance
    NN_BATCH_SIZE = 200                          # 批处理大小
    # =========================================================================
    
    # 🤖 CausalEngine参数 - 测试4种有效模式
    CAUSAL_MODES = ['deterministic', 'exogenous', 'endogenous', 'standard']
    CAUSAL_HIDDEN_SIZES = NN_HIDDEN_SIZES        # 使用统一神经网络配置
    CAUSAL_MAX_EPOCHS = NN_MAX_EPOCHS            # 使用统一神经网络配置
    CAUSAL_LR = NN_LEARNING_RATE                 # 使用统一神经网络配置
    CAUSAL_PATIENCE = NN_PATIENCE                # 使用统一神经网络配置
    CAUSAL_TOL = NN_TOLERANCE                    # 使用统一神经网络配置
    CAUSAL_GAMMA_INIT = 1.0                      # gamma初始化
    CAUSAL_B_NOISE_INIT = 1.0                    # b_noise初始化
    CAUSAL_B_NOISE_TRAINABLE = True              # b_noise是否可训练
    CAUSAL_ALPHA = 0.0                           # CausalEngine L2正则化
    
    # 🧠 传统神经网络方法参数 - 使用统一配置
    SKLEARN_HIDDEN_LAYERS = NN_HIDDEN_SIZES      # 使用统一神经网络配置
    SKLEARN_MAX_ITER = NN_MAX_EPOCHS             # 使用统一神经网络配置
    SKLEARN_LR = NN_LEARNING_RATE                # 使用统一神经网络配置
    SKLEARN_ALPHA = 0.0                          # sklearn MLP L2正则化
    
    PYTORCH_EPOCHS = NN_MAX_EPOCHS               # 使用统一神经网络配置
    PYTORCH_LR = NN_LEARNING_RATE                # 使用统一神经网络配置
    PYTORCH_PATIENCE = NN_PATIENCE               # 使用统一神经网络配置
    PYTORCH_ALPHA = 0.0                          # PyTorch MLP L2正则化
    
    # 📊 实验参数
    ANOMALY_RATIO = 0.3                          # 标签异常比例 (核心实验默认值: 40%噪声挑战)
    SAVE_PLOTS = True                            # 是否保存图表
    VERBOSE = True                               # 是否显示详细输出
    
    # 🎯 要测试的方法列表
    METHODS_TO_TEST = {
        'sklearn_mlp': True,         # sklearn MLPRegressor
        'pytorch_mlp': True,         # PyTorch MLP
        'mlp_huber': True,           # Huber损失MLP
        'mlp_pinball': True,         # Pinball损失MLP  
        'mlp_cauchy': True,          # Cauchy损失MLP
        'random_forest': True,       # Random Forest
        'xgboost': True,            # XGBoost (如果可用)
        'lightgbm': True,           # LightGBM (如果可用)
        'catboost': True,           # CatBoost (如果可用)
        'causal_deterministic': True,  # CausalEngine deterministic
        'causal_exogenous': True,      # CausalEngine exogenous
        'causal_endogenous': True,     # CausalEngine endogenous
        'causal_standard': True,       # CausalEngine standard
    }
    
    # 🌲 传统机器学习方法参数
    RANDOM_FOREST_N_ESTIMATORS = 100
    RANDOM_FOREST_MAX_DEPTH = 10
    RANDOM_FOREST_MIN_SAMPLES_SPLIT = 5
    
    XGBOOST_N_ESTIMATORS = 100
    XGBOOST_MAX_DEPTH = 6
    XGBOOST_LEARNING_RATE = 0.1
    
    LIGHTGBM_N_ESTIMATORS = 100
    LIGHTGBM_MAX_DEPTH = 6
    LIGHTGBM_LEARNING_RATE = 0.1
    
    CATBOOST_ITERATIONS = 100
    CATBOOST_DEPTH = 6
    CATBOOST_LEARNING_RATE = 0.1
    
    # 🛑 树方法早停配置（仅XGBoost, LightGBM, CatBoost支持）
    TREE_EARLY_STOPPING_ROUNDS = 10     # 早停patience，与神经网络NN_PATIENCE/5保持合理比例
    
    # 📈 可视化参数
    FIGURE_DPI = 300                             # 图表分辨率
    FIGURE_SIZE_ANALYSIS = (16, 12)              # 数据分析图表大小
    FIGURE_SIZE_PERFORMANCE = (24, 20)           # 性能对比图表大小（更大以容纳13个方法）
    FIGURE_SIZE_MODES_COMPARISON = (18, 12)      # CausalEngine模式对比图表大小
    
    # 📁 输出目录参数
    OUTPUT_DIR = "results/comprehensive_causal_modes_sklearn_style"


class SklearnStyleCausalModesTutorial:
    """
    全面CausalEngine模式教程类 - Sklearn-Style版本
    
    演示所有CausalEngine推理模式在真实世界回归任务中的性能特点
    """
    
    def __init__(self, config=None):
        self.config = config if config is not None else SklearnStyleTutorialConfig()
        self.X = None
        self.y = None
        self.feature_names = None
        self.results = {}
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.config.OUTPUT_DIR):
            os.makedirs(self.config.OUTPUT_DIR)
            print(f"📁 创建输出目录: {self.config.OUTPUT_DIR}/")
    
    def _get_output_path(self, filename):
        """获取输出文件的完整路径"""
        return os.path.join(self.config.OUTPUT_DIR, filename)
        
    def load_and_explore_data(self, verbose=True):
        """加载并探索加州房价数据集"""
        if verbose:
            print("🏠 全面CausalEngine模式教程 - 加州房价预测 (Sklearn-Style版本)")
            print("=" * 80)
            print("📊 正在加载加州房价数据集...")
        
        # 加载数据
        housing = fetch_california_housing()
        self.X, self.y = housing.data, housing.target
        self.feature_names = housing.feature_names
        
        if verbose:
            print(f"✅ 数据加载完成")
            print(f"   - 样本数量: {self.X.shape[0]:,}")
            print(f"   - 特征数量: {self.X.shape[1]}")
            print(f"   - 特征名称: {', '.join(self.feature_names)}")
            print(f"   - 目标范围: ${self.y.min():.2f} - ${self.y.max():.2f} (百万美元)")
            print(f"   - 目标均值: ${self.y.mean():.2f}")
            print(f"   - 目标标准差: ${self.y.std():.2f}")
        
        return self.X, self.y
    
    def visualize_data(self, save_plots=None):
        """数据可视化分析"""
        if save_plots is None:
            save_plots = self.config.SAVE_PLOTS
            
        print("\n📈 数据分布分析")
        print("-" * 30)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ANALYSIS)
        fig.suptitle('California Housing Dataset Analysis - Sklearn-Style CausalEngine Tutorial', fontsize=16, fontweight='bold')
        
        # 1. 目标变量分布
        axes[0, 0].hist(self.y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('House Price Distribution')
        axes[0, 0].set_xlabel('House Price ($100k)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.y.mean(), color='red', linestyle='--', label=f'Mean: ${self.y.mean():.2f}')
        axes[0, 0].legend()
        
        # 2. 特征相关性热力图
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df['MedHouseVal'] = self.y
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0, 1], cbar_kws={'shrink': 0.8})
        axes[0, 1].set_title('Feature Correlation Matrix')
        
        # 3. 特征分布箱线图
        df_features = pd.DataFrame(self.X, columns=self.feature_names)
        df_features_normalized = (df_features - df_features.mean()) / df_features.std()
        df_features_normalized.boxplot(ax=axes[1, 0])
        axes[1, 0].set_title('Feature Distribution (Standardized)')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Standardized Values')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 最重要特征与目标的散点图
        most_corr_feature = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False).index[1]
        most_corr_idx = list(self.feature_names).index(most_corr_feature)
        axes[1, 1].scatter(self.X[:, most_corr_idx], self.y, alpha=0.5, s=1)
        axes[1, 1].set_title(f'Most Correlated Feature: {most_corr_feature}')
        axes[1, 1].set_xlabel(most_corr_feature)
        axes[1, 1].set_ylabel('House Price ($100k)')
        
        plt.tight_layout()
        
        if save_plots:
            output_path = self._get_output_path('sklearn_style_data_analysis.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 数据分析图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 数据统计摘要
        print("\n📋 数据统计摘要:")
        print(f"  - 最相关特征: {most_corr_feature} (相关系数: {corr_matrix.loc[most_corr_feature, 'MedHouseVal']:.3f})")
        print(f"  - 异常值检测: {np.sum(np.abs(self.y - self.y.mean()) > 3 * self.y.std())} 个潜在异常值")
        print(f"  - 数据完整性: 无缺失值" if not np.any(np.isnan(self.X)) else "  - 警告: 存在缺失值")
    
    def prepare_data(self, verbose=True):
        """准备数据 - 科学严谨的标准化策略"""
        if verbose:
            print("\n📊 数据准备 - 科学严谨的标准化策略")
            print("=" * 60)
        
        # 1. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
        
        if verbose:
            print(f"✅ 数据分割完成: 训练集 {len(X_train)} | 测试集 {len(X_test)}")
        
        # 2. 🎯 标准化策略（基于干净的训练数据）
        if verbose:
            print("\n🎯 标准化策略 - 基于干净训练数据学习统计量:")
        
        # 特征标准化 - 只在训练集上fit
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # 目标标准化 - 关键：在干净的训练集上fit
        scaler_y = StandardScaler()
        y_train_clean_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        if verbose:
            print(f"   - 特征标准化器基于训练集学习: mean={scaler_X.mean_[:3]}, std={scaler_X.scale_[:3]}")
            print(f"   - 目标标准化器基于干净训练目标学习: mean={scaler_y.mean_[0]:.3f}, std={scaler_y.scale_[0]:.3f}")
        
        # 3. 噪声注入（在标准化后进行）
        if self.config.ANOMALY_RATIO > 0:
            y_train_scaled_noisy, noise_indices = inject_shuffle_noise(
                y_train_clean_scaled,
                noise_ratio=self.config.ANOMALY_RATIO,
                random_state=self.config.RANDOM_STATE
            )
            y_train_scaled = y_train_scaled_noisy
            
            # 同时对原始尺度的训练目标应用相同的噪声（用于评估）
            y_train_noisy, _ = inject_shuffle_noise(
                y_train,
                noise_ratio=self.config.ANOMALY_RATIO,
                random_state=self.config.RANDOM_STATE
            )
            y_train_original = y_train_noisy
            
            if verbose:
                print(f"\n✅ 噪声注入完成: {self.config.ANOMALY_RATIO:.1%} ({len(noise_indices)}/{len(y_train_scaled)} 样本受影响)")
                print(f"   - 噪声在标准化后注入，保证标准化器基于干净数据")
        else:
            y_train_scaled = y_train_clean_scaled
            y_train_original = y_train
            if verbose:
                print("\n✅ 无噪声注入: 纯净环境")
        
        if verbose:
            print(f"\n📊 最终数据状态:")
            print(f"   - 训练集: X标准化 + y标准化 + {self.config.ANOMALY_RATIO:.0%}噪声")
            print(f"   - 测试集: X标准化 + y标准化 + 纯净无噪声")
            print(f"   - 标准化器基于干净训练数据，确保无泄露")
        
        return {
            # 原始数据（用于最终评估）
            'X_train_original': X_train, 'X_test_original': X_test,
            'y_train_original': y_train_original, 'y_test_original': y_test,
            
            # 标准化数据（用于模型训练）
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_train': y_train_scaled, 'y_test': y_test_scaled,
            
            # 标准化器（用于逆变换）
            'scaler_X': scaler_X, 'scaler_y': scaler_y
        }
    
    def train_sklearn_mlp(self, data, verbose=True):
        """训练sklearn MLPRegressor"""
        if not self.config.METHODS_TO_TEST.get('sklearn_mlp'):
            return None
            
        if verbose:
            print("🔧 训练 sklearn MLPRegressor...")
        
        start_time = time.time()
        model = MLPRegressor(
            hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
            max_iter=self.config.SKLEARN_MAX_ITER,
            learning_rate_init=self.config.SKLEARN_LR,
            early_stopping=True,
            validation_fraction=self.config.VAL_SIZE,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            batch_size=self.config.NN_BATCH_SIZE,
            alpha=self.config.SKLEARN_ALPHA,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成: {model.n_iter_} epochs (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def train_pytorch_mlp(self, data, verbose=True):
        """训练PyTorch MLPRegressor"""
        if not self.config.METHODS_TO_TEST.get('pytorch_mlp'):
            return None
            
        if verbose:
            print("🔧 训练 PyTorch MLPRegressor...")
        
        start_time = time.time()
        model = MLPPytorchRegressor(
            hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
            max_iter=self.config.PYTORCH_EPOCHS,
            learning_rate=self.config.PYTORCH_LR,
            early_stopping=True,
            validation_fraction=self.config.VAL_SIZE,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            batch_size=self.config.NN_BATCH_SIZE,
            alpha=self.config.PYTORCH_ALPHA,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成: {model.n_iter_} epochs (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def train_robust_mlp(self, data, method_name, regressor_class, verbose=True):
        """训练稳健回归器（Huber, Pinball, Cauchy）"""
        if not self.config.METHODS_TO_TEST.get(method_name):
            return None
            
        if verbose:
            print(f"🔧 训练 {regressor_class.__name__}...")
        
        start_time = time.time()
        model = regressor_class(
            hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
            max_iter=self.config.SKLEARN_MAX_ITER,
            learning_rate=self.config.SKLEARN_LR,
            early_stopping=True,
            validation_fraction=self.config.VAL_SIZE,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            batch_size=self.config.NN_BATCH_SIZE,
            alpha=self.config.PYTORCH_ALPHA,  # 使用与PyTorch相同的alpha
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成: {model.n_iter_} epochs (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def train_causal_regressor(self, data, mode, verbose=True):
        """训练CausalEngine回归器"""
        method_key = f'causal_{mode}'
        if not self.config.METHODS_TO_TEST.get(method_key):
            return None
            
        if verbose:
            print(f"🔧 训练 CausalEngine ({mode})...")
        
        start_time = time.time()
        model = MLPCausalRegressor(
            perception_hidden_layers=self.config.CAUSAL_HIDDEN_SIZES,
            mode=mode,
            max_iter=self.config.CAUSAL_MAX_EPOCHS,
            learning_rate=self.config.CAUSAL_LR,
            early_stopping=True,
            validation_fraction=self.config.VAL_SIZE,
            n_iter_no_change=self.config.CAUSAL_PATIENCE,
            tol=self.config.CAUSAL_TOL,
            batch_size=self.config.NN_BATCH_SIZE,
            alpha=self.config.CAUSAL_ALPHA,
            gamma_init=self.config.CAUSAL_GAMMA_INIT,
            b_noise_init=self.config.CAUSAL_B_NOISE_INIT,
            b_noise_trainable=self.config.CAUSAL_B_NOISE_TRAINABLE,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成: {model.n_iter_} epochs (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def train_random_forest(self, data, verbose=True):
        """训练Random Forest"""
        if not self.config.METHODS_TO_TEST.get('random_forest'):
            return None
            
        if verbose:
            print("🌲 训练 Random Forest...")
        
        start_time = time.time()
        model = RandomForestRegressor(
            n_estimators=self.config.RANDOM_FOREST_N_ESTIMATORS,
            max_depth=self.config.RANDOM_FOREST_MAX_DEPTH,
            min_samples_split=self.config.RANDOM_FOREST_MIN_SAMPLES_SPLIT,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(data['X_train'], data['y_train'])
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成 (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def train_xgboost(self, data, verbose=True):
        """训练XGBoost"""
        if not self.config.METHODS_TO_TEST.get('xgboost') or not XGBOOST_AVAILABLE:
            return None
            
        if verbose:
            print("🚀 训练 XGBoost...")
        
        start_time = time.time()
        
        # 准备验证集用于早停（使用标准化数据）
        X_train_std = data['X_train']
        y_train_std = data['y_train']
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            X_train_std, y_train_std,
            test_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_STATE
        )
        
        model = xgb.XGBRegressor(
            n_estimators=self.config.XGBOOST_N_ESTIMATORS,
            max_depth=self.config.XGBOOST_MAX_DEPTH,
            learning_rate=self.config.XGBOOST_LEARNING_RATE,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=self.config.TREE_EARLY_STOPPING_ROUNDS
        )
        
        # 使用早停训练（统一使用标准化数据）
        model.fit(
            X_train_val, y_train_val,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成: {model.best_iteration} 轮 (早停) (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def train_lightgbm(self, data, verbose=True):
        """训练LightGBM"""
        if not self.config.METHODS_TO_TEST.get('lightgbm') or not LIGHTGBM_AVAILABLE:
            return None
            
        if verbose:
            print("⚡ 训练 LightGBM...")
        
        start_time = time.time()
        model = lgb.LGBMRegressor(
            n_estimators=self.config.LIGHTGBM_N_ESTIMATORS,
            max_depth=self.config.LIGHTGBM_MAX_DEPTH,
            learning_rate=self.config.LIGHTGBM_LEARNING_RATE,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        )
        
        model.fit(data['X_train'], data['y_train'])
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成 (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def train_catboost(self, data, verbose=True):
        """训练CatBoost"""
        if not self.config.METHODS_TO_TEST.get('catboost') or not CATBOOST_AVAILABLE:
            return None
            
        if verbose:
            print("🐱 训练 CatBoost...")
        
        start_time = time.time()
        model = cb.CatBoostRegressor(
            iterations=self.config.CATBOOST_ITERATIONS,
            depth=self.config.CATBOOST_DEPTH,
            learning_rate=self.config.CATBOOST_LEARNING_RATE,
            random_seed=self.config.RANDOM_STATE,
            thread_count=-1,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   训练完成 (用时: {training_time:.2f}s)")
        
        return model, training_time
    
    def evaluate_model(self, model, data, model_name):
        """评估模型性能 - 统一逆变换逻辑"""
        # 🎯 统一策略：所有方法在标准化空间预测，然后逆变换到原始尺度评估
        test_pred_scaled = model.predict(data['X_test'])
        
        # 将预测结果转换回原始尺度进行评估
        test_pred_original = data['scaler_y'].inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        
        # 在原始尺度下计算性能指标
        results = {
            'test': {
                'MAE': mean_absolute_error(data['y_test_original'], test_pred_original),
                'MdAE': median_absolute_error(data['y_test_original'], test_pred_original),
                'RMSE': np.sqrt(mean_squared_error(data['y_test_original'], test_pred_original)),
                'R²': r2_score(data['y_test_original'], test_pred_original)
            }
        }
        
        return results
    
    def run_comprehensive_benchmark(self, verbose=None):
        """运行全面的基准测试 - 包含所有CausalEngine模式"""
        if verbose is None:
            verbose = self.config.VERBOSE
            
        if verbose:
            print("\n🚀 开始全面基准测试 - Sklearn-Style实现")
            print("=" * 80)
            print(f"🔧 实验配置:")
            print(f"   - 测试集比例: {self.config.TEST_SIZE:.1%}")
            print(f"   - 验证集比例: {self.config.VAL_SIZE:.1%}")
            print(f"   - 异常标签比例: {self.config.ANOMALY_RATIO:.1%}")
            print(f"   - 随机种子: {self.config.RANDOM_STATE}")
            print(f"   - CausalEngine模式: {', '.join(self.config.CAUSAL_MODES)}")
            print(f"   - 网络结构: {self.config.CAUSAL_HIDDEN_SIZES}")
            print(f"   - 最大训练轮数: {self.config.CAUSAL_MAX_EPOCHS}")
            print(f"   - 早停patience: {self.config.CAUSAL_PATIENCE}")
            enabled_methods = [k for k, v in self.config.METHODS_TO_TEST.items() if v]
            print(f"   - 总计对比方法: {len(enabled_methods)} 种")
            print(f"   - 方法列表: {', '.join(enabled_methods)}")
        
        # 准备数据
        data = self.prepare_data(verbose=verbose)
        self.results = {}
        training_times = {}
        
        # 1. 训练sklearn MLPRegressor
        result = self.train_sklearn_mlp(data, verbose=verbose)
        if result:
            model, train_time = result
            self.results['sklearn'] = self.evaluate_model(model, data, 'sklearn')
            training_times['sklearn'] = train_time
        
        # 2. 训练PyTorch MLPRegressor
        result = self.train_pytorch_mlp(data, verbose=verbose)
        if result:
            model, train_time = result
            self.results['pytorch'] = self.evaluate_model(model, data, 'pytorch')
            training_times['pytorch'] = train_time
        
        # 3. 训练稳健回归器
        robust_methods = [
            ('mlp_huber', MLPHuberRegressor),
            ('mlp_pinball', MLPPinballRegressor),
            ('mlp_cauchy', MLPCauchyRegressor)
        ]
        
        for method_name, regressor_class in robust_methods:
            result = self.train_robust_mlp(data, method_name, regressor_class, verbose=verbose)
            if result:
                model, train_time = result
                # 映射到结果键名
                result_key = method_name.replace('mlp_', '')
                if method_name == 'mlp_pinball':
                    result_key = 'pinball'
                self.results[result_key] = self.evaluate_model(model, data, result_key)
                training_times[result_key] = train_time
        
        # 4. 训练传统机器学习方法
        traditional_ml_methods = [
            ('random_forest', self.train_random_forest),
            ('xgboost', self.train_xgboost),
            ('lightgbm', self.train_lightgbm),
            ('catboost', self.train_catboost)
        ]
        
        for method_name, train_func in traditional_ml_methods:
            result = train_func(data, verbose=verbose)
            if result:
                model, train_time = result
                self.results[method_name] = self.evaluate_model(model, data, method_name)
                training_times[method_name] = train_time
        
        # 5. 训练CausalEngine模式
        for mode in self.config.CAUSAL_MODES:
            result = self.train_causal_regressor(data, mode, verbose=verbose)
            if result:
                model, train_time = result
                self.results[mode] = self.evaluate_model(model, data, mode)
                training_times[mode] = train_time
        
        if verbose:
            print(f"\n📊 全面基准测试结果 (异常比例: {self.config.ANOMALY_RATIO:.0%})")
            self.print_results(training_times)
        
        return self.results
    
    def print_results(self, training_times=None):
        """打印测试结果"""
        if not self.results:
            print("❌ 没有可显示的结果")
            return
        
        print("\n" + "=" * 120)
        print(f"{'方法':<20} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10} {'训练时间(s)':<12}")
        print("-" * 120)
        
        # 按MdAE排序（越小越好）
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['test']['MdAE'])
        
        for method, metrics in sorted_results:
            test_m = metrics['test']
            train_time = training_times.get(method, 0.0) if training_times else 0.0
            
            # 方法名显示优化
            if method in self.config.CAUSAL_MODES:
                display_name = f"CausalEngine ({method})"
            elif method == 'random_forest':
                display_name = "Random Forest"
            elif method == 'xgboost':
                display_name = "XGBoost"
            elif method == 'lightgbm':
                display_name = "LightGBM"
            elif method == 'catboost':
                display_name = "CatBoost"
            else:
                display_name = method.replace('_', ' ').title()
            
            print(f"{display_name:<20} {test_m['MAE']:<10.4f} {test_m['MdAE']:<10.4f} "
                  f"{test_m['RMSE']:<10.4f} {test_m['R²']:<10.4f} {train_time:<12.2f}")
        
        print("=" * 120)
    
    def analyze_causal_modes_performance(self, verbose=True):
        """专门分析CausalEngine不同模式的性能特点"""
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        if verbose:
            print("\n🔬 CausalEngine模式深度分析")
            print("=" * 70)
        
        # 提取CausalEngine模式结果
        causal_results = {}
        traditional_results = {}
        
        for method, metrics in self.results.items():
            if method in self.config.CAUSAL_MODES:
                causal_results[method] = metrics
            else:
                traditional_results[method] = metrics
        
        if verbose:
            print(f"🎯 CausalEngine模式性能对比 (共{len(causal_results)}种模式):")
            print("-" * 50)
            
            # 按MdAE分数排序（越小越好）
            causal_mdae_scores = {mode: metrics['test']['MdAE'] for mode, metrics in causal_results.items()}
            sorted_causal = sorted(causal_mdae_scores.items(), key=lambda x: x[1])
            
            for i, (mode, mdae) in enumerate(sorted_causal, 1):
                mae = causal_results[mode]['test']['MAE']
                r2 = causal_results[mode]['test']['R²']
                print(f"   {i}. {mode:<12} - MdAE: {mdae:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")
            
            # 模式特点分析
            print(f"\n📊 模式特点分析:")
            print("-" * 30)
            
            best_mode = sorted_causal[0][0]
            worst_mode = sorted_causal[-1][0]
            performance_gap = sorted_causal[-1][1] - sorted_causal[0][1]
            
            print(f"   🏆 最佳模式: {best_mode} (MdAE = {sorted_causal[0][1]:.3f})")
            print(f"   📉 最弱模式: {worst_mode} (MdAE = {sorted_causal[-1][1]:.3f})")
            print(f"   📏 性能差距: {performance_gap:.3f} ({performance_gap/sorted_causal[0][1]*100:.1f}%)")
            
            # 与传统方法比较
            if traditional_results:
                print(f"\n🆚 CausalEngine vs 传统方法:")
                print("-" * 40)
                
                traditional_mdae_scores = {method: metrics['test']['MdAE'] for method, metrics in traditional_results.items()}
                best_traditional = min(traditional_mdae_scores.keys(), key=lambda x: traditional_mdae_scores[x])
                best_traditional_mdae = traditional_mdae_scores[best_traditional]
                
                print(f"   最佳传统方法: {best_traditional} (MdAE = {best_traditional_mdae:.3f})")
                print(f"   最佳CausalEngine: {best_mode} (MdAE = {sorted_causal[0][1]:.3f})")
                
                improvement = (best_traditional_mdae - sorted_causal[0][1]) / best_traditional_mdae * 100
                print(f"   性能提升: {improvement:+.2f}%")
                
                better_modes = sum(1 for _, mdae in sorted_causal if mdae < best_traditional_mdae)
                print(f"   优于传统方法的CausalEngine模式: {better_modes}/{len(sorted_causal)}")
        
        return causal_results, traditional_results
    
    def create_comprehensive_performance_visualization(self, save_plot=None, extended=False):
        """创建全面的性能可视化图表 - 支持标准版和扩展版"""
        if save_plot is None:
            save_plot = self.config.SAVE_PLOTS
            
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        chart_type = "扩展版" if extended else "标准版"
        print(f"\n📊 创建全面性能可视化图表 ({chart_type})")
        print("-" * 40)
        
        # 准备数据 - 根据扩展标志决定包含的方法
        if extended:
            # 扩展版：包含所有可用方法
            all_available_methods = list(self.results.keys())
            # 按类型排序：先传统方法，后CausalEngine
            traditional_methods = [m for m in all_available_methods if m not in self.config.CAUSAL_MODES]
            causal_methods = [m for m in self.config.CAUSAL_MODES if m in self.results]
            methods = traditional_methods + causal_methods
        else:
            # 标准版：包含9种核心方法（除了3种robust MLP）
            robust_mlp_methods = ['huber', 'pinball', 'cauchy']  # 排除的robust MLP方法
            standard_traditional = [m for m in self.results.keys() 
                                  if m not in self.config.CAUSAL_MODES and m not in robust_mlp_methods]
            causal_methods = [m for m in self.config.CAUSAL_MODES if m in self.results]
            methods = standard_traditional + causal_methods
        
        # 为不同类型的方法设置颜色
        colors = []
        for method in methods:
            if method in self.config.CAUSAL_MODES:
                colors.append('#2ca02c')  # 绿色系 - CausalEngine
            else:
                colors.append('#1f77b4')  # 蓝色系 - 传统方法
        
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_PERFORMANCE)
        title_suffix = " (Extended with All Methods)" if extended else ""
        fig.suptitle(f'Sklearn-Style CausalEngine Modes vs Traditional Methods{title_suffix}\nCalifornia Housing Performance (40% Label Noise)', 
                     fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.results[method]['test'][metric] for method in methods]
            
            bars = axes[i].bar(range(len(methods)), values, color=colors, alpha=0.8, edgecolor='black')
            axes[i].set_title(f'{metric} (Test Set)', fontweight='bold')
            axes[i].set_ylabel(metric)
            
            # 设置X轴标签
            method_labels = []
            for method in methods:
                if method in self.config.CAUSAL_MODES:
                    method_labels.append(f'CausalEngine\n({method})')
                elif method == 'sklearn':
                    method_labels.append('sklearn\nMLP')
                elif method == 'pytorch':
                    method_labels.append('PyTorch\nMLP')
                else:
                    display_name = method.replace('_', ' ').title()
                    if len(display_name) > 12:
                        words = display_name.split()
                        if len(words) > 1:
                            display_name = f"{words[0]}\n{' '.join(words[1:])}"
                    method_labels.append(display_name)
            
            axes[i].set_xticks(range(len(methods)))
            axes[i].set_xticklabels(method_labels, rotation=45, ha='right', fontsize=8)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            # 高亮最佳结果
            if metric == 'R²':
                best_idx = values.index(max(values))
            else:
                best_idx = values.index(min(values))
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        if save_plot:
            if extended:
                filename = 'sklearn_style_performance_comparison_extended.png'
            else:
                filename = 'sklearn_style_performance_comparison.png'
            output_path = self._get_output_path(filename)
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 {chart_type}性能对比图表已保存为 {output_path}")
        
        plt.close()
    
    def create_causal_modes_comparison(self, save_plot=None):
        """创建专门的CausalEngine模式对比图表"""
        if save_plot is None:
            save_plot = self.config.SAVE_PLOTS
            
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\n📊 创建CausalEngine模式专项对比图表")
        print("-" * 45)
        
        # 提取CausalEngine模式结果
        causal_methods = [m for m in self.config.CAUSAL_MODES if m in self.results]
        
        if len(causal_methods) < 2:
            print("❌ 需要至少2种CausalEngine模式来进行对比")
            return
        
        # 创建雷达图显示CausalEngine模式的多维性能
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.FIGURE_SIZE_MODES_COMPARISON)
        fig.suptitle('Sklearn-Style CausalEngine Modes Detailed Comparison', fontsize=16, fontweight='bold')
        
        # 左图：性能条形图
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        colors = plt.cm.Set3(np.linspace(0, 1, len(causal_methods)))
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, method in enumerate(causal_methods):
            values = [self.results[method]['test'][metric] for metric in metrics]
            ax1.bar(x + i * width, values, width, label=f'{method}', color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title('CausalEngine Modes Performance Comparison')
        ax1.set_xticks(x + width * (len(causal_methods) - 1) / 2)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：MdAE性能排名（越小越好）
        mdae_scores = [(method, self.results[method]['test']['MdAE']) for method in causal_methods]
        mdae_scores.sort(key=lambda x: x[1])  # 按升序排列，因为MdAE越小越好
        
        methods_sorted = [item[0] for item in mdae_scores]
        mdae_values = [item[1] for item in mdae_scores]
        
        bars = ax2.bar(range(len(methods_sorted)), mdae_values, color=colors[:len(methods_sorted)], alpha=0.8)
        ax2.set_xlabel('CausalEngine Modes')
        ax2.set_ylabel('MdAE (Median Absolute Error)')
        ax2.set_title('CausalEngine Modes MdAE Performance Ranking')
        ax2.set_xticks(range(len(methods_sorted)))
        ax2.set_xticklabels(methods_sorted, rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, mdae_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 高亮最佳模式（MdAE最小的）
        bars[0].set_color('gold')
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(3)
        
        plt.tight_layout()
        
        if save_plot:
            output_path = self._get_output_path('sklearn_style_causal_modes_comparison.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 CausalEngine模式对比图表已保存为 {output_path}")
        
        plt.close()
    
    def print_comprehensive_summary(self):
        """打印全面的总结报告"""
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\n📋 全面实验总结报告 - Sklearn-Style版本")
        print("=" * 80)
        
        # 统计信息
        total_methods = len(self.results)
        causal_methods = len([m for m in self.results if m in self.config.CAUSAL_MODES])
        traditional_methods = total_methods - causal_methods
        
        print(f"🔢 实验规模:")
        print(f"   - 总计测试方法: {total_methods}")
        print(f"   - CausalEngine模式: {causal_methods}")
        print(f"   - 传统方法: {traditional_methods}")
        print(f"   - 数据集大小: {self.X.shape[0]:,} 样本 × {self.X.shape[1]} 特征")
        print(f"   - 异常标签比例: {self.config.ANOMALY_RATIO:.1%}")
        
        # 性能排名
        print(f"\n🏆 总体性能排名 (按MdAE分数):")
        print("-" * 50)
        
        all_mdae_scores = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items()]
        all_mdae_scores.sort(key=lambda x: x[1])
        
        for i, (method, mdae) in enumerate(all_mdae_scores, 1):
            method_type = "CausalEngine" if method in self.config.CAUSAL_MODES else "Traditional"
            r2 = self.results[method]['test']['R²']
            print(f"   {i:2d}. {method:<15} ({method_type:<12}) - MdAE: {mdae:.3f}, R²: {r2:.4f}")
        
        # CausalEngine优势分析
        print(f"\n🎯 CausalEngine模式分析:")
        print("-" * 40)
        
        causal_results = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items() 
                         if method in self.config.CAUSAL_MODES]
        traditional_results = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items() 
                              if method not in self.config.CAUSAL_MODES]
        
        if causal_results and traditional_results:
            best_causal = min(causal_results, key=lambda x: x[1])
            best_traditional = min(traditional_results, key=lambda x: x[1])
            
            print(f"   最佳CausalEngine模式: {best_causal[0]} (MdAE = {best_causal[1]:.3f})")
            print(f"   最佳传统方法: {best_traditional[0]} (MdAE = {best_traditional[1]:.3f})")
            
            improvement = (best_traditional[1] - best_causal[1]) / best_traditional[1] * 100
            print(f"   性能提升: {improvement:+.2f}%")
            
            better_causal_count = sum(1 for _, mdae in causal_results if mdae < best_traditional[1])
            print(f"   优于最佳传统方法的CausalEngine模式: {better_causal_count}/{len(causal_results)}")
        
        # 关键发现
        print(f"\n💡 关键发现:")
        print("-" * 20)
        
        if len(all_mdae_scores) > 0:
            top_method = all_mdae_scores[0]
            if top_method[0] in self.config.CAUSAL_MODES:
                print(f"   ✅ CausalEngine模式 '{top_method[0]}' 在MdAE指标上取得最佳性能")
                print(f"   ✅ 因果推理在稳健性方面显示出明显优势")
            else:
                print(f"   ⚠️ 传统方法 '{top_method[0]}' 在MdAE指标上表现最优")
                print(f"   ⚠️ 建议进一步调优CausalEngine参数")


def main():
    """主函数：运行完整的全面CausalEngine模式教程 - Sklearn-Style版本"""
    print("🏠 全面CausalEngine模式教程 - Sklearn-Style版本")
    print("🎯 目标：测试所有CausalEngine推理模式在真实世界回归任务中的表现")
    print("=" * 90)
    
    # 检查包可用性
    print("📦 检查依赖包可用性:")
    available_packages = []
    if XGBOOST_AVAILABLE:
        available_packages.append("XGBoost")
    if LIGHTGBM_AVAILABLE:
        available_packages.append("LightGBM")
    if CATBOOST_AVAILABLE:
        available_packages.append("CatBoost")
    
    print(f"   ✅ 可用的传统机器学习包: {', '.join(available_packages) if available_packages else '无'}")
    
    missing_packages = []
    if not XGBOOST_AVAILABLE:
        missing_packages.append("xgboost")
    if not LIGHTGBM_AVAILABLE:
        missing_packages.append("lightgbm")
    if not CATBOOST_AVAILABLE:
        missing_packages.append("catboost")
    
    if missing_packages:
        print(f"   ⚠️ 缺失的包: {', '.join(missing_packages)}")
        print(f"   💡 提示: pip install {' '.join(missing_packages)}")
    print()
    
    # 创建配置实例
    config = SklearnStyleTutorialConfig()
    
    print(f"🔧 当前配置:")
    print(f"   - CausalEngine模式: {', '.join(config.CAUSAL_MODES)} (共{len(config.CAUSAL_MODES)}种)")
    print(f"   - 网络架构: {config.CAUSAL_HIDDEN_SIZES}")
    print(f"   - 最大轮数: {config.CAUSAL_MAX_EPOCHS}")
    print(f"   - 早停patience: {config.CAUSAL_PATIENCE}")
    print(f"   - 异常比例: {config.ANOMALY_RATIO:.1%}")
    enabled_methods = [k for k, v in config.METHODS_TO_TEST.items() if v]
    print(f"   - 总计对比方法: {len(enabled_methods)} 种")
    print(f"   - 输出目录: {config.OUTPUT_DIR}/")
    print()
    
    # 创建教程实例
    tutorial = SklearnStyleCausalModesTutorial(config)
    
    # 1. 加载和探索数据
    tutorial.load_and_explore_data()
    
    # 2. 数据可视化
    tutorial.visualize_data()
    
    # 3. 运行全面基准测试
    tutorial.run_comprehensive_benchmark()
    
    # 4. 专门分析CausalEngine模式性能
    tutorial.analyze_causal_modes_performance()
    
    # 5. 创建全面性能可视化 - 生成标准版和扩展版
    tutorial.create_comprehensive_performance_visualization(extended=False)  # 标准版
    tutorial.create_comprehensive_performance_visualization(extended=True)   # 扩展版
    
    # 6. 创建CausalEngine模式专项对比
    tutorial.create_causal_modes_comparison()
    
    # 7. 打印全面总结报告
    tutorial.print_comprehensive_summary()
    
    print("\n🎉 全面CausalEngine模式教程完成！(Sklearn-Style版本)")
    print("📋 实验总结:")
    print(f"   - 使用了真实世界的加州房价数据集 ({tutorial.X.shape[0]:,} 样本)")
    print(f"   - 测试了所有 {len(config.CAUSAL_MODES)} 种CausalEngine推理模式")
    print(f"   - 与传统方法进行了全面对比")
    print(f"   - 在 {config.ANOMALY_RATIO:.0%} 标签噪声环境下验证了鲁棒性")
    print(f"   - 使用sklearn-style实现，与Legacy版本形成对比")
    
    print("\n📊 生成的文件:")
    if config.SAVE_PLOTS:
        print(f"   - {config.OUTPUT_DIR}/sklearn_style_data_analysis.png                   (数据分析图)")
        print(f"   - {config.OUTPUT_DIR}/sklearn_style_performance_comparison.png          (标准性能对比图)")
        print(f"   - {config.OUTPUT_DIR}/sklearn_style_performance_comparison_extended.png (扩展性能对比图)")
        print(f"   - {config.OUTPUT_DIR}/sklearn_style_causal_modes_comparison.png         (CausalEngine模式专项对比图)")
    
    print("\n💡 提示：通过修改SklearnStyleTutorialConfig类来自定义实验参数！")
    print("🔬 对比建议：运行Legacy版本的comprehensive_causal_modes_tutorial.py进行性能对比")


if __name__ == "__main__":
    main()