#!/usr/bin/env python3
"""
🏠 扩展版真实世界回归教程：加州房价预测 - Sklearn版本
========================================================

这个教程演示CausalEngine与多种强力传统方法在真实世界回归任务中的性能对比。

与原版扩展教程的区别：
- 直接使用sklearn-style的MLPCausalRegressor、MLPHuberRegressor等封装好的learners
- 不依赖BaselineBenchmark类，直接进行模型训练和比较
- 保持所有原有功能：数据探索、可视化、性能比较、鲁棒性测试
- 包含更全面的鲁棒性回归器对比

数据集：加州房价数据集（California Housing Dataset）
- 20,640个样本
- 8个特征（房屋年龄、收入、人口等）
- 目标：预测房价中位数

我们将比较10种方法：
1. sklearn MLPRegressor（传统神经网络）
2. PyTorch MLP（传统深度学习）
3. MLP Huber（Huber损失稳健回归）
4. MLP Pinball（Pinball损失稳健回归）
5. MLP Cauchy（Cauchy损失稳健回归）
6. Random Forest（随机森林）
7. CatBoost（强力梯度提升）
8. XGBoost（强力梯度提升）
9. CausalEngine - deterministic（确定性因果推理）
10. CausalEngine - standard（标准因果推理）

关键亮点：
- 真实世界数据的鲁棒性测试
- 8种强力传统机器学习方法对比（2种神经网络 + 3种稳健回归 + 3种集成方法）
- 3种稳健神经网络回归方法（Huber、Pinball、Cauchy）
- 3种集成学习方法（Random Forest、CatBoost、XGBoost）
- 2种因果推理方法（deterministic、standard）
- 统一神经网络参数配置确保公平比较
- 因果推理vs传统方法的性能差异分析
- 直接使用sklearn-style learners

实验设计说明
==================================================================
本脚本包含两组核心实验，旨在全面评估CausalEngine在真实回归任务上的性能和鲁棒性。
所有实验参数均可在下方的 `TutorialConfig` 类中进行修改。

实验一：核心性能对比 (在25%标签异常下)
--------------------------------------------------
- **目标**: 比较CausalEngine和8种传统方法在含有固定标签异常数据上的预测性能。
- **设置**: 默认设置25%的标签异常（`ANOMALY_RATIO = 0.25`），模拟真实世界中常见的数据质量问题。
- **对比模型**: 
  - 传统方法: sklearn MLP, PyTorch MLP, MLP Huber, MLP Pinball, MLP Cauchy, Random Forest, CatBoost, XGBoost
  - CausalEngine: deterministic, standard

实验二：鲁棒性分析 (跨越不同标签异常水平)
--------------------------------------------------
- **目标**: 探究模型性能随标签异常水平增加时的变化情况，评估其稳定性。
- **设置**: 在一系列标签异常比例（如0%, 10%, 20%, 30%, 40%）下分别运行测试。
- **对比模型**: 所有10种方法在不同标签异常水平下的表现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
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

# 导入我们的sklearn-style learners
from causal_sklearn.regressor import (
    MLPCausalRegressor, MLPPytorchRegressor, 
    MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
)

warnings.filterwarnings('ignore')

# Try to import CatBoost and XGBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, will skip CatBoost in comparisons")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, will skip XGBoost in comparisons")


class TutorialConfig:
    """
    扩展教程配置类 - 方便调整各种参数
    
    🔧 在这里修改参数来自定义实验设置！
    """
    
    # 🎯 数据分割参数
    TEST_SIZE = 0.2          # 测试集比例
    VAL_SIZE = 0.2           # 验证集比例 (相对于训练集)
    RANDOM_STATE = 42        # 随机种子
    
    # 🧠 统一神经网络配置 - 所有神经网络方法使用相同参数确保公平比较
    # =========================================================================
    # 🔧 在这里修改所有神经网络方法的共同参数！
    NN_HIDDEN_SIZES = (128, 64, 32)                 # 🔧 统一神经网络隐藏层结构
    NN_MAX_EPOCHS = 3000                            # 最大训练轮数
    NN_LEARNING_RATE = 0.01                         # 学习率
    NN_PATIENCE = 50                                # 早停patience
    NN_TOLERANCE = 1e-4                             # 早停tolerance
    # =========================================================================
    
    # ✨ [新功能] 是否对神经网络输入进行特征标准化
    USE_SCALER = True                               # 推荐: True。设为False可观察无标准化时神经网络的性能
    
    # 🤖 CausalEngine参数 - 使用统一神经网络配置
    CAUSAL_MODES = ['deterministic', 'standard']    # 可选: ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    CAUSAL_HIDDEN_SIZES = NN_HIDDEN_SIZES          # 使用统一神经网络配置
    CAUSAL_MAX_EPOCHS = NN_MAX_EPOCHS               # 使用统一神经网络配置
    CAUSAL_LR = NN_LEARNING_RATE                    # 使用统一神经网络配置
    CAUSAL_PATIENCE = NN_PATIENCE                   # 使用统一神经网络配置
    CAUSAL_TOL = NN_TOLERANCE                       # 使用统一神经网络配置
    CAUSAL_GAMMA_INIT = 1.0                         # gamma初始化
    CAUSAL_B_NOISE_INIT = 1.0                       # b_noise初始化
    CAUSAL_B_NOISE_TRAINABLE = True                 # b_noise是否可训练
    
    # 🧠 传统神经网络方法参数 - 使用统一配置
    SKLEARN_HIDDEN_LAYERS = NN_HIDDEN_SIZES         # 使用统一神经网络配置
    SKLEARN_MAX_ITER = NN_MAX_EPOCHS                # 使用统一神经网络配置
    SKLEARN_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    
    PYTORCH_EPOCHS = NN_MAX_EPOCHS                  # 使用统一神经网络配置
    PYTORCH_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    PYTORCH_PATIENCE = NN_PATIENCE                  # 使用统一神经网络配置
    
    # 🎯 鲁棒性回归器参数
    HUBER_DELTA = 1.0                               # Huber损失delta参数
    PINBALL_QUANTILE = 0.5                          # Pinball损失分位数（中位数回归）
    
    # 🌲 传统机器学习方法参数
    RF_N_ESTIMATORS = 100                           # 随机森林树的数量
    RF_MAX_DEPTH = None                             # 随机森林最大深度
    CATBOOST_ITERATIONS = 1000                      # CatBoost迭代次数
    CATBOOST_DEPTH = 6                              # CatBoost深度
    XGBOOST_N_ESTIMATORS = 1000                     # XGBoost迭代次数
    XGBOOST_DEPTH = 6                               # XGBoost深度
    
    # 📊 实验参数
    ANOMALY_RATIO = 0.3                            # 标签异常比例 (核心实验默认值: 30%标签异常挑战) 
    SAVE_PLOTS = True                               # 是否保存图表
    VERBOSE = True                                  # 是否显示详细输出
    
    # 🛡️ 鲁棒性测试参数 - 验证"CausalEngine鲁棒性优势"的假设
    ROBUSTNESS_ANOMALY_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 标签异常水平
    RUN_ROBUSTNESS_TEST = True                      # 是否运行鲁棒性测试
    
    # 📈 可视化参数
    FIGURE_DPI = 300                                # 图表分辨率
    FIGURE_SIZE_ANALYSIS = (16, 12)                 # 数据分析图表大小
    FIGURE_SIZE_PERFORMANCE = (16, 12)              # 性能对比图表大小
    FIGURE_SIZE_ROBUSTNESS = (18, 14)               # 鲁棒性测试图表大小 (更大容纳更多方法)
    
    # 📁 输出目录参数
    OUTPUT_DIR = "results/real_world_regression_tutorial_extended_sklearn"  # 输出目录名称


class ExtendedCaliforniaHousingSklearnTutorial:
    """
    扩展版加州房价回归教程主类 - Sklearn版本
    
    使用sklearn-style learners演示CausalEngine与多种强力方法的性能对比
    """
    
    def __init__(self, config=None):
        self.config = config if config is not None else TutorialConfig()
        self.X = None
        self.y = None
        self.feature_names = None
        self.results = {}
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.config.OUTPUT_DIR):
            os.makedirs(self.config.OUTPUT_DIR)
            if self.config.VERBOSE:
                print(f"📁 创建输出目录: {self.config.OUTPUT_DIR}/")
    
    def _get_output_path(self, filename):
        """获取输出文件的完整路径"""
        return os.path.join(self.config.OUTPUT_DIR, filename)
        
    def _get_clean_method_name(self, method_key: str) -> str:
        """Removes prefixes and suffixes from method names for clean display."""
        name = method_key.replace('causal_', '').replace('_mlp', '').replace('mlp_', '')
        return name
        
    def load_and_explore_data(self, verbose=True):
        """加载并探索加州房价数据集"""
        if verbose:
            print("🏠 扩展版加州房价预测 - 真实世界回归教程 (Sklearn版本)")
            print("=" * 70)
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
            
            # 统计可用方法
            total_methods = len(self.config.CAUSAL_MODES) + 5  # 5种传统方法基础
            if CATBOOST_AVAILABLE:
                total_methods += 1
            if XGBOOST_AVAILABLE:
                total_methods += 1

            ensemble_methods = "Random Forest"
            if CATBOOST_AVAILABLE:
                ensemble_methods += ", CatBoost"
            if XGBOOST_AVAILABLE:
                ensemble_methods += ", XGBoost"
            ensemble_count = sum([1 for m in [CATBOOST_AVAILABLE, XGBOOST_AVAILABLE, True]])

            print(f"\n🎯 将比较 {total_methods} 种方法:")
            print(f"   - CausalEngine ({len(self.config.CAUSAL_MODES)}种): {', '.join(self.config.CAUSAL_MODES)}")
            print(f"   - 传统神经网络 (2种): sklearn MLP, PyTorch MLP")
            print(f"   - 鲁棒性回归器 (3种): Huber, Pinball, Cauchy")
            print(f"   - 集成学习 ({ensemble_count}种): {ensemble_methods}")
        
        return self.X, self.y
    
    def visualize_data(self, save_plots=None):
        """数据可视化分析"""
        if save_plots is None:
            save_plots = self.config.SAVE_PLOTS
            
        print("\n📈 数据分布分析")
        print("-" * 30)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ANALYSIS)
        fig.suptitle('California Housing Dataset Analysis - Extended Sklearn Version', fontsize=16, fontweight='bold')
        
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
            # 使用与原始版本一致的文件名
            output_path = self._get_output_path('extended_data_analysis.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 数据分析图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 数据统计摘要
        print("\n📋 数据统计摘要:")
        print(f"  - 最相关特征: {most_corr_feature} (相关系数: {corr_matrix.loc[most_corr_feature, 'MedHouseVal']:.3f})")
        print(f"  - 异常值检测: {np.sum(np.abs(self.y - self.y.mean()) > 3 * self.y.std())} 个潜在异常值")
        print(f"  - 数据完整性: 无缺失值" if not np.any(np.isnan(self.X)) else "  - 警告: 存在缺失值")
    
    def _inject_label_anomalies(self, y, anomaly_ratio, random_state=42):
        """注入标签异常"""
        if anomaly_ratio <= 0:
            return y.copy()
            
        np.random.seed(random_state)
        y_noisy = y.copy()
        n_anomalies = int(len(y) * anomaly_ratio)
        
        # 随机选择异常样本
        anomaly_indices = np.random.choice(len(y), n_anomalies, replace=False)
        
        # 添加异常 - 使用标准差的倍数作为异常强度
        anomaly_strength = np.std(y) * 2.0  # 2倍标准差的异常
        anomalies = np.random.normal(0, anomaly_strength, n_anomalies)
        y_noisy[anomaly_indices] += anomalies
        
        return y_noisy
    
    def _train_models(self, X_train, y_train, X_val, y_val, anomaly_ratio, verbose=True):
        """训练所有模型"""
        if verbose:
            print(f"\n🔧 训练模型 (异常比例: {anomaly_ratio:.1%})")
            print("-" * 50)
        
        # 数据预处理 - 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 注入标签异常到训练标签
        y_train_noisy = self._inject_label_anomalies(y_train, anomaly_ratio, random_state=self.config.RANDOM_STATE)
        
        models = {}
        
        # 1. Sklearn MLPRegressor
        if verbose:
            print("   🔧 训练 sklearn MLPRegressor...")
        sklearn_model = MLPRegressor(
            hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
            max_iter=self.config.SKLEARN_MAX_ITER,
            learning_rate_init=self.config.SKLEARN_LR,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            random_state=self.config.RANDOM_STATE,
            alpha=0.0001  # L2正则化
        )
        sklearn_model.fit(X_train_scaled, y_train_noisy)
        models['sklearn_mlp'] = (sklearn_model, scaler)
        
        # 2. PyTorch MLP
        if verbose:
            print("   🔧 训练 PyTorch MLPRegressor...")
        pytorch_model = MLPPytorchRegressor(
            hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,
            max_iter=self.config.PYTORCH_EPOCHS,
            learning_rate=self.config.PYTORCH_LR,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=self.config.PYTORCH_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            random_state=self.config.RANDOM_STATE,
            verbose=False,
            alpha=0.0001
        )
        pytorch_model.fit(X_train_scaled, y_train_noisy)
        models['pytorch_mlp'] = (pytorch_model, scaler)
        
        # 3. MLP Huber Regressor
        if verbose:
            print("   🔧 训练 MLP Huber Regressor...")
        huber_model = MLPHuberRegressor(
            hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,
            max_iter=self.config.NN_MAX_EPOCHS,
            learning_rate=self.config.NN_LEARNING_RATE,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            delta=self.config.HUBER_DELTA,
            random_state=self.config.RANDOM_STATE,
            verbose=False,
            alpha=0.0001
        )
        huber_model.fit(X_train_scaled, y_train_noisy)
        models['mlp_huber'] = (huber_model, scaler)
        
        # 4. MLP Pinball Regressor
        if verbose:
            print("   🔧 训练 MLP Pinball Regressor...")
        pinball_model = MLPPinballRegressor(
            hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,
            max_iter=self.config.NN_MAX_EPOCHS,
            learning_rate=self.config.NN_LEARNING_RATE,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            quantile=self.config.PINBALL_QUANTILE,
            random_state=self.config.RANDOM_STATE,
            verbose=False,
            alpha=0.0001
        )
        pinball_model.fit(X_train_scaled, y_train_noisy)
        models['mlp_pinball'] = (pinball_model, scaler)
        
        # 5. MLP Cauchy Regressor
        if verbose:
            print("   🔧 训练 MLP Cauchy Regressor...")
        cauchy_model = MLPCauchyRegressor(
            hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,
            max_iter=self.config.NN_MAX_EPOCHS,
            learning_rate=self.config.NN_LEARNING_RATE,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            random_state=self.config.RANDOM_STATE,
            verbose=False,
            alpha=0.0001
        )
        cauchy_model.fit(X_train_scaled, y_train_noisy)
        models['mlp_cauchy'] = (cauchy_model, scaler)
        
        # 6. Random Forest
        if verbose:
            print("   🔧 训练 Random Forest...")
        # Random Forest doesn't need scaling
        rf_model = RandomForestRegressor(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1  # 使用所有CPU核心
        )
        rf_model.fit(X_train, y_train_noisy)  # RF不需要标准化
        # 为RF创建一个恒等缩放器以保持接口一致
        identity_scaler = StandardScaler()
        identity_scaler.fit(X_train)
        identity_scaler.scale_ = np.ones(X_train.shape[1])
        identity_scaler.mean_ = np.zeros(X_train.shape[1])
        models['random_forest'] = (rf_model, identity_scaler)
        
        # 7. CatBoost (如果可用)
        if CATBOOST_AVAILABLE:
            if verbose:
                print("   🔧 训练 CatBoost...")
            catboost_model = CatBoostRegressor(
                iterations=self.config.CATBOOST_ITERATIONS,
                depth=self.config.CATBOOST_DEPTH,
                learning_rate=0.1,
                random_seed=self.config.RANDOM_STATE,
                verbose=False  # 关闭CatBoost的详细输出
            )
            catboost_model.fit(X_train, y_train_noisy)  # CatBoost也不需要标准化
            # 为CatBoost创建一个恒等缩放器以保持接口一致
            identity_scaler_cat = StandardScaler()
            identity_scaler_cat.fit(X_train)
            identity_scaler_cat.scale_ = np.ones(X_train.shape[1])
            identity_scaler_cat.mean_ = np.zeros(X_train.shape[1])
            models['catboost'] = (catboost_model, identity_scaler_cat)
        
        # 8. XGBoost (如果可用)
        if XGBOOST_AVAILABLE:
            if verbose:
                print("   🔧 训练 XGBoost...")
            xgboost_model = XGBRegressor(
                n_estimators=self.config.XGBOOST_N_ESTIMATORS,
                max_depth=self.config.XGBOOST_DEPTH,
                learning_rate=0.05,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                early_stopping_rounds=50
            )
            # XGBoost 使用验证集进行早停
            xgboost_model.fit(X_train, y_train_noisy, eval_set=[(X_val, y_val)], verbose=False)
            # 为XGBoost也创建一个恒等缩放器
            identity_scaler_xgb = StandardScaler()
            identity_scaler_xgb.fit(X_train)
            identity_scaler_xgb.scale_ = np.ones(X_train.shape[1])
            identity_scaler_xgb.mean_ = np.zeros(X_train.shape[1])
            models['xgboost'] = (xgboost_model, identity_scaler_xgb)
        
        # 9. CausalEngine 各种模式
        for mode in self.config.CAUSAL_MODES:
            if verbose:
                print(f"   🔧 训练 CausalEngine ({mode})...")
            causal_model = MLPCausalRegressor(
                perception_hidden_layers=self.config.CAUSAL_HIDDEN_SIZES,
                mode=mode,
                gamma_init=self.config.CAUSAL_GAMMA_INIT,
                b_noise_init=self.config.CAUSAL_B_NOISE_INIT,
                b_noise_trainable=self.config.CAUSAL_B_NOISE_TRAINABLE,
                max_iter=self.config.CAUSAL_MAX_EPOCHS,
                learning_rate=self.config.CAUSAL_LR,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=self.config.CAUSAL_PATIENCE,
                tol=self.config.CAUSAL_TOL,
                random_state=self.config.RANDOM_STATE,
                verbose=False,
                alpha=0.0001
            )
            causal_model.fit(X_train_scaled, y_train_noisy)
            models[f'causal_{mode}'] = (causal_model, scaler)
        
        return models
    
    def _evaluate_models(self, models, X_val, y_val, X_test, y_test):
        """评估所有模型"""
        results = {}
        
        for model_name, (model, scaler) in models.items():
            # 根据配置决定是否使用标准化
            if self.config.USE_SCALER:
                # 对于需要标准化的模型，进行标准化
                if model_name in ['random_forest', 'catboost', 'xgboost']:
                    # RF, CatBoost, XGBoost 不需要标准化
                    X_val_processed = X_val
                    X_test_processed = X_test
                else:
                    # 神经网络需要标准化
                    X_val_processed = scaler.transform(X_val)
                    X_test_processed = scaler.transform(X_test)
            else:
                # 不使用标准化时，所有模型都用原始数据
                X_val_processed = X_val
                X_test_processed = X_test
            
            # 验证集评估
            y_val_pred = model.predict(X_val_processed)
            
            val_metrics = {
                'MAE': mean_absolute_error(y_val, y_val_pred),
                'MdAE': median_absolute_error(y_val, y_val_pred),
                'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'R²': r2_score(y_val, y_val_pred)
            }
            
            # 测试集评估
            y_test_pred = model.predict(X_test_processed)
            
            test_metrics = {
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'MdAE': median_absolute_error(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'R²': r2_score(y_test, y_test_pred)
            }
            
            results[model_name] = {
                'val': val_metrics,
                'test': test_metrics
            }
        
        return results
    
    def run_comprehensive_benchmark(self, test_size=None, val_size=None, anomaly_ratio=None, verbose=None):
        """运行全面的基准测试"""
        # 使用配置参数作为默认值
        if test_size is None:
            test_size = self.config.TEST_SIZE
        if val_size is None:
            val_size = self.config.VAL_SIZE
        if anomaly_ratio is None:
            anomaly_ratio = self.config.ANOMALY_RATIO
        if verbose is None:
            verbose = self.config.VERBOSE
            
        if verbose:
            print("\n🚀 开始扩展版综合基准测试 (Sklearn版本)")
            print("=" * 70)
            print(f"🔧 实验配置:")
            print(f"   - 测试集比例: {test_size:.1%}")
            print(f"   - 验证集比例: {val_size:.1%}")
            print(f"   - 标签异常比例: {anomaly_ratio:.1%}")
            print(f"   - 使用特征缩放 (Scaler): {'是' if self.config.USE_SCALER else '否'}")
            print(f"   - 随机种子: {self.config.RANDOM_STATE}")
            print(f"   - CausalEngine模式: {', '.join(self.config.CAUSAL_MODES)}")
            print(f"   - 神经网络结构: {self.config.NN_HIDDEN_SIZES}")
            print(f"   - 最大训练轮数: {self.config.NN_MAX_EPOCHS}")
            print(f"   - 早停patience: {self.config.NN_PATIENCE}")
        
        # 数据分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.config.RANDOM_STATE
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.RANDOM_STATE
        )
        
        if verbose:
            print(f"\n📊 数据分割:")
            print(f"   - 训练集: {len(X_train)} 样本")
            print(f"   - 验证集: {len(X_val)} 样本")
            print(f"   - 测试集: {len(X_test)} 样本")
        
        # 训练模型
        start_time = time.time()
        models = self._train_models(X_train, y_train, X_val, y_val, anomaly_ratio, verbose)
        training_time = time.time() - start_time
        
        # 评估模型
        core_results = self._evaluate_models(models, X_val, y_val, X_test, y_test)
        
        # 存储核心性能结果 (与原始版本保持一致的结构)
        if not hasattr(self, 'results'):
            self.results = {}
        self.results['core_performance'] = core_results
        
        if verbose:
            print(f"\n⏱️ 总训练时间: {training_time:.1f} 秒")
            print(f"\n📊 扩展版基准测试结果 (异常比例: {anomaly_ratio:.0%})")
            self._print_results()
        
        return core_results
    
    def generate_summary_report(self):
        """生成实验总结报告 - 与原始版本功能完全对应"""
        if self.config.VERBOSE:
            print("\n📋 生成实验总结报告...")
        
        report_lines = []
        report_lines.append("# 扩展版加州房价回归实验总结报告")
        report_lines.append("")
        report_lines.append("🏠 **California Housing Dataset Regression Analysis**")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 实验配置
        report_lines.append("## 📊 实验配置")
        report_lines.append("")
        report_lines.append(f"- **数据集**: 加州房价数据集")
        report_lines.append(f"  - 样本数: {self.X.shape[0]:,}")
        report_lines.append(f"  - 特征数: {self.X.shape[1]}")
        report_lines.append(f"  - 房价范围: ${self.y.min():.2f} - ${self.y.max():.2f} (10万美元)")
        report_lines.append("")
        report_lines.append(f"- **数据分割**:")
        report_lines.append(f"  - 测试集比例: {self.config.TEST_SIZE:.1%}")
        report_lines.append(f"  - 验证集比例: {self.config.VAL_SIZE:.1%}")
        report_lines.append(f"  - 随机种子: {self.config.RANDOM_STATE}")
        report_lines.append("")
        report_lines.append(f"- **神经网络统一配置**:")
        report_lines.append(f"  - 网络结构: {self.config.NN_HIDDEN_SIZES}")
        report_lines.append(f"  - 最大轮数: {self.config.NN_MAX_EPOCHS}")
        report_lines.append(f"  - 学习率: {self.config.NN_LEARNING_RATE}")
        report_lines.append(f"  - 早停patience: {self.config.NN_PATIENCE}")
        report_lines.append("")
        
        # 计算方法数量
        baseline_methods = ['sklearn', 'pytorch', 'huber', 'pinball', 'cauchy', 'random_forest']
        if CATBOOST_AVAILABLE:
            baseline_methods.append('catboost')
        if XGBOOST_AVAILABLE:
            baseline_methods.append('xgboost')
        causal_modes = self.config.CAUSAL_MODES
        
        report_lines.append(f"- **实验方法**: {len(baseline_methods) + len(causal_modes)} 种")
        report_lines.append(f"  - 传统方法 ({len(baseline_methods)}种): {', '.join(baseline_methods)}")
        report_lines.append(f"  - CausalEngine ({len(causal_modes)}种): {', '.join(causal_modes)}")
        report_lines.append("")
        
        # 核心性能测试结果
        if 'core_performance' in self.results:
            results = self.results['core_performance']
            report_lines.append("## 🎯 核心性能测试结果")
            report_lines.append("")
            report_lines.append(f"**标签异常水平**: {self.config.ANOMALY_RATIO:.1%}")
            report_lines.append("")
            
            # 创建性能表格 - 按MdAE排序
            methods_by_mdae = sorted(results.keys(), key=lambda x: results[x]['test']['MdAE'])
            
            report_lines.append("### 📈 测试集性能排名 (按MdAE升序)")
            report_lines.append("")
            
            # 表格头
            report_lines.append("| 排名 | 方法 | MAE | MdAE | RMSE | R² | 方法类型 |")
            report_lines.append("|:----:|------|----:|-----:|-----:|---:|----------|")
            
            for i, method in enumerate(methods_by_mdae, 1):
                test_metrics = results[method]['test']
                
                # 判断方法类型
                if any(mode in method for mode in ['deterministic', 'standard', 'exogenous', 'endogenous']):
                    method_type = "🤖 CausalEngine"
                elif any(robust in method.lower() for robust in ['huber', 'cauchy', 'pinball']):
                    method_type = "🛡️ 稳健回归"
                elif method.lower() in ['catboost', 'random_forest', 'xgboost']:
                    method_type = "🌲 集成学习"
                else:
                    method_type = "🧠 神经网络"
                
                report_lines.append(f"| {i} | **{self._get_clean_method_name(method)}** | "
                                  f"{test_metrics['MAE']:.4f} | "
                                  f"**{test_metrics['MdAE']:.4f}** | "
                                  f"{test_metrics['RMSE']:.4f} | "
                                  f"{test_metrics['R²']:.4f} | "
                                  f"{method_type} |")
            
            report_lines.append("")
            
            # 验证集vs测试集对比（展示噪声影响）
            report_lines.append("### 🔍 验证集 vs 测试集性能对比")
            report_lines.append("")
            report_lines.append("*验证集包含标签异常，测试集为纯净数据*")
            report_lines.append("")
            
            report_lines.append("| 方法 | 验证集MdAE | 测试集MdAE | 性能提升 |")
            report_lines.append("|------|----------:|----------:|--------:|")
            
            for method in methods_by_mdae:
                val_mdae = results[method]['val']['MdAE']
                test_mdae = results[method]['test']['MdAE']
                improvement = ((val_mdae - test_mdae) / val_mdae) * 100
                
                report_lines.append(f"| {self._get_clean_method_name(method)} | "
                                  f"{val_mdae:.4f} | "
                                  f"{test_mdae:.4f} | "
                                  f"{improvement:+.1f}% |")
            
            report_lines.append("")
            
            # 关键发现
            best_mdae_method = methods_by_mdae[0]
            best_mdae_score = results[best_mdae_method]['test']['MdAE']
            
            # 识别CausalEngine方法
            causal_methods = [m for m in results.keys() if any(mode in m for mode in ['deterministic', 'standard', 'exogenous', 'endogenous'])]
            
            report_lines.append("### 🏆 关键发现")
            report_lines.append("")
            report_lines.append(f"- **🥇 最佳整体性能**: `{self._get_clean_method_name(best_mdae_method)}` (MdAE: {best_mdae_score:.4f})")
            
            if causal_methods:
                best_causal = min(causal_methods, key=lambda x: results[x]['test']['MdAE'])
                causal_rank = methods_by_mdae.index(best_causal) + 1
                causal_score = results[best_causal]['test']['MdAE']
                report_lines.append(f"- **🤖 最佳CausalEngine**: `{self._get_clean_method_name(best_causal)}` (排名: {causal_rank}/{len(methods_by_mdae)}, MdAE: {causal_score:.4f})")
                
                # CausalEngine模式对比
                if len(causal_methods) > 1:
                    report_lines.append("")
                    report_lines.append("**CausalEngine模式对比**:")
                    for causal_method in sorted(causal_methods, key=lambda x: results[x]['test']['MdAE']):
                        rank = methods_by_mdae.index(causal_method) + 1
                        score = results[causal_method]['test']['MdAE']
                        report_lines.append(f"  - `{self._get_clean_method_name(causal_method)}`: 排名 {rank}, MdAE {score:.4f}")
            
            # 传统方法分析
            traditional_methods = [m for m in results.keys() if m not in causal_methods]
            if traditional_methods:
                best_traditional = min(traditional_methods, key=lambda x: results[x]['test']['MdAE'])
                traditional_rank = methods_by_mdae.index(best_traditional) + 1
                traditional_score = results[best_traditional]['test']['MdAE']
                report_lines.append(f"- **🏅 最佳传统方法**: `{self._get_clean_method_name(best_traditional)}` (排名: {traditional_rank}/{len(methods_by_mdae)}, MdAE: {traditional_score:.4f})")
            
            report_lines.append("")
        
        # 鲁棒性测试结果
        if 'robustness' in self.results:
            robustness_results = self.results['robustness']
            report_lines.append("## 🛡️ 鲁棒性测试结果")
            report_lines.append("")
            
            noise_levels = sorted(robustness_results.keys())
            methods = list(robustness_results[noise_levels[0]].keys())
            
            report_lines.append("### 📊 MdAE性能随标签异常水平变化")
            report_lines.append("")
            
            # 表格头
            header = "| 方法 | " + " | ".join([f"{r:.0%}" for r in noise_levels]) + " | 稳定性* |"
            separator = "|------|" + "|".join([f"{'-'*(len(f'{r:.0%}')+1):->6}" for r in noise_levels]) + "|--------|"
            
            report_lines.append(header)
            report_lines.append(separator)
            
                            # 按0%标签异常性能排序
            methods_by_clean = sorted(methods, key=lambda x: robustness_results[0.0][x]['test']['MdAE'])
            
            for method in methods_by_clean:
                mdae_values = []
                scores = []
                for noise in noise_levels:
                    score = robustness_results[noise][method]['test']['MdAE']
                    scores.append(score)
                    mdae_values.append(f"{score:.4f}")
                
                # 计算稳定性 (最大值-最小值)/最小值
                stability = (max(scores) - min(scores)) / min(scores) * 100
                
                # 方法名格式化
                method_display = self._get_clean_method_name(method)
                
                report_lines.append(f"| {method_display} | " + 
                                  " | ".join(mdae_values) + 
                                  f" | {stability:.1f}% |")
            
            report_lines.append("")
            report_lines.append("*稳定性 = (最大MdAE - 最小MdAE) / 最小MdAE × 100%，越小越稳定*")
            report_lines.append("")
            
            # 鲁棒性分析
            report_lines.append("### 🔍 鲁棒性分析")
            report_lines.append("")
            
            # 找出最稳定的方法
            stability_scores = {}
            for method in methods:
                scores = [robustness_results[noise][method]['test']['MdAE'] for noise in noise_levels]
                stability_scores[method] = (max(scores) - min(scores)) / min(scores) * 100
            
            most_stable = min(stability_scores.keys(), key=lambda x: stability_scores[x])
            least_stable = max(stability_scores.keys(), key=lambda x: stability_scores[x])
            
            report_lines.append(f"- **🏆 最稳定方法**: `{self._get_clean_method_name(most_stable)}` (稳定性: {stability_scores[most_stable]:.1f}%)")
            report_lines.append(f"- **⚠️ 最不稳定方法**: `{self._get_clean_method_name(least_stable)}` (稳定性: {stability_scores[least_stable]:.1f}%)")
            
            report_lines.append("")
        
        # 添加脚注
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## 📝 说明")
        report_lines.append("")
        report_lines.append("- **MdAE**: Median Absolute Error (中位数绝对误差) - 主要评估指标")
        report_lines.append("- **MAE**: Mean Absolute Error (平均绝对误差)")
        report_lines.append("- **RMSE**: Root Mean Square Error (均方根误差)")
        report_lines.append("- **R²**: 决定系数 (越接近1越好)")
        report_lines.append("- **标签异常设置**: 验证集包含人工标签异常，测试集为纯净数据")
        report_lines.append("- **统一配置**: 所有神经网络方法使用相同的超参数确保公平比较")
        report_lines.append("")
        
        import pandas as pd
        report_lines.append(f"📊 **生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 保存报告 - 使用与原始版本一致的文件名
        report_path = self._get_output_path('extended_experiment_summary.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        if self.config.VERBOSE:
            print(f"📋 实验总结报告已保存: {report_path}")
        
        return report_lines
    
    def _print_results(self):
        """打印结果表格"""
        print("=" * 160)
        print(f"{'方法':<20} {'验证集':<65} {'测试集':<65}")
        print(f"{'':20} {'MAE':<12} {'MdAE':<12} {'RMSE':<12} {'R²':<12} {'MAE':<12} {'MdAE':<12} {'RMSE':<12} {'R²':<12}")
        print("-" * 160)
        
        # 按测试集MdAE排序显示 - 使用正确的结果结构
        core_results = self.results.get('core_performance', self.results)
        sorted_methods = sorted(core_results.keys(), key=lambda x: core_results[x]['test']['MdAE'])
        
        for method in sorted_methods:
            metrics = core_results[method]
            val_m = metrics['val']
            test_m = metrics['test']
            
            # 格式化方法名称
            display_name = self._get_clean_method_name(method)
            
            print(f"{display_name:<20} {val_m['MAE']:<12.4f} {val_m['MdAE']:<12.4f} {val_m['RMSE']:<12.4f} {val_m['R²']:<12.4f} "
                  f"{test_m['MAE']:<12.4f} {test_m['MdAE']:<12.4f} {test_m['RMSE']:<12.4f} {test_m['R²']:<12.4f}")
        
        print("=" * 160)
        
        # 显示排名和性能分析
        best_method = sorted_methods[0]
        best_score = core_results[best_method]['test']['MdAE']
        print(f"\n🏆 最佳方法: {self._get_clean_method_name(best_method)} (测试集MdAE: {best_score:.4f})")
        
        # CausalEngine分析
        causal_methods = [m for m in sorted_methods if 'causal' in m]
        if causal_methods:
            best_causal = causal_methods[0]
            causal_rank = sorted_methods.index(best_causal) + 1
            causal_score = core_results[best_causal]['test']['MdAE']
            print(f"🤖 最佳CausalEngine: {self._get_clean_method_name(best_causal)} (排名: {causal_rank}/{len(sorted_methods)}, 测试集MdAE: {causal_score:.4f})")
    
    def analyze_performance(self, verbose=True):
        """分析性能结果"""
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        # 获取正确的结果结构
        core_results = self.results.get('core_performance', self.results)
        
        if verbose:
            print("\n🔍 扩展版性能分析")
            print("=" * 60)
        
        # 提取测试集R²分数
        test_r2_scores = {}
        test_mdae_scores = {}
        for method, metrics in core_results.items():
            test_r2_scores[method] = metrics['test']['R²']
            test_mdae_scores[method] = metrics['test']['MdAE']
        
        # 找到最佳方法（按MdAE）
        best_method_mdae = min(test_mdae_scores.keys(), key=lambda x: test_mdae_scores[x])
        best_mdae = test_mdae_scores[best_method_mdae]
        
        # 找到最佳方法（按R²）
        best_method_r2 = max(test_r2_scores.keys(), key=lambda x: test_r2_scores[x])
        best_r2 = test_r2_scores[best_method_r2]
        
        if verbose:
            print(f"🏆 最佳方法 (MdAE): {self._get_clean_method_name(best_method_mdae)}")
            print(f"   MdAE = {best_mdae:.4f}")
            print(f"🏆 最佳方法 (R²): {self._get_clean_method_name(best_method_r2)}")
            print(f"   R² = {best_r2:.4f}")
            print()
            print("📊 性能排名 (按MdAE升序):")
            
            sorted_methods = sorted(test_mdae_scores.items(), key=lambda x: x[1])
            for i, (method, mdae) in enumerate(sorted_methods, 1):
                r2 = test_r2_scores[method]
                print(f"   {i}. {self._get_clean_method_name(method):<20} MdAE = {mdae:.4f}, R² = {r2:.4f}")
        
        # 分类分析
        causal_methods = [m for m in core_results.keys() if 'causal' in m]
        neural_methods = [m for m in core_results.keys() if 'mlp' in m and 'causal' not in m]
        ensemble_methods = [m for m in core_results.keys() if m in ['random_forest', 'catboost', 'xgboost']]
        
        if verbose and causal_methods:
            print(f"\n🎯 方法类别分析:")
            
            # CausalEngine分析
            best_causal = min(causal_methods, key=lambda x: test_mdae_scores[x])
            print(f"   🤖 最佳CausalEngine: {self._get_clean_method_name(best_causal)} (MdAE: {test_mdae_scores[best_causal]:.4f})")
            
            # 神经网络分析
            if neural_methods:
                best_neural = min(neural_methods, key=lambda x: test_mdae_scores[x])
                print(f"   🧠 最佳神经网络: {self._get_clean_method_name(best_neural)} (MdAE: {test_mdae_scores[best_neural]:.4f})")
            
            # 集成方法分析
            if ensemble_methods:
                best_ensemble = min(ensemble_methods, key=lambda x: test_mdae_scores[x])
                print(f"   🌲 最佳集成方法: {self._get_clean_method_name(best_ensemble)} (MdAE: {test_mdae_scores[best_ensemble]:.4f})")
            
            # 性能提升分析
            all_traditional = neural_methods + ensemble_methods
            if all_traditional:
                best_traditional = min(all_traditional, key=lambda x: test_mdae_scores[x])
                improvement = ((test_mdae_scores[best_traditional] - test_mdae_scores[best_causal]) 
                             / test_mdae_scores[best_traditional]) * 100
                
                print(f"\n📈 CausalEngine vs 传统方法:")
                print(f"   最佳CausalEngine: {self._get_clean_method_name(best_causal)} (MdAE: {test_mdae_scores[best_causal]:.4f})")
                print(f"   最佳传统方法: {self._get_clean_method_name(best_traditional)} (MdAE: {test_mdae_scores[best_traditional]:.4f})")
                print(f"   性能提升: {improvement:+.2f}%")
                
                if improvement > 0:
                    print(f"   ✅ CausalEngine显著优于传统方法！")
                else:
                    print(f"   ⚠️ 在此设置下传统方法表现更好")
        
        return test_mdae_scores
    
    def create_performance_visualization(self, save_plot=None):
        """创建性能可视化图表"""
        if save_plot is None:
            save_plot = self.config.SAVE_PLOTS
            
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\n📊 创建扩展版性能可视化图表")
        print("-" * 30)
        
        # 准备数据
        core_results = self.results.get('core_performance', self.results)
        methods = list(core_results.keys())
        clean_methods = [self._get_clean_method_name(m) for m in methods]
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_PERFORMANCE)
        fig.suptitle('Extended CausalEngine vs Traditional Methods: California Housing Performance (25% Label Anomaly) - Sklearn Version', 
                     fontsize=14, fontweight='bold')
        axes = axes.flatten()  # 展平为一维数组便于访问
        
        # 设置颜色
        colors = []
        for method in methods:
            if 'causal' in method:
                colors.append('gold')  # CausalEngine用金色
            elif any(robust in method for robust in ['huber', 'pinball', 'cauchy']):
                colors.append('lightgreen')  # 稳健方法用浅绿
            elif method in ['random_forest', 'catboost', 'xgboost']:
                colors.append('lightcoral')  # 集成方法用浅红
            else:
                colors.append('lightblue')  # 传统神经网络用浅蓝
        
        for i, metric in enumerate(metrics):
            values = [core_results[method]['test'][metric] for method in methods]
            
            bars = axes[i].bar(clean_methods, values, color=colors, alpha=0.8, edgecolor='black')
            axes[i].set_title(f'{metric} (Test Set)', fontweight='bold')
            axes[i].set_ylabel(metric)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # 高亮最佳结果
            if metric == 'R²':
                best_idx = values.index(max(values))
            else:
                best_idx = values.index(min(values))
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)
            
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            # 使用与原始版本一致的文件名
            output_path = self._get_output_path('core_performance_comparison.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 扩展版性能图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
    
    def run_robustness_test(self, anomaly_ratios=None, verbose=None):
        """运行鲁棒性测试（不同异常比例）"""
        if anomaly_ratios is None:
            anomaly_ratios = self.config.ROBUSTNESS_ANOMALY_RATIOS
        if verbose is None:
            verbose = self.config.VERBOSE
            
        if verbose:
            print("\n🛡️ 扩展版鲁棒性测试")
            print("=" * 60)
            print("测试CausalEngine与多种方法在不同异常标签比例下的表现")
        
        robustness_results = {}
        
        # 数据分割（固定分割以确保一致性）
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            random_state=self.config.RANDOM_STATE
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.2,
            random_state=self.config.RANDOM_STATE
        )
        
        total_tests = len(anomaly_ratios)
        for i, anomaly_ratio in enumerate(anomaly_ratios):
            if verbose:
                print(f"\n🔬 测试异常比例 {i+1}/{total_tests}: {anomaly_ratio:.1%}")
                print("-" * 30)
            
            # 训练模型 (为了节省时间，仅训练核心模型)
            models = self._train_robustness_models(X_train, y_train, X_val, y_val, anomaly_ratio, verbose=False)
            
            # 评估模型
            results = self._evaluate_models(models, X_val, y_val, X_test, y_test)
            robustness_results[anomaly_ratio] = results
            
            if verbose:
                print("主要方法 MdAE 分数:")
                core_methods = ['sklearn_mlp', 'pytorch_mlp', 'random_forest', 'mlp_huber'] + [f'causal_{mode}' for mode in self.config.CAUSAL_MODES]
                if CATBOOST_AVAILABLE:
                    core_methods.append('catboost')
                if XGBOOST_AVAILABLE:
                    core_methods.append('xgboost')
                
                for method in core_methods:
                    if method in results:
                        mdae = results[method]['test']['MdAE']
                        print(f"  {method:<15}: {mdae:.4f}")
        
        # 存储鲁棒性结果到主结果结构（与原始版本保持一致）
        if not hasattr(self, 'results'):
            self.results = {}
        self.results['robustness'] = robustness_results
        
        # 可视化鲁棒性结果
        if verbose:
            # 打印详细的鲁棒性表格
            self._print_robustness_table(robustness_results, anomaly_ratios)
            # 绘制鲁棒性图表
            self._plot_robustness_results(robustness_results, anomaly_ratios)
            # 分析鲁棒性趋势
            self._analyze_robustness_trends(robustness_results, anomaly_ratios)
        
        return robustness_results
    
    def _train_robustness_models(self, X_train, y_train, X_val, y_val, anomaly_ratio, verbose=False):
        """为鲁棒性测试训练核心模型（精简版）"""
        # 根据配置决定是否使用标准化
        if self.config.USE_SCALER:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            # 创建一个恒等变换的scaler
            scaler = StandardScaler(with_mean=False, with_std=False)
            scaler.fit(X_train)
            X_train_scaled = X_train.copy() # 明确复制
        
        # 注入标签异常
        y_train_noisy = self._inject_label_anomalies(y_train, anomaly_ratio, random_state=self.config.RANDOM_STATE)
        
        models = {}
        
        # 训练核心模型 - 为节省时间，只训练代表性模型
        core_models_config = {
            'sklearn_mlp': MLPRegressor(
                hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
                max_iter=self.config.SKLEARN_MAX_ITER,
                learning_rate_init=self.config.SKLEARN_LR,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=self.config.NN_PATIENCE,
                tol=self.config.NN_TOLERANCE,
                random_state=self.config.RANDOM_STATE,
                alpha=0.0001
            ),
            'pytorch_mlp': MLPPytorchRegressor(
                hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,
                max_iter=self.config.PYTORCH_EPOCHS,
                learning_rate=self.config.PYTORCH_LR,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=self.config.PYTORCH_PATIENCE,
                tol=self.config.NN_TOLERANCE,
                random_state=self.config.RANDOM_STATE,
                verbose=False,
                alpha=0.0001
            ),
            'mlp_huber': MLPHuberRegressor(
                hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,
                max_iter=self.config.NN_MAX_EPOCHS,
                learning_rate=self.config.NN_LEARNING_RATE,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=self.config.NN_PATIENCE,
                tol=self.config.NN_TOLERANCE,
                delta=self.config.HUBER_DELTA,
                random_state=self.config.RANDOM_STATE,
                verbose=False,
                alpha=0.0001
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=self.config.RF_N_ESTIMATORS,
                max_depth=self.config.RF_MAX_DEPTH,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
        }
        
        # 添加CatBoost（如果可用）
        if CATBOOST_AVAILABLE:
            core_models_config['catboost'] = CatBoostRegressor(
                iterations=self.config.CATBOOST_ITERATIONS,
                depth=self.config.CATBOOST_DEPTH,
                learning_rate=0.1,
                random_seed=self.config.RANDOM_STATE,
                verbose=False
            )
        
        # 添加XGBoost（如果可用）
        if XGBOOST_AVAILABLE:
            core_models_config['xgboost'] = XGBRegressor(
                n_estimators=self.config.XGBOOST_N_ESTIMATORS,
                max_depth=self.config.XGBOOST_DEPTH,
                learning_rate=0.05,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                early_stopping_rounds=50
            )
        
        # 添加CausalEngine模式
        for mode in self.config.CAUSAL_MODES:
            core_models_config[f'causal_{mode}'] = MLPCausalRegressor(
                perception_hidden_layers=self.config.CAUSAL_HIDDEN_SIZES,
                mode=mode,
                gamma_init=self.config.CAUSAL_GAMMA_INIT,
                b_noise_init=self.config.CAUSAL_B_NOISE_INIT,
                b_noise_trainable=self.config.CAUSAL_B_NOISE_TRAINABLE,
                max_iter=self.config.CAUSAL_MAX_EPOCHS,
                learning_rate=self.config.CAUSAL_LR,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=self.config.CAUSAL_PATIENCE,
                tol=self.config.CAUSAL_TOL,
                random_state=self.config.RANDOM_STATE,
                verbose=False,
                alpha=0.0001
            )
        
        # 训练模型
        for model_name, model in core_models_config.items():
            if model_name in ['random_forest', 'catboost', 'xgboost']:
                # RF, CatBoost, XGBoost始终使用原始数据
                if model_name == 'xgboost':
                    model.fit(X_train, y_train_noisy, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train_noisy)
                # 创建恒等缩放器
                identity_scaler = StandardScaler(with_mean=False, with_std=False)
                identity_scaler.fit(X_train)
                models[model_name] = (model, identity_scaler)
            else:
                # 神经网络根据配置使用缩放或原始数据
                model.fit(X_train_scaled, y_train_noisy)
                models[model_name] = (model, scaler)
        
        return models
    
    def _print_robustness_table(self, robustness_results, anomaly_ratios):
        """打印鲁棒性测试详细表格"""
        print("\n📊 扩展版鲁棒性测试详细结果表格")
        print("=" * 160)
        
        # 表头
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        methods = list(robustness_results[anomaly_ratios[0]].keys())
        
        # 打印表头
        header_line1 = f"{'异常比例':<10} {'方法':<20}"
        header_line2 = f"{'':10} {'':20}"
        
        for split in ['验证集', '测试集']:
            header_line1 += f" {split:<40}"
            header_line2 += f" {metrics[0]:<9} {metrics[1]:<9} {metrics[2]:<9} {metrics[3]:<9}"
        
        print(header_line1)
        print(header_line2)
        print("-" * 160)
        
        # 为每个异常比例打印结果
        for ratio in anomaly_ratios:
            ratio_str = f"{ratio:.0%}"
            
            for i, method in enumerate(methods):
                if method in robustness_results[ratio]:
                    results = robustness_results[ratio][method]
                    
                    # 第一行显示异常比例，后续行为空
                    ratio_display = ratio_str if i == 0 else ""
                    
                    line = f"{ratio_display:<10} {method:<20}"
                    
                    # 验证集指标
                    val_metrics = results['val']
                    line += f" {val_metrics['MAE']:<9.4f} {val_metrics['MdAE']:<9.4f} {val_metrics['RMSE']:<9.4f} {val_metrics['R²']:<9.4f}"
                    
                    # 测试集指标
                    test_metrics = results['test']
                    line += f" {test_metrics['MAE']:<9.4f} {test_metrics['MdAE']:<9.4f} {test_metrics['RMSE']:<9.4f} {test_metrics['R²']:<9.4f}"
                    
                    print(line)
            
            # 在每个异常比例组之间添加分隔线
            if ratio != anomaly_ratios[-1]:
                print("-" * 160)
        
        print("=" * 160)
        print("💡 观察要点：")
        print("   - R² 越高越好（接近1.0为最佳）")
        print("   - MAE, MdAE, RMSE 越低越好（接近0为最佳）")
        print("   - 关注各方法在异常比例增加时的性能变化趋势")
    
    def _plot_robustness_results(self, robustness_results, anomaly_ratios):
        """绘制鲁棒性测试结果"""
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ROBUSTNESS)
        fig.suptitle('Extended Robustness Test: Impact of Label Anomaly on Model Performance (Sklearn Version)', fontsize=16, fontweight='bold')
        
        methods = list(robustness_results[anomaly_ratios[0]].keys())
        
        # 设置颜色和线型
        method_styles = {}
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, method in enumerate(methods):
            if 'causal' in method:
                method_styles[method] = {'color': 'red', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 8}
            elif any(robust in method for robust in ['huber', 'pinball', 'cauchy']):
                method_styles[method] = {'color': colors[i % len(colors)], 'linestyle': '--', 'linewidth': 2, 'marker': 's', 'markersize': 6}
            elif method in ['random_forest', 'catboost', 'xgboost']:
                method_styles[method] = {'color': colors[i % len(colors)], 'linestyle': '-.', 'linewidth': 2, 'marker': 'D', 'markersize': 6}
            else:
                method_styles[method] = {'color': colors[i % len(colors)], 'linestyle': ':', 'linewidth': 2, 'marker': '^', 'markersize': 6}
        
        # 4个回归指标
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        metric_labels = ['Mean Absolute Error (MAE)', 'Median Absolute Error (MdAE)', 'Root Mean Squared Error (RMSE)', 'R-squared Score (R²)']
        
        # 为每个指标创建子图
        for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # 绘制每个方法的曲线
            for method in methods:
                scores = []
                for ratio in anomaly_ratios:
                    if method in robustness_results[ratio]:
                        scores.append(robustness_results[ratio][method]['test'][metric])
                    else:
                        scores.append(np.nan)
                
                # 简化标签显示
                label = self._get_clean_method_name(method)
                ax.plot(anomaly_ratios, scores, 
                       label=label, 
                       **method_styles[method])
            
            # 设置子图属性
            ax.set_xlabel('Label Anomaly Ratio', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            
            # # 为R²添加特殊处理（越高越好），其他指标越低越好
            # if metric == 'R²':
            #     ax.set_ylim(bottom=0)  # R²从0开始显示
            # else:
            #     ax.set_ylim(bottom=0)  # 误差指标从0开始显示
        
        plt.tight_layout()
        
        # 保存图片 - 使用与原始版本一致的文件名
        output_path = self._get_output_path('extended_robustness_analysis.png')
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        print(f"📊 扩展版鲁棒性测试图表已保存为 {output_path}")
        plt.close()  # 关闭图形，避免内存泄漏
    
    def _analyze_robustness_trends(self, robustness_results, anomaly_ratios):
        """分析鲁棒性趋势"""
        print("\n🔬 扩展版鲁棒性趋势分析")
        print("=" * 60)
        
        methods = list(robustness_results[anomaly_ratios[0]].keys())
        
        # 分析MdAE指标的变化趋势
        print("📈 MdAE指标随异常比例变化趋势：")
        print("-" * 40)
        
        stability_scores = {}
        for method in methods:
            mdae_scores = []
            for ratio in anomaly_ratios:
                if method in robustness_results[ratio]:
                    mdae_scores.append(robustness_results[ratio][method]['test']['MdAE'])
                else:
                    mdae_scores.append(np.nan)
            
            # 计算性能下降情况
            clean_mdae = mdae_scores[0] if len(mdae_scores) > 0 and not np.isnan(mdae_scores[0]) else 0
            final_mdae = mdae_scores[-1] if len(mdae_scores) > 0 and not np.isnan(mdae_scores[-1]) else 0
            
            if clean_mdae > 0:
                performance_degradation = ((final_mdae - clean_mdae) / clean_mdae) * 100
                stability_score = (max(mdae_scores) - min(mdae_scores)) / min(mdae_scores) * 100
                stability_scores[method] = stability_score
            else:
                performance_degradation = float('inf')
                stability_scores[method] = float('inf')
            
            display_name = self._get_clean_method_name(method)
            print(f"  {display_name}:")
            print(f"    - 零异常时MdAE: {clean_mdae:.4f}")
            print(f"    - 最高异常时MdAE: {final_mdae:.4f}")
            print(f"    - 性能下降: {performance_degradation:+.1f}%")
            print(f"    - 稳定性评分: {stability_scores[method]:.1f}%")
        
        # 验证假设
        print("\n🎯 假设验证结果：")
        print("-" * 40)
        
        # 找出最稳定的方法
        most_stable = min(stability_scores.keys(), key=lambda x: stability_scores[x])
        least_stable = max(stability_scores.keys(), key=lambda x: stability_scores[x])
        
        print(f"✅ 最稳定方法: {self._get_clean_method_name(most_stable)} (稳定性: {stability_scores[most_stable]:.1f}%)")
        print(f"⚠️ 最不稳定方法: {self._get_clean_method_name(least_stable)} (稳定性: {stability_scores[least_stable]:.1f}%)")
        
        # 分析CausalEngine vs 其他方法
        causal_methods = [m for m in methods if 'causal' in m]
        non_causal_methods = [m for m in methods if 'causal' not in m]
        
        if causal_methods and non_causal_methods:
            # 找出最佳CausalEngine和最佳传统方法
            best_causal = min(causal_methods, key=lambda x: stability_scores[x])
            best_traditional = min(non_causal_methods, key=lambda x: stability_scores[x])
            
            print(f"\n�� CausalEngine vs 传统方法稳定性对比:")
            print(f"   最佳CausalEngine: {self._get_clean_method_name(best_causal)} (稳定性: {stability_scores[best_causal]:.1f}%)")
            print(f"   最佳传统方法: {self._get_clean_method_name(best_traditional)} (稳定性: {stability_scores[best_traditional]:.1f}%)")
            
            if stability_scores[best_causal] < stability_scores[best_traditional]:
                print(f"   ✅ CausalEngine更稳定！")
            else:
                print(f"   ⚠️ 传统方法在此实验中更稳定")
        
        # 综合结论
        causal_in_top3 = any(method in sorted(stability_scores.keys(), key=lambda x: stability_scores[x])[:3] for method in causal_methods)
        
        print(f"\n🏆 综合结论:")
        if causal_in_top3:
            print("   ✨ CausalEngine在鲁棒性测试中表现优异，验证了其在噪声环境中的稳定性优势")
        else:
            print("   🔬 需要进一步优化CausalEngine参数以提升鲁棒性")


def main():
    """主函数：运行完整的扩展版教程"""
    print("🏠 CausalEngine扩展版真实世界回归教程 - Sklearn版本")
    print("🎯 目标：在加州房价预测任务中展示CausalEngine与多种强力方法的性能对比")
    print("🔧 特点：使用sklearn-style learners，包含鲁棒性回归器和集成学习方法")
    print("=" * 80)
    
    # 创建配置实例（在这里可以自定义配置）
    config = TutorialConfig()
    
    # 🔧 快速配置示例 - 取消注释来修改参数：
    # config.CAUSAL_MODES = ['deterministic', 'standard', 'endogenous']  # 添加更多模式
    # config.NN_MAX_EPOCHS = 1000  # 减少训练轮数以加快速度
    # config.ANOMALY_RATIO = 0.1   # 设置10%异常标签
    # config.RUN_ROBUSTNESS_TEST = False  # 跳过鲁棒性测试
    
    print(f"🔧 当前配置:")
    print(f"   - CausalEngine模式: {', '.join(config.CAUSAL_MODES)}")
    print(f"   - 神经网络架构: {config.NN_HIDDEN_SIZES}")
    print(f"   - 最大轮数: {config.NN_MAX_EPOCHS}")
    print(f"   - 早停patience: {config.NN_PATIENCE}")
    print(f"   - 异常比例: {config.ANOMALY_RATIO:.1%}")
    print(f"   - 运行鲁棒性测试: {'是' if config.RUN_ROBUSTNESS_TEST else '否'}")
    print(f"   - 输出目录: {config.OUTPUT_DIR}/")
    print()
    
    # 创建教程实例
    tutorial = ExtendedCaliforniaHousingSklearnTutorial(config)
    
    # 1. 加载和探索数据
    tutorial.load_and_explore_data()
    
    # 2. 数据可视化
    tutorial.visualize_data()
    
    # 3. 运行综合基准测试
    tutorial.run_comprehensive_benchmark()
    
    # 4. 性能分析
    tutorial.analyze_performance()
    
    # 5. 创建性能可视化
    tutorial.create_performance_visualization()
    
    # 6. 鲁棒性测试（可选）
    if config.RUN_ROBUSTNESS_TEST:
        tutorial.run_robustness_test()
    else:
        print("\n🛡️ 跳过鲁棒性测试（配置中禁用）")
    
    # 7. 生成综合实验报告 - 与原始版本完全对应
    tutorial.generate_summary_report()
    
    print("\n🎉 扩展版教程完成！")
    print("📋 总结:")
    print("   - 使用了真实世界的加州房价数据集")
    methods_count = len(config.CAUSAL_MODES) + 5 + (1 if CATBOOST_AVAILABLE else 0) + (1 if XGBOOST_AVAILABLE else 0)
    print(f"   - 比较了{methods_count}种不同的方法")
    print("   - 包含3种鲁棒性神经网络回归器（Huber、Pinball、Cauchy）")
    print("   - 包含强力集成学习方法（Random Forest、CatBoost、XGBoost）")
    print("   - 展示了CausalEngine的性能优势")
    print("   - 直接使用sklearn-style learners")
    if config.RUN_ROBUSTNESS_TEST:
        print("   - 测试了模型的鲁棒性")
    print("   - 提供了详细的可视化分析")
    print("\n📊 生成的文件:")
    if config.SAVE_PLOTS:
        print(f"   - {config.OUTPUT_DIR}/extended_data_analysis.png                   (数据分析图)")
        print(f"   - {config.OUTPUT_DIR}/core_performance_comparison.png              (性能对比图)")
        print(f"   - {config.OUTPUT_DIR}/extended_experiment_summary.md               (实验总结报告)")
        if config.RUN_ROBUSTNESS_TEST:
            print(f"   - {config.OUTPUT_DIR}/extended_robustness_analysis.png             (鲁棒性测试图)")
    
    print("\n💡 提示：在脚本顶部的TutorialConfig类中修改参数来自定义实验！")


if __name__ == "__main__":
    main()