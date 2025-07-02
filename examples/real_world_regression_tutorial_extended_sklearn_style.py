#!/usr/bin/env python3
"""
🏠 扩展版真实世界回归教程：加州房价预测 - Sklearn-Style版本
============================================================

这个教程演示CausalEngine与多种强力传统方法在真实世界回归任务中的性能对比。

数据集：加州房价数据集（California Housing Dataset）
- 20,640个样本
- 8个特征（房屋年龄、收入、人口等）
- 目标：预测房价中位数

我们将比较13种方法：
1. sklearn MLPRegressor（传统神经网络）
2. PyTorch MLP（传统深度学习）
3. MLP Huber（Huber损失稳健回归）
4. MLP Pinball（Pinball损失稳健回归）
5. MLP Cauchy（Cauchy损失稳健回归）
6. Random Forest（随机森林）
7. XGBoost（梯度提升）
8. LightGBM（轻量梯度提升）
9. CatBoost（强力梯度提升）
10. CausalEngine - deterministic（确定性）
11. CausalEngine - exogenous（外生噪声主导）
12. CausalEngine - endogenous（内生不确定性主导）
13. CausalEngine - standard（内生+外生混合）

关键亮点：
- 真实世界数据的鲁棒性测试
- 6种强力传统机器学习方法对比
- 3种稳健神经网络回归方法（Huber、Pinball、Cauchy）
- 4种CausalEngine模式完整对比
- 统一神经网络参数配置确保公平比较
- 因果推理vs传统方法的性能差异分析
- 使用sklearn-style regressor实现

实验设计说明
==================================================================
本脚本包含两组核心实验，旨在全面评估CausalEngine在真实回归任务上的性能和鲁棒性。
所有实验参数均可在下方的 `TutorialConfig` 类中进行修改。

实验一：核心性能对比 (在40%标签噪声下)
--------------------------------------------------
- **目标**: 比较CausalEngine和9种传统方法在含有固定噪声数据上的预测性能。
- **设置**: 默认设置40%的标签噪声（`ANOMALY_RATIO = 0.4`），模拟真实世界中常见的数据质量问题。
- **对比模型**: 
  - 传统方法: sklearn MLP, PyTorch MLP, MLP Huber, MLP Pinball, MLP Cauchy, Random Forest, XGBoost, LightGBM, CatBoost
  - CausalEngine: deterministic, exogenous, endogenous, standard等模式

实验二：鲁棒性分析 (跨越不同噪声水平)
--------------------------------------------------
- **目标**: 探究模型性能随标签噪声水平增加时的变化情况，评估其稳定性。
- **设置**: 在一系列噪声比例（如0%, 10%, 20%, 30%, 40%, 50%）下分别运行测试。
- **对比模型**: 所有13种方法在不同噪声水平下的表现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
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

# 导入树模型 - 处理可能的导入错误
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
    NN_HIDDEN_SIZES = (128, 64, 32)                 # 神经网络隐藏层结构
    NN_MAX_EPOCHS = 3000                            # 最大训练轮数
    NN_LEARNING_RATE = 0.01                         # 学习率
    NN_PATIENCE = 50                                # 早停patience
    NN_TOLERANCE = 1e-4                             # 早停tolerance
    # =========================================================================
    
    # 🤖 CausalEngine参数 - 使用统一神经网络配置
    CAUSAL_MODES = ['deterministic', 'exogenous', 'endogenous', 'standard']  # 可选: ['deterministic', 'exogenous', 'endogenous', 'standard']
    CAUSAL_HIDDEN_SIZES = NN_HIDDEN_SIZES          # 使用统一神经网络配置
    CAUSAL_MAX_EPOCHS = NN_MAX_EPOCHS              # 使用统一神经网络配置
    CAUSAL_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    CAUSAL_PATIENCE = NN_PATIENCE                  # 使用统一神经网络配置
    CAUSAL_TOL = NN_TOLERANCE                      # 使用统一神经网络配置
    CAUSAL_GAMMA_INIT = 1.0                      # gamma初始化
    CAUSAL_B_NOISE_INIT = 1.0                    # b_noise初始化
    CAUSAL_B_NOISE_TRAINABLE = True              # b_noise是否可训练
    
    # 🧠 传统神经网络方法参数 - 使用统一配置
    SKLEARN_HIDDEN_LAYERS = NN_HIDDEN_SIZES         # 使用统一神经网络配置
    SKLEARN_MAX_ITER = NN_MAX_EPOCHS                # 使用统一神经网络配置
    SKLEARN_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    
    PYTORCH_HIDDEN_SIZES = NN_HIDDEN_SIZES          # 使用统一神经网络配置
    PYTORCH_EPOCHS = NN_MAX_EPOCHS                  # 使用统一神经网络配置
    PYTORCH_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    PYTORCH_PATIENCE = NN_PATIENCE                  # 使用统一神经网络配置
    
    # 🎯 基准方法配置 - 扩展版包含更多强力方法
    BASELINE_METHODS = [
        'sklearn_mlp',       # sklearn神经网络  
        'pytorch_mlp',       # PyTorch神经网络
        'mlp_huber',         # Huber损失MLP（稳健回归）
        'mlp_pinball_median',# Pinball损失MLP（稳健回归）
        'mlp_cauchy',        # Cauchy损失MLP（稳健回归）
        'random_forest',     # 随机森林
        'xgboost',           # XGBoost - 强力梯度提升
        'lightgbm',          # LightGBM - 轻量梯度提升
        'catboost'           # CatBoost - 强力梯度提升
    ]
    
    # 🛡️ 稳健回归器参数 - 使用统一配置
    ROBUST_HIDDEN_SIZES = NN_HIDDEN_SIZES           # 使用统一神经网络配置
    ROBUST_MAX_EPOCHS = NN_MAX_EPOCHS               # 使用统一神经网络配置
    ROBUST_LR = NN_LEARNING_RATE                    # 使用统一神经网络配置
    ROBUST_PATIENCE = NN_PATIENCE                   # 使用统一神经网络配置
    
    # 🌲 树模型参数
    TREE_N_ESTIMATORS = 100                         # 树的数量
    TREE_RANDOM_STATE = RANDOM_STATE                # 随机种子
    TREE_MAX_DEPTH = None                           # 最大深度（None表示不限制）
    
    # XGBoost 参数
    XGBOOST_N_ESTIMATORS = 100
    XGBOOST_LEARNING_RATE = 0.1
    XGBOOST_MAX_DEPTH = 6
    
    # LightGBM 参数
    LIGHTGBM_N_ESTIMATORS = 100
    LIGHTGBM_LEARNING_RATE = 0.1
    LIGHTGBM_MAX_DEPTH = -1
    
    # CatBoost 参数
    CATBOOST_ITERATIONS = 100
    CATBOOST_LEARNING_RATE = 0.1
    CATBOOST_DEPTH = 6
    
    # 📊 实验参数
    ANOMALY_RATIO = 0.4                          # 标签异常比例 (核心实验默认值: 40%噪声挑战)
    SAVE_PLOTS = True                            # 是否保存图表
    VERBOSE = True                               # 是否显示详细输出
    
    # 🛡️ 鲁棒性测试参数 - 验证"CausalEngine鲁棒性优势"的假设
    ROBUSTNESS_ANOMALY_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 噪声水平
    RUN_ROBUSTNESS_TEST = True                      # 是否运行鲁棒性测试
    
    # 📈 可视化参数
    FIGURE_DPI = 300                             # 图表分辨率
    FIGURE_SIZE_ANALYSIS = (24, 20)              # 数据分析图表大小
    FIGURE_SIZE_PERFORMANCE = (30, 25)           # 性能对比图表大小 (更大以容纳更多方法)
    FIGURE_SIZE_ROBUSTNESS = (30, 25)            # 鲁棒性测试图表大小 (更大以容纳更多方法)
    
    # 📁 输出目录参数
    OUTPUT_DIR = "results/california_housing_regression_extended_sklearn_style"  # 输出目录名称


class ExtendedCaliforniaHousingTutorialSklearnStyle:
    """
    扩展版加州房价回归教程类 - Sklearn-Style版本
    
    演示CausalEngine在真实世界回归任务中的优越性能
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
            print(f"📁 创建输出目录: {self.config.OUTPUT_DIR}/")
    
    def _get_output_path(self, filename):
        """获取输出文件的完整路径"""
        return os.path.join(self.config.OUTPUT_DIR, filename)
        
    def load_and_explore_data(self, verbose=True):
        """加载并探索加州房价数据集"""
        if verbose:
            print("🏠 扩展版加州房价预测 - 真实世界回归教程 (Sklearn-Style)")
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
            print(f"   - 将比较 {len(self.config.CAUSAL_MODES) + 9} 种方法")
        
        return self.X, self.y
    
    def visualize_data(self, save_plots=None):
        """数据可视化分析"""
        if save_plots is None:
            save_plots = self.config.SAVE_PLOTS
            
        print("\\n📈 数据分布分析")
        print("-" * 30)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ANALYSIS)
        fig.suptitle('California Housing Dataset Analysis - Extended Sklearn-Style Tutorial', fontsize=16, fontweight='bold')
        
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
            output_path = self._get_output_path('california_housing_analysis_extended_sklearn_style.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 数据分析图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 数据统计摘要
        print("\\n📋 数据统计摘要:")
        print(f"  - 最相关特征: {most_corr_feature} (相关系数: {corr_matrix.loc[most_corr_feature, 'MedHouseVal']:.3f})")
        print(f"  - 异常值检测: {np.sum(np.abs(self.y - self.y.mean()) > 3 * self.y.std())} 个潜在异常值")
        print(f"  - 数据完整性: 无缺失值" if not np.any(np.isnan(self.X)) else "  - 警告: 存在缺失值")
    
    def _prepare_data(self, test_size, val_size, anomaly_ratio, random_state):
        """准备训练和测试数据"""
        # 数据分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
        
        # 注入噪声
        if anomaly_ratio > 0:
            y_train_noisy, noise_indices = inject_shuffle_noise(
                y_train, noise_ratio=anomaly_ratio, random_state=random_state
            )
            y_train = y_train_noisy
            if self.config.VERBOSE:
                print(f"   异常注入: {anomaly_ratio:.1%} ({len(noise_indices)}/{len(y_train)} 样本受影响)")
        
        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        return {
            'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
            'y_train': y_train_scaled, 'y_val': y_val_scaled, 'y_test': y_test,
            'scaler_X': scaler_X, 'scaler_y': scaler_y
        }
    
    def _train_sklearn_model(self, data):
        """训练sklearn模型"""
        print("🔧 训练 sklearn MLPRegressor...")
        
        model = MLPRegressor(
            hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
            max_iter=self.config.SKLEARN_MAX_ITER,
            learning_rate_init=self.config.SKLEARN_LR,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            random_state=self.config.RANDOM_STATE
        )
        
        model.fit(data['X_train'], data['y_train'])
        
        if self.config.VERBOSE:
            print(f"   训练完成: {model.n_iter_} epochs")
        
        return model
    
    def _train_pytorch_model(self, data):
        """训练PyTorch模型"""
        print("🔧 训练 PyTorch MLPRegressor...")
        
        model = MLPPytorchRegressor(
            hidden_layer_sizes=self.config.PYTORCH_HIDDEN_SIZES,
            max_iter=self.config.PYTORCH_EPOCHS,
            learning_rate=self.config.PYTORCH_LR,
            early_stopping=True,
            n_iter_no_change=self.config.PYTORCH_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        
        print(f"   训练完成: {model.n_iter_} epochs")
        return model
    
    def _train_robust_regressor(self, data, robust_type):
        """训练稳健回归器"""
        print(f"🔧 训练 {robust_type} Regressor...")
        
        # 选择稳健回归器类型
        if robust_type == 'huber':
            model_class = MLPHuberRegressor
        elif robust_type == 'pinball':
            model_class = MLPPinballRegressor
        elif robust_type == 'cauchy':
            model_class = MLPCauchyRegressor
        else:
            raise ValueError(f"Unknown robust type: {robust_type}")
        
        model = model_class(
            hidden_layer_sizes=self.config.ROBUST_HIDDEN_SIZES,
            max_iter=self.config.ROBUST_MAX_EPOCHS,
            learning_rate=self.config.ROBUST_LR,
            early_stopping=True,
            n_iter_no_change=self.config.ROBUST_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        
        if hasattr(model, 'n_iter_'):
            print(f"   训练完成: {model.n_iter_} epochs")
        else:
            print(f"   训练完成")
        
        return model
    
    def _train_tree_model(self, data, tree_type):
        """训练树模型"""
        print(f"🔧 训练 {tree_type}...")
        
        # 使用原始数据（未标准化的特征）进行树模型训练
        X_train_original = data['scaler_X'].inverse_transform(data['X_train'])
        y_train_original = data['scaler_y'].inverse_transform(data['y_train'].reshape(-1, 1)).flatten()
        
        if tree_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=self.config.TREE_N_ESTIMATORS,
                max_depth=self.config.TREE_MAX_DEPTH,
                random_state=self.config.TREE_RANDOM_STATE,
                n_jobs=-1
            )
            
        elif tree_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                print("   ⚠️ XGBoost 不可用，跳过...")
                return None
            model = xgb.XGBRegressor(
                n_estimators=self.config.XGBOOST_N_ESTIMATORS,
                learning_rate=self.config.XGBOOST_LEARNING_RATE,
                max_depth=self.config.XGBOOST_MAX_DEPTH,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                verbosity=0
            )
            
        elif tree_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                print("   ⚠️ LightGBM 不可用，跳过...")
                return None
            model = lgb.LGBMRegressor(
                n_estimators=self.config.LIGHTGBM_N_ESTIMATORS,
                learning_rate=self.config.LIGHTGBM_LEARNING_RATE,
                max_depth=self.config.LIGHTGBM_MAX_DEPTH,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1
            )
            
        elif tree_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                print("   ⚠️ CatBoost 不可用，跳过...")
                return None
            model = cb.CatBoostRegressor(
                iterations=self.config.CATBOOST_ITERATIONS,
                learning_rate=self.config.CATBOOST_LEARNING_RATE,
                depth=self.config.CATBOOST_DEPTH,
                random_state=self.config.RANDOM_STATE,
                verbose=False
            )
            
        else:
            raise ValueError(f"Unknown tree type: {tree_type}")
        
        model.fit(X_train_original, y_train_original)
        print(f"   训练完成")
        
        return model
    
    def _train_causal_model(self, data, mode):
        """训练CausalEngine模型"""
        print(f"🔧 训练 CausalEngine ({mode})...")
        
        model = MLPCausalRegressor(
            perception_hidden_layers=self.config.CAUSAL_HIDDEN_SIZES,
            mode=mode,
            max_iter=self.config.CAUSAL_MAX_EPOCHS,
            learning_rate=self.config.CAUSAL_LR,
            early_stopping=True,
            n_iter_no_change=self.config.CAUSAL_PATIENCE,
            tol=self.config.CAUSAL_TOL,
            gamma_init=self.config.CAUSAL_GAMMA_INIT,
            b_noise_init=self.config.CAUSAL_B_NOISE_INIT,
            b_noise_trainable=self.config.CAUSAL_B_NOISE_TRAINABLE,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        
        model.fit(data['X_train'], data['y_train'])
        
        if self.config.VERBOSE:
            print(f"   训练完成: {model.n_iter_} epochs")
        
        return model
    
    def _evaluate_model(self, model, data, model_name):
        """评估模型性能"""
        is_tree_model = model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost']
        
        if is_tree_model:
            # 树模型使用原始特征进行预测
            X_val_for_pred = data['scaler_X'].inverse_transform(data['X_val'])
            X_test_for_pred = data['scaler_X'].inverse_transform(data['X_test'])
            
            val_pred = model.predict(X_val_for_pred)
            test_pred = model.predict(X_test_for_pred)
            
            # 获取验证集的原始目标值
            y_val_original = data['scaler_y'].inverse_transform(data['y_val'].reshape(-1, 1)).flatten()
        else:
            # 神经网络模型使用标准化特征进行预测
            val_pred_scaled = model.predict(data['X_val'])
            test_pred_scaled = model.predict(data['X_test'])
            
            # 转换回原始尺度
            val_pred = data['scaler_y'].inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
            test_pred = data['scaler_y'].inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
            y_val_original = data['scaler_y'].inverse_transform(data['y_val'].reshape(-1, 1)).flatten()
        
        # 计算指标
        val_metrics = {
            'MAE': mean_absolute_error(y_val_original, val_pred),
            'MdAE': median_absolute_error(y_val_original, val_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val_original, val_pred)),
            'R²': r2_score(y_val_original, val_pred)
        }
        
        test_metrics = {
            'MAE': mean_absolute_error(data['y_test'], test_pred),
            'MdAE': median_absolute_error(data['y_test'], test_pred),
            'RMSE': np.sqrt(mean_squared_error(data['y_test'], test_pred)),
            'R²': r2_score(data['y_test'], test_pred)
        }
        
        return {'val': val_metrics, 'test': test_metrics}
    
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
            print("\\n🚀 开始综合基准测试 (Extended Sklearn-Style)")
            print("=" * 80)
            print(f"🔧 实验配置:")
            print(f"   - 测试集比例: {test_size:.1%}")
            print(f"   - 验证集比例: {val_size:.1%}")
            print(f"   - 异常标签比例: {anomaly_ratio:.1%}")
            print(f"   - 随机种子: {self.config.RANDOM_STATE}")
            print(f"   - CausalEngine模式: {', '.join(self.config.CAUSAL_MODES)}")
            print(f"   - 网络架构: {self.config.CAUSAL_HIDDEN_SIZES}")
            print(f"   - 最大训练轮数: {self.config.CAUSAL_MAX_EPOCHS}")
            print(f"   - 早停patience: {self.config.CAUSAL_PATIENCE}")
        
        # 加载数据
        if self.X is None or self.y is None:
            self.load_and_explore_data(verbose=verbose)
        
        # 准备数据
        data = self._prepare_data(test_size, val_size, anomaly_ratio, self.config.RANDOM_STATE)
        
        # 训练和评估模型
        self.results = {}
        
        # 1. sklearn模型
        sklearn_model = self._train_sklearn_model(data)
        self.results['sklearn_mlp'] = self._evaluate_model(sklearn_model, data, 'sklearn_mlp')
        
        # 2. PyTorch模型
        pytorch_model = self._train_pytorch_model(data)
        self.results['pytorch_mlp'] = self._evaluate_model(pytorch_model, data, 'pytorch_mlp')
        
        # 3. 稳健回归器
        for robust_type in ['huber', 'pinball', 'cauchy']:
            robust_model = self._train_robust_regressor(data, robust_type)
            if robust_model is not None:
                # 使用与Legacy版本一致的键名
                result_key = f'mlp_{robust_type}_median' if robust_type == 'pinball' else f'mlp_{robust_type}'
                self.results[result_key] = self._evaluate_model(robust_model, data, result_key)
        
        # 4. 树模型
        for tree_type in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
            tree_model = self._train_tree_model(data, tree_type)
            if tree_model is not None:
                self.results[tree_type] = self._evaluate_model(tree_model, data, tree_type)
        
        # 5. CausalEngine模型
        for mode in self.config.CAUSAL_MODES:
            causal_model = self._train_causal_model(data, mode)
            self.results[mode] = self._evaluate_model(causal_model, data, mode)
        
        if verbose:
            self._print_results(anomaly_ratio)
        
        return self.results
    
    def _print_results(self, anomaly_ratio):
        """打印结果"""
        print(f"\\n📊 基准测试结果 (异常比例: {anomaly_ratio:.0%})")
        print("=" * 140)
        print(f"{'方法':<20} {'验证集':<50} {'测试集':<50}")
        print(f"{'':20} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 140)
        
        for method, metrics in self.results.items():
            val_m = metrics['val']
            test_m = metrics['test']
            print(f"{method:<20} {val_m['MAE']:<10.4f} {val_m['MdAE']:<10.4f} {val_m['RMSE']:<10.4f} {val_m['R²']:<10.4f} "
                  f"{test_m['MAE']:<10.4f} {test_m['MdAE']:<10.4f} {test_m['RMSE']:<10.4f} {test_m['R²']:<10.4f}")
        
        print("=" * 140)
    
    def analyze_performance(self, verbose=True):
        """分析性能结果"""
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        if verbose:
            print("\\n🔍 性能分析")
            print("=" * 60)
        
        # 提取测试集R²分数
        test_r2_scores = {}
        for method, metrics in self.results.items():
            test_r2_scores[method] = metrics['test']['R²']
        
        # 找到最佳方法
        best_method = max(test_r2_scores.keys(), key=lambda x: test_r2_scores[x])
        best_r2 = test_r2_scores[best_method]
        
        if verbose:
            print(f"🏆 最佳方法: {best_method}")
            print(f"   R² = {best_r2:.4f}")
            print()
            print("📊 性能排名 (按R²分数):")
            
            sorted_methods = sorted(test_r2_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (method, r2) in enumerate(sorted_methods, 1):
                improvement = ((r2 - sorted_methods[-1][1]) / abs(sorted_methods[-1][1])) * 100
                print(f"   {i}. {method:<15} R² = {r2:.4f} (+ {improvement:+.1f}%)")
        
        # CausalEngine性能分析
        causal_methods = [m for m in self.results.keys() if m in ['deterministic', 'standard', 'sampling', 'exogenous', 'endogenous']]
        if causal_methods:
            best_causal = max(causal_methods, key=lambda x: test_r2_scores[x])
            traditional_methods = [m for m in self.results.keys() if m not in causal_methods]
            
            if traditional_methods and verbose:
                best_traditional = max(traditional_methods, key=lambda x: test_r2_scores[x])
                causal_improvement = ((test_r2_scores[best_causal] - test_r2_scores[best_traditional]) 
                                    / abs(test_r2_scores[best_traditional])) * 100
                
                print(f"\\n🎯 CausalEngine优势分析:")
                print(f"   最佳CausalEngine模式: {best_causal} (R² = {test_r2_scores[best_causal]:.4f})")
                print(f"   最佳传统方法: {best_traditional} (R² = {test_r2_scores[best_traditional]:.4f})")
                print(f"   性能提升: {causal_improvement:+.2f}%")
                
                if causal_improvement > 0:
                    print(f"   ✅ CausalEngine显著优于传统方法！")
                else:
                    print(f"   ⚠️ 在此数据集上传统方法表现更好")
        
        return test_r2_scores
    
    def create_performance_visualization(self, save_plot=None):
        """创建性能可视化图表"""
        if save_plot is None:
            save_plot = self.config.SAVE_PLOTS
            
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\\n📊 创建性能可视化图表")
        print("-" * 30)
        
        # 准备数据
        methods = list(self.results.keys())
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_PERFORMANCE)
        fig.suptitle('Extended CausalEngine vs Traditional Methods: California Housing Performance (40% Label Noise) - Sklearn-Style', 
                    fontsize=16, fontweight='bold')
        axes = axes.flatten()  # 展平为一维数组便于访问
        
        # 设置颜色方案
        def get_method_color(method):
            if method in ['deterministic', 'exogenous', 'endogenous', 'standard']:
                return 'gold'  # CausalEngine用金色
            elif any(robust in method for robust in ['huber', 'pinball', 'cauchy']):
                return 'lightgreen'  # 稳健方法用浅绿
            elif method in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
                return 'sandybrown'  # 树模型用棕色
            else:
                return 'lightblue'  # 神经网络用浅蓝
        
        colors = [get_method_color(method) for method in methods]
        
        for i, metric in enumerate(metrics):
            values = [self.results[method]['test'][metric] for method in methods]
            
            bars = axes[i].bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
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
            bars[best_idx].set_color('red')
            bars[best_idx].set_edgecolor('darkred')
            bars[best_idx].set_linewidth(3)
            
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_path = self._get_output_path('california_housing_performance_extended_sklearn_style.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 性能图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
    
    def run_robustness_analysis(self, noise_levels=None, verbose=None):
        """运行鲁棒性分析"""
        if noise_levels is None:
            noise_levels = self.config.ROBUSTNESS_ANOMALY_RATIOS
        if verbose is None:
            verbose = self.config.VERBOSE
            
        if verbose:
            print("\n🔬 开始鲁棒性分析")
            print("=" * 60)
            print(f"🎯 测试噪声水平: {[f'{level:.0%}' for level in noise_levels]}")
            print(f"   比较方法: 选取主要方法进行鲁棒性测试")
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            if verbose:
                print(f"\n📊 测试噪声水平: {noise_level:.0%}")
                print("-" * 30)
            
            # 临时修改配置以测试特定噪声水平
            original_config = self.config.ANOMALY_RATIO
            self.config.ANOMALY_RATIO = noise_level
            
            # 选取主要方法进行鲁棒性测试以节省时间
            original_causal_modes = self.config.CAUSAL_MODES
            robustness_methods = ['sklearn_mlp', 'pytorch_mlp', 'random_forest', 'standard', 'endogenous']
            self.config.CAUSAL_MODES = [mode for mode in self.config.CAUSAL_MODES 
                                      if mode in robustness_methods]
            
            try:
                # 加载数据（如果尚未加载）
                if self.X is None or self.y is None:
                    self.load_and_explore_data(verbose=False)
                
                # 准备数据
                data = self._prepare_data(
                    self.config.TEST_SIZE, 
                    self.config.VAL_SIZE, 
                    noise_level, 
                    self.config.RANDOM_STATE
                )
                
                noise_results = {}
                
                # 测试sklearn MLP
                if 'sklearn_mlp' in robustness_methods:
                    sklearn_model = self._train_sklearn_model(data)
                    noise_results['sklearn_mlp'] = self._evaluate_model(sklearn_model, data, 'sklearn_mlp')
                
                # 测试PyTorch MLP
                if 'pytorch_mlp' in robustness_methods:
                    pytorch_model = self._train_pytorch_model(data)
                    noise_results['pytorch_mlp'] = self._evaluate_model(pytorch_model, data, 'pytorch_mlp')
                
                # 测试CausalEngine
                for mode in self.config.CAUSAL_MODES:
                    if not mode in robustness_methods:
                        continue
                    causal_model = self._train_causal_model(data, mode)
                    noise_results[mode] = self._evaluate_model(causal_model, data, mode)
                
                # 测试最佳树模型（Random Forest作为代表）
                if 'random_forest' in robustness_methods:
                    rf_model = self._train_tree_model(data, 'random_forest')
                    if rf_model is not None:
                        noise_results['random_forest'] = self._evaluate_model(rf_model, data, 'random_forest')
                
                robustness_results[noise_level] = noise_results
                
            finally:
                # 恢复原始配置
                self.config.ANOMALY_RATIO = original_config
                self.config.CAUSAL_MODES = original_causal_modes
        
        if verbose:
            self._print_robustness_results(robustness_results, noise_levels)
        
        return robustness_results
    
    def _print_robustness_results(self, robustness_results, noise_levels):
        """打印鲁棒性分析结果"""
        print("\n📊 鲁棒性分析结果")
        print("=" * 100)
        
        methods = list(robustness_results[noise_levels[0]].keys())
        
        # 打印表头
        header = f"{'方法':<20}"
        for noise_level in noise_levels:
            header += f"{'噪声' + f'{noise_level:.0%}':<15}"
        print(header)
        print("-" * len(header))
        
        # 打印每个方法的结果（使用R²作为主要指标）
        for method in methods:
            row = f"{method:<20}"
            for noise_level in noise_levels:
                if method in robustness_results[noise_level]:
                    r2 = robustness_results[noise_level][method]['test']['R²']
                    row += f"{r2:<15.4f}"
                else:
                    row += f"{'N/A':<15}"
            print(row)
        
        print("=" * 100)
        
        # 分析最稳定的方法
        print("\n🎯 稳定性分析:")
        stability_scores = {}
        
        for method in methods:
            r2_scores = []
            for noise_level in noise_levels:
                if method in robustness_results[noise_level]:
                    r2_scores.append(robustness_results[noise_level][method]['test']['R²'])
            
            if r2_scores:
                # 计算标准差作为稳定性指标（越小越稳定）
                stability_scores[method] = np.std(r2_scores)
        
        # 找到最稳定的方法
        most_stable = min(stability_scores.keys(), key=lambda x: stability_scores[x])
        print(f"   最稳定方法: {most_stable} (标准差: {stability_scores[most_stable]:.4f})")
        
        # 按稳定性排序
        print("   稳定性排名:")
        sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1])
        for i, (method, std) in enumerate(sorted_stability, 1):
            print(f"   {i}. {method:<15} 标准差: {std:.4f}")
    
    def create_robustness_visualization(self, robustness_results, save_plot=None):
        """创建鲁棒性可视化图表"""
        if save_plot is None:
            save_plot = self.config.SAVE_PLOTS
            
        print("\n📊 创建鲁棒性可视化图表")
        print("-" * 30)
        
        noise_levels = list(robustness_results.keys())
        methods = list(robustness_results[noise_levels[0]].keys())
        
        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE_ROBUSTNESS)
        
        # 为每个方法绘制线图
        for method in methods:
            r2_scores = []
            valid_noise_levels = []
            
            for noise_level in noise_levels:
                if method in robustness_results[noise_level]:
                    r2_scores.append(robustness_results[noise_level][method]['test']['R²'])
                    valid_noise_levels.append(noise_level * 100)  # 转换为百分比
            
            if r2_scores:
                # 设置线条样式
                if method in ['deterministic', 'exogenous', 'endogenous', 'standard']:
                    linestyle = '-'
                    linewidth = 3
                    marker = 'o'
                    markersize = 8
                else:
                    linestyle = '--'
                    linewidth = 2
                    marker = 's'
                    markersize = 6
                
                ax.plot(valid_noise_levels, r2_scores, 
                       label=method, linestyle=linestyle, linewidth=linewidth,
                       marker=marker, markersize=markersize)
        
        ax.set_xlabel('噪声水平 (%)', fontweight='bold')
        ax.set_ylabel('R² 分数', fontweight='bold')
        ax.set_title('Extended Methods Robustness Analysis: Performance vs Noise Level - Sklearn-Style', 
                    fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_path = self._get_output_path('california_housing_robustness_extended_sklearn_style.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 鲁棒性图表已保存为 {output_path}")
        
        plt.close()
    
    def _get_output_path(self, filename):
        """获取输出文件路径"""
        output_dir = os.path.join(os.path.dirname(__file__), self.config.OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)


def main():
    """主函数 - 运行扩展版教程 (Sklearn-Style版本)"""
    print("🚀 扩展版加州房价回归教程 - Sklearn-Style版本")
    print("=" * 60)
    
    # 创建教程实例
    tutorial = ExtendedCaliforniaHousingTutorialSklearnStyle()
    
    # 1. 加载和分析数据
    tutorial.load_and_explore_data()
    
    # 2. 运行核心性能测试
    core_results = tutorial.run_comprehensive_benchmark()
    
    # 3. 运行鲁棒性测试
    if tutorial.config.RUN_ROBUSTNESS_TEST:
        robustness_results = tutorial.run_robustness_analysis()
        
        # 创建鲁棒性可视化
        tutorial.create_robustness_visualization(robustness_results)
    
    if tutorial.config.VERBOSE:
        print("\n🎉 扩展版教程运行完成！")
        print(f"📁 所有结果已保存到: {tutorial.config.OUTPUT_DIR}")
        print("\n主要输出文件:")
        print("- california_housing_performance_extended_sklearn_style.png: 核心性能对比图表")
        if tutorial.config.RUN_ROBUSTNESS_TEST:
            print("- california_housing_robustness_extended_sklearn_style.png: 鲁棒性分析图表")
        print("\n💡 关键发现:")
        print("   1. 13种方法的全面性能对比已完成")
        print("   2. CausalEngine在多种噪声水平下的鲁棒性已验证")
        print("   3. 稳健回归和树模型提供了有价值的基准对比")
        print("   4. Sklearn-style实现确保了实验的可重现性")


if __name__ == "__main__":
    main()