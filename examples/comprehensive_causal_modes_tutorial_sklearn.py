#!/usr/bin/env python3
"""
🏠 全面CausalEngine模式教程：加州房价预测 - Sklearn版本
=========================================================

这个教程演示所有4种CausalEngine推理模式在真实世界回归任务中的性能表现。

与原版教程的区别：
- 直接使用sklearn-style的MLPCausalRegressor等封装好的learners
- 不依赖BaselineBenchmark类，直接进行模型训练和比较
- 保持所有原有功能：数据探索、可视化、性能比较、模式深度分析
- 包含更全面的传统方法对比（sklearn MLP, PyTorch MLP, 稳健回归器）

数据集：加州房价数据集（California Housing Dataset）
- 20,640个样本
- 8个特征（房屋年龄、收入、人口等）
- 目标：预测房价中位数

我们将比较所有方法：
1. sklearn MLPRegressor（传统神经网络）
2. PyTorch MLP（传统深度学习）
3. MLP Huber（Huber损失稳健回归）
4. MLP Pinball（Pinball损失稳健回归）
5. MLP Cauchy（Cauchy损失稳健回归）
6. Random Forest（随机森林）
7. CatBoost（强力梯度提升）
8. CausalEngine - deterministic（确定性推理）
9. CausalEngine - exogenous（外生噪声主导）
10. CausalEngine - endogenous（内生不确定性主导）
11. CausalEngine - standard（内生+外生混合）

关键亮点：
- 4种CausalEngine推理模式的全面对比
- 6种强力传统机器学习方法（包含3种稳健回归）
- 真实世界数据的鲁棒性测试
- 因果推理vs传统方法的性能差异分析
- 直接使用sklearn-style learners

实验设计说明
==================================================================
本脚本专注于全面评估CausalEngine的4种推理模式，旨在揭示不同因果推理策略
在真实回归任务上的性能特点和适用场景。

核心实验：全模式性能对比 (在25%标签噪声下)
--------------------------------------------------
- **目标**: 比较所有4种CausalEngine模式和传统方法的预测性能
- **设置**: 25%标签噪声，模拟真实世界数据质量挑战
- **对比模型**: 
  - 传统方法: sklearn MLP, PyTorch MLP, Huber MLP, Pinball MLP, Cauchy MLP, Random Forest, CatBoost
  - CausalEngine: deterministic, exogenous, endogenous, standard
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
from causal_sklearn.utils import causal_split, add_label_anomalies

warnings.filterwarnings('ignore')

# Try to import CatBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, will skip CatBoost in comparisons")


class ComprehensiveTutorialSklearnConfig:
    """
    全面教程配置类 - Sklearn版本
    
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
    NN_LEARNING_RATE = 0.001                         # 学习率
    NN_PATIENCE = 50                                # 早停patience
    NN_TOLERANCE = 1e-4                             # 早停tolerance
    # 
    # 🎯 统一性保证：
    # - 所有方法使用相同的批次大小策略（全量批次）
    # - 所有方法使用相同的数据预处理（StandardScaler）
    # - 所有方法使用相同的早停配置
    # - 所有方法使用相同的随机种子确保可重复性
    # =========================================================================
    
    # 🤖 CausalEngine参数 - 测试4种有效模式
    CAUSAL_MODES = ['deterministic', 'exogenous', 'endogenous', 'standard']
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
    
    # 🛡️ 稳健回归器参数 - 使用统一配置
    HUBER_EPOCHS = NN_MAX_EPOCHS                    # 使用统一神经网络配置
    HUBER_LR = NN_LEARNING_RATE                     # 使用统一神经网络配置
    HUBER_PATIENCE = NN_PATIENCE                    # 使用统一神经网络配置
    
    PINBALL_EPOCHS = NN_MAX_EPOCHS                  # 使用统一神经网络配置
    PINBALL_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    PINBALL_PATIENCE = NN_PATIENCE                  # 使用统一神经网络配置
    
    CAUCHY_EPOCHS = NN_MAX_EPOCHS                   # 使用统一神经网络配置
    CAUCHY_LR = NN_LEARNING_RATE                    # 使用统一神经网络配置
    CAUCHY_PATIENCE = NN_PATIENCE                   # 使用统一神经网络配置
    
    # 🌲 随机森林参数
    RF_N_ESTIMATORS = 100                           # 树的数量
    RF_MAX_DEPTH = None                             # 最大深度
    RF_MIN_SAMPLES_SPLIT = 2                        # 最小分割样本数
    
    # 🚀 CatBoost参数
    CATBOOST_ITERATIONS = 1000                      # 迭代次数
    CATBOOST_LR = 0.1                               # 学习率
    CATBOOST_DEPTH = 6                              # 树深度
    
    # 📊 实验参数
    ANOMALY_RATIO = 0.3                            # 标签异常比例 (核心实验默认值: 30%噪声挑战)
    SAVE_PLOTS = True                               # 是否保存图表
    VERBOSE = True                                  # 是否显示详细输出
    
    # 📈 可视化参数
    FIGURE_DPI = 300                                # 图表分辨率
    FIGURE_SIZE_ANALYSIS = (16, 12)                 # 数据分析图表大小
    FIGURE_SIZE_PERFORMANCE = (26, 16)              # 性能对比图表大小（更大以容纳11个方法）
    FIGURE_SIZE_MODES_COMPARISON = (18, 12)         # CausalEngine模式对比图表大小
    
    # 📁 输出目录参数
    OUTPUT_DIR = "results/comprehensive_causal_modes_tutorial_sklearn"


class ComprehensiveCausalModesSklearnTutorial:
    """
    全面CausalEngine模式教程类 - Sklearn版本
    
    使用sklearn-style learners演示所有4种CausalEngine推理模式在真实世界回归任务中的性能特点
    """
    
    def __init__(self, config=None):
        self.config = config if config is not None else ComprehensiveTutorialSklearnConfig()
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
            print("🏠 全面CausalEngine模式教程 - 加州房价预测 (Sklearn版本)")
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
            
        print("\\n📈 数据分布分析")
        print("-" * 30)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ANALYSIS)
        fig.suptitle('California Housing Dataset Analysis - Comprehensive CausalEngine Modes Tutorial (Sklearn Version)', fontsize=16, fontweight='bold')
        
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
            output_path = self._get_output_path('comprehensive_data_analysis_sklearn.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 数据分析图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 数据统计摘要
        print("\\n📋 数据统计摘要:")
        print(f"  - 最相关特征: {most_corr_feature} (相关系数: {corr_matrix.loc[most_corr_feature, 'MedHouseVal']:.3f})")
        print(f"  - 异常值检测: {np.sum(np.abs(self.y - self.y.mean()) > 3 * self.y.std())} 个潜在异常值")
        print(f"  - 数据完整性: 无缺失值" if not np.any(np.isnan(self.X)) else "  - 警告: 存在缺失值")
    
    def _inject_label_anomalies(self, y, anomaly_ratio, random_state=42):
        """注入标签异常 (使用utils.py中的'shuffle'策略)"""
        if anomaly_ratio <= 0:
            return y.copy()
            
        np.random.seed(random_state)
        return add_label_anomalies(y, ratio=anomaly_ratio, task_type='regression', strategy='shuffle')
    
    def _train_all_models(self, X_train, y_train, X_val, y_val, anomaly_ratio, verbose=True):
        """训练所有模型"""
        if verbose:
            print(f"\\n🔧 训练所有模型 (异常比例: {anomaly_ratio:.1%})")
            print("-" * 60)
        
        # 数据预处理 - 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 注入标签异常到训练标签
        y_train_noisy = self._inject_label_anomalies(y_train, anomaly_ratio, random_state=self.config.RANDOM_STATE)
        
        models = {}
        
        # 1. sklearn MLPRegressor
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
            alpha=0,  # L2正则化
            batch_size=X_train_scaled.shape[0]  # 🔧 统一使用全量批次
        )
        sklearn_model.fit(X_train_scaled, y_train_noisy)
        models['sklearn'] = (sklearn_model, scaler)
        
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
            alpha=0,
            batch_size=X_train_scaled.shape[0]  # 🔧 统一使用全量批次
        )
        pytorch_model.fit(X_train_scaled, y_train_noisy)
        models['pytorch'] = (pytorch_model, scaler)
        
        # 3. 稳健回归器系列
        robust_regressors = {
            'huber': (MLPHuberRegressor, self.config.HUBER_EPOCHS, self.config.HUBER_LR, self.config.HUBER_PATIENCE),
            'pinball': (MLPPinballRegressor, self.config.PINBALL_EPOCHS, self.config.PINBALL_LR, self.config.PINBALL_PATIENCE),
            'cauchy': (MLPCauchyRegressor, self.config.CAUCHY_EPOCHS, self.config.CAUCHY_LR, self.config.CAUCHY_PATIENCE)
        }
        
        for name, (regressor_class, epochs, lr, patience) in robust_regressors.items():
            if verbose:
                print(f"   🔧 训练 MLP {name.capitalize()} Regressor...")
            robust_model = regressor_class(
                hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,
                max_iter=epochs,
                learning_rate=lr,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=patience,
                tol=self.config.NN_TOLERANCE,
                random_state=self.config.RANDOM_STATE,
                verbose=False,
                alpha=0,
                batch_size=X_train_scaled.shape[0]  # 🔧 统一使用全量批次
            )
            robust_model.fit(X_train_scaled, y_train_noisy)
            models[name] = (robust_model, scaler)
        
        # 4. 随机森林
        if verbose:
            print("   🔧 训练 Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            min_samples_split=self.config.RF_MIN_SAMPLES_SPLIT,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train_noisy)
        models['random_forest'] = (rf_model, scaler)
        
        # 5. CatBoost (可选)
        if CATBOOST_AVAILABLE:
            if verbose:
                print("   🔧 训练 CatBoost...")
            catboost_model = CatBoostRegressor(
                iterations=self.config.CATBOOST_ITERATIONS,
                learning_rate=self.config.CATBOOST_LR,
                depth=self.config.CATBOOST_DEPTH,
                random_state=self.config.RANDOM_STATE,
                verbose=False
            )
            catboost_model.fit(X_train_scaled, y_train_noisy)
            models['catboost'] = (catboost_model, scaler)
        
        # 6. CausalEngine 各种模式
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
                alpha=0,
                batch_size=X_train_scaled.shape[0]  # 🔧 统一使用全量批次
            )
            causal_model.fit(X_train_scaled, y_train_noisy)
            models[mode] = (causal_model, scaler)
        
        return models
    
    def _evaluate_models(self, models, X_val, y_val, X_test, y_test):
        """评估所有模型"""
        results = {}
        
        for model_name, (model, scaler) in models.items():
            # 验证集评估
            X_val_scaled = scaler.transform(X_val)
            y_val_pred = model.predict(X_val_scaled)
            
            val_metrics = {
                'MAE': mean_absolute_error(y_val, y_val_pred),
                'MdAE': median_absolute_error(y_val, y_val_pred),
                'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'R²': r2_score(y_val, y_val_pred)
            }
            
            # 测试集评估
            X_test_scaled = scaler.transform(X_test)
            y_test_pred = model.predict(X_test_scaled)
            
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
        """运行全面的基准测试 - 包含所有4种CausalEngine模式"""
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
            print("\\n🚀 开始全面基准测试 - 测试所有4种CausalEngine模式 (Sklearn版本)")
            print("=" * 90)
            print(f"🔧 实验配置:")
            print(f"   - 测试集比例: {test_size:.1%}")
            print(f"   - 验证集比例: {val_size:.1%}")
            print(f"   - 异常标签比例: {anomaly_ratio:.1%}")
            print(f"   - 随机种子: {self.config.RANDOM_STATE}")
            print(f"   - CausalEngine模式: {', '.join(self.config.CAUSAL_MODES)}")
            print(f"   - CausalEngine网络: {self.config.CAUSAL_HIDDEN_SIZES}")
            print(f"   - 最大训练轮数: {self.config.CAUSAL_MAX_EPOCHS}")
            print(f"   - 早停patience: {self.config.CAUSAL_PATIENCE}")
            traditional_count = 6 + (1 if CATBOOST_AVAILABLE else 0)  # sklearn, pytorch, huber, pinball, cauchy, rf, (catboost)
            total_methods = len(self.config.CAUSAL_MODES) + traditional_count
            print(f"   - 总计对比方法: {total_methods} 种 ({len(self.config.CAUSAL_MODES)}种CausalEngine + {traditional_count}种传统)")
        
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
            print(f"\\n📊 数据分割:")
            print(f"   - 训练集: {len(X_train)} 样本")
            print(f"   - 验证集: {len(X_val)} 样本")
            print(f"   - 测试集: {len(X_test)} 样本")
        
        # 训练模型
        models = self._train_all_models(X_train, y_train, X_val, y_val, anomaly_ratio, verbose)
        
        # 评估模型
        self.results = self._evaluate_models(models, X_val, y_val, X_test, y_test)
        
        if verbose:
            print(f"\\n📊 全面基准测试结果 (异常比例: {anomaly_ratio:.0%})")
            self._print_results()
        
        return self.results
    
    def _print_results(self):
        """打印结果表格"""
        print("=" * 150)
        print(f"{'方法':<20} {'验证集':<60} {'测试集':<60}")
        print(f"{'':20} {'MAE':<12} {'MdAE':<12} {'RMSE':<12} {'R²':<12} {'MAE':<12} {'MdAE':<12} {'RMSE':<12} {'R²':<12}")
        print("-" * 150)
        
        for method, metrics in self.results.items():
            val_m = metrics['val']
            test_m = metrics['test']
            print(f"{method:<20} {val_m['MAE']:<12.4f} {val_m['MdAE']:<12.4f} {val_m['RMSE']:<12.4f} {val_m['R²']:<12.4f} "
                  f"{test_m['MAE']:<12.4f} {test_m['MdAE']:<12.4f} {test_m['RMSE']:<12.4f} {test_m['R²']:<12.4f}")
        
        print("=" * 150)
    
    def analyze_causal_modes_performance(self, verbose=True):
        """专门分析CausalEngine不同模式的性能特点"""
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        if verbose:
            print("\\n🔬 CausalEngine模式深度分析 (Sklearn版本)")
            print("=" * 80)
        
        # 提取CausalEngine模式结果
        causal_results = {}
        traditional_results = {}
        
        for method, metrics in self.results.items():
            if method in self.config.CAUSAL_MODES:
                causal_results[method] = metrics
            else:
                # 所有非CausalEngine的方法都算作传统方法
                traditional_results[method] = metrics
        
        if verbose:
            print(f"🎯 CausalEngine模式性能对比 (共{len(causal_results)}种模式):")
            print("-" * 50)
            
            # 按MdAE分数排序（越小越好）
            causal_mdae_scores = {mode: metrics['test']['MdAE'] for mode, metrics in causal_results.items()}
            sorted_causal = sorted(causal_mdae_scores.items(), key=lambda x: x[1])  # 升序排列
            
            for i, (mode, mdae) in enumerate(sorted_causal, 1):
                mae = causal_results[mode]['test']['MAE']
                r2 = causal_results[mode]['test']['R²']
                print(f"   {i}. {mode:<12} - MdAE: {mdae:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")
            
            # 模式特点分析
            print(f"\\n📊 模式特点分析:")
            print("-" * 30)
            
            best_mode = sorted_causal[0][0]
            worst_mode = sorted_causal[-1][0]
            performance_gap = sorted_causal[-1][1] - sorted_causal[0][1]  # 最差 - 最好 (因为MdAE越小越好)
            
            print(f"   🏆 最佳模式: {best_mode} (MdAE = {sorted_causal[0][1]:.3f})")
            print(f"   📉 最弱模式: {worst_mode} (MdAE = {sorted_causal[-1][1]:.3f})")
            print(f"   📏 性能差距: {performance_gap:.3f} ({performance_gap/sorted_causal[0][1]*100:.1f}%)")
            
            # 与传统方法比较（基于MdAE）
            if traditional_results:
                print(f"\\n🆚 CausalEngine vs 传统方法:")
                print("-" * 40)
                
                traditional_mdae_scores = {method: metrics['test']['MdAE'] for method, metrics in traditional_results.items()}
                best_traditional = min(traditional_mdae_scores.keys(), key=lambda x: traditional_mdae_scores[x])  # 最小MdAE最好
                best_traditional_mdae = traditional_mdae_scores[best_traditional]
                
                print(f"   最佳传统方法: {best_traditional} (MdAE = {best_traditional_mdae:.3f})")
                print(f"   最佳CausalEngine: {best_mode} (MdAE = {sorted_causal[0][1]:.3f})")
                
                improvement = (best_traditional_mdae - sorted_causal[0][1]) / best_traditional_mdae * 100  # 正值表示CausalEngine更好
                print(f"   性能提升: {improvement:+.2f}%")
                
                # 统计有多少CausalEngine模式优于最佳传统方法
                better_modes = sum(1 for _, mdae in sorted_causal if mdae < best_traditional_mdae)
                print(f"   优于传统方法的CausalEngine模式: {better_modes}/{len(sorted_causal)}")
        
        return causal_results, traditional_results
    
    def create_comprehensive_performance_visualization(self, save_plot=None):
        """创建全面的性能可视化图表 - 展示所有方法"""
        if save_plot is None:
            save_plot = self.config.SAVE_PLOTS
            
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\\n📊 创建全面性能可视化图表")
        print("-" * 40)
        
        # 准备数据 - 分类排列：传统方法 + CausalEngine模式
        traditional_methods = [m for m in self.results.keys() if m not in self.config.CAUSAL_MODES]
        causal_methods = [m for m in self.config.CAUSAL_MODES if m in self.results]
        methods = traditional_methods + causal_methods
        
        # 为不同类型的方法设置颜色
        colors = []
        for method in methods:
            if method in self.config.CAUSAL_MODES:
                colors.append('#2ca02c')  # 绿色系 - CausalEngine
            else:
                colors.append('#1f77b4')  # 蓝色系 - 传统方法
        
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        
        # 创建子图 - 2x2布局展示4个指标
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_PERFORMANCE)
        fig.suptitle('Comprehensive CausalEngine Modes vs Traditional Methods (Sklearn Version)\\nCalifornia Housing Performance (25% Label Noise)', 
                     fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.results[method]['test'][metric] for method in methods]
            
            bars = axes[i].bar(range(len(methods)), values, color=colors, alpha=0.8, edgecolor='black')
            axes[i].set_title(f'{metric} (Test Set)', fontweight='bold')
            axes[i].set_ylabel(metric)
            
            # 设置X轴标签 - 智能处理各种方法名
            method_labels = []
            for method in methods:
                if method in self.config.CAUSAL_MODES:
                    method_labels.append(f'CausalEngine\\n({method})')
                elif method == 'sklearn':
                    method_labels.append('sklearn\\nMLP')
                elif method == 'pytorch':
                    method_labels.append('PyTorch\\nMLP')
                else:
                    # 其他传统方法，简化显示名称
                    display_name = method.replace('_', ' ').title()
                    if len(display_name) > 12:
                        # 长名称分行显示
                        words = display_name.split()
                        if len(words) > 1:
                            display_name = f"{words[0]}\\n{' '.join(words[1:])}"
                    method_labels.append(display_name)
            
            axes[i].set_xticks(range(len(methods)))
            axes[i].set_xticklabels(method_labels, rotation=45, ha='right', fontsize=8)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
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
            output_path = self._get_output_path('comprehensive_performance_comparison_sklearn.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 全面性能图表已保存为 {output_path}")
        
        plt.close()
    
    def create_causal_modes_comparison(self, save_plot=None):
        """创建专门的CausalEngine模式对比图表"""
        if save_plot is None:
            save_plot = self.config.SAVE_PLOTS
            
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\\n📊 创建CausalEngine模式专项对比图表")
        print("-" * 45)
        
        # 提取CausalEngine模式结果
        causal_methods = [m for m in self.config.CAUSAL_MODES if m in self.results]
        
        if len(causal_methods) < 2:
            print("❌ 需要至少2种CausalEngine模式来进行对比")
            return
        
        # 创建雷达图显示CausalEngine模式的多维性能
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.FIGURE_SIZE_MODES_COMPARISON)
        fig.suptitle('CausalEngine Modes Detailed Comparison (Sklearn Version)', fontsize=16, fontweight='bold')
        
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
            output_path = self._get_output_path('causal_modes_detailed_comparison_sklearn.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 CausalEngine模式对比图表已保存为 {output_path}")
        
        plt.close()
    
    def print_comprehensive_summary(self):
        """打印全面的总结报告"""
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\\n📋 全面实验总结报告 (Sklearn版本)")
        print("=" * 90)
        
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
        
        # 性能排名（按MdAE分数，越小越好）
        print(f"\\n🏆 总体性能排名 (按MdAE分数):")
        print("-" * 50)
        
        all_mdae_scores = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items()]
        all_mdae_scores.sort(key=lambda x: x[1])  # 升序排列，MdAE越小越好
        
        for i, (method, mdae) in enumerate(all_mdae_scores, 1):
            method_type = "CausalEngine" if method in self.config.CAUSAL_MODES else "Traditional"
            r2 = self.results[method]['test']['R²']
            print(f"   {i:2d}. {method:<15} ({method_type:<12}) - MdAE: {mdae:.3f}, R²: {r2:.4f}")
        
        # CausalEngine优势分析（基于MdAE）
        print(f"\\n🎯 CausalEngine模式分析:")
        print("-" * 40)
        
        causal_results = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items() 
                         if method in self.config.CAUSAL_MODES]
        traditional_results = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items() 
                              if method not in self.config.CAUSAL_MODES]
        
        if causal_results and traditional_results:
            best_causal = min(causal_results, key=lambda x: x[1])  # 最小MdAE最好
            best_traditional = min(traditional_results, key=lambda x: x[1])  # 最小MdAE最好
            
            print(f"   最佳CausalEngine模式: {best_causal[0]} (MdAE = {best_causal[1]:.3f})")
            print(f"   最佳传统方法: {best_traditional[0]} (MdAE = {best_traditional[1]:.3f})")
            
            improvement = (best_traditional[1] - best_causal[1]) / best_traditional[1] * 100  # 正值表示CausalEngine更好
            print(f"   性能提升: {improvement:+.2f}%")
            
            # 统计优于传统方法的CausalEngine模式数量
            better_causal_count = sum(1 for _, mdae in causal_results if mdae < best_traditional[1])
            print(f"   优于最佳传统方法的CausalEngine模式: {better_causal_count}/{len(causal_results)}")
        
        # 关键发现（基于MdAE）
        print(f"\\n💡 关键发现:")
        print("-" * 20)
        
        if len(all_mdae_scores) > 0:
            top_method = all_mdae_scores[0]  # MdAE最小的方法最好
            if top_method[0] in self.config.CAUSAL_MODES:
                print(f"   ✅ CausalEngine模式 '{top_method[0]}' 在MdAE指标上取得最佳性能")
                print(f"   ✅ 因果推理在稳健性方面显示出明显优势")
            else:
                print(f"   ⚠️ 传统方法 '{top_method[0]}' 在MdAE指标上表现最优")
                print(f"   ⚠️ 建议进一步调优CausalEngine参数")
            
            # 检查CausalEngine模式间的差异（基于MdAE）
            if len(causal_results) > 1:
                causal_mdae_values = [mdae for _, mdae in causal_results]
                causal_std = np.std(causal_mdae_values)
                print(f"   📊 CausalEngine模式间MdAE标准差: {causal_std:.4f}")
                if causal_std < 0.02:
                    print(f"   📈 不同CausalEngine模式MdAE性能较为接近")
                else:
                    print(f"   📈 不同CausalEngine模式MdAE存在显著性能差异")


def main():
    """主函数：运行完整的全面CausalEngine模式教程"""
    print("🏠 全面CausalEngine模式教程 - Sklearn版本")
    print("🎯 目标：测试所有4种CausalEngine推理模式在真实世界回归任务中的表现")
    print("🔧 特点：使用sklearn-style learners，无需BaselineBenchmark")
    print("=" * 100)
    
    # 创建配置实例
    config = ComprehensiveTutorialSklearnConfig()
    
    print(f"🔧 当前配置:")
    print(f"   - CausalEngine模式: {', '.join(config.CAUSAL_MODES)} (共{len(config.CAUSAL_MODES)}种)")
    print(f"   - 网络架构: {config.CAUSAL_HIDDEN_SIZES}")
    print(f"   - 最大轮数: {config.CAUSAL_MAX_EPOCHS}")
    print(f"   - 早停patience: {config.CAUSAL_PATIENCE}")
    print(f"   - 异常比例: {config.ANOMALY_RATIO:.1%}")
    traditional_count = 6 + (1 if CATBOOST_AVAILABLE else 0)
    total_methods = len(config.CAUSAL_MODES) + traditional_count
    print(f"   - 总计对比方法: {total_methods} 种")
    print(f"   - 输出目录: {config.OUTPUT_DIR}/")
    print()
    
    # 创建教程实例
    tutorial = ComprehensiveCausalModesSklearnTutorial(config)
    
    # 1. 加载和探索数据
    tutorial.load_and_explore_data()
    
    # 2. 数据可视化
    tutorial.visualize_data()
    
    # 3. 运行全面基准测试 - 测试所有4种CausalEngine模式
    tutorial.run_comprehensive_benchmark()
    
    # 4. 专门分析CausalEngine模式性能
    tutorial.analyze_causal_modes_performance()
    
    # 5. 创建全面性能可视化
    tutorial.create_comprehensive_performance_visualization()
    
    # 6. 创建CausalEngine模式专项对比
    tutorial.create_causal_modes_comparison()
    
    # 7. 打印全面总结报告
    tutorial.print_comprehensive_summary()
    
    print("\\n🎉 全面CausalEngine模式教程完成！")
    print("📋 实验总结:")
    print(f"   - 使用了真实世界的加州房价数据集 ({tutorial.X.shape[0]:,} 样本)")
    print(f"   - 测试了所有 {len(config.CAUSAL_MODES)} 种CausalEngine推理模式")
    traditional_count = 6 + (1 if CATBOOST_AVAILABLE else 0)
    print(f"   - 与 {traditional_count} 种传统方法进行了全面对比")
    print(f"   - 传统方法包括: sklearn MLP, PyTorch MLP, 稳健回归器, 集成方法等")
    print(f"   - 在 {config.ANOMALY_RATIO:.0%} 标签噪声环境下验证了鲁棒性")
    print("   - 提供了详细的模式特点分析和可视化")
    print("   - 直接使用sklearn-style learners")
    
    print("\\n📊 生成的文件:")
    if config.SAVE_PLOTS:
        print(f"   - {config.OUTPUT_DIR}/comprehensive_data_analysis_sklearn.png           (数据分析图)")
        print(f"   - {config.OUTPUT_DIR}/comprehensive_performance_comparison_sklearn.png  (全面性能对比图)")
        print(f"   - {config.OUTPUT_DIR}/causal_modes_detailed_comparison_sklearn.png      (CausalEngine模式专项对比图)")
    
    print("\\n💡 提示：通过修改ComprehensiveTutorialSklearnConfig类来自定义实验参数！")
    print("🔬 下一步：可以尝试不同的数据集或调整模型参数来进一步验证CausalEngine的优越性")


if __name__ == "__main__":
    main()