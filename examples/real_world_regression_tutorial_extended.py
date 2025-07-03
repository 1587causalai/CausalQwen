#!/usr/bin/env python3
"""
🏠 扩展版真实世界回归教程：加州房价预测
==========================================

这个教程演示CausalEngine与多种强力传统方法在真实世界回归任务中的性能对比。

数据集：加州房价数据集（California Housing Dataset）
- 20,640个样本
- 8个特征（房屋年龄、收入、人口等）
- 目标：预测房价中位数

我们将比较12种方法：
1. sklearn MLPRegressor（传统神经网络）
2. PyTorch MLP（传统深度学习）
3. MLP Huber（Huber损失稳健回归）
4. MLP Pinball（Pinball损失稳健回归）
5. MLP Cauchy（Cauchy损失稳健回归）
6. Random Forest（随机森林）
7. XGBoost（梯度提升）
8. LightGBM（轻量梯度提升）
9. CatBoost（强力梯度提升）
10. CausalEngine - exogenous（外生噪声主导）
11. CausalEngine - endogenous（内生不确定性主导）
12. CausalEngine - standard（内生+外生混合）

关键亮点：
- 真实世界数据的鲁棒性测试
- 6种强力传统机器学习方法对比
- 3种稳健神经网络回归方法（Huber、Pinball、Cauchy）
- 统一神经网络参数配置确保公平比较
- 因果推理vs传统方法的性能差异分析

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
  - CausalEngine: exogenous, endogenous, standard等模式

实验二：鲁棒性分析 (跨越不同噪声水平)
--------------------------------------------------
- **目标**: 探究模型性能随标签噪声水平增加时的变化情况，评估其稳定性。
- **设置**: 在一系列噪声比例（如0%, 10%, 20%, 30%, 40%, 50%）下分别运行测试。
- **对比模型**: 所有12种方法在不同噪声水平下的表现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import warnings
import os
import sys

# 设置matplotlib后端为非交互式，避免弹出窗口
plt.switch_backend('Agg')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的基准测试模块
from causal_sklearn.benchmarks import BaselineBenchmark

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
    NN_PATIENCE = 200                               # 早停patience
    NN_TOLERANCE = 1e-4                             # 早停tolerance
    # =========================================================================
    
    # 🤖 CausalEngine参数 - 使用统一神经网络配置
    CAUSAL_MODES = ['deterministic', 'exogenous', 'endogenous', 'standard']       # 可选: ['deterministic', 'exogenous', 'endogenous', 'standard']
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
    
    # 📊 实验参数
    ANOMALY_RATIO = 0.4                             # 标签异常比例 (核心实验默认值: 40%噪声挑战)
    SAVE_PLOTS = True                               # 是否保存图表
    VERBOSE = True                                  # 是否显示详细输出
    
    # 🛡️ 鲁棒性测试参数 - 验证"CausalEngine鲁棒性优势"的假设
    ROBUSTNESS_ANOMALY_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 噪声水平
    RUN_ROBUSTNESS_TEST = True                      # 是否运行鲁棒性测试
    
    # 📈 可视化参数
    FIGURE_DPI = 300                                # 图表分辨率
    FIGURE_SIZE_ANALYSIS = (24, 20)                 # 数据分析图表大小
    FIGURE_SIZE_PERFORMANCE = (24, 20)              # 性能对比图表大小
    FIGURE_SIZE_ROBUSTNESS = (24, 20)               # 鲁棒性测试图表大小 (更大容纳更多方法)
    
    # 📁 输出目录参数
    OUTPUT_DIR = "results/california_housing_regression_extended"  # 输出目录名称


class ExtendedCaliforniaHousingTutorial:
    """
    扩展版加州房价回归教程主类
    
    实现了真实世界数据上的全面性能评估，包含多种强力传统方法对比
    """
    
    def __init__(self, config=None):
        self.config = config or TutorialConfig()
        self.benchmark = BaselineBenchmark()
        self.results = {}
        
        # 确保输出目录存在
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        if self.config.VERBOSE:
            print("🏠 扩展版加州房价回归教程")
            print("=" * 60)
            print(f"📊 将比较 {len(self.config.BASELINE_METHODS) + len(self.config.CAUSAL_MODES)} 种方法")
            print(f"🎯 基准方法 ({len(self.config.BASELINE_METHODS)}种): {', '.join(self.config.BASELINE_METHODS)}")
            print(f"🤖 CausalEngine模式 ({len(self.config.CAUSAL_MODES)}种): {', '.join(self.config.CAUSAL_MODES)}")
            print(f"📁 结果保存到: {self.config.OUTPUT_DIR}")
            print()
    
    def load_and_analyze_data(self):
        """加载并分析加州房价数据集"""
        if self.config.VERBOSE:
            print("📥 加载加州房价数据集...")
        
        # 加载数据
        california_housing = fetch_california_housing()
        X = california_housing.data
        y = california_housing.target
        feature_names = california_housing.feature_names
        
        if self.config.VERBOSE:
            print(f"✅ 数据加载完成:")
            print(f"   - 样本数: {X.shape[0]:,}")
            print(f"   - 特征数: {X.shape[1]}")
            print(f"   - 特征名: {', '.join(feature_names)}")
            print(f"   - 房价范围: ${y.min():.2f} - ${y.max():.2f} (单位: 10万美元)")
            print(f"   - 房价均值: ${y.mean():.2f} ± ${y.std():.2f}")
            print()
        
        # 保存数据信息供后续使用
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
        # 生成数据分析图表
        if self.config.SAVE_PLOTS:
            self._create_data_analysis_plots()
        
        return X, y, feature_names
    
    def _create_data_analysis_plots(self):
        """创建数据分析图表"""
        if self.config.VERBOSE:
            print("📊 生成数据分析图表...")
        
        # 创建综合数据分析图
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ANALYSIS, dpi=self.config.FIGURE_DPI)
        fig.suptitle('California Housing Dataset Analysis - Extended Regression Tutorial', fontsize=16, fontweight='bold')
        
        # 1. 房价分布
        axes[0,0].hist(self.y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(self.y.mean(), color='red', linestyle='--', alpha=0.8, 
                         label=f'Mean: ${self.y.mean():.2f}')
        axes[0,0].set_xlabel('House Price ($100k)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('House Price Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 特征相关矩阵
        feature_data = pd.DataFrame(self.X, columns=self.feature_names)
        feature_data['MedHouseVal'] = self.y
        corr_matrix = feature_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[0,1], cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('Feature Correlation Matrix')
        
        # 3. 特征分布（标准化后）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        feature_data_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        feature_data_scaled.boxplot(ax=axes[1,0])
        axes[1,0].set_title('Feature Distribution (Standardized)')
        axes[1,0].set_ylabel('Standardized Value')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 最相关特征散点图
        # 找到与房价最相关的特征
        target_corr = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False)
        most_correlated_feature = target_corr.index[1]  # 除了自己之外最相关的
        
        axes[1,1].scatter(feature_data[most_correlated_feature], self.y, 
                         alpha=0.6, c='green', s=1)
        axes[1,1].set_xlabel(most_correlated_feature)
        axes[1,1].set_ylabel('MedHouseVal')
        axes[1,1].set_title(f'Most Correlated Feature: {most_correlated_feature}')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        analysis_plot_path = os.path.join(self.config.OUTPUT_DIR, 'extended_data_analysis.png')
        plt.savefig(analysis_plot_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        if self.config.VERBOSE:
            print(f"📊 数据分析图表已保存: {analysis_plot_path}")
    
    def run_core_performance_test(self):
        """运行核心性能测试（固定25%噪声）"""
        if self.config.VERBOSE:
            print(f"🎯 运行核心性能测试 (异常比例: {self.config.ANOMALY_RATIO:.1%})")
            print("=" * 60)
        
        # 运行性能对比
        results = self.benchmark.compare_models(
            X=self.X,
            y=self.y,
            task_type='regression',
            test_size=self.config.TEST_SIZE,
            val_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_STATE,
            anomaly_ratio=self.config.ANOMALY_RATIO,
            
            # 基准方法配置
            baseline_methods=self.config.BASELINE_METHODS,
            
            # CausalEngine配置
            causal_modes=self.config.CAUSAL_MODES,
            
            # --- 统一参数，适配所有模型 ---
            # 神经网络结构 (适配 'hidden_sizes' 和 'hidden_layer_sizes')
            hidden_sizes=self.config.NN_HIDDEN_SIZES,
            hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,

            # 训练轮数 (适配 'max_epochs' 和 'max_iter')
            max_epochs=self.config.NN_MAX_EPOCHS,
            max_iter=self.config.NN_MAX_EPOCHS,

            # 学习率 (适配 'lr' 和 'learning_rate')
            lr=self.config.NN_LEARNING_RATE,
            learning_rate=self.config.NN_LEARNING_RATE,

            # 早停参数
            patience=self.config.NN_PATIENCE,
            tol=self.config.NN_TOLERANCE,
            
            # CausalEngine专属参数
            gamma_init=self.config.CAUSAL_GAMMA_INIT,
            b_noise_init=self.config.CAUSAL_B_NOISE_INIT,
            b_noise_trainable=self.config.CAUSAL_B_NOISE_TRAINABLE,
            
            verbose=self.config.VERBOSE
        )
        
        self.results['core_performance'] = results
        
        if self.config.SAVE_PLOTS:
            self._create_performance_plots(results, 'core_performance')
        
        return results
    
    def run_robustness_test(self):
        """运行鲁棒性测试（多个噪声水平）"""
        if not self.config.RUN_ROBUSTNESS_TEST:
            if self.config.VERBOSE:
                print("⏭️  跳过鲁棒性测试 (RUN_ROBUSTNESS_TEST=False)")
            return None
        
        if self.config.VERBOSE:
            print(f"🛡️  运行鲁棒性测试")
            print(f"   噪声水平: {[f'{r:.1%}' for r in self.config.ROBUSTNESS_ANOMALY_RATIOS]}")
            print("=" * 60)
        
        robustness_results = {}
        
        for i, anomaly_ratio in enumerate(self.config.ROBUSTNESS_ANOMALY_RATIOS):
            if self.config.VERBOSE:
                print(f"\\n🔄 测试噪声水平 {i+1}/{len(self.config.ROBUSTNESS_ANOMALY_RATIOS)}: {anomaly_ratio:.1%}")
            
            results = self.benchmark.compare_models(
                X=self.X,
                y=self.y,
                task_type='regression',
                test_size=self.config.TEST_SIZE,
                val_size=self.config.VAL_SIZE,
                random_state=self.config.RANDOM_STATE,
                anomaly_ratio=anomaly_ratio,
                
                # 基准方法配置
                baseline_methods=self.config.BASELINE_METHODS,
                
                # --- 统一参数，适配所有模型 ---
                # CausalEngine配置
                causal_modes=self.config.CAUSAL_MODES,

                # 神经网络结构 (适配 'hidden_sizes' 和 'hidden_layer_sizes')
                hidden_sizes=self.config.NN_HIDDEN_SIZES,
                hidden_layer_sizes=self.config.NN_HIDDEN_SIZES,

                # 训练轮数 (适配 'max_epochs' 和 'max_iter')
                max_epochs=self.config.NN_MAX_EPOCHS,
                max_iter=self.config.NN_MAX_EPOCHS,

                # 学习率 (适配 'lr' 和 'learning_rate')
                lr=self.config.NN_LEARNING_RATE,
                learning_rate=self.config.NN_LEARNING_RATE,

                # 早停参数
                patience=self.config.NN_PATIENCE,
                tol=self.config.NN_TOLERANCE,

                # CausalEngine专属参数
                gamma_init=self.config.CAUSAL_GAMMA_INIT,
                b_noise_init=self.config.CAUSAL_B_NOISE_INIT,
                b_noise_trainable=self.config.CAUSAL_B_NOISE_TRAINABLE,
                
                verbose=False  # 降低输出量
            )
            
            robustness_results[anomaly_ratio] = results
        
        self.results['robustness'] = robustness_results
        
        if self.config.SAVE_PLOTS:
            self._create_robustness_plots(robustness_results)
        
        return robustness_results
    
    def _create_performance_plots(self, results, experiment_name):
        """创建性能对比图表"""
        if self.config.VERBOSE:
            print(f"📊 生成{experiment_name}性能图表...")
        
        # 创建性能对比图
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_PERFORMANCE, dpi=self.config.FIGURE_DPI)
        fig.suptitle(f'Extended California Housing Test Set Performance\\nNoise Level: {self.config.ANOMALY_RATIO:.1%}', 
                    fontsize=16, fontweight='bold')
        
        # 准备数据
        methods = list(results.keys())
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        
        test_data = {metric: [results[method]['test'][metric] for method in methods] for metric in metrics}
        val_data = {metric: [results[method]['val'][metric] for method in methods] for metric in metrics}
        
        # 设置颜色
        colors = []
        for method in methods:
            if 'causal' in method.lower() or any(mode in method for mode in ['deterministic', 'standard', 'exogenous', 'endogenous']):
                colors.append('gold')  # CausalEngine用金色
            elif any(robust in method.lower() for robust in ['huber', 'cauchy', 'pinball']):
                colors.append('lightgreen')  # 稳健方法用浅绿
            else:
                colors.append('lightblue')  # 传统方法用浅蓝
        
        # 为每个指标创建子图
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            x_pos = np.arange(len(methods))
            
            # 只显示测试集性能（更清爽）
            bars = ax.bar(x_pos, test_data[metric], color=colors, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{metric} (Test Set)', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # 高亮最佳结果
            if metric == 'R²':
                best_idx = test_data[metric].index(max(test_data[metric]))
            else:
                best_idx = test_data[metric].index(min(test_data[metric]))
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        
        # 保存图表
        performance_plot_path = os.path.join(self.config.OUTPUT_DIR, f'{experiment_name}_comparison.png')
        plt.savefig(performance_plot_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        if self.config.VERBOSE:
            print(f"📊 性能对比图表已保存: {performance_plot_path}")
    
    def _create_robustness_plots(self, robustness_results):
        """创建鲁棒性测试图表 - 4个指标的2x2子图布局"""
        if self.config.VERBOSE:
            print("📊 生成鲁棒性测试图表...")
        
        # 创建鲁棒性测试图
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ROBUSTNESS, dpi=self.config.FIGURE_DPI)
        fig.suptitle('Extended Robustness Analysis: Performance vs Noise Level', fontsize=16, fontweight='bold')
        
        # 准备数据
        noise_levels = list(robustness_results.keys())
        methods = list(robustness_results[noise_levels[0]].keys())
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        metric_labels = ['Mean Absolute Error (MAE)', 'Median Absolute Error (MdAE)', 'Root Mean Squared Error (RMSE)', 'R-squared Score (R²)']
        
        # 设置颜色和线型
        method_styles = {}
        causal_methods = [m for m in methods if 'causal' in m.lower() or any(mode in m for mode in ['deterministic', 'standard', 'exogenous', 'endogenous'])]
        robust_methods = [m for m in methods if any(robust in m.lower() for robust in ['huber', 'cauchy', 'pinball'])]
        traditional_methods = [m for m in methods if m not in causal_methods and m not in robust_methods]
        
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        markers = ['o', 's', 'v', '^', 'D', 'P', 'X', 'h', '+', '*']
        
        for i, method in enumerate(methods):
            if method in causal_methods:
                method_styles[method] = {'color': '#d62728', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 8}
            elif method in robust_methods:
                method_styles[method] = {'color': colors[i % len(colors)], 'linestyle': '--', 'linewidth': 2, 'marker': 's', 'markersize': 6}
            else:
                method_styles[method] = {'color': colors[i % len(colors)], 'linestyle': ':', 'linewidth': 2, 'marker': markers[i % len(markers)], 'markersize': 6}
        
        # 为每个指标创建子图
        for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            for method in methods:
                # 收集该方法在各噪声水平下的测试集性能
                values = []
                for noise in noise_levels:
                    if method in robustness_results[noise]:
                        values.append(robustness_results[noise][method]['test'][metric])
                    else:
                        values.append(np.nan)
                
                ax.plot(noise_levels, values, 
                       label=method, 
                       **method_styles[method])
            
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Label Noise Ratio', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 设置x轴刻度标签
            ax.set_xticks(noise_levels)
            ax.set_xticklabels([f'{r:.1%}' for r in noise_levels])
            
            # 为R²添加特殊处理（越高越好），其他指标越低越好
            if metric == 'R²':
                ax.set_ylim(bottom=0)  # R²从0开始显示
            else:
                ax.set_ylim(bottom=0)  # 误差指标从0开始显示
        
        plt.tight_layout()
        
        # 保存图表
        robustness_plot_path = os.path.join(self.config.OUTPUT_DIR, 'extended_robustness_analysis.png')
        plt.savefig(robustness_plot_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        if self.config.VERBOSE:
            print(f"📊 鲁棒性分析图表已保存: {robustness_plot_path}")
    
    def generate_summary_report(self):
        """生成实验总结报告"""
        if self.config.VERBOSE:
            print("\\n📋 生成实验总结报告...")
        
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
        report_lines.append(f"- **实验方法**: {len(self.config.BASELINE_METHODS) + len(self.config.CAUSAL_MODES)} 种")
        report_lines.append(f"  - 传统方法 ({len(self.config.BASELINE_METHODS)}种): {', '.join(self.config.BASELINE_METHODS)}")
        report_lines.append(f"  - CausalEngine ({len(self.config.CAUSAL_MODES)}种): {', '.join(self.config.CAUSAL_MODES)}")
        report_lines.append("")
        
        # 核心性能测试结果
        if 'core_performance' in self.results:
            results = self.results['core_performance']
            report_lines.append("## 🎯 核心性能测试结果")
            report_lines.append("")
            report_lines.append(f"**噪声水平**: {self.config.ANOMALY_RATIO:.1%}")
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
                elif method.lower() in ['catboost', 'random_forest']:
                    method_type = "🌲 集成学习"
                else:
                    method_type = "🧠 神经网络"
                
                report_lines.append(f"| {i} | **{method}** | "
                                  f"{test_metrics['MAE']:.4f} | "
                                  f"**{test_metrics['MdAE']:.4f}** | "
                                  f"{test_metrics['RMSE']:.4f} | "
                                  f"{test_metrics['R²']:.4f} | "
                                  f"{method_type} |")
            
            report_lines.append("")
            
            # 验证集vs测试集对比（展示噪声影响）
            report_lines.append("### 🔍 验证集 vs 测试集性能对比")
            report_lines.append("")
            report_lines.append("*验证集包含噪声，测试集为纯净数据*")
            report_lines.append("")
            
            report_lines.append("| 方法 | 验证集MdAE | 测试集MdAE | 性能提升 |")
            report_lines.append("|------|----------:|----------:|--------:|")
            
            for method in methods_by_mdae:
                val_mdae = results[method]['val']['MdAE']
                test_mdae = results[method]['test']['MdAE']
                improvement = ((val_mdae - test_mdae) / val_mdae) * 100
                
                report_lines.append(f"| {method} | "
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
            report_lines.append(f"- **🥇 最佳整体性能**: `{best_mdae_method}` (MdAE: {best_mdae_score:.4f})")
            
            if causal_methods:
                best_causal = min(causal_methods, key=lambda x: results[x]['test']['MdAE'])
                causal_rank = methods_by_mdae.index(best_causal) + 1
                causal_score = results[best_causal]['test']['MdAE']
                report_lines.append(f"- **🤖 最佳CausalEngine**: `{best_causal}` (排名: {causal_rank}/{len(methods_by_mdae)}, MdAE: {causal_score:.4f})")
                
                # CausalEngine模式对比
                if len(causal_methods) > 1:
                    report_lines.append("")
                    report_lines.append("**CausalEngine模式对比**:")
                    for causal_method in sorted(causal_methods, key=lambda x: results[x]['test']['MdAE']):
                        rank = methods_by_mdae.index(causal_method) + 1
                        score = results[causal_method]['test']['MdAE']
                        report_lines.append(f"  - `{causal_method}`: 排名 {rank}, MdAE {score:.4f}")
            
            # 传统方法分析
            traditional_methods = [m for m in results.keys() if m not in causal_methods]
            if traditional_methods:
                best_traditional = min(traditional_methods, key=lambda x: results[x]['test']['MdAE'])
                traditional_rank = methods_by_mdae.index(best_traditional) + 1
                traditional_score = results[best_traditional]['test']['MdAE']
                report_lines.append(f"- **🏅 最佳传统方法**: `{best_traditional}` (排名: {traditional_rank}/{len(methods_by_mdae)}, MdAE: {traditional_score:.4f})")
            
            report_lines.append("")
        
        # 鲁棒性测试结果
        if 'robustness' in self.results:
            robustness_results = self.results['robustness']
            report_lines.append("## 🛡️ 鲁棒性测试结果")
            report_lines.append("")
            
            noise_levels = sorted(robustness_results.keys())
            methods = list(robustness_results[noise_levels[0]].keys())
            
            report_lines.append("### 📊 MdAE性能随噪声水平变化")
            report_lines.append("")
            
            # 表格头
            header = "| 方法 | " + " | ".join([f"{r:.0%}" for r in noise_levels]) + " | 稳定性* |"
            separator = "|" + "|".join(["-" * max(6, len(f"{r:.0%}")) for r in [0] + noise_levels + [0]]) + "|"
            separator = "|------|" + "|".join([f"{'-'*(len(f'{r:.0%}')+1):->6}" for r in noise_levels]) + "|--------|"
            
            report_lines.append(header)
            report_lines.append(separator)
            
            # 按0%噪声性能排序
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
                method_display = f"**{method}**" if any(mode in method for mode in ['deterministic', 'standard']) else method
                
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
            
            report_lines.append(f"- **🏆 最稳定方法**: `{most_stable}` (稳定性: {stability_scores[most_stable]:.1f}%)")
            report_lines.append(f"- **⚠️ 最不稳定方法**: `{least_stable}` (稳定性: {stability_scores[least_stable]:.1f}%)")
            
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
        report_lines.append("- **噪声设置**: 验证集包含人工噪声，测试集为纯净数据")
        report_lines.append("- **统一配置**: 所有神经网络方法使用相同的超参数确保公平比较")
        report_lines.append("")
        report_lines.append(f"📊 **生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 保存报告
        report_path = os.path.join(self.config.OUTPUT_DIR, 'extended_experiment_summary.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(report_lines))
        
        if self.config.VERBOSE:
            print(f"📋 实验总结报告已保存: {report_path}")
        
        return report_lines


def main():
    """主函数 - 运行扩展版教程"""
    print("🚀 扩展版加州房价回归教程")
    print("=" * 60)
    
    # 创建教程实例
    tutorial = ExtendedCaliforniaHousingTutorial()
    
    # 1. 加载和分析数据
    tutorial.load_and_analyze_data()
    
    # 2. 运行核心性能测试
    core_results = tutorial.run_core_performance_test()
    
    # 3. 运行鲁棒性测试
    robustness_results = tutorial.run_robustness_test()
    
    # 4. 生成总结报告
    tutorial.generate_summary_report()
    
    if tutorial.config.VERBOSE:
        print("\\n🎉 扩展版教程运行完成！")
        print(f"📁 所有结果已保存到: {tutorial.config.OUTPUT_DIR}")
        print("\\n主要输出文件:")
        print("- extended_data_analysis.png: 数据分析图表")
        print("- core_performance_comparison.png: 核心性能对比图表")
        if tutorial.config.RUN_ROBUSTNESS_TEST:
            print("- extended_robustness_analysis.png: 鲁棒性分析图表")
        print("- extended_experiment_summary.md: 实验总结报告")


if __name__ == "__main__":
    main()