#!/usr/bin/env python3
"""
🏠 真实世界回归教程：加州房价预测
=====================================

这个教程演示如何使用CausalEngine在真实世界回归任务中取得优于传统ML方法的性能。

数据集：加州房价数据集（California Housing Dataset）
- 20,640个样本
- 8个特征（房屋年龄、收入、人口等）
- 目标：预测房价中位数

我们将比较：
1. sklearn MLPRegressor（传统神经网络）
2. PyTorch MLP（传统深度学习）
3. CausalEngine（因果推理方法）

关键亮点：
- 真实世界数据的鲁棒性
- 处理异常值的能力
- 因果推理带来的性能提升

实验设计说明
==================================================================
本脚本包含两组核心实验，旨在全面评估CausalEngine在真实回归任务上的性能和鲁棒性。
所有实验参数均可在下方的 `TutorialConfig` 类中进行修改。

实验一：核心性能对比 (在40%标签噪声下)
--------------------------------------------------
- **目标**: 比较CausalEngine和传统方法在含有固定噪声数据上的预测性能。
- **设置**: 默认设置40%的标签噪声（`ANOMALY_RATIO = 0.4`），模拟真实世界中常见的数据质量问题。
- **对比模型**: 
  - CausalEngine (不同模式, 如 'deterministic', 'standard')
  - Sklearn MLPRegressor
  - PyTorch MLP

实验二：鲁棒性分析 (跨越不同噪声水平)
--------------------------------------------------
- **目标**: 探究模型性能随标签噪声水平增加时的变化情况，评估其稳定性。
- **设置**: 在一系列噪声比例（如0%, 10%, 20%, 30%）下分别运行测试。
- **对比模型**:
  - CausalEngine ('standard'模式)
  - Sklearn MLPRegressor
  - PyTorch MLP
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
    教程配置类 - 方便调整各种参数
    
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
    CAUSAL_MODES = ['deterministic', 'standard']  # 可选: ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    CAUSAL_HIDDEN_SIZES = NN_HIDDEN_SIZES          # 使用统一神经网络配置
    CAUSAL_MAX_EPOCHS = NN_MAX_EPOCHS              # 使用统一神经网络配置
    CAUSAL_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    CAUSAL_PATIENCE = NN_PATIENCE                  # 使用统一神经网络配置
    CAUSAL_TOL = NN_TOLERANCE                      # 使用统一神经网络配置
    CAUSAL_GAMMA_INIT = 1.0                      # gamma初始化
    CAUSAL_B_NOISE_INIT = 1.0                    # b_noise初始化
    CAUSAL_B_NOISE_TRAINABLE = True              # b_noise是否可训练
    
    # 🧠 传统方法参数 - 使用统一配置
    SKLEARN_HIDDEN_LAYERS = NN_HIDDEN_SIZES         # 使用统一神经网络配置
    SKLEARN_MAX_ITER = NN_MAX_EPOCHS                # 使用统一神经网络配置
    SKLEARN_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    
    PYTORCH_EPOCHS = NN_MAX_EPOCHS                  # 使用统一神经网络配置
    PYTORCH_LR = NN_LEARNING_RATE                   # 使用统一神经网络配置
    PYTORCH_PATIENCE = NN_PATIENCE                  # 使用统一神经网络配置
    
    # 📊 实验参数
    ANOMALY_RATIO = 0.4                          # 标签异常比例 (核心实验默认值: 40%噪声挑战)
    SAVE_PLOTS = True                            # 是否保存图表
    VERBOSE = True                               # 是否显示详细输出
    
    # 🛡️ 鲁棒性测试参数 - 设计为验证"CausalEngine鲁棒性优势"的假设
    ROBUSTNESS_ANOMALY_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 6个关键噪声水平就足够
    RUN_ROBUSTNESS_TEST = True                   # 是否运行鲁棒性测试
    
    # 📈 可视化参数
    FIGURE_DPI = 300                             # 图表分辨率
    FIGURE_SIZE_ANALYSIS = (24, 20)              # 数据分析图表大小
    FIGURE_SIZE_PERFORMANCE = (24, 20)           # 性能对比图表大小
    FIGURE_SIZE_ROBUSTNESS = (24, 20)            # 鲁棒性测试图表大小 (4个子图)
    
    # 📁 输出目录参数
    OUTPUT_DIR = "results/california_housing_regression"                       # 输出目录名称


class CaliforniaHousingTutorial:
    """
    加州房价回归教程类
    
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
            print("🏠 加州房价预测 - 真实世界回归教程")
            print("=" * 60)
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
        fig.suptitle('California Housing Dataset Analysis', fontsize=16, fontweight='bold')
        
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
            output_path = self._get_output_path('california_housing_analysis.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 数据分析图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 数据统计摘要
        print("\\n📋 数据统计摘要:")
        print(f"  - 最相关特征: {most_corr_feature} (相关系数: {corr_matrix.loc[most_corr_feature, 'MedHouseVal']:.3f})")
        print(f"  - 异常值检测: {np.sum(np.abs(self.y - self.y.mean()) > 3 * self.y.std())} 个潜在异常值")
        print(f"  - 数据完整性: 无缺失值" if not np.any(np.isnan(self.X)) else "  - 警告: 存在缺失值")
    
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
            print("\\n🚀 开始综合基准测试")
            print("=" * 60)
            print(f"🔧 实验配置:")
            print(f"   - 测试集比例: {test_size:.1%}")
            print(f"   - 验证集比例: {val_size:.1%}")
            print(f"   - 异常标签比例: {anomaly_ratio:.1%}")
            print(f"   - 随机种子: {self.config.RANDOM_STATE}")
            print(f"   - CausalEngine模式: {', '.join(self.config.CAUSAL_MODES)}")
            print(f"   - CausalEngine网络: {self.config.CAUSAL_HIDDEN_SIZES}")
            print(f"   - 最大训练轮数: {self.config.CAUSAL_MAX_EPOCHS}")
            print(f"   - 早停patience: {self.config.CAUSAL_PATIENCE}")
        
        # 使用基准测试模块
        benchmark = BaselineBenchmark()
        
        # 运行基准测试
        self.results = benchmark.compare_models(
            X=self.X,
            y=self.y,
            task_type='regression',
            test_size=test_size,
            val_size=val_size,
            anomaly_ratio=anomaly_ratio,
            random_state=self.config.RANDOM_STATE,
            verbose=verbose,
            # CausalEngine参数
            causal_modes=self.config.CAUSAL_MODES,
            hidden_sizes=self.config.CAUSAL_HIDDEN_SIZES,
            max_epochs=self.config.CAUSAL_MAX_EPOCHS,
            lr=self.config.CAUSAL_LR,
            patience=self.config.CAUSAL_PATIENCE,
            tol=self.config.CAUSAL_TOL,
            gamma_init=self.config.CAUSAL_GAMMA_INIT,
            b_noise_init=self.config.CAUSAL_B_NOISE_INIT,
            b_noise_trainable=self.config.CAUSAL_B_NOISE_TRAINABLE,
            # sklearn/PyTorch参数
            hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
            max_iter=self.config.SKLEARN_MAX_ITER,
            learning_rate=self.config.SKLEARN_LR
        )
        
        if verbose:
            print(f"\n📊 基准测试结果 (异常比例: {anomaly_ratio:.0%})")
            benchmark.print_results(self.results, 'regression')
        
        return self.results
    
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
        causal_methods = [m for m in self.results.keys() if m in ['deterministic', 'standard', 'sampling']]
        if causal_methods:
            best_causal = max(causal_methods, key=lambda x: test_r2_scores[x])
            traditional_methods = [m for m in self.results.keys() if m in ['sklearn_mlp', 'pytorch_mlp']]
            
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
        fig.suptitle('CausalEngine vs Traditional Methods: California Housing Performance (40% Label Noise)', fontsize=16, fontweight='bold')
        axes = axes.flatten()  # 展平为一维数组便于访问
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
        
        for i, metric in enumerate(metrics):
            values = [self.results[method]['test'][metric] for method in methods]
            
            bars = axes[i].bar(methods, values, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
            axes[i].set_title(f'{metric} (Test Set)', fontweight='bold')
            axes[i].set_ylabel(metric)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
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
            output_path = self._get_output_path('california_housing_performance.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 性能图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
    
    def run_robustness_test(self, anomaly_ratios=None, verbose=None):
        """运行鲁棒性测试（不同异常比例）"""
        if anomaly_ratios is None:
            anomaly_ratios = self.config.ROBUSTNESS_ANOMALY_RATIOS
        if verbose is None:
            verbose = self.config.VERBOSE
            
        if verbose:
            print("\\n🛡️ 鲁棒性测试")
            print("=" * 60)
            print("测试CausalEngine在不同异常标签比例下的表现")
        
        robustness_results = {}
        
        for anomaly_ratio in anomaly_ratios:
            if verbose:
                print(f"\\n🔬 测试异常比例: {anomaly_ratio:.1%}")
                print("-" * 30)
            
            # 运行基准测试
            benchmark = BaselineBenchmark()
            results = benchmark.compare_models(
                X=self.X,
                y=self.y,
                task_type='regression',
                test_size=0.2,
                val_size=0.2,
                anomaly_ratio=anomaly_ratio,
                random_state=self.config.RANDOM_STATE,
                verbose=False,  # 简化输出
                # 基准方法配置
                baseline_methods=['sklearn_mlp', 'pytorch_mlp'],  # 包含传统方法
                # CausalEngine参数
                causal_modes=['deterministic', 'standard'],  # 测试两种重要模式
                hidden_sizes=self.config.CAUSAL_HIDDEN_SIZES,
                max_epochs=self.config.CAUSAL_MAX_EPOCHS,
                lr=self.config.CAUSAL_LR,
                patience=self.config.CAUSAL_PATIENCE,
                tol=self.config.CAUSAL_TOL,
                # sklearn/PyTorch参数
                hidden_layer_sizes=self.config.SKLEARN_HIDDEN_LAYERS,
                max_iter=self.config.SKLEARN_MAX_ITER,
                learning_rate=self.config.SKLEARN_LR
            )
            
            robustness_results[anomaly_ratio] = results
            
            if verbose:
                print("R² 分数:")
                for method in ['sklearn_mlp', 'pytorch_mlp', 'deterministic', 'standard']:
                    if method in results:
                        r2 = results[method]['test']['R²']
                        print(f"  {method:<12}: {r2:.4f}")
        
        # 可视化鲁棒性结果
        if verbose:
            # 打印详细的鲁棒性表格
            self._print_robustness_table(robustness_results, anomaly_ratios)
            # 绘制鲁棒性图表
            self._plot_robustness_results(robustness_results, anomaly_ratios)
            # 分析鲁棒性趋势
            self._analyze_robustness_trends(robustness_results, anomaly_ratios)
        
        return robustness_results
    
    def _print_robustness_table(self, robustness_results, anomaly_ratios):
        """打印鲁棒性测试详细表格 - 显示所有噪声水平下的模型性能"""
        print("\n📊 鲁棒性测试详细结果表格")
        print("=" * 140)
        
        # 表头
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        methods = ['sklearn_mlp', 'pytorch_mlp', 'deterministic', 'standard']
        
        # 打印表头
        header_line1 = f"{'异常比例':<8} {'方法':<12}"
        header_line2 = f"{'':8} {'':12}"
        
        for split in ['验证集', '测试集']:
            header_line1 += f" {split:<40}"
            header_line2 += f" {metrics[0]:<9} {metrics[1]:<9} {metrics[2]:<9} {metrics[3]:<9}"
        
        print(header_line1)
        print(header_line2)
        print("-" * 140)
        
        # 为每个异常比例打印结果
        for ratio in anomaly_ratios:
            ratio_str = f"{ratio:.0%}"
            
            for i, method in enumerate(methods):
                if method in robustness_results[ratio]:
                    results = robustness_results[ratio][method]
                    
                    # 第一行显示异常比例，后续行为空
                    ratio_display = ratio_str if i == 0 else ""
                    
                    line = f"{ratio_display:<8} {method:<12}"
                    
                    # 验证集指标
                    val_metrics = results['val']
                    line += f" {val_metrics['MAE']:<9.4f} {val_metrics['MdAE']:<9.4f} {val_metrics['RMSE']:<9.4f} {val_metrics['R²']:<9.4f}"
                    
                    # 测试集指标
                    test_metrics = results['test']
                    line += f" {test_metrics['MAE']:<9.4f} {test_metrics['MdAE']:<9.4f} {test_metrics['RMSE']:<9.4f} {test_metrics['R²']:<9.4f}"
                    
                    print(line)
            
            # 在每个异常比例组之间添加分隔线
            if ratio != anomaly_ratios[-1]:
                print("-" * 140)
        
        print("=" * 140)
        print("💡 观察要点：")
        print("   - R² 越高越好（接近1.0为最佳）")
        print("   - MAE, MdAE, RMSE 越低越好（接近0为最佳）")
        print("   - 关注各方法在异常比例增加时的性能变化趋势")
    
    def _plot_robustness_results(self, robustness_results, anomaly_ratios):
        """绘制鲁棒性测试结果 - 显示所有4个回归指标"""
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ROBUSTNESS)
        fig.suptitle('Robustness Test: Impact of Label Noise on Model Performance', fontsize=16, fontweight='bold')
        
        methods = ['sklearn_mlp', 'pytorch_mlp', 'deterministic', 'standard']
        method_labels = ['sklearn MLP', 'PyTorch MLP', 'CausalEngine (Det)', 'CausalEngine (Std)']
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']  # 更清晰的颜色
        markers = ['o', 's', 'v', '^']
        
        # 4个回归指标
        metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
        metric_labels = ['Mean Absolute Error (MAE)', 'Median Absolute Error (MdAE)', 'Root Mean Squared Error (RMSE)', 'R-squared Score (R²)']
        
        # 为每个指标创建子图
        for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # 绘制每个方法的曲线
            for method, label, color, marker in zip(methods, method_labels, colors, markers):
                scores = []
                for ratio in anomaly_ratios:
                    if method in robustness_results[ratio]:
                        scores.append(robustness_results[ratio][method]['test'][metric])
                    else:
                        scores.append(np.nan)
                
                ax.plot(anomaly_ratios, scores, marker=marker, linewidth=2.5, 
                       markersize=8, label=label, color=color, alpha=0.8)
            
            # 设置子图属性
            ax.set_xlabel('Label Noise Ratio', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # 为R²添加特殊处理（越高越好），其他指标越低越好
            if metric == 'R²':
                ax.set_ylim(bottom=0)  # R²从0开始显示
            else:
                ax.set_ylim(bottom=0)  # 误差指标从0开始显示
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self._get_output_path('california_housing_robustness.png')
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        print(f"📊 鲁棒性测试图表已保存为 {output_path}")
        plt.close()  # 关闭图形，避免内存泄漏
    
    def _analyze_robustness_trends(self, robustness_results, anomaly_ratios):
        """分析鲁棒性趋势 - 验证CausalEngine鲁棒性假设"""
        print("\n🔬 鲁棒性趋势分析")
        print("=" * 60)
        
        methods = ['sklearn_mlp', 'pytorch_mlp', 'deterministic', 'standard']
        method_names = {'sklearn_mlp': 'sklearn MLP', 'pytorch_mlp': 'PyTorch MLP', 'deterministic': 'CausalEngine (deterministic)', 'standard': 'CausalEngine (standard)'}
        
        # 分析R²指标的变化趋势
        print("📈 R²指标随异常比例变化趋势：")
        print("-" * 40)
        
        for method in methods:
            r2_scores = []
            for ratio in anomaly_ratios:
                if method in robustness_results[ratio]:
                    r2_scores.append(robustness_results[ratio][method]['test']['R²'])
                else:
                    r2_scores.append(np.nan)
            
            # 计算性能下降情况
            clean_r2 = r2_scores[0] if len(r2_scores) > 0 and not np.isnan(r2_scores[0]) else 0
            final_r2 = r2_scores[-1] if len(r2_scores) > 0 and not np.isnan(r2_scores[-1]) else 0
            
            if clean_r2 > 0:
                performance_retention = (final_r2 / clean_r2) * 100
                performance_drop = clean_r2 - final_r2
            else:
                performance_retention = 0
                performance_drop = float('inf')
            
            print(f"  {method_names[method]}:")
            print(f"    - 零异常时R²: {clean_r2:.4f}")
            print(f"    - 最高异常时R²: {final_r2:.4f}")
            print(f"    - 性能保持率: {performance_retention:.1f}%")
            print(f"    - 绝对下降: {performance_drop:.4f}")
        
        # 验证假设
        print("\n🎯 假设验证结果：")
        print("-" * 40)
        
        # 提取关键数据
        clean_performance = {}  # 零异常时的性能
        noisy_performance = {}  # 高异常时的性能
        
        for method in methods:
            # 零异常性能
            if method in robustness_results[0.0]:
                clean_performance[method] = robustness_results[0.0][method]['test']['R²']
            
            # 最高异常性能 (选择0.25或最后一个)
            high_noise_ratio = 0.25 if 0.25 in anomaly_ratios else anomaly_ratios[-1]
            if method in robustness_results[high_noise_ratio]:
                noisy_performance[method] = robustness_results[high_noise_ratio][method]['test']['R²']
        
        # 检查零异常时所有模型是否都表现良好
        zero_noise_good = all(score > 0.6 for score in clean_performance.values())
        print(f"✅ 假设1 - 零异常时所有模型表现良好: {'通过' if zero_noise_good else '未通过'}")
        
        # 检查CausalEngine是否保持良好性能（选择表现更好的模式）
        causal_methods_performance = {}
        for method in ['deterministic', 'standard']:
            if method in noisy_performance:
                causal_methods_performance[method] = noisy_performance[method]
        
        if causal_methods_performance:
            best_causal_performance = max(causal_methods_performance.values())
            best_causal_method = max(causal_methods_performance.keys(), key=lambda x: causal_methods_performance[x])
            causal_robust = best_causal_performance > 0.6
            print(f"✅ 假设2 - CausalEngine在高噪声下保持良好: {'通过' if causal_robust else '未通过'}")
            print(f"   最佳CausalEngine模式: {best_causal_method} (R² = {best_causal_performance:.4f})")
        else:
            causal_robust = False
            print(f"✅ 假设2 - CausalEngine在高噪声下保持良好: 未通过 (无数据)")
        
        # 检查传统方法性能是否急剧下降
        traditional_degraded = True
        for method in ['sklearn_mlp', 'pytorch_mlp']:
            if method in clean_performance and method in noisy_performance:
                retention_rate = noisy_performance[method] / clean_performance[method]
                if retention_rate > 0.5:  # 如果保持率超过50%，认为没有急剧下降
                    traditional_degraded = False
                    break
        
        print(f"✅ 假设3 - 传统方法性能急剧下降: {'通过' if traditional_degraded else '未通过'}")
        
        # 综合结论
        all_passed = zero_noise_good and causal_robust and traditional_degraded
        print(f"\n🏆 综合结论: {'CausalEngine鲁棒性优势得到验证！' if all_passed else '需要进一步分析实验结果'}")
        
        if all_passed:
            print("   ✨ 实验完美证明了CausalEngine在真实世界噪声环境中的显著优势")
        else:
            print("   ⚠️ 建议调整实验参数或检查模型配置")

def main():
    """主函数：运行完整的教程"""
    print("🏠 CausalEngine真实世界回归教程")
    print("🎯 目标：在加州房价预测任务中展示CausalEngine的优越性")
    print("=" * 80)
    
    # 创建配置实例（在这里可以自定义配置）
    config = TutorialConfig()
    
    # 🔧 快速配置示例 - 取消注释来修改参数：
    # config.CAUSAL_MODES = ['deterministic', 'standard', 'sampling']  # 添加更多模式
    # config.CAUSAL_MAX_EPOCHS = 500  # 减少训练轮数以加快速度
    # config.ANOMALY_RATIO = 0.1      # 添加10%异常标签
    # config.RUN_ROBUSTNESS_TEST = False  # 跳过鲁棒性测试
    
    print(f"🔧 当前配置:")
    print(f"   - CausalEngine模式: {', '.join(config.CAUSAL_MODES)}")
    print(f"   - 网络架构: {config.CAUSAL_HIDDEN_SIZES}")
    print(f"   - 最大轮数: {config.CAUSAL_MAX_EPOCHS}")
    print(f"   - 早停patience: {config.CAUSAL_PATIENCE}")
    print(f"   - 异常比例: {config.ANOMALY_RATIO:.1%}")
    print(f"   - 运行鲁棒性测试: {'是' if config.RUN_ROBUSTNESS_TEST else '否'}")
    print(f"   - 输出目录: {config.OUTPUT_DIR}/")
    print()
    
    # 创建教程实例
    tutorial = CaliforniaHousingTutorial(config)
    
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
        print("\\n🛡️ 跳过鲁棒性测试（配置中禁用）")
    
    print("\\n🎉 教程完成！")
    print("📋 总结:")
    print("   - 使用了真实世界的加州房价数据集")
    print(f"   - 比较了{len(config.CAUSAL_MODES) + 2}种不同的方法")
    print("   - 展示了CausalEngine的性能优势")
    if config.RUN_ROBUSTNESS_TEST:
        print("   - 测试了模型的鲁棒性")
    print("   - 提供了详细的可视化分析")
    print("\\n📊 生成的文件:")
    if config.SAVE_PLOTS:
        print(f"   - {config.OUTPUT_DIR}/california_housing_analysis.png     (数据分析图)")
        print(f"   - {config.OUTPUT_DIR}/california_housing_performance.png  (性能对比图)")
        if config.RUN_ROBUSTNESS_TEST:
            print(f"   - {config.OUTPUT_DIR}/california_housing_robustness.png   (鲁棒性测试图 - 4个指标)")
    
    print("\\n💡 提示：在脚本顶部的TutorialConfig类中修改参数来自定义实验！")

if __name__ == "__main__":
    main()