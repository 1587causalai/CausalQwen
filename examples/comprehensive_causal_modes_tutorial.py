#!/usr/bin/env python3
"""
🏠 全面CausalEngine模式教程：加州房价预测
=========================================

这个教程演示所有5种CausalEngine推理模式在真实世界回归任务中的性能表现。

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

实验设计说明
==================================================================
本脚本专注于全面评估CausalEngine的4种推理模式，旨在揭示不同因果推理策略
在真实回归任务上的性能特点和适用场景。

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


class ComprehensiveTutorialConfig:
    """
    全面教程配置类 - 测试所有CausalEngine模式
    
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
    NN_PATIENCE = 200                            # 早停patience
    NN_TOLERANCE = 1e-4                          # 早停tolerance
    # =========================================================================
    
    # 🤖 CausalEngine参数 - 测试4种有效模式（移除sampling）
    CAUSAL_MODES = ['deterministic', 'exogenous', 'endogenous', 'standard']
    CAUSAL_HIDDEN_SIZES = NN_HIDDEN_SIZES        # 使用统一神经网络配置
    CAUSAL_MAX_EPOCHS = NN_MAX_EPOCHS            # 使用统一神经网络配置
    CAUSAL_LR = NN_LEARNING_RATE                 # 使用统一神经网络配置
    CAUSAL_PATIENCE = NN_PATIENCE                # 使用统一神经网络配置
    CAUSAL_TOL = NN_TOLERANCE                    # 使用统一神经网络配置
    CAUSAL_GAMMA_INIT = 1.0                      # gamma初始化
    CAUSAL_B_NOISE_INIT = 1.0                    # b_noise初始化
    CAUSAL_B_NOISE_TRAINABLE = True              # b_noise是否可训练
    
    # 🧠 传统神经网络方法参数 - 使用统一配置
    SKLEARN_HIDDEN_LAYERS = NN_HIDDEN_SIZES      # 使用统一神经网络配置
    SKLEARN_MAX_ITER = NN_MAX_EPOCHS             # 使用统一神经网络配置
    SKLEARN_LR = NN_LEARNING_RATE                # 使用统一神经网络配置
    
    PYTORCH_EPOCHS = NN_MAX_EPOCHS               # 使用统一神经网络配置
    PYTORCH_LR = NN_LEARNING_RATE                # 使用统一神经网络配置
    PYTORCH_PATIENCE = NN_PATIENCE               # 使用统一神经网络配置
    
    # 📊 实验参数
    ANOMALY_RATIO = 0.4                         # 标签异常比例 (核心实验默认值: 40%噪声挑战)
    SAVE_PLOTS = True                            # 是否保存图表
    VERBOSE = True                               # 是否显示详细输出
    
    # 🎯 基准方法配置 - 包含9种传统方法（与sklearn-style版本完全一致）
    BASELINE_METHODS = [
        'sklearn_mlp',          # sklearn MLPRegressor
        'pytorch_mlp',          # PyTorch MLP
        'random_forest',        # 随机森林
        'xgboost',              # XGBoost
        'lightgbm',             # LightGBM  
        'catboost',             # CatBoost - 强力梯度提升
        'mlp_huber',            # MLP Huber（Huber损失稳健回归）
        'mlp_pinball_median',   # MLP Pinball Median（中位数回归）
        'mlp_cauchy'            # MLP Cauchy（Cauchy损失稳健回归）
    ]
    
    # 或者使用预定义组合：
    # BASELINE_METHODS = 'group:comprehensive'  # 使用预定义的comprehensive组合
    # BASELINE_METHODS = 'group:competitive'    # 使用预定义的competitive组合
    
    # 📈 可视化参数
    FIGURE_DPI = 300                             # 图表分辨率
    FIGURE_SIZE_ANALYSIS = (16, 12)              # 数据分析图表大小
    FIGURE_SIZE_PERFORMANCE = (24, 20)           # 性能对比图表大小（更大以容纳13个方法）
    FIGURE_SIZE_MODES_COMPARISON = (18, 12)      # CausalEngine模式对比图表大小
    
    # 📁 输出目录参数
    OUTPUT_DIR = "results/comprehensive_causal_modes"


class ComprehensiveCausalModesTutorial:
    """
    全面CausalEngine模式教程类
    
    演示所有5种CausalEngine推理模式在真实世界回归任务中的性能特点
    """
    
    def __init__(self, config=None):
        self.config = config if config is not None else ComprehensiveTutorialConfig()
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
            print("🏠 全面CausalEngine模式教程 - 加州房价预测")
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
        
        return self.X, self.y
    
    def visualize_data(self, save_plots=None):
        """数据可视化分析"""
        if save_plots is None:
            save_plots = self.config.SAVE_PLOTS
            
        print("\n📈 数据分布分析")
        print("-" * 30)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_ANALYSIS)
        fig.suptitle('California Housing Dataset Analysis - Comprehensive CausalEngine Modes Tutorial', fontsize=16, fontweight='bold')
        
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
            output_path = self._get_output_path('comprehensive_data_analysis.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 数据分析图表已保存为 {output_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 数据统计摘要
        print("\n📋 数据统计摘要:")
        print(f"  - 最相关特征: {most_corr_feature} (相关系数: {corr_matrix.loc[most_corr_feature, 'MedHouseVal']:.3f})")
        print(f"  - 异常值检测: {np.sum(np.abs(self.y - self.y.mean()) > 3 * self.y.std())} 个潜在异常值")
        print(f"  - 数据完整性: 无缺失值" if not np.any(np.isnan(self.X)) else "  - 警告: 存在缺失值")
    
    def run_comprehensive_benchmark(self, test_size=None, val_size=None, anomaly_ratio=None, verbose=None):
        """运行全面的基准测试 - 包含所有5种CausalEngine模式"""
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
            print("\n🚀 开始全面基准测试 - 测试所有5种CausalEngine模式")
            print("=" * 80)
            print(f"🔧 实验配置:")
            print(f"   - 测试集比例: {test_size:.1%}")
            print(f"   - 验证集比例: {val_size:.1%}")
            print(f"   - 异常标签比例: {anomaly_ratio:.1%}")
            print(f"   - 随机种子: {self.config.RANDOM_STATE}")
            print(f"   - CausalEngine模式: {', '.join(self.config.CAUSAL_MODES)}")
            print(f"   - CausalEngine网络: {self.config.CAUSAL_HIDDEN_SIZES}")
            print(f"   - 最大训练轮数: {self.config.CAUSAL_MAX_EPOCHS}")
            print(f"   - 早停patience: {self.config.CAUSAL_PATIENCE}")
            baseline_count = len(self.config.BASELINE_METHODS)
            total_methods = len(self.config.CAUSAL_MODES) + baseline_count
            print(f"   - 基准方法: {self.config.BASELINE_METHODS}")
            print(f"   - 总计对比方法: {total_methods} 种 ({len(self.config.CAUSAL_MODES)}种CausalEngine + {baseline_count}种传统)")
        
        # 使用基准测试模块
        benchmark = BaselineBenchmark()
        
        # 打印可用方法报告
        if verbose:
            benchmark.print_method_availability()
        
        # 运行基准测试
        self.results = benchmark.compare_models(
            X=self.X,
            y=self.y,
            task_type='regression',
            baseline_methods=self.config.BASELINE_METHODS,  # 新增：使用配置的基准方法
            test_size=test_size,
            val_size=val_size,
            anomaly_ratio=anomaly_ratio,
            random_state=self.config.RANDOM_STATE,
            verbose=verbose,
            # CausalEngine参数 - 包含所有5种模式
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
            print(f"\n📊 全面基准测试结果 (异常比例: {anomaly_ratio:.0%})")
            benchmark.print_results(self.results, 'regression')
        
        return self.results
    
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
            print(f"\n📊 模式特点分析:")
            print("-" * 30)
            
            best_mode = sorted_causal[0][0]
            worst_mode = sorted_causal[-1][0]
            performance_gap = sorted_causal[-1][1] - sorted_causal[0][1]  # 最差 - 最好 (因为MdAE越小越好)
            
            print(f"   🏆 最佳模式: {best_mode} (MdAE = {sorted_causal[0][1]:.3f})")
            print(f"   📉 最弱模式: {worst_mode} (MdAE = {sorted_causal[-1][1]:.3f})")
            print(f"   📏 性能差距: {performance_gap:.3f} ({performance_gap/sorted_causal[0][1]*100:.1f}%)")
            
            # 与传统方法比较（基于MdAE）
            if traditional_results:
                print(f"\n🆚 CausalEngine vs 传统方法:")
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
            # 标准版：包含9种核心方法（除了robust MLP方法）
            robust_mlp_methods = ['mlp_huber', 'mlp_pinball_median', 'mlp_cauchy']  # 排除的robust MLP方法
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
        
        # 创建子图 - 2x2布局展示4个指标
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE_PERFORMANCE)
        title_suffix = " (Extended with All Methods)" if extended else ""
        fig.suptitle(f'Comprehensive CausalEngine Modes vs Traditional Methods{title_suffix}\nCalifornia Housing Performance (40% Label Noise)', 
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
                    method_labels.append(f'CausalEngine\n({method})')
                elif 'sklearn' in method.lower() or method == 'sklearn':
                    method_labels.append('sklearn\nMLP')
                elif 'pytorch' in method.lower() or method == 'pytorch':
                    method_labels.append('PyTorch\nMLP')
                else:
                    # 其他传统方法，简化显示名称
                    display_name = method.replace('_', ' ').title()
                    if len(display_name) > 12:
                        # 长名称分行显示
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
            filename = 'comprehensive_performance_comparison_extended.png' if extended else 'comprehensive_performance_comparison.png'
            output_path = self._get_output_path(filename)
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 {chart_type}全面性能图表已保存为 {output_path}")
        
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
        fig.suptitle('CausalEngine Modes Detailed Comparison', fontsize=16, fontweight='bold')
        
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
            output_path = self._get_output_path('causal_modes_detailed_comparison.png')
            plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            print(f"📊 CausalEngine模式对比图表已保存为 {output_path}")
        
        plt.close()
    
    def print_comprehensive_summary(self):
        """打印全面的总结报告"""
        if not self.results:
            print("❌ 请先运行基准测试")
            return
        
        print("\n📋 全面实验总结报告")
        print("=" * 80)
        
        # 统计信息
        total_methods = len(self.results)
        causal_methods = len([m for m in self.results if m in self.config.CAUSAL_MODES])
        traditional_methods = len([m for m in self.results if m in ['sklearn', 'pytorch']])
        
        print(f"🔢 实验规模:")
        print(f"   - 总计测试方法: {total_methods}")
        print(f"   - CausalEngine模式: {causal_methods}")
        print(f"   - 传统方法: {traditional_methods}")
        print(f"   - 数据集大小: {self.X.shape[0]:,} 样本 × {self.X.shape[1]} 特征")
        print(f"   - 异常标签比例: {self.config.ANOMALY_RATIO:.1%}")
        
        # 性能排名（按MdAE分数，越小越好）
        print(f"\n🏆 总体性能排名 (按MdAE分数):")
        print("-" * 50)
        
        all_mdae_scores = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items()]
        all_mdae_scores.sort(key=lambda x: x[1])  # 升序排列，MdAE越小越好
        
        for i, (method, mdae) in enumerate(all_mdae_scores, 1):
            method_type = "CausalEngine" if method in self.config.CAUSAL_MODES else "Traditional"
            r2 = self.results[method]['test']['R²']
            print(f"   {i:2d}. {method:<15} ({method_type:<12}) - MdAE: {mdae:.3f}, R²: {r2:.4f}")
        
        # CausalEngine优势分析（基于MdAE）
        print(f"\n🎯 CausalEngine模式分析:")
        print("-" * 40)
        
        causal_results = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items() 
                         if method in self.config.CAUSAL_MODES]
        traditional_results = [(method, metrics['test']['MdAE']) for method, metrics in self.results.items() 
                              if method in ['sklearn', 'pytorch']]
        
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
        print(f"\n💡 关键发现:")
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
    print("🏠 全面CausalEngine模式教程")
    print("🎯 目标：测试所有5种CausalEngine推理模式在真实世界回归任务中的表现")
    print("=" * 90)
    
    # 创建配置实例
    config = ComprehensiveTutorialConfig()
    
    print(f"🔧 当前配置:")
    print(f"   - CausalEngine模式: {', '.join(config.CAUSAL_MODES)} (共{len(config.CAUSAL_MODES)}种)")
    print(f"   - 网络架构: {config.CAUSAL_HIDDEN_SIZES}")
    print(f"   - 最大轮数: {config.CAUSAL_MAX_EPOCHS}")
    print(f"   - 早停patience: {config.CAUSAL_PATIENCE}")
    print(f"   - 异常比例: {config.ANOMALY_RATIO:.1%}")
    print(f"   - 总计对比方法: {len(config.CAUSAL_MODES) + len(config.BASELINE_METHODS)} 种")
    print(f"   - 输出目录: {config.OUTPUT_DIR}/")
    print()
    
    # 创建教程实例
    tutorial = ComprehensiveCausalModesTutorial(config)
    
    # 1. 加载和探索数据
    tutorial.load_and_explore_data()
    
    # 2. 数据可视化
    tutorial.visualize_data()
    
    # 3. 运行全面基准测试 - 测试所有5种CausalEngine模式
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
    
    print("\n🎉 全面CausalEngine模式教程完成！")
    print("📋 实验总结:")
    print(f"   - 使用了真实世界的加州房价数据集 ({tutorial.X.shape[0]:,} 样本)")
    print(f"   - 测试了所有 {len(config.CAUSAL_MODES)} 种CausalEngine推理模式")
    print(f"   - 与 {len(config.BASELINE_METHODS)} 种传统方法进行了全面对比")
    print(f"   - 基准方法包括: {', '.join(config.BASELINE_METHODS[:3])}等")
    print(f"   - 在 {config.ANOMALY_RATIO:.0%} 标签噪声环境下验证了鲁棒性")
    print("   - 提供了详细的模式特点分析和可视化")
    
    print("\n📊 生成的文件:")
    if config.SAVE_PLOTS:
        print(f"   - {config.OUTPUT_DIR}/comprehensive_data_analysis.png                    (数据分析图)")
        print(f"   - {config.OUTPUT_DIR}/comprehensive_performance_comparison.png           (标准性能对比图)")
        print(f"   - {config.OUTPUT_DIR}/comprehensive_performance_comparison_extended.png  (扩展性能对比图)")
        print(f"   - {config.OUTPUT_DIR}/causal_modes_detailed_comparison.png               (CausalEngine模式专项对比图)")
    
    print("\n💡 提示：通过修改ComprehensiveTutorialConfig类来自定义实验参数！")
    print("🔬 下一步：可以尝试不同的数据集或调整模型参数来进一步验证CausalEngine的优越性")


if __name__ == "__main__":
    main()