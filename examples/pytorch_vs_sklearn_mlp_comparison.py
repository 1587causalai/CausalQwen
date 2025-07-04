#!/usr/bin/env python3
"""
PyTorch vs Sklearn MLP 对比脚本 - 2024重构版

本脚本实现了三个版本的 MLP:
1. 从零开始手动实现的 PyTorch 版本
2. Sklearn 的 MLPRegressor
3. Causal-Sklearn 库中封装的 MLPPytorchRegressor

🎯 目标：
    - 在含有标签噪声的真实世界数据集上，公平地比较三种MLP实现的性能。
    - 采用科学的数据处理流程（先标准化，后加噪），避免数据泄露。
    - 多次运行实验以分析不同实现的性能稳定性和固有方差。

✨ 借鉴 comprehensive_causal_modes_tutorial_sklearn_style.py 的优点：
    1. 引入配置类（ComparisonConfig），集中管理所有参数。
    2. 采用面向对象的结构（MLPComparisonTutorial），封装整个实验流程。
    3. 纠正数据处理逻辑：先在干净数据上标准化，再对标准化后的训练标签注入噪声。
    4. 增强了日志输出，使实验流程更清晰。
    5. 统一的输出目录管理。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import time
import warnings
import os
import sys

# 添加项目根目录到Python路径，以便导入causal_sklearn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from causal_sklearn.regressor import MLPPytorchRegressor
from causal_sklearn.data_processing import inject_shuffle_noise

warnings.filterwarnings('ignore')
plt.switch_backend('Agg') # 设置matplotlib后端，避免弹出窗口


class ComparisonConfig:
    """
    MLP对比实验配置类
    🔧 在这里修改参数来自定义实验设置！
    """
    # 实验控制
    N_RUNS = 5              # 实验运行次数
    BASE_SEED = 42          # 基础随机种子
    
    # 数据集与预处理
    TEST_SIZE = 0.2         # 测试集比例
    ANOMALY_RATIO = 0.25    # 训练集标签噪声比例
    
    # 统一神经网络配置 - 所有模型使用相同参数以确保公平
    MLP_HIDDEN_SIZES = (100, 50)         # 隐藏层结构
    MLP_MAX_ITER = 1000                  # 最大迭代次数
    MLP_LEARNING_RATE = 0.001            # 学习率
    MLP_ALPHA = 0.0001                   # L2正则化系数
    
    # 统一早停配置
    EARLY_STOPPING_ENABLED = True
    VALIDATION_FRACTION = 0.1
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_TOL = 1e-4
    
    # 输出和可视化
    OUTPUT_DIR = "results/mlp_comparison"
    FIGURE_DPI = 300
    FIGURE_SIZE = (15, 18)


class PyTorchMLP(nn.Module):
    """手动实现的 PyTorch MLP"""
    
    def __init__(self, input_size, hidden_sizes, output_size=1, random_state=None):
        super(PyTorchMLP, self).__init__()
        if random_state is not None:
            torch.manual_seed(random_state)
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.n_iter_ = 0
        
    def fit(self, X, y, epochs, lr, alpha, early_stopping, validation_fraction, n_iter_no_change, tol, random_state):
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=alpha)
        criterion = nn.MSELoss()
        
        if early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=validation_fraction, random_state=random_state
            )
        else:
            X_train, y_train = X_tensor, y_tensor
            X_val, y_val = None, None
            
        self.train()
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None

        for epoch in range(epochs):
            outputs = self.network(X_train)
            loss = criterion(outputs, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if early_stopping and X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.network(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    
                    if val_loss < best_val_loss - tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = self.state_dict().copy()
                    else:
                        no_improve_count += 1
                self.train()
                        
                if no_improve_count >= n_iter_no_change:
                    self.n_iter_ = epoch + 1
                    break
        
        if self.n_iter_ == 0:
            self.n_iter_ = epoch + 1

        if early_stopping and best_state_dict is not None:
            self.load_state_dict(best_state_dict)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.network(X_tensor).squeeze().cpu().numpy()
        return predictions


class MLPComparisonTutorial:
    """封装 MLP 对比实验流程的类"""

    def __init__(self, config):
        self.config = config
        self.X_train_orig, self.X_test_orig, self.y_train_orig, self.y_test_orig = None, None, None, None
        
        self.results = {
            'pytorch': [],
            'sklearn': [],
            'causal_sklearn': []
        }
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        if not os.path.exists(self.config.OUTPUT_DIR):
            os.makedirs(self.config.OUTPUT_DIR)
            print(f"📁 创建输出目录: {self.config.OUTPUT_DIR}")

    def _get_output_path(self, filename):
        return os.path.join(self.config.OUTPUT_DIR, filename)

    def load_and_split_data(self):
        print("\n📊 步骤 1: 加载 California Housing 数据集并分割")
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        print(f"   - 数据集加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        self.X_train_orig, self.X_test_orig, self.y_train_orig, self.y_test_orig = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.BASE_SEED
        )
        print(f"   - 数据分割完成: 训练集 {self.X_train_orig.shape[0]} | 测试集 {self.X_test_orig.shape[0]}")

    def _prepare_data_for_run(self):
        """
        准备单次运行所需的数据。
        此函数实现了科学的数据处理流程：先标准化，后注入噪声。
        """
        # 1. 特征标准化 (Scaler在干净的原始训练集上fit)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(self.X_train_orig)
        X_test_scaled = scaler_X.transform(self.X_test_orig)
        
        # 2. 目标标准化 (Scaler在干净的原始训练集上fit)
        scaler_y = StandardScaler()
        y_train_clean_scaled = scaler_y.fit_transform(self.y_train_orig.reshape(-1, 1)).flatten()
        
        # 3. 在标准化后的训练标签上注入噪声
        y_train_noisy_scaled, noise_indices = inject_shuffle_noise(
            y_train_clean_scaled, 
            noise_ratio=self.config.ANOMALY_RATIO,
            random_state=self.config.BASE_SEED
        )
        
        return {
            "X_train": X_train_scaled,
            "y_train": y_train_noisy_scaled,
            "X_test": X_test_scaled,
            "y_test_orig": self.y_test_orig,
            "scaler_y": scaler_y,
            "noise_indices": noise_indices
        }

    def _run_pytorch_mlp(self, data, run_id, random_state):
        print(f"   - 正在运行手动 PyTorch MLP (随机种子: {random_state})")
        start_time = time.time()
        
        model = PyTorchMLP(
            input_size=data['X_train'].shape[1], 
            hidden_sizes=self.config.MLP_HIDDEN_SIZES, 
            random_state=random_state
        )
        
        model.fit(
            data['X_train'], data['y_train'], 
            epochs=self.config.MLP_MAX_ITER, 
            lr=self.config.MLP_LEARNING_RATE,
            alpha=self.config.MLP_ALPHA,
            early_stopping=self.config.EARLY_STOPPING_ENABLED,
            validation_fraction=self.config.VALIDATION_FRACTION,
            n_iter_no_change=self.config.EARLY_STOPPING_PATIENCE,
            tol=self.config.EARLY_STOPPING_TOL,
            random_state=random_state
        )
        
        y_pred_scaled = model.predict(data['X_test'])
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'predictions_scaled': y_pred_scaled,
            'n_iter': model.n_iter_
        }

    def _run_sklearn_mlp(self, data, run_id, random_state):
        print(f"   - 正在运行 Sklearn MLP (随机种子: {random_state})")
        start_time = time.time()
        
        model_params = {
            'hidden_layer_sizes': self.config.MLP_HIDDEN_SIZES,
            'max_iter': self.config.MLP_MAX_ITER,
            'learning_rate_init': self.config.MLP_LEARNING_RATE,
            'alpha': self.config.MLP_ALPHA,
            'random_state': random_state,
            'solver': 'adam',
            'batch_size': 'auto' # Sklearn默认
        }
        if self.config.EARLY_STOPPING_ENABLED:
            model_params.update({
                'early_stopping': True,
                'validation_fraction': self.config.VALIDATION_FRACTION,
                'n_iter_no_change': self.config.EARLY_STOPPING_PATIENCE,
                'tol': self.config.EARLY_STOPPING_TOL
            })

        model = MLPRegressor(**model_params)
        model.fit(data['X_train'], data['y_train'])
        y_pred_scaled = model.predict(data['X_test'])
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'predictions_scaled': y_pred_scaled,
            'n_iter': model.n_iter_
        }

    def _run_causal_sklearn_mlp(self, data, run_id, random_state):
        print(f"   - 正在运行 Causal-Sklearn MLP (随机种子: {random_state})")
        start_time = time.time()

        model_params = {
            'hidden_layer_sizes': self.config.MLP_HIDDEN_SIZES,
            'max_iter': self.config.MLP_MAX_ITER,
            'learning_rate': self.config.MLP_LEARNING_RATE,
            'alpha': self.config.MLP_ALPHA,
            'random_state': random_state,
            'batch_size': None, # None表示全批量
            'verbose': False
        }
        if self.config.EARLY_STOPPING_ENABLED:
            model_params.update({
                'early_stopping': True,
                'validation_fraction': self.config.VALIDATION_FRACTION,
                'n_iter_no_change': self.config.EARLY_STOPPING_PATIENCE,
                'tol': self.config.EARLY_STOPPING_TOL
            })

        model = MLPPytorchRegressor(**model_params)
        model.fit(data['X_train'], data['y_train'])
        y_pred_scaled = model.predict(data['X_test'])
        training_time = time.time() - start_time

        return {
            'training_time': training_time,
            'predictions_scaled': y_pred_scaled,
            'n_iter': model.n_iter_
        }

    def _evaluate_and_store_run(self, run_result, y_test_orig, scaler_y, result_container):
        y_pred_orig_scale = scaler_y.inverse_transform(run_result['predictions_scaled'].reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_test_orig, y_pred_orig_scale)
        
        metrics = {
            'mse': mse,
            'r2': r2_score(y_test_orig, y_pred_orig_scale),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig_scale),
            'mdae': median_absolute_error(y_test_orig, y_pred_orig_scale),
            'rmse': np.sqrt(mse),
            'training_time': run_result['training_time'],
            'n_iter': run_result['n_iter']
        }
        result_container.append(metrics)
        return metrics

    def run_comparison(self):
        print("\n📊 步骤 2: 准备数据 (采用科学的'先标准化、后加噪'策略)")
        data = self._prepare_data_for_run()
        print(f"   - 数据标准化完成 (基于干净数据)")
        print(f"   - 对训练集注入 {self.config.ANOMALY_RATIO:.0%} 的异常值 ({len(data['noise_indices'])}个样本)")
        
        print(f"\n🚀 步骤 3: 开始进行 {self.config.N_RUNS} 次对比实验")
        for i in range(self.config.N_RUNS):
            run_seed = self.config.BASE_SEED + i * 100
            print(f"\n--- 实验 {i+1}/{self.config.N_RUNS} (随机种子: {run_seed}) ---")
            
            # 运行模型
            pytorch_run = self._run_pytorch_mlp(data, i+1, run_seed)
            sklearn_run = self._run_sklearn_mlp(data, i+1, run_seed)
            causal_sklearn_run = self._run_causal_sklearn_mlp(data, i+1, run_seed)
            
            # 评估并存储结果
            p_metrics = self._evaluate_and_store_run(pytorch_run, data['y_test_orig'], data['scaler_y'], self.results['pytorch'])
            s_metrics = self._evaluate_and_store_run(sklearn_run, data['y_test_orig'], data['scaler_y'], self.results['sklearn'])
            c_metrics = self._evaluate_and_store_run(causal_sklearn_run, data['y_test_orig'], data['scaler_y'], self.results['causal_sklearn'])
            
            print(f"   PyTorch        - MdAE: {p_metrics['mdae']:.4f}, R2: {p_metrics['r2']:.4f}, Time: {p_metrics['training_time']:.2f}s, Iters: {p_metrics['n_iter']}")
            print(f"   Sklearn        - MdAE: {s_metrics['mdae']:.4f}, R2: {s_metrics['r2']:.4f}, Time: {s_metrics['training_time']:.2f}s, Iters: {s_metrics['n_iter']}")
            print(f"   Causal-Sklearn - MdAE: {c_metrics['mdae']:.4f}, R2: {c_metrics['r2']:.4f}, Time: {c_metrics['training_time']:.2f}s, Iters: {c_metrics['n_iter']}")
        
        print("\n✅ 所有实验运行完成！")

    def print_summary_stats(self):
        print("\n" + "=" * 60)
        print("📊 步骤 4: 结果汇总分析")
        print("=" * 60)
        
        metrics_to_extract = ['mse', 'r2', 'mae', 'mdae', 'rmse', 'training_time', 'n_iter']
        
        model_metrics = {
            "手动 PyTorch MLP": {m: [r[m] for r in self.results['pytorch']] for m in metrics_to_extract},
            "Sklearn MLP": {m: [r[m] for r in self.results['sklearn']] for m in metrics_to_extract},
            "Causal-Sklearn MLP": {m: [r[m] for r in self.results['causal_sklearn']] for m in metrics_to_extract}
        }

        def print_stats(name, metrics):
            print(f"\n{name} 统计结果 (共 {self.config.N_RUNS} 次运行):")
            for key, values in metrics.items():
                print(f"   - {key.upper():<13} - 平均值: {np.mean(values):.4f}, 标准差: {np.std(values):.4f}")
        
        for name, metrics in model_metrics.items():
            print_stats(name, metrics)

    def plot_results(self):
        print("\n" + "=" * 60)
        print("📊 步骤 5: 生成可视化图表")
        print("=" * 60)
        
        fig, axes = plt.subplots(3, 2, figsize=self.config.FIGURE_SIZE)
        fig.suptitle(f'Model Performance Comparison (Boxplot)\n{self.config.N_RUNS} runs on California Housing with {self.config.ANOMALY_RATIO:.0%} training noise', fontsize=16)
        
        all_metrics_map = {
            'MSE': 'mse', 'RMSE': 'rmse', 'MAE': 'mae', 
            'MdAE': 'mdae', 'R²': 'r2', 'Training Time (s)': 'training_time'
        }
        
        labels = ['PyTorch', 'Sklearn', 'Causal-Sklearn']
        axes_flat = axes.flatten()
        
        for i, (title, metric_key) in enumerate(all_metrics_map.items()):
            ax = axes_flat[i]
            data_to_plot = [
                [r[metric_key] for r in self.results['pytorch']],
                [r[metric_key] for r in self.results['sklearn']],
                [r[metric_key] for r in self.results['causal_sklearn']]
            ]
            ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            ax.set_title(title)
            ax.set_ylabel(title)
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = self._get_output_path('pytorch_vs_sklearn_mlp_comparison.png')
        plt.savefig(save_path, dpi=self.config.FIGURE_DPI)
        print(f"   - 对比图表已保存至: {save_path}")
        plt.close()
        return save_path


def main():
    """主函数，运行完整的对比实验"""
    print("=" * 60)
    print("PyTorch vs Sklearn vs Causal-Sklearn MLP 实现对比")
    print("=" * 60)
    
    # 1. 创建配置和教程实例
    config = ComparisonConfig()
    tutorial = MLPComparisonTutorial(config)
    
    # 2. 运行实验流程
    tutorial.load_and_split_data()
    tutorial.run_comparison()
    
    # 3. 分析和可视化结果
    tutorial.print_summary_stats()
    save_path = tutorial.plot_results()
    
    # 4. 打印最终总结
    print("\n" + "=" * 60)
    print("✅ 分析完成")
    print("=" * 60)
    print(f"   - 结果图已保存至: {save_path}")
    print("\n关键洞见:")
    print("1. 三种实现使用相同的预处理、网络结构和超参数，在含噪数据上训练。")
    print("2. 采用科学的'先标准化、后加噪'流程，确保了对比的公平性。")
    print("3. 多次运行的结果（箱线图）展示了不同实现的性能稳定性和固有方差。")
    print("4. 这为三种MLP实现在真实且有挑战的场景下提供了一个公平的性能基准。")


if __name__ == "__main__":
    main()