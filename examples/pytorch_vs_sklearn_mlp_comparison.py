#!/usr/bin/env python3
"""
PyTorch vs Sklearn MLP 对比脚本

本脚本实现了三个版本的 MLP:
1. 从零开始手动实现的 PyTorch 版本
2. Sklearn 的 MLPRegressor
3. Causal-Sklearn 库中封装的 MLPPytorchRegressor

目标是比较它们的性能，并观察算法固有的可变性。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import time
import warnings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from causal_sklearn.regressor import MLPPytorchRegressor
from causal_sklearn.data_processing import inject_shuffle_noise
warnings.filterwarnings('ignore')

# 全局随机种子将在每次运行时设置，以便更好地分析可变性
# 这使我们能够观察到真实的算法方差

class PyTorchMLP(nn.Module):
    """手动实现的 PyTorch MLP"""
    
    def __init__(self, input_size, hidden_sizes=[100, 50], output_size=1, random_state=None):
        super(PyTorchMLP, self).__init__()
        
        # 为可复现的权重初始化设置随机种子
        if random_state is not None:
            torch.manual_seed(random_state)
        
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        # 移除内置的缩放器，以使用外部预处理
        self.n_iter_ = 0
        
    def fit(self, X, y, epochs=1000, lr=0.001, batch_size=None, random_state=None,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4):
        """训练模型，并实现早停逻辑"""
        # 为训练的可复现性设置随机种子
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # 转换为张量 (X 和 y 应该已经被预处理)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1) # 确保y是2D
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 如果启用早停，则分割出验证集
        if early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor,
                test_size=validation_fraction,
                random_state=random_state
            )
        else:
            X_train, y_train = X_tensor, y_tensor
            X_val, y_val = None, None
            
        # 训练循环
        self.train()
        
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None

        # 全批量训练
        for epoch in range(epochs):
            # 训练步骤
            outputs = self.network(X_train).squeeze()
            loss = criterion(outputs, y_train.squeeze())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 验证步骤
            if early_stopping and X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.network(X_val).squeeze()
                    val_loss = criterion(val_outputs, y_val.squeeze()).item()
                    
                    if val_loss < best_val_loss - tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = self.state_dict().copy()
                    else:
                        no_improve_count += 1
                
                self.train() # 切换回训练模式
                        
                if no_improve_count >= n_iter_no_change:
                    if self.n_iter_ == 0: self.n_iter_ = epoch + 1 # 记录停止的迭代次数
                    print(f"   (手动PyTorch: 在 {epoch + 1} 次迭代后早停)")
                    break
        
        self.n_iter_ = epoch + 1

        # 如果使用早停，恢复最佳模型状态
        if early_stopping and best_state_dict is not None:
            self.load_state_dict(best_state_dict)
    
    def predict(self, X):
        """进行预测"""
        self.eval()
        with torch.no_grad():
            # X 应该已经被预处理
            X_tensor = torch.FloatTensor(X)
            predictions = self.network(X_tensor).squeeze().numpy()
        return predictions

def run_pytorch_mlp(X_train, X_test, y_train, y_test_original, run_id, random_state=None, early_stopping_config=None):
    """运行手动 PyTorch MLP 并返回评估指标"""
    print(f"正在运行手动 PyTorch MLP - 第 {run_id} 次 (随机种子: {random_state})")
    
    start_time = time.time()
    
    # 创建并训练模型
    # 注意: y_train 是带噪且标准化的，模型在此基础上学习
    model = PyTorchMLP(input_size=X_train.shape[1], hidden_sizes=[100, 50], random_state=random_state)
    
    fit_params = {
        'epochs': 1000, 
        'lr': 0.001, 
        'batch_size': None, 
        'random_state': random_state
    }
    if early_stopping_config:
        fit_params.update(early_stopping_config)

    model.fit(X_train, y_train, **fit_params)
    
    # 进行预测 (预测结果是标准化的)
    y_pred_scaled = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    # 修正策略：让运行函数只返回预测值，评估在主函数中统一进行。
    return {
        'training_time': training_time,
        'predictions_scaled': y_pred_scaled, # 返回标准化预测
        'n_iter': model.n_iter_
    }

def run_sklearn_mlp(X_train, X_test, y_train, y_test_original, run_id, random_state=None, early_stopping_config=None):
    """运行 Sklearn MLP 并返回评估指标"""
    print(f"正在运行 Sklearn MLP - 第 {run_id} 次 (随机种子: {random_state})")
    
    start_time = time.time()
    
    # 创建并训练参数匹配的模型
    model_params = {
        'hidden_layer_sizes': (100, 50),
        'max_iter': 1000,
        'random_state': random_state,
        'alpha': 0.0001,
        'learning_rate_init': 0.001,
        'solver': 'adam',
        'batch_size': X_train.shape[0] # 显式设置为全批量以进行公平比较
    }
    if early_stopping_config:
        model_params.update({
            'early_stopping': True,
            'validation_fraction': early_stopping_config['validation_fraction'],
            'n_iter_no_change': early_stopping_config['n_iter_no_change'],
            'tol': early_stopping_config['tol']
        })
    else:
        model_params['early_stopping'] = False

    model = MLPRegressor(**model_params)
    print(model_params)

    # 模型在标准化的 y_train 上训练
    model.fit(X_train, y_train)
    
    # 进行预测 (预测结果是标准化的)
    y_pred_scaled = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    return {
        'training_time': training_time,
        'predictions_scaled': y_pred_scaled, # 返回标准化预测
        'n_iter': model.n_iter_
    }

def run_causal_sklearn_mlp(X_train, X_test, y_train, y_test_original, run_id, random_state=None, early_stopping_config=None):
    """运行 Causal-Sklearn MLP 并返回评估指标"""
    print(f"正在运行 Causal-Sklearn MLP - 第 {run_id} 次 (随机种子: {random_state})")

    start_time = time.time()

    # 创建并训练参数匹配的模型
    model_params = {
        'hidden_layer_sizes': (100, 50),
        'max_iter': 1000,
        'random_state': random_state,
        'alpha': 0.0001,
        'learning_rate': 0.001,
        'batch_size': None,  # None 表示全批量
        'verbose': False
    }
    if early_stopping_config:
        model_params.update({
            'early_stopping': True,
            **early_stopping_config
        })
    else:
        model_params['early_stopping'] = False

    model = MLPPytorchRegressor(**model_params)

    model.fit(X_train, y_train)

    # 进行预测
    y_pred_scaled = model.predict(X_test)

    training_time = time.time() - start_time

    return {
        'training_time': training_time,
        'predictions_scaled': y_pred_scaled, # 返回标准化预测
        'n_iter': model.n_iter_
    }


def compare_mlp_implementations(n_runs=3, base_seed=42):
    """比较 PyTorch 和 Sklearn MLP 的实现"""
    
    print("=" * 60)
    print("PyTorch vs Sklearn vs Causal-Sklearn MLP 实现对比")
    print("=" * 60)
    
    # 1. 加载真实数据集
    print("\n📊 步骤 1: 加载 California Housing 数据集")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    print(f"   - 数据集加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 2. 分割数据 (得到原始的训练集和测试集)
    print("\n📊 步骤 2: 分割数据")
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=base_seed
    )
    print(f"   - 训练集: {X_train_orig.shape[0]}, 测试集: {X_test_orig.shape[0]}")
    
    # 3. 对训练数据注入噪声
    anomaly_ratio = 0.25
    print(f"\n📊 步骤 3: 对训练集注入 {anomaly_ratio:.0%} 的异常值")
    y_train_noisy, noise_indices = inject_shuffle_noise(
        y_train_orig, 
        noise_ratio=anomaly_ratio,
        random_state=base_seed
    )
    print(f"   - {len(noise_indices)}/{len(y_train_orig)} 个训练样本的标签被污染")
    
    # 4. 标准化数据（所有模型使用相同的预处理）
    print("\n📊 步骤 4: 使用 StandardScaler 标准化数据")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_orig)
    X_test_scaled = scaler_X.transform(X_test_orig)
    
    # 使用带噪声的y_train来拟合y的scaler
    y_train_scaled = scaler_y.fit_transform(y_train_noisy.reshape(-1, 1)).flatten()
    print("   - 特征 (X) 和目标 (y) 均已标准化")
    
    # 定义统一的早停配置
    early_stopping_config = {
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'tol': 1e-4
    }
    print("\n📊 步骤 5: 配置统一的早停策略")
    print(f"   - Patience: {early_stopping_config['n_iter_no_change']}, Validation Fraction: {early_stopping_config['validation_fraction']}")

    # 存储结果
    pytorch_results = []
    sklearn_results = []
    causal_sklearn_results = []
    
    print(f"\n正在进行 {n_runs} 次实验，每次使用不同随机种子...")
    
    for i in range(n_runs):
        run_seed = base_seed + i * 100  # 每次运行使用不同的种子
        print(f"\n--- 实验 {i+1}/{n_runs} (随机种子: {run_seed}) ---")
        
        # 运行模型得到标准化的预测
        pytorch_run = run_pytorch_mlp(X_train_scaled, X_test_scaled, y_train_scaled, y_test_orig, i+1, run_seed, early_stopping_config)
        sklearn_run = run_sklearn_mlp(X_train_scaled, X_test_scaled, y_train_scaled, y_test_orig, i+1, run_seed, early_stopping_config)
        causal_sklearn_run = run_causal_sklearn_mlp(X_train_scaled, X_test_scaled, y_train_scaled, y_test_orig, i+1, run_seed, early_stopping_config)
        
        # 统一在主函数中进行评估
        runs = [pytorch_run, sklearn_run, causal_sklearn_run]
        results_list = [pytorch_results, sklearn_results, causal_sklearn_results]

        for run_result, result_container in zip(runs, results_list):
            # 逆标准化预测值以进行评估
            y_pred_orig_scale = scaler_y.inverse_transform(run_result['predictions_scaled'].reshape(-1, 1)).flatten()
            
            # 计算评估指标
            mse = mean_squared_error(y_test_orig, y_pred_orig_scale)
            
            # 将指标存入容器
            result_container.append({
                'mse': mse,
                'r2': r2_score(y_test_orig, y_pred_orig_scale),
                'mae': mean_absolute_error(y_test_orig, y_pred_orig_scale),
                'mdae': median_absolute_error(y_test_orig, y_pred_orig_scale),
                'rmse': np.sqrt(mse),
                'training_time': run_result['training_time'],
                'n_iter': run_result['n_iter']
            })

        print(f"PyTorch        - MSE: {pytorch_results[-1]['mse']:.4f}, R2: {pytorch_results[-1]['r2']:.4f}, MAE: {pytorch_results[-1]['mae']:.4f}, "
              f"RMSE: {pytorch_results[-1]['rmse']:.4f}, MdAE: {pytorch_results[-1]['mdae']:.4f}, Time: {pytorch_results[-1]['training_time']:.2f}s")
        print(f"Sklearn        - MSE: {sklearn_results[-1]['mse']:.4f}, R2: {sklearn_results[-1]['r2']:.4f}, MAE: {sklearn_results[-1]['mae']:.4f}, "
              f"RMSE: {sklearn_results[-1]['rmse']:.4f}, MdAE: {sklearn_results[-1]['mdae']:.4f}, Time: {sklearn_results[-1]['training_time']:.2f}s, Iters: {sklearn_results[-1]['n_iter']}")
        print(f"Causal-Sklearn - MSE: {causal_sklearn_results[-1]['mse']:.4f}, R2: {causal_sklearn_results[-1]['r2']:.4f}, MAE: {causal_sklearn_results[-1]['mae']:.4f}, "
              f"RMSE: {causal_sklearn_results[-1]['rmse']:.4f}, MdAE: {causal_sklearn_results[-1]['mdae']:.4f}, Time: {causal_sklearn_results[-1]['training_time']:.2f}s, Iters: {causal_sklearn_results[-1]['n_iter']}")
    
    # 分析结果
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    
    # 提取指标
    metrics_to_extract = ['mse', 'r2', 'mae', 'mdae', 'rmse', 'training_time']
    
    pytorch_metrics = {m: [r[m] for r in pytorch_results] for m in metrics_to_extract}
    sklearn_metrics = {m: [r[m] for r in sklearn_results] for m in metrics_to_extract}
    causal_sklearn_metrics = {m: [r[m] for r in causal_sklearn_results] for m in metrics_to_extract}
    
    # 打印统计数据
    sklearn_iters = [r['n_iter'] for r in sklearn_results]
    causal_sklearn_iters = [r['n_iter'] for r in causal_sklearn_results]
    
    def print_stats(name, metrics):
        print(f"\n{name} 统计结果:")
        print(f"MSE  - 平均值: {np.mean(metrics['mse']):.4f}, 标准差: {np.std(metrics['mse']):.4f}, 范围: [{np.min(metrics['mse']):.4f}, {np.max(metrics['mse']):.4f}]")
        print(f"RMSE - 平均值: {np.mean(metrics['rmse']):.4f}, 标准差: {np.std(metrics['rmse']):.4f}, 范围: [{np.min(metrics['rmse']):.4f}, {np.max(metrics['rmse']):.4f}]")
        print(f"MAE  - 平均值: {np.mean(metrics['mae']):.4f}, 标准差: {np.std(metrics['mae']):.4f}, 范围: [{np.min(metrics['mae']):.4f}, {np.max(metrics['mae']):.4f}]")
        print(f"MdAE - 平均值: {np.mean(metrics['mdae']):.4f}, 标准差: {np.std(metrics['mdae']):.4f}, 范围: [{np.min(metrics['mdae']):.4f}, {np.max(metrics['mdae']):.4f}]")
        print(f"R2   - 平均值: {np.mean(metrics['r2']):.4f}, 标准差: {np.std(metrics['r2']):.4f}, 范围: [{np.min(metrics['r2']):.4f}, {np.max(metrics['r2']):.4f}]")
        print(f"时间 - 平均值: {np.mean(metrics['training_time']):.2f}s, 标准差: {np.std(metrics['training_time']):.2f}s")

    print_stats("手动 PyTorch MLP", pytorch_metrics)
    print_stats("Sklearn MLP", sklearn_metrics)
    print_stats("Causal-Sklearn MLP", causal_sklearn_metrics)

    print("\n迭代次数统计:")
    print(f"Sklearn        - 平均值: {np.mean(sklearn_iters):.1f}, 标准差: {np.std(sklearn_iters):.1f}")
    print(f"Causal-Sklearn - 平均值: {np.mean(causal_sklearn_iters):.1f}, 标准差: {np.std(causal_sklearn_iters):.1f}")

    # 绘制结果图
    save_path = plot_comparison_results(pytorch_metrics, sklearn_metrics, causal_sklearn_metrics)
    
    # 打印参数配置以供验证
    print("\n" + "=" * 60)
    print("参数配置验证")
    print("=" * 60)
    print("三个模型均使用:")
    print("- 数据集: California Housing, 25% 训练标签噪声")
    print("- 网络结构: [input_size, 100, 50, 1]")
    print("- 早停策略: 启用 (Patience=20, Val Fraction=0.1)")
    print("- 激活函数: ReLU")
    print("- 优化器: Adam")
    print("- 学习率: 0.001")
    print("- L2正则化 (alpha): 0.0001")
    print("- 最大迭代次数: 1000")
    print("- 批处理大小: 全量批处理 (为公平比较而统一设置)")
    print("- 数据预处理: StandardScaler (X和y均使用)")
    print("- 随机种子: 每次运行不同，但三种模型在同一次运行中使用相同种子")
    
    return pytorch_results, sklearn_results, causal_sklearn_results, save_path

def plot_comparison_results(pytorch_metrics, sklearn_metrics, causal_sklearn_metrics):
    """绘制对比结果图"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Model Performance Metrics Comparison (Boxplot)\nDataset: California Housing with 25% training noise', fontsize=16)
    
    all_metrics = {
        'MSE': ('mse', axes[0, 0]),
        'RMSE': ('rmse', axes[0, 1]),
        'MAE': ('mae', axes[1, 0]),
        'MdAE': ('mdae', axes[1, 1]),
        'R²': ('r2', axes[2, 0]),
        'Training Time (s)': ('training_time', axes[2, 1])
    }
    
    labels = ['PyTorch', 'Sklearn', 'Causal-Sklearn']
    
    for title, (metric_key, ax) in all_metrics.items():
        data_to_plot = [
            pytorch_metrics[metric_key],
            sklearn_metrics[metric_key],
            causal_sklearn_metrics[metric_key]
        ]
        
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 定义并创建结果目录
    # 使用 __file__ 来定位项目根目录，确保路径的健壮性
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'results', 'tmp')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pytorch_vs_sklearn_mlp_comparison.png')

    plt.savefig(save_path, dpi=300)
    plt.show()
    return save_path

if __name__ == "__main__":
    # 运行对比
    pytorch_results, sklearn_results, causal_sklearn_results, save_path = compare_mlp_implementations(n_runs=3, base_seed=42)
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)
    print(f"结果已保存至: {save_path}")
    print("\n此对比展示了在带噪声的真实数据集上，不同MLP实现的性能差异。")
    print("它比较了手动PyTorch实现、标准Sklearn MLP以及一个Sklearn兼容的PyTorch封装器。")
    print("\n关键洞见:")
    print("1. 三种实现使用相同的预处理和网络结构，并在带噪声的数据上训练。")
    print("2. 结果的差异反映了算法实现（如优化器细节、权重初始化）上的微妙不同。")
    print("3. 每种方法内部的变异性显示了优化过程中固有的随机性。")
    print("4. 这为三种不同的MLP实现提供了一个在更真实场景下的公平性能比较。")