"""
CausalEngine基准测试基础模块 - 全局标准化重构版
===================================================

提供统一的基准测试框架，用于比较CausalEngine与传统机器学习方法的性能。
此版本经过重构，遵循"全局标准化"和"职责分离"原则，确保实验的公平性和可复现性。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings

from .._causal_engine import create_causal_regressor, create_causal_classifier
from .methods import BaselineMethodFactory, MethodDependencyChecker, filter_available_methods
from ..data_processing import inject_shuffle_noise
from .method_configs import (
    DEFAULT_METHOD_CONFIGS, get_method_group, get_task_recommendations, 
    validate_methods, expand_method_groups, list_available_methods
)

warnings.filterwarnings('ignore')


class PyTorchBaseline(nn.Module):
    """PyTorch基线模型（传统MLP）"""
    
    def __init__(self, input_size, output_size, hidden_sizes=(128, 64)):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class BaselineBenchmark:
    """
    基准测试基类
    
    提供统一的接口来比较CausalEngine与传统机器学习方法的性能。
    支持配置驱动的基准方法选择，包括神经网络、集成方法、SVM、线性方法等。
    """
    
    def __init__(self):
        self.results = {}
        self.method_factory = BaselineMethodFactory()
        self.dependency_checker = MethodDependencyChecker()
    
    def train_causal_engine(self, X_train, y_train, X_val, y_val, task_type='regression', mode='standard',
                           hidden_sizes=(128, 64), max_epochs=5000, lr=0.01, patience=500, tol=1e-8,
                           gamma_init=1.0, b_noise_init=1.0, b_noise_trainable=True, ovr_threshold=0.0, verbose=True):
        """训练CausalEngine模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_size = X_train.shape[1]
        if task_type == 'regression':
            output_size = 1
            model = create_causal_regressor(
                input_size=input_size,
                output_size=output_size,
                repre_size=hidden_sizes[0] if hidden_sizes else None,
                causal_size=hidden_sizes[0] if hidden_sizes else None,
                perception_hidden_layers=hidden_sizes,
                abduction_hidden_layers=(),
                gamma_init=gamma_init,
                b_noise_init=b_noise_init,
                b_noise_trainable=b_noise_trainable
            )
        else:
            # 确定类别数量时，使用传递进来的y_train，它可能已经是LabelEncoder之后的结果
            n_classes = len(np.unique(y_train))
            model = create_causal_classifier(
                input_size=input_size,
                n_classes=n_classes,
                repre_size=hidden_sizes[0] if hidden_sizes else None,
                causal_size=hidden_sizes[0] if hidden_sizes else None,
                perception_hidden_layers=hidden_sizes,
                abduction_hidden_layers=(),
                gamma_init=gamma_init,
                b_noise_init=b_noise_init,
                b_noise_trainable=b_noise_trainable,
                ovr_threshold=ovr_threshold
            )
        
        if verbose:
            print(f"   为模式构建模型: {mode}")
            print(f"   ==> 模型已构建。总可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        model = model.to(device)
        
        X_train_torch = torch.FloatTensor(X_train).to(device)
        y_train_torch = torch.FloatTensor(y_train).to(device)
        X_val_torch = torch.FloatTensor(X_val).to(device)
        y_val_torch = torch.FloatTensor(y_val).to(device)
        
        if task_type == 'classification':
            y_train_torch = y_train_torch.long()
            y_val_torch = y_val_torch.long()
        else:
            if len(y_train_torch.shape) == 1:
                y_train_torch = y_train_torch.unsqueeze(1)
                y_val_torch = y_val_torch.unsqueeze(1)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None
        
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = model.compute_loss(X_train_torch, y_train_torch, mode)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_loss = model.compute_loss(X_val_torch, y_val_torch, mode).item()
            
            if epoch % 500 == 0 and verbose:
                print(f"      Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss:.6f}")
            
            if val_loss < best_val_loss - tol:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"   Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.6f}")
                if best_state_dict:
                    model.load_state_dict(best_state_dict)
                break
        
        return model

    def compare_models(self, X, y, task_type='regression', test_size=0.2, val_size=0.25,
                       anomaly_ratio=0.0, random_state=42, verbose=True, **kwargs):
        """
        通用模型比较方法 - 全局标准化重构版

        数据预处理黄金准则:
        1. 分割出干净的训练/测试集
        2. 在【干净】的y_train上拟合StandardScaler
        3. 在原始尺度的y_train上注入shuffle噪声
        4. 使用【干净】的scaler来转换带噪声的y_train
        5. 所有模型都在完全标准化的(X, y)空间中训练
        6. 所有模型的预测结果都逆转换为原始尺度进行评估
        """
        # 1. 数据分割
        stratify_option = y if task_type == 'classification' else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_option
        )
        if verbose:
            print(f"🔥 数据准备: 分割数据集...")
            print(f"   - 原始训练集: {len(X_train_full)}, 原始测试集: {len(X_test)}")

        # 2. 噪声注入（在原始尺度上）
        y_train_noisy = y_train_full
        if anomaly_ratio > 0:
            y_train_noisy, noise_indices = inject_shuffle_noise(
                y_train_full,
                noise_ratio=anomaly_ratio,
                random_state=random_state
            )
            if verbose:
                print(f"   - 注入 {anomaly_ratio:.1%} shuffle噪声: {len(noise_indices)}个样本受影响")

        # 3. 标准化（遵循黄金准则）
        scaler_X = StandardScaler()
        X_train_full_scaled = scaler_X.fit_transform(X_train_full)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = None
        y_train_for_model = y_train_noisy
        if task_type == 'regression':
            scaler_y = StandardScaler()
            scaler_y.fit(y_train_full.reshape(-1, 1)) # 在干净的y上拟合
            y_train_for_model = scaler_y.transform(y_train_noisy.reshape(-1, 1)).flatten() # 转换带噪的y
            if verbose:
                print(f"   - X和y都已标准化 (y在干净数据上拟合scaler)")
        else:
             if verbose:
                print(f"   - 仅X被标准化 (分类任务)")

        # 4. 从（可能带噪的）训练集中分割出验证集
        stratify_val_option = y_train_noisy if task_type == 'classification' else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full_scaled, y_train_for_model,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_val_option
        )
        if verbose:
            print(f"   - 最终训练集: {len(X_train)}, 验证集: {len(X_val)}")
        
        results = {}
        
        # 5. 确定要使用的基准方法
        baseline_methods = self._get_baseline_methods(task_type, **kwargs)
        causal_modes = kwargs.get('causal_modes', ['deterministic', 'standard'])
        
        if verbose:
            print(f"\n📊 选择的基准方法: {baseline_methods}")
            print(f"🧠 CausalEngine模式: {causal_modes}")

        all_methods_to_run = baseline_methods + causal_modes
        
        # 6. 统一的训练和评估循环
        for method_name in all_methods_to_run:
            if verbose:
                print("-" * 50)
                print(f"🚀 正在处理: {method_name}")

            model = None
            try:
                if method_name in causal_modes:
                    causal_kwargs = {k: v for k, v in kwargs.items() if k in [
                        'hidden_sizes', 'max_epochs', 'lr', 'patience', 'tol',
                        'gamma_init', 'b_noise_init', 'b_noise_trainable', 'ovr_threshold'
                    ]}
                    model = self.train_causal_engine(
                        X_train, y_train, X_val, y_val, task_type, method_name, verbose=verbose, **causal_kwargs
                    )
                else:
                    method_config = DEFAULT_METHOD_CONFIGS.get(method_name)
                    if not method_config:
                        print(f"❌ 未知方法: {method_name}，跳过")
                        continue
                    
                    params = method_config['params'].copy()
                    if 'baseline_config' in kwargs:
                        config = kwargs['baseline_config']
                        if isinstance(config, dict) and 'method_params' in config:
                            user_params = config['method_params'].get(method_name, {})
                            params.update(user_params)
                    
                    model = self.method_factory.create_model(method_name, task_type, **params)
                    model.fit(X_train, y_train)
            
            except Exception as e:
                if verbose:
                    print(f"❌ 训练 {method_name} 时出错: {str(e)}")
                continue

            # 统一评估
            if model:
                # 对于CausalEngine模型，需要确保输入是torch tensor
                if method_name in causal_modes:
                    import torch
                    device = next(model.parameters()).device
                    X_test_torch = torch.FloatTensor(X_test_scaled).to(device)
                    X_val_torch = torch.FloatTensor(X_val).to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        pred_test_scaled = model.predict(X_test_torch).cpu().numpy()
                        pred_val_scaled = model.predict(X_val_torch).cpu().numpy()
                else:
                    pred_test_scaled = model.predict(X_test_scaled)
                    pred_val_scaled = model.predict(X_val)

                if task_type == 'regression' and scaler_y:
                    pred_test = scaler_y.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
                    pred_val = scaler_y.inverse_transform(pred_val_scaled.reshape(-1, 1)).flatten()
                    y_val_original = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
                    
                    y_test_original = y_test # 测试集y始终是干净、原始的
                    
                    results[method_name] = {
                        'test': {'MAE': mean_absolute_error(y_test_original, pred_test),
                                 'MdAE': median_absolute_error(y_test_original, pred_test),
                                 'RMSE': np.sqrt(mean_squared_error(y_test_original, pred_test)),
                                 'R²': r2_score(y_test_original, pred_test)},
                        'val': {'MAE': mean_absolute_error(y_val_original, pred_val),
                                'MdAE': median_absolute_error(y_val_original, pred_val),
                                'RMSE': np.sqrt(mean_squared_error(y_val_original, pred_val)),
                                'R²': r2_score(y_val_original, pred_val)}
                    }
                else:
                    n_classes = len(np.unique(y_train_full))
                    avg_method = 'binary' if n_classes == 2 else 'macro'
                    
                    results[method_name] = {
                        'test': {'Acc': accuracy_score(y_test, pred_test_scaled),
                                 'Precision': precision_score(y_test, pred_test_scaled, average=avg_method, zero_division=0),
                                 'Recall': recall_score(y_test, pred_test_scaled, average=avg_method, zero_division=0),
                                 'F1': f1_score(y_test, pred_test_scaled, average=avg_method, zero_division=0)},
                        'val': {'Acc': accuracy_score(y_val, pred_val_scaled),
                                'Precision': precision_score(y_val, pred_val_scaled, average=avg_method, zero_division=0),
                                'Recall': recall_score(y_val, pred_val_scaled, average=avg_method, zero_division=0),
                                'F1': f1_score(y_val, pred_val_scaled, average=avg_method, zero_division=0)}
                    }

        return results
    
    def format_results_table(self, results, task_type='regression'):
        """格式化结果为表格字符串"""
        if task_type == 'regression':
            metrics = ['MAE', 'MdAE', 'RMSE', 'R²']
            title = "📊 回归性能对比"
        else:
            metrics = ['Acc', 'Precision', 'Recall', 'F1']
            title = "📊 分类性能对比"
        
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * 120)
        lines.append(f"{'方法':<25} {'验证集':<50} {'测试集':<50}")
        lines.append(f"{'':25} {metrics[0]:<10} {metrics[1]:<10} {metrics[2]:<10} {metrics[3]:<10} "
                    f"{metrics[0]:<10} {metrics[1]:<10} {metrics[2]:<10} {metrics[3]:<10}")
        lines.append("-" * 120)
        
        # 获取所有方法名称的配置
        method_configs = {key: val['name'] for key, val in DEFAULT_METHOD_CONFIGS.items()}

        for method, res_dict in sorted(results.items(), key=lambda item: item[1]['test'].get('MdAE', float('inf'))):
            display_name = method_configs.get(method, method.capitalize())
            val_m = res_dict.get('val', {})
            test_m = res_dict.get('test', {})
            
            val_values = [val_m.get(m, float('nan')) for m in metrics]
            test_values = [test_m.get(m, float('nan')) for m in metrics]
            
            lines.append(f"{display_name:<25} {val_values[0]:<10.4f} {val_values[1]:<10.4f} "
                        f"{val_values[2]:<10.4f} {val_values[3]:<10.4f} "
                        f"{test_values[0]:<10.4f} {test_values[1]:<10.4f} "
                        f"{test_values[2]:<10.4f} {test_values[3]:<10.4f}")
        
        lines.append("=" * 120)
        return '\n'.join(lines)
    
    def print_results(self, results, task_type='regression'):
        """打印基准测试结果"""
        print(self.format_results_table(results, task_type))
    
    def benchmark_synthetic_data(self, task_type='regression', n_samples=1000, n_features=20, 
                                anomaly_ratio=0.0, verbose=True, **kwargs):
        """在合成数据上进行基准测试"""
        from sklearn.datasets import make_regression, make_classification
        
        if task_type == 'regression':
            X, y = make_regression(
                n_samples=n_samples, 
                n_features=n_features, 
                noise=0.1, 
                random_state=kwargs.get('random_state', 42)
            )
        else:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(n_features//2, 2),
                n_redundant=0,
                n_clusters_per_class=1,
                random_state=kwargs.get('random_state', 42)
            )
        
        if verbose:
            print(f"\n🧪 {task_type.title()} 基准测试")
            print(f"数据集: {n_samples} 样本, {n_features} 特征")
            if anomaly_ratio > 0:
                print(f"标签异常: {anomaly_ratio:.1%}")
        
        results = self.compare_models(
            X, y, task_type=task_type, anomaly_ratio=anomaly_ratio, 
            verbose=verbose, **kwargs
        )
        
        if verbose:
            self.print_results(results, task_type)
        
        return results
    
    def _get_baseline_methods(self, task_type: str, **kwargs) -> list:
        """
        确定要使用的基准方法列表
        """
        # 方式1: 直接指定方法列表
        if 'baseline_methods' in kwargs:
            methods = kwargs['baseline_methods']
            if isinstance(methods, str):
                methods = [methods]
            
            methods = expand_method_groups(methods)
            available_methods, _ = filter_available_methods(methods)
            return available_methods
        
        # 方式2: 使用预定义方法组合
        if 'method_group' in kwargs:
            group_name = kwargs['method_group']
            methods = get_method_group(group_name)
            available_methods, _ = filter_available_methods(methods)
            return available_methods
        
        # 方式3: 任务特定推荐
        if 'recommendation_type' in kwargs:
            rec_type = kwargs['recommendation_type']
            methods = get_task_recommendations(task_type, rec_type)
            available_methods, _ = filter_available_methods(methods)
            return available_methods
        
        # 默认方式：向后兼容
        return ['sklearn_mlp', 'pytorch_mlp']
    
    def list_available_baseline_methods(self) -> dict:
        """列出所有可用的基准方法"""
        all_methods = list_available_methods()
        available = {}
        
        for method in all_methods:
            config = DEFAULT_METHOD_CONFIGS.get(method)
            available[method] = {
                'name': config['name'],
                'type': config['type'],
                'available': self.method_factory.is_method_available(method)
            }
        
        return available
    
    def print_method_availability(self):
        """打印方法可用性报告"""
        print("\n📦 基准方法可用性报告")
        print("=" * 80)
        
        methods = self.list_available_baseline_methods()
        
        by_type = {}
        for method, info in methods.items():
            method_type = info['type']
            if method_type not in by_type:
                by_type[method_type] = []
            by_type[method_type].append((method, info))
        
        for method_type, method_list in by_type.items():
            print(f"\n📊 {method_type.title()} Methods:")
            print("-" * 40)
            
            for method, info in method_list:
                status = "✅" if info['available'] else "❌"
                print(f"  {status} {method:<20} - {info['name']}")
        
        self.dependency_checker.print_dependency_status()