"""
CausalEngine基准测试基础模块

提供统一的基准测试框架，用于比较CausalEngine与传统机器学习方法的性能。
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
    
    提供统一的接口来比较CausalEngine与sklearn和PyTorch基线的性能。
    """
    
    def __init__(self):
        self.results = {}
    
    def add_label_anomalies(self, y, anomaly_ratio=0.1, anomaly_type='regression'):
        """
        给标签添加异常 - 用于测试模型的鲁棒性
        
        Args:
            y: 原始标签
            anomaly_ratio: 异常比例 (0.0-1.0)
            anomaly_type: 'regression'(回归异常) 或 'classification'(分类翻转)
        """
        y_noisy = y.copy()
        n_anomalies = int(len(y) * anomaly_ratio)
        
        if n_anomalies == 0:
            return y_noisy
            
        anomaly_indices = np.random.choice(len(y), n_anomalies, replace=False)
        
        if anomaly_type == 'regression':
            # 回归异常：简单而强烈的异常
            y_std = np.std(y)
            
            for idx in anomaly_indices:
                # 随机选择异常类型
                if np.random.random() < 0.5:
                    # 策略1: 3倍标准差偏移
                    sign = np.random.choice([-1, 1])
                    y_noisy[idx] = y[idx] + sign * 3.0 * y_std
                else:
                    # 策略2: 10倍缩放
                    scale_factor = np.random.choice([0.1, 10.0])  # 极端缩放
                    y_noisy[idx] = y[idx] * scale_factor
                
        elif anomaly_type == 'classification':
            # 分类异常：标签翻转
            unique_labels = np.unique(y)
            for idx in anomaly_indices:
                other_labels = unique_labels[unique_labels != y[idx]]
                if len(other_labels) > 0:
                    y_noisy[idx] = np.random.choice(other_labels)
        
        return y_noisy
    
    def train_pytorch_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                          epochs=1000, lr=0.001, task='regression', patience=50, tol=1e-4):
        """训练PyTorch基线模型"""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        if task == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
            y_train_tensor = y_train_tensor.long()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 早停
        best_loss = float('inf')
        no_improve = 0
        best_model_path = f"/tmp/pytorch_best_model_{id(model)}.pkl"
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            if task == 'regression':
                loss = criterion(outputs.squeeze(), y_train_tensor)
            else:
                loss = criterion(outputs, y_train_tensor)
            
            loss.backward()
            optimizer.step()
            
            # 验证集早停
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val)
                    val_outputs = model(X_val_tensor)
                    if task == 'regression':
                        y_val_tensor = torch.FloatTensor(y_val)
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                    else:
                        y_val_tensor = torch.LongTensor(y_val)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if val_loss < best_loss - tol:
                    best_loss = val_loss
                    no_improve = 0
                    # 保存最佳模型
                    import pickle
                    with open(best_model_path, 'wb') as f:
                        pickle.dump(model.state_dict(), f)
                    if epoch == 0:
                        print(f"   最佳模型临时存储: {best_model_path}")
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    break
        
        # 恢复最佳模型
        import pickle
        if os.path.exists(best_model_path):
            with open(best_model_path, 'rb') as f:
                model.load_state_dict(pickle.load(f))
            print(f"   已恢复最佳模型，删除临时文件: {best_model_path}")
            os.remove(best_model_path)
        
        # 将实际训练轮数作为属性添加到模型
        model.n_iter_ = epoch + 1
        model.final_loss_ = best_loss
        return model
    
    def train_causal_engine(self, X_train, y_train, X_val, y_val, task_type='regression', mode='standard',
                           hidden_sizes=(128, 64), max_epochs=5000, lr=0.01, patience=500, tol=1e-8,
                           gamma_init=1.0, b_noise_init=1.0, b_noise_trainable=True, ovr_threshold=0.0, verbose=True):
        """训练CausalEngine模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_size = X_train.shape[1]
        if task_type == 'regression':
            output_size = 1 if len(y_train.shape) == 1 else y_train.shape[1]
            model = create_causal_regressor(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_sizes[0],
                hidden_layers=hidden_sizes[1:],
                gamma_init=gamma_init,
                b_noise_init=b_noise_init,
                b_noise_trainable=b_noise_trainable
            )
        else:
            n_classes = len(np.unique(y_train))
            model = create_causal_classifier(
                input_size=input_size,
                n_classes=n_classes,
                hidden_size=hidden_sizes[0],
                hidden_layers=hidden_sizes[1:],
                gamma_init=gamma_init,
                b_noise_init=b_noise_init,
                b_noise_trainable=b_noise_trainable,
                ovr_threshold=ovr_threshold
            )
        
        if verbose:
            print(f"\n为模式构建模型: {mode}")
            print(f"==> 模型已构建。总可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
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
            # 训练阶段
            model.train()
            optimizer.zero_grad()
            loss = model.compute_loss(X_train_torch, y_train_torch, mode)
            loss.backward()
            optimizer.step()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_loss = model.compute_loss(X_val_torch, y_val_torch, mode).item()
            
            # 打印进度
            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            
            # 早停检查
            if val_loss < best_val_loss - tol:
                best_val_loss = val_loss
                patience_counter = 0
                if verbose:
                    print(f"New best validation loss: {val_loss:.6f} at epoch {epoch + 1}")
                # 保存最佳状态
                best_state_dict = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                    print(f"Restored best model from validation loss: {best_val_loss:.6f}")
                # 恢复最佳模型
                if best_state_dict is not None:
                    model.load_state_dict(best_state_dict)
                break
        
        return model
    
    def compare_models(self, X, y, task_type='regression', test_size=0.2, val_size=0.25, 
                      anomaly_ratio=0.0, random_state=42, verbose=True, **kwargs):
        """
        通用模型比较方法
        
        Args:
            X: 特征数据
            y: 标签数据
            task_type: 'regression' 或 'classification'
            test_size: 测试集比例
            val_size: 验证集比例（相对于训练集）
            anomaly_ratio: 标签异常比例
            random_state: 随机种子
            verbose: 是否显示详细信息
            **kwargs: 其他参数
        """
        # 数据分割
        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
        
        # 添加标签异常（仅训练集）
        if anomaly_ratio > 0:
            y_train = self.add_label_anomalies(y_train, anomaly_ratio, task_type)
        
        # 分割训练和验证集
        if task_type == 'classification':
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state)
        
        # 标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        if task_type == 'regression':
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = y_train
            y_val_scaled = y_val
            y_test_scaled = y_test
        
        results = {}
        
        # 训练和评估所有模型
        results.update(self._train_sklearn_baseline(
            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, task_type, verbose, **kwargs
        ))
        
        results.update(self._train_pytorch_baseline(
            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, task_type, verbose, **kwargs
        ))
        
        results.update(self._train_causal_engines(
            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, task_type, verbose, **kwargs
        ))
        
        return results
    
    def _train_sklearn_baseline(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                               task_type, verbose, **kwargs):
        """训练sklearn基线"""
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (128, 64))
        max_iter = kwargs.get('max_iter', 5000)
        learning_rate = kwargs.get('learning_rate', 0.01)
        random_state = kwargs.get('random_state', 42)
        
        if verbose: print("训练 sklearn 基线...")
        
        if task_type == 'regression':
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=50,
                tol=1e-4,
                random_state=random_state,
                alpha=0.0001
            )
            model.fit(X_train, y_train)
            pred_test = model.predict(X_test)
            pred_val = model.predict(X_val)
            
            return {
                'sklearn': {
                    'test': {
                        'MAE': mean_absolute_error(y_test, pred_test),
                        'MdAE': median_absolute_error(y_test, pred_test), 
                        'RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
                        'R²': r2_score(y_test, pred_test)
                    },
                    'val': {
                        'MAE': mean_absolute_error(y_val, pred_val),
                        'MdAE': median_absolute_error(y_val, pred_val), 
                        'RMSE': np.sqrt(mean_squared_error(y_val, pred_val)),
                        'R²': r2_score(y_val, pred_val)
                    }
                }
            }
        else:
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=50,
                tol=1e-4,
                random_state=random_state,
                alpha=0.0001
            )
            model.fit(X_train, y_train)
            pred_test = model.predict(X_test)
            pred_val = model.predict(X_val)
            
            n_classes = len(np.unique(y_test))
            avg_method = 'binary' if n_classes == 2 else 'macro'
            
            return {
                'sklearn': {
                    'test': {
                        'Acc': accuracy_score(y_test, pred_test),
                        'Precision': precision_score(y_test, pred_test, average=avg_method, zero_division=0),
                        'Recall': recall_score(y_test, pred_test, average=avg_method, zero_division=0),
                        'F1': f1_score(y_test, pred_test, average=avg_method, zero_division=0)
                    },
                    'val': {
                        'Acc': accuracy_score(y_val, pred_val),
                        'Precision': precision_score(y_val, pred_val, average=avg_method, zero_division=0),
                        'Recall': recall_score(y_val, pred_val, average=avg_method, zero_division=0),
                        'F1': f1_score(y_val, pred_val, average=avg_method, zero_division=0)
                    }
                }
            }
    
    def _train_pytorch_baseline(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                               task_type, verbose, **kwargs):
        """训练PyTorch基线"""
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (128, 64))
        max_iter = kwargs.get('max_iter', 5000)
        learning_rate = kwargs.get('learning_rate', 0.01)
        
        if verbose: print("训练 PyTorch 基线...")
        
        n_features = X_train.shape[1]
        if task_type == 'regression':
            output_size = 1
        else:
            output_size = len(np.unique(y_train))
        
        model = PyTorchBaseline(n_features, output_size, hidden_layer_sizes)
        model = self.train_pytorch_model(
            model, X_train, y_train, X_val, y_val, 
            epochs=max_iter, lr=learning_rate, task=task_type,
            patience=50, tol=1e-4)
        
        model.eval()
        with torch.no_grad():
            if task_type == 'regression':
                pred_test = model(torch.FloatTensor(X_test)).squeeze().numpy()
                pred_val = model(torch.FloatTensor(X_val)).squeeze().numpy()
                
                return {
                    'pytorch': {
                        'test': {
                            'MAE': mean_absolute_error(y_test, pred_test),
                            'MdAE': median_absolute_error(y_test, pred_test),
                            'RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
                            'R²': r2_score(y_test, pred_test)
                        },
                        'val': {
                            'MAE': mean_absolute_error(y_val, pred_val),
                            'MdAE': median_absolute_error(y_val, pred_val),
                            'RMSE': np.sqrt(mean_squared_error(y_val, pred_val)),
                            'R²': r2_score(y_val, pred_val)
                        }
                    }
                }
            else:
                outputs_test = model(torch.FloatTensor(X_test))
                pred_test = torch.argmax(outputs_test, dim=1).numpy()
                outputs_val = model(torch.FloatTensor(X_val))
                pred_val = torch.argmax(outputs_val, dim=1).numpy()
                
                n_classes = len(np.unique(y_test))
                avg_method = 'binary' if n_classes == 2 else 'macro'
                
                return {
                    'pytorch': {
                        'test': {
                            'Acc': accuracy_score(y_test, pred_test),
                            'Precision': precision_score(y_test, pred_test, average=avg_method, zero_division=0),
                            'Recall': recall_score(y_test, pred_test, average=avg_method, zero_division=0),
                            'F1': f1_score(y_test, pred_test, average=avg_method, zero_division=0)
                        },
                        'val': {
                            'Acc': accuracy_score(y_val, pred_val),
                            'Precision': precision_score(y_val, pred_val, average=avg_method, zero_division=0),
                            'Recall': recall_score(y_val, pred_val, average=avg_method, zero_division=0),
                            'F1': f1_score(y_val, pred_val, average=avg_method, zero_division=0)
                        }
                    }
                }
    
    def _train_causal_engines(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                             task_type, verbose, **kwargs):
        """训练CausalEngine模型（多种模式）"""
        modes = kwargs.get('causal_modes', ['deterministic', 'standard'])
        results = {}
        
        for mode in modes:
            if verbose: print(f"训练 CausalEngine ({mode})...")
            
            # 过滤CausalEngine相关参数
            causal_kwargs = {k: v for k, v in kwargs.items() if k in [
                'hidden_sizes', 'max_epochs', 'lr', 'patience', 'tol',
                'gamma_init', 'b_noise_init', 'b_noise_trainable', 'ovr_threshold'
            ]}
            
            model = self.train_causal_engine(
                X_train, y_train, X_val, y_val, task_type, mode, verbose=verbose, **causal_kwargs
            )
            
            # 预测
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                X_test_torch = torch.FloatTensor(X_test).to(device)
                X_val_torch = torch.FloatTensor(X_val).to(device)
                
                if task_type == 'regression':
                    pred_test = model.predict(X_test_torch, mode).cpu().numpy().flatten()
                    pred_val = model.predict(X_val_torch, mode).cpu().numpy().flatten()
                    
                    results[mode] = {
                        'test': {
                            'MAE': mean_absolute_error(y_test, pred_test),
                            'MdAE': median_absolute_error(y_test, pred_test),
                            'RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
                            'R²': r2_score(y_test, pred_test)
                        },
                        'val': {
                            'MAE': mean_absolute_error(y_val, pred_val),
                            'MdAE': median_absolute_error(y_val, pred_val),
                            'RMSE': np.sqrt(mean_squared_error(y_val, pred_val)),
                            'R²': r2_score(y_val, pred_val)
                        }
                    }
                else:
                    pred_test = model.predict(X_test_torch, mode).cpu().numpy()
                    pred_val = model.predict(X_val_torch, mode).cpu().numpy()
                    
                    n_classes = len(np.unique(y_test))
                    avg_method = 'binary' if n_classes == 2 else 'macro'
                    
                    results[mode] = {
                        'test': {
                            'Acc': accuracy_score(y_test, pred_test),
                            'Precision': precision_score(y_test, pred_test, average=avg_method, zero_division=0),
                            'Recall': recall_score(y_test, pred_test, average=avg_method, zero_division=0),
                            'F1': f1_score(y_test, pred_test, average=avg_method, zero_division=0)
                        },
                        'val': {
                            'Acc': accuracy_score(y_val, pred_val),
                            'Precision': precision_score(y_val, pred_val, average=avg_method, zero_division=0),
                            'Recall': recall_score(y_val, pred_val, average=avg_method, zero_division=0),
                            'F1': f1_score(y_val, pred_val, average=avg_method, zero_division=0)
                        }
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
        lines.append(f"{'方法':<15} {'验证集':<50} {'测试集':<50}")
        lines.append(f"{'':15} {metrics[0]:<10} {metrics[1]:<10} {metrics[2]:<10} {metrics[3]:<10} "
                    f"{metrics[0]:<10} {metrics[1]:<10} {metrics[2]:<10} {metrics[3]:<10}")
        lines.append("-" * 120)
        
        for method, results_dict in results.items():
            val_m = results_dict['val']
            test_m = results_dict['test']
            lines.append(f"{method:<15} {val_m[metrics[0]]:<10.4f} {val_m[metrics[1]]:<10.4f} "
                        f"{val_m[metrics[2]]:<10.4f} {val_m[metrics[3]]:<10.4f} "
                        f"{test_m[metrics[0]]:<10.4f} {test_m[metrics[1]]:<10.4f} "
                        f"{test_m[metrics[2]]:<10.4f} {test_m[metrics[3]]:<10.4f}")
        
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