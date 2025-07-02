"""
CausalEngine基准测试基础模块

提供统一的基准测试框架，用于比较CausalEngine与传统机器学习方法的性能。
支持多种基准方法：神经网络、集成方法、SVM、线性方法等。
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
    get_method_config, get_method_group, get_task_recommendations, 
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
    
    数据预处理策略：
    - 特征标准化：所有方法都接收StandardScaler标准化后的特征
    - 目标变量：保持原始尺度，确保与Sklearn-Style实现一致性
    """
    
    def __init__(self):
        self.results = {}
        self.method_factory = BaselineMethodFactory()
        self.dependency_checker = MethodDependencyChecker()
    
    
    def train_pytorch_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                          epochs=1000, lr=0.001, task='regression', patience=50, tol=1e-4, criterion=None):
        """训练PyTorch基线模型"""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        if criterion is None:
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

            if callable(criterion): # Handle custom loss functions
                loss = criterion(outputs.squeeze(), y_train_tensor)
            else: # Handle nn.Module losses
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

                    if callable(criterion): # Handle custom loss functions
                        y_val_tensor = torch.FloatTensor(y_val)
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                    else: # Handle nn.Module losses
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
                repre_size=hidden_sizes[0] if hidden_sizes else None,
                causal_size=hidden_sizes[0] if hidden_sizes else None,
                perception_hidden_layers=hidden_sizes,
                abduction_hidden_layers=(),
                gamma_init=gamma_init,
                b_noise_init=b_noise_init,
                b_noise_trainable=b_noise_trainable
            )
        else:
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
                      anomaly_ratio=0.0, random_state=42, verbose=True, global_standardization=False, **kwargs):
        """
        通用模型比较方法
        
        数据预处理策略：
        - 特征(X): 使用StandardScaler进行标准化
        - 目标(y): 默认保持原始尺度，如果global_standardization=True则也进行标准化
        - 支持全局标准化策略以确保与Sklearn-Style实现的完全一致性
        
        Args:
            X: 特征数据
            y: 标签数据
            task_type: 'regression' 或 'classification'
            test_size: 测试集比例
            val_size: 验证集比例（相对于训练集）
            anomaly_ratio: 标签异常比例
            random_state: 随机种子
            verbose: 是否显示详细信息
            global_standardization: 是否对y也进行标准化（用于与Sklearn-Style实现完全一致）
            **kwargs: 其他参数
        """
        # 1. 统一数据分割和异常注入
        if verbose and anomaly_ratio > 0:
            print(f"🔥 数据准备: 分割数据集并注入 {anomaly_ratio:.1%} 的标签异常...")
        
        # 使用标准train_test_split进行数据分割
        stratify_option = y if task_type == 'classification' else None
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_option
        )
        
        # 对训练集标签进行异常注入
        if anomaly_ratio > 0:
            y_train_full, noise_indices = inject_shuffle_noise(
                y_train_full,
                noise_ratio=anomaly_ratio,
                random_state=random_state
            )
            if verbose:
                print(f"   异常注入完成: {anomaly_ratio:.1%} ({len(noise_indices)}/{len(y_train_full)} 样本受影响)")
        
        # 2. 从(可能带噪的)训练集中分割出验证集
        # 注意：这里的y_train_full可能已经带有噪声
        stratify_val_option = y_train_full if task_type == 'classification' else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=val_size, 
            random_state=random_state, 
            stratify=stratify_val_option
        )
        
        # 3. 标准化
        # 特征标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        # 目标变量处理：根据global_standardization参数决定是否标准化
        scaler_y = None
        if global_standardization and task_type == 'regression':
            # 全局标准化策略：对y也进行标准化
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            if verbose:
                print(f"   🎯 全局标准化：X和y都已标准化")
        else:
            # 传统策略：只对特征进行标准化，目标变量保持原始尺度
            y_train_scaled = y_train
            y_val_scaled = y_val  
            y_test_scaled = y_test
            if verbose and not global_standardization:
                print(f"   📊 传统标准化：只对X标准化，y保持原始尺度")

        # 4. 异常注入已完成，此处无需额外操作
        
        results = {}
        
        # 5. 确定要使用的基准方法
        baseline_methods = self._get_baseline_methods(task_type, **kwargs)
        causal_modes = kwargs.get('causal_modes', ['deterministic', 'standard'])
        
        if verbose:
            print(f"\n📊 选择的基准方法: {baseline_methods}")
            print(f"🧠 CausalEngine模式: {causal_modes}")
        
        # 6. 准备逆变换参数（用于全局标准化模式）
        inverse_transform_params = {}
        if global_standardization and task_type == 'regression' and scaler_y is not None:
            inverse_transform_params = {
                'scaler_y': scaler_y,
                'y_original_val': y_val,
                'y_original_test': y_test
            }
        
        # 7. 训练和评估传统基准方法
        for method_name in baseline_methods:
            if method_name in ['sklearn', 'sklearn_mlp']:
                # 保持向后兼容
                method_config = get_method_config('sklearn_mlp') or {'params': {}}
                params = method_config['params'].copy()
                params.update(kwargs)

                results.update(self._train_sklearn_baseline(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
                    X_test_scaled, y_test_scaled, task_type, verbose, **inverse_transform_params, **params
                ))
            elif method_name in ['pytorch', 'pytorch_mlp']:
                # 保持向后兼容
                method_config = get_method_config('pytorch_mlp') or {'params': {}}
                params = method_config['params'].copy()
                params.update(kwargs)

                results.update(self._train_pytorch_baseline(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
                    X_test_scaled, y_test_scaled, task_type, verbose, **inverse_transform_params, **params
                ))
            elif method_name == 'mlp_huber':
                method_config = get_method_config('mlp_huber') or {'params': {}}
                params = method_config['params'].copy()
                params.update(kwargs)

                results.update(self._train_huber_baseline(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                    X_test_scaled, y_test_scaled, task_type, verbose, **inverse_transform_params, **params
                ))
            elif method_name == 'mlp_pinball_median':
                method_config = get_method_config('mlp_pinball_median') or {'params': {}}
                params = method_config['params'].copy()
                params.update(kwargs)

                results.update(self._train_pinball_baseline(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                    X_test_scaled, y_test_scaled, task_type, verbose, **inverse_transform_params, **params
                ))
            elif method_name == 'mlp_cauchy':
                method_config = get_method_config('mlp_cauchy') or {'params': {}}
                params = method_config['params'].copy()
                params.update(kwargs)
                
                results.update(self._train_cauchy_baseline(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                    X_test_scaled, y_test_scaled, task_type, verbose, **inverse_transform_params, **params
                ))
            else:
                # 新的基准方法
                config = get_method_config(method_name)
                # 统一使用未缩放的y，与其他方法保持一致
                result = self._train_baseline_method(
                    method_name, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                    X_test_scaled, y_test_scaled, task_type, verbose, **inverse_transform_params, **kwargs
                )
                if result:
                    results.update(result)
        
        # 7. 训练和评估CausalEngine模型
        results.update(self._train_causal_engines(
            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, task_type, verbose, **inverse_transform_params, **kwargs
        ))
        
        # 8. 全局标准化模式下的结果已经在正确的尺度上
        # 因为我们现在让调用方传递正确的数据并设置合适的参数
        
        return results
    
    def _train_sklearn_baseline(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                               task_type, verbose, scaler_y=None, y_original_val=None, y_original_test=None, **kwargs):
        """训练sklearn基线"""
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (128, 64))
        max_iter = kwargs.get('max_iter', 5000)
        learning_rate = kwargs.get('learning_rate_init', 0.01)
        random_state = kwargs.get('random_state', 42)
        patience = kwargs.get('patience', 50)
        tol = kwargs.get('tol', 1e-4)
        
        if verbose: print("训练 sklearn 基线...")
        
        if task_type == 'regression':
            # 使用外部验证集进行早停，而不是内部划分
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate,
                early_stopping=False,  # 关闭内部早停
                random_state=random_state,
                alpha=kwargs.get('alpha', 0.0001)
            )
            
            # 手动实现早停策略，使用外部验证集
            model = self._train_sklearn_with_external_validation(
                model, X_train, y_train, X_val, y_val, 
                patience=patience, tol=tol, task_type='regression'
            )
            
            pred_test = model.predict(X_test)
            pred_val = model.predict(X_val)
            
            # 如果使用了全局标准化，需要进行逆变换
            if scaler_y is not None and y_original_test is not None and y_original_val is not None:
                # 将预测结果转换回原始尺度
                pred_test_original = scaler_y.inverse_transform(pred_test.reshape(-1, 1)).flatten()
                pred_val_original = scaler_y.inverse_transform(pred_val.reshape(-1, 1)).flatten()
                
                # 在原始尺度上评估
                eval_y_test = y_original_test
                eval_y_val = y_original_val
                eval_pred_test = pred_test_original
                eval_pred_val = pred_val_original
            else:
                # 在当前尺度上评估（传统模式）
                eval_y_test = y_test
                eval_y_val = y_val
                eval_pred_test = pred_test
                eval_pred_val = pred_val
            
            return {
                'sklearn': {
                    'test': {
                        'MAE': mean_absolute_error(eval_y_test, eval_pred_test),
                        'MdAE': median_absolute_error(eval_y_test, eval_pred_test), 
                        'RMSE': np.sqrt(mean_squared_error(eval_y_test, eval_pred_test)),
                        'R²': r2_score(eval_y_test, eval_pred_test)
                    },
                    'val': {
                        'MAE': mean_absolute_error(eval_y_val, eval_pred_val),
                        'MdAE': median_absolute_error(eval_y_val, eval_pred_val), 
                        'RMSE': np.sqrt(mean_squared_error(eval_y_val, eval_pred_val)),
                        'R²': r2_score(eval_y_val, eval_pred_val)
                    }
                }
            }
        else:
            # 使用外部验证集进行早停，而不是内部划分
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate,
                early_stopping=False,  # 关闭内部早停
                random_state=random_state,
                alpha=kwargs.get('alpha', 0.0001)
            )
            
            # 手动实现早停策略，使用外部验证集
            model = self._train_sklearn_with_external_validation(
                model, X_train, y_train, X_val, y_val, 
                patience=patience, tol=tol, task_type='classification'
            )
            
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
                               task_type, verbose, scaler_y=None, y_original_val=None, y_original_test=None, **kwargs):
        """训练PyTorch基线"""
        if verbose: print("训练 PyTorch 基线 (legacy)...")
        return self._train_generic_pytorch_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            task_type, nn.MSELoss(), 'pytorch', verbose, scaler_y, y_original_val, y_original_test, **kwargs)
    
    def _train_huber_baseline(self, X_train, y_train, X_val, y_val, X_test, y_test,
                              task_type, verbose, scaler_y=None, y_original_val=None, y_original_test=None, **kwargs):
        """训练Huber Loss MLP基线（遗产实现）"""
        if verbose: print("训练 mlp_huber (legacy)...")
        # Huber Loss 不需要标准化y
        return self._train_generic_pytorch_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            task_type, nn.HuberLoss(), 'mlp_huber', verbose, scaler_y, y_original_val, y_original_test, **kwargs)

    def _train_pinball_baseline(self, X_train, y_train, X_val, y_val, X_test, y_test,
                                task_type, verbose, scaler_y=None, y_original_val=None, y_original_test=None, **kwargs):
        """训练Pinball Loss MLP基线（遗产实现）"""
        if verbose: print("训练 mlp_pinball_median (legacy)...")
        # Pinball Loss 不需要标准化y
        return self._train_generic_pytorch_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            task_type, self._pinball_loss, 'mlp_pinball_median', verbose, scaler_y, y_original_val, y_original_test, **kwargs)

    def _train_cauchy_baseline(self, X_train, y_train, X_val, y_val, X_test, y_test,
                               task_type, verbose, scaler_y=None, y_original_val=None, y_original_test=None, **kwargs):
        """训练Cauchy Loss MLP基线（遗产实现）"""
        if verbose: print("训练 mlp_cauchy (legacy)...")
        # Cauchy Loss 不需要标准化y
        return self._train_generic_pytorch_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            task_type, self._cauchy_loss, 'mlp_cauchy', verbose, scaler_y, y_original_val, y_original_test, **kwargs)
    
    def _pinball_loss(self, y_pred, y_true, quantile=0.5):
        """Pinball loss (quantile loss) for PyTorch."""
        error = y_true - y_pred
        loss = torch.where(error >= 0,
                           quantile * error,
                           (quantile - 1) * error)
        return loss.mean()
        
    def _cauchy_loss(self, y_pred, y_true):
        """Cauchy loss function: log(1 + (y_pred - y_true)^2)."""
        error = y_pred - y_true
        loss = torch.log(1 + error**2)
        return loss.mean()

    def _train_generic_pytorch_baseline(self, X_train, y_train, X_val, y_val, X_test, y_test,
                                        task_type, criterion, method_name, verbose, 
                                        scaler_y=None, y_original_val=None, y_original_test=None,
                                        hidden_sizes=(128, 64), epochs=3000, lr=0.01, 
                                        patience=50, tol=1e-4, **kwargs):
        """通用的PyTorch模型训练函数（用于各类稳健回归器）"""
        n_features = X_train.shape[1]
        output_size = 1

        model = PyTorchBaseline(n_features, output_size, hidden_sizes)
        
        # 使用通用的PyTorch训练器，但传入特定的损失函数
        model = self.train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, lr=lr, task=task_type,
            patience=patience, tol=tol, criterion=criterion)

        model.eval()
        with torch.no_grad():
            pred_test = model(torch.FloatTensor(X_test)).squeeze().numpy()
            pred_val = model(torch.FloatTensor(X_val)).squeeze().numpy()

            # 如果使用了全局标准化，需要进行逆变换
            if scaler_y is not None and y_original_test is not None and y_original_val is not None:
                # 将预测结果转换回原始尺度
                pred_test_original = scaler_y.inverse_transform(pred_test.reshape(-1, 1)).flatten()
                pred_val_original = scaler_y.inverse_transform(pred_val.reshape(-1, 1)).flatten()
                
                # 在原始尺度上评估
                eval_y_test = y_original_test
                eval_y_val = y_original_val
                eval_pred_test = pred_test_original
                eval_pred_val = pred_val_original
            else:
                # 在当前尺度上评估（传统模式）
                eval_y_test = y_test
                eval_y_val = y_val
                eval_pred_test = pred_test
                eval_pred_val = pred_val

            return {
                method_name: {
                    'test': {
                        'MAE': mean_absolute_error(eval_y_test, eval_pred_test),
                        'MdAE': median_absolute_error(eval_y_test, eval_pred_test),
                        'RMSE': np.sqrt(mean_squared_error(eval_y_test, eval_pred_test)),
                        'R²': r2_score(eval_y_test, eval_pred_test)
                    },
                    'val': {
                        'MAE': mean_absolute_error(eval_y_val, eval_pred_val),
                        'MdAE': median_absolute_error(eval_y_val, eval_pred_val),
                        'RMSE': np.sqrt(mean_squared_error(eval_y_val, eval_pred_val)),
                        'R²': r2_score(eval_y_val, eval_pred_val)
                    }
                }
            }

    def _train_causal_engines(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                             task_type, verbose, scaler_y=None, y_original_val=None, y_original_test=None, **kwargs):
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
                    
                    # 如果使用了全局标准化，需要进行逆变换
                    if scaler_y is not None and y_original_test is not None and y_original_val is not None:
                        # 将预测结果转换回原始尺度
                        pred_test_original = scaler_y.inverse_transform(pred_test.reshape(-1, 1)).flatten()
                        pred_val_original = scaler_y.inverse_transform(pred_val.reshape(-1, 1)).flatten()
                        
                        # 在原始尺度上评估
                        eval_y_test = y_original_test
                        eval_y_val = y_original_val
                        eval_pred_test = pred_test_original
                        eval_pred_val = pred_val_original
                    else:
                        # 在当前尺度上评估（传统模式）
                        eval_y_test = y_test
                        eval_y_val = y_val
                        eval_pred_test = pred_test
                        eval_pred_val = pred_val
                    
                    results[mode] = {
                        'test': {
                            'MAE': mean_absolute_error(eval_y_test, eval_pred_test),
                            'MdAE': median_absolute_error(eval_y_test, eval_pred_test),
                            'RMSE': np.sqrt(mean_squared_error(eval_y_test, eval_pred_test)),
                            'R²': r2_score(eval_y_test, eval_pred_test)
                        },
                        'val': {
                            'MAE': mean_absolute_error(eval_y_val, eval_pred_val),
                            'MdAE': median_absolute_error(eval_y_val, eval_pred_val),
                            'RMSE': np.sqrt(mean_squared_error(eval_y_val, eval_pred_val)),
                            'R²': r2_score(eval_y_val, eval_pred_val)
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
        
        # 创建方法名显示映射，用于更好的对齐
        display_name_mapping = {
            'MLP Pinball Median': 'MLP Pinball',  # 配置文件中的显示名称
            'MLP Huber': 'MLP Huber',
            'MLP Cauchy': 'MLP Cauchy', 
            'sklearn MLP': 'sklearn',
            'PyTorch MLP': 'pytorch',
            'Random Forest': 'Random Forest',
            'LightGBM': 'LightGBM',
            'XGBoost': 'XGBoost',
            'Ridge Regression': 'Ridge Regression',
            # 兼容原始method_name
            'mlp_pinball_median': 'MLP Pinball',
            'mlp_huber': 'MLP Huber',
            'mlp_cauchy': 'MLP Cauchy',
            'sklearn_mlp': 'sklearn',
            'pytorch_mlp': 'pytorch',
            'sklearn': 'sklearn',  # 向后兼容
            'pytorch': 'pytorch'   # 向后兼容
        }
        
        for method, results_dict in results.items():
            val_m = results_dict['val']
            test_m = results_dict['test']
            # 使用显示名称或原名称
            display_name = display_name_mapping.get(method, method)
            lines.append(f"{display_name:<15} {val_m[metrics[0]]:<10.4f} {val_m[metrics[1]]:<10.4f} "
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
    
    def _get_baseline_methods(self, task_type: str, **kwargs) -> list:
        """
        确定要使用的基准方法列表
        
        支持多种配置方式：
        1. baseline_methods: 直接指定方法列表
        2. baseline_config: 配置字典，包含方法列表和参数
        3. method_group: 使用预定义的方法组合
        4. 默认方式：向后兼容的传统方法
        """
        # 方式1: 直接指定方法列表
        if 'baseline_methods' in kwargs:
            methods = kwargs['baseline_methods']
            if isinstance(methods, str):
                methods = [methods]
            
            # 展开方法组合
            methods = expand_method_groups(methods)
            
            # 过滤可用方法
            available_methods, unavailable_methods = filter_available_methods(methods)
            
            if unavailable_methods:
                print(f"⚠️ 跳过不可用的方法: {unavailable_methods}")
            
            return available_methods
        
        # 方式2: 配置字典
        if 'baseline_config' in kwargs:
            config = kwargs['baseline_config']
            if isinstance(config, dict) and 'traditional_methods' in config:
                methods = config['traditional_methods']
                methods = expand_method_groups(methods)
                available_methods, unavailable_methods = filter_available_methods(methods)
                
                if unavailable_methods:
                    print(f"⚠️ 跳过不可用的方法: {unavailable_methods}")
                
                return available_methods
        
        # 方式3: 使用预定义方法组合
        if 'method_group' in kwargs:
            group_name = kwargs['method_group']
            methods = get_method_group(group_name)
            if not methods:
                print(f"⚠️ 未知的方法组合: {group_name}，使用默认方法")
                methods = ['sklearn_mlp', 'pytorch_mlp']
            
            available_methods, unavailable_methods = filter_available_methods(methods)
            
            if unavailable_methods:
                print(f"⚠️ 跳过不可用的方法: {unavailable_methods}")
            
            return available_methods
        
        # 方式4: 任务特定推荐
        if 'recommendation_type' in kwargs:
            rec_type = kwargs['recommendation_type']
            methods = get_task_recommendations(task_type, rec_type)
            available_methods, unavailable_methods = filter_available_methods(methods)
            
            if unavailable_methods:
                print(f"⚠️ 跳过不可用的方法: {unavailable_methods}")
            
            return available_methods
        
        # 默认方式：向后兼容
        return ['sklearn', 'pytorch']
    
    def _train_baseline_method(self, method_name: str, X_train, y_train, X_val, y_val, 
                              X_test, y_test, task_type: str, verbose: bool, **kwargs):
        """
        训练指定的基准方法
        
        Returns:
            包含方法结果的字典，格式: {method_name: {val: {...}, test: {...}}}
        """
        try:
            if verbose:
                print(f"训练 {method_name}...")
            
            # 获取方法配置
            method_config = get_method_config(method_name)
            if not method_config:
                print(f"❌ 未知方法: {method_name}")
                return None
            
            # 合并参数：默认配置 + 用户传入的参数
            method_params = method_config['params'].copy()
            
            # 从kwargs中提取相关参数
            if 'baseline_config' in kwargs:
                config = kwargs['baseline_config']
                if isinstance(config, dict) and 'method_params' in config:
                    user_params = config['method_params'].get(method_name, {})
                    method_params.update(user_params)
            
            # 创建模型
            model = self.method_factory.create_model(method_name, task_type, **method_params)
            
            # 训练和评估
            results = self.method_factory.train_and_evaluate(
                method_name, model, X_train, y_train, X_val, y_val, X_test, y_test, task_type
            )
            
            # 返回格式化结果
            display_name = method_config.get('name', method_name)
            return {display_name: results}
            
        except Exception as e:
            if verbose:
                print(f"❌ 训练 {method_name} 时出错: {str(e)}")
            return None
    
    def list_available_baseline_methods(self) -> dict:
        """列出所有可用的基准方法"""
        all_methods = list_available_methods()
        available = {}
        
        for method in all_methods:
            config = get_method_config(method)
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
        
        # 按类型分组
        by_type = {}
        for method, info in methods.items():
            method_type = info['type']
            if method_type not in by_type:
                by_type[method_type] = []
            by_type[method_type].append((method, info))
        
        # 打印各类型的方法
        for method_type, method_list in by_type.items():
            print(f"\n📊 {method_type.title()} Methods:")
            print("-" * 40)
            
            for method, info in method_list:
                status = "✅" if info['available'] else "❌"
                print(f"  {status} {method:<20} - {info['name']}")
        
        # 打印依赖状态
        self.dependency_checker.print_dependency_status()
    
    def _train_sklearn_with_external_validation(self, model, X_train, y_train, X_val, y_val, 
                                              patience=50, tol=1e-4, task_type='regression'):
        """
        使用外部验证集训练sklearn模型并实现早停
        
        Args:
            model: sklearn模型实例
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            patience: 早停patience
            tol: 早停tolerance
            task_type: 任务类型
        
        Returns:
            训练好的模型
        """
        from sklearn.metrics import mean_squared_error, log_loss
        
        best_score = float('inf')
        patience_counter = 0
        best_model = None
        
        # sklearn的增量训练策略
        for epoch in range(model.max_iter):
            # 执行一轮训练（使用partial_fit或设置max_iter=1）
            if hasattr(model, 'partial_fit'):
                # 支持增量训练的模型
                if epoch == 0:
                    model.partial_fit(X_train, y_train)
                else:
                    model.partial_fit(X_train, y_train)
            else:
                # 不支持增量训练的模型，设置较小的max_iter并重新训练
                temp_model = model.__class__(**model.get_params())
                temp_model.max_iter = epoch + 1
                temp_model.warm_start = True
                temp_model.fit(X_train, y_train)
                model = temp_model
            
            # 在验证集上评估
            try:
                val_pred = model.predict(X_val)
                
                if task_type == 'regression':
                    val_score = mean_squared_error(y_val, val_pred)
                else:
                    try:
                        val_proba = model.predict_proba(X_val)
                        val_score = log_loss(y_val, val_proba)
                    except:
                        # 如果predict_proba失败，使用简单的错误率
                        val_score = 1.0 - model.score(X_val, y_val)
                
                # 早停检查
                if val_score < best_score - tol:
                    best_score = val_score
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model = model.__class__(**model.get_params())
                    if hasattr(model, 'coefs_'):
                        # 深拷贝训练好的参数
                        import copy
                        best_model = copy.deepcopy(model)
                    else:
                        best_model.fit(X_train, y_train)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
                    
            except Exception as e:
                # 如果评估失败，继续训练
                continue
        
        # 返回最佳模型或当前模型
        return best_model if best_model is not None else model