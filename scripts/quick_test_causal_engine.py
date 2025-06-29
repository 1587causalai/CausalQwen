#!/usr/bin/env python3
"""
CausalEngine 快速测试脚本 - causal-sklearn分支版本
简单灵活的端对端测试，支持回归和分类任务
基于原始CausalEngine分支脚本完整复现功能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
import os
import sys
import warnings

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的CausalEngine实现
from causal_sklearn._causal_engine import create_causal_regressor, create_causal_classifier
from causal_sklearn.regressor import MLPCausalRegressor
from causal_sklearn.utils import causal_split

warnings.filterwarnings('ignore')


class QuickTester:
    """
    CausalEngine快速测试器
    
    使用方法:
    tester = QuickTester()
    tester.test_regression() 或 tester.test_classification()
    """
    
    def __init__(self):
        self.results = {}
    
    
    def create_pytorch_model(self, input_size, output_size, hidden_sizes):
        """创建PyTorch基线模型"""
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def train_pytorch_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                          epochs=1000, lr=0.001, task='regression', patience=50, tol=1e-4):
        """训练PyTorch模型"""
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
    
    def test_regression(self, 
                       # 数据设置
                       n_samples=2000, n_features=12, noise=0.3, random_state=42,
                       anomaly_ratio=0.30,
                       
                       # 网络结构
                       hidden_layer_sizes=(128, 64), causal_size=None,
                       
                       # CausalEngine参数
                       gamma_init=1.0, b_noise_init=1.0, b_noise_trainable=True,
                       
                       # 训练参数
                       max_iter=5000, learning_rate=0.01,
                       patience=500, tol=1e-8,
                       
                       # 显示设置
                       verbose=True):
        """
        回归任务快速测试
        
        可调参数说明:
        - noise: 数据噪声标准差 (sklearn make_regression参数)
        - anomaly_ratio: 标签异常比例 (0.0-0.5, 仅影响train/val)
        - gamma_init: γ_U初始化值 (建议1.0-20.0)
        - b_noise_init: 外生噪声初始值 (建议0.01-1.0)
        - b_noise_trainable: b_noise是否可训练
        """
        if verbose:
            print("🔬 CausalEngine回归测试")
            print("=" * 60)
            print(f"数据: {n_samples}样本, {n_features}特征, 噪声{noise}")
            print(f"标签异常: {anomaly_ratio:.1%} (复合异常) - 仅影响train+val，test保持干净")
            print(f"网络结构: {hidden_layer_sizes}, causal_size={causal_size}")
            print(f"CausalEngine: γ_init={gamma_init}, b_noise_init={b_noise_init}, trainable={b_noise_trainable}")
            print()
        
        # 生成数据
        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                              noise=noise, random_state=random_state)
        
        # 使用 causal_split 进行3分割，自动处理异常注入
        # 注意：MLPCausalRegressor有自己的验证集分割，所以我们只需要train/test分割
        # 但为了与PyTorch基线公平比较（它需要外部验证集），我们仍然进行3分割
        X_train, X_val, X_test, y_train, y_val, y_test = causal_split(
            X, y, test_size=0.2, val_size=0.25, random_state=random_state,
            anomaly_ratio=anomaly_ratio, anomaly_type='regression')
        
        # 为了与sklearn/pytorch基线模型公平比较，我们先对它们进行训练
        # 它们需要手动进行数据缩放
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        # y_test不进行缩放，因为我们将在原始尺度上评估所有模型
        
        results = {}
        
        # 1. sklearn MLPRegressor
        if verbose: print("训练 sklearn MLPRegressor...")
        sklearn_reg = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate,
            early_stopping=True,
            validation_fraction=0.2, # Sklearn MLP自己的验证集分割
            n_iter_no_change=50,
            tol=1e-4,
            random_state=random_state,
            alpha=0.0001
        )
        # sklearn模型在缩放后的数据上训练
        sklearn_reg.fit(np.vstack([X_train_scaled, X_val_scaled]), np.concatenate([y_train_scaled, y_val_scaled]))
        
        # 预测并在原始尺度上评估
        sklearn_pred_test_scaled = sklearn_reg.predict(X_test_scaled)
        sklearn_pred_val_scaled = sklearn_reg.predict(X_val_scaled)
        
        sklearn_pred_test = scaler_y.inverse_transform(sklearn_pred_test_scaled.reshape(-1,1)).ravel()
        sklearn_pred_val = scaler_y.inverse_transform(sklearn_pred_val_scaled.reshape(-1,1)).ravel()

        results['sklearn'] = {
            'test': {
                'MAE': mean_absolute_error(y_test, sklearn_pred_test),
                'MdAE': median_absolute_error(y_test, sklearn_pred_test), 
                'RMSE': np.sqrt(mean_squared_error(y_test, sklearn_pred_test)),
                'R²': r2_score(y_test, sklearn_pred_test)
            },
            'val': {
                'MAE': mean_absolute_error(y_val, sklearn_pred_val),
                'MdAE': median_absolute_error(y_val, sklearn_pred_val), 
                'RMSE': np.sqrt(mean_squared_error(y_val, sklearn_pred_val)),
                'R²': r2_score(y_val, sklearn_pred_val)
            }
        }
        
        # 2. PyTorch基线
        if verbose: print("训练 PyTorch基线...")
        pytorch_model = self.create_pytorch_model(n_features, 1, hidden_layer_sizes)
        pytorch_model = self.train_pytorch_model(
            pytorch_model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            epochs=max_iter, lr=learning_rate, task='regression',
            patience=50, tol=1e-4)
        
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_pred_test_scaled = pytorch_model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()
            pytorch_pred_val_scaled = pytorch_model(torch.FloatTensor(X_val_scaled)).squeeze().numpy()
        
        pytorch_pred_test = scaler_y.inverse_transform(pytorch_pred_test_scaled.reshape(-1,1)).ravel()
        pytorch_pred_val = scaler_y.inverse_transform(pytorch_pred_val_scaled.reshape(-1,1)).ravel()

        results['pytorch'] = {
            'test': {
                'MAE': mean_absolute_error(y_test, pytorch_pred_test),
                'MdAE': median_absolute_error(y_test, pytorch_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, pytorch_pred_test)),
                'R²': r2_score(y_test, pytorch_pred_test)
            },
            'val': {
                'MAE': mean_absolute_error(y_val, pytorch_pred_val),
                'MdAE': median_absolute_error(y_val, pytorch_pred_val),
                'RMSE': np.sqrt(mean_squared_error(y_val, pytorch_pred_val)),
                'R²': r2_score(y_val, pytorch_pred_val)
            }
        }
        
        # Causal Regressor 公共训练数据（不缩放，模型内部处理）
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        # 3. CausalEngine deterministic模式
        if verbose: print("训练 MLPCausalRegressor (deterministic)...")
        causal_det = MLPCausalRegressor(
            perception_hidden_layers=hidden_layer_sizes,
            abduction_hidden_layers=(),
            mode='deterministic',
            gamma_init=gamma_init,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable,
            max_iter=max_iter,
            learning_rate=learning_rate,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=patience,
            tol=tol,
            random_state=random_state,
            verbose=verbose
        )
        causal_det.fit(X_train_full, y_train_full)
        
        # 预测并评估
        causal_det_pred_test = causal_det.predict(X_test)
        causal_det_pred_val = causal_det.predict(X_val)
        
        results['deterministic'] = {
            'test': {
                'MAE': mean_absolute_error(y_test, causal_det_pred_test),
                'MdAE': median_absolute_error(y_test, causal_det_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, causal_det_pred_test)),
                'R²': r2_score(y_test, causal_det_pred_test)
            },
            'val': {
                'MAE': mean_absolute_error(y_val, causal_det_pred_val),
                'MdAE': median_absolute_error(y_val, causal_det_pred_val),
                'RMSE': np.sqrt(mean_squared_error(y_val, causal_det_pred_val)),
                'R²': r2_score(y_val, causal_det_pred_val)
            }
        }
        
        # 4. CausalEngine standard模式
        if verbose: print("训练 MLPCausalRegressor (standard)...")
        causal_std = MLPCausalRegressor(
            perception_hidden_layers=hidden_layer_sizes,
            abduction_hidden_layers=(),
            mode='standard',
            gamma_init=gamma_init,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable,
            max_iter=max_iter,
            learning_rate=learning_rate,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=patience,
            tol=tol,
            random_state=random_state,
            verbose=verbose
        )
        causal_std.fit(X_train_full, y_train_full)

        # 预测并评估
        causal_std_pred_test = causal_std.predict(X_test)
        causal_std_pred_val = causal_std.predict(X_val)

        results['standard'] = {
            'test': {
                'MAE': mean_absolute_error(y_test, causal_std_pred_test),
                'MdAE': median_absolute_error(y_test, causal_std_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, causal_std_pred_test)),
                'R²': r2_score(y_test, causal_std_pred_test)
            },
            'val': {
                'MAE': mean_absolute_error(y_val, causal_std_pred_val),
                'MdAE': median_absolute_error(y_val, causal_std_pred_val),
                'RMSE': np.sqrt(mean_squared_error(y_val, causal_std_pred_val)),
                'R²': r2_score(y_val, causal_std_pred_val)
            }
        }
        
        # 显示结果
        if verbose:
            print("\n📊 回归结果对比:")
            print("=" * 120)
            print(f"{'方法':<15} {'验证集':<50} {'测试集':<50}")
            print(f"{'':15} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10}")
            print("-" * 120)
            for method, metrics in results.items():
                val_m = metrics['val']
                test_m = metrics['test']
                print(f"{method:<15} {val_m['MAE']:<10.4f} {val_m['MdAE']:<10.4f} {val_m['RMSE']:<10.4f} {val_m['R²']:<10.4f} "
                      f"{test_m['MAE']:<10.4f} {test_m['MdAE']:<10.4f} {test_m['RMSE']:<10.4f} {test_m['R²']:<10.4f}")
            print("=" * 120)
        
        self.results['regression'] = results
        return results
    
    def test_classification(self,
                           # 数据设置  
                           n_samples=3000, n_features=15, n_classes=3, n_informative=None,
                           class_sep=0.3, random_state=42,
                           label_noise_ratio=0.3, label_noise_type='flip',
                           
                           # 网络结构
                           hidden_layer_sizes=(128, 64), causal_size=None,
                           
                           # CausalEngine参数
                           gamma_init=1.0, b_noise_init=1.0, ovr_threshold_init=0.0,
                           b_noise_trainable=True,
                           
                           # 训练参数
                           max_iter=5000, learning_rate=0.01, early_stopping=True,
                           patience=500, tol=1e-8,
                           
                           # 显示设置
                           verbose=True):
        """
        分类任务快速测试
        
        可调参数说明:
        - n_classes: 类别数 (2-10)
        - class_sep: 类别分离度 (0.5-2.0，越大越容易分类)
        - label_noise_ratio: 标签噪声比例 (0.0-0.5)
        - ovr_threshold_init: OvR阈值初始化 (-2.0到2.0)
        """
        if verbose:
            print("🎯 CausalEngine分类测试")
            print("=" * 60)
            print(f"数据: {n_samples}样本, {n_features}特征, {n_classes}类别")
            print(f"标签噪声: {label_noise_ratio:.1%} ({label_noise_type}) - 仅影响train+val，test保持干净")
            print(f"网络结构: {hidden_layer_sizes}, causal_size={causal_size}")
            print(f"CausalEngine: γ_init={gamma_init}, b_noise_init={b_noise_init}, ovr_threshold={ovr_threshold_init}")
            print()
        
        # 生成数据
        if n_informative is None:
            n_informative = min(n_features, max(2, n_features // 2))
            
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            n_informative=n_informative, n_redundant=0, n_clusters_per_class=1,
            class_sep=class_sep, random_state=random_state
        )
        
        # 使用 causal_split 进行3分割，自动处理异常注入
        X_train, X_val, X_test, y_train, y_val, y_test = causal_split(
            X, y, test_size=0.2, val_size=0.25, random_state=random_state, stratify=y,
            anomaly_ratio=label_noise_ratio, anomaly_type='classification', 
            classification_anomaly_strategy='shuffle')
        
        # 标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        results = {}
        
        # 评估函数
        def evaluate_classification(y_true, y_pred):
            avg_method = 'binary' if n_classes == 2 else 'macro'
            return {
                'Acc': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average=avg_method, zero_division=0),
                'Recall': recall_score(y_true, y_pred, average=avg_method, zero_division=0),
                'F1': f1_score(y_true, y_pred, average=avg_method, zero_division=0)
            }
        
        # 1. sklearn MLPClassifier
        if verbose: print("训练 sklearn MLPClassifier...")
        sklearn_clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate,
            early_stopping=early_stopping,
            validation_fraction=0.2,
            n_iter_no_change=50,
            tol=1e-4,
            random_state=random_state,
            alpha=0.0001
        )
        sklearn_clf.fit(X_train_scaled, y_train)
        
        # 显示sklearn训练信息
        if verbose:
            print(f"   sklearn MLPClassifier训练完成:")
            print(f"   - 实际训练轮数: {sklearn_clf.n_iter_}")
            if hasattr(sklearn_clf, 'loss_curve_') and sklearn_clf.loss_curve_:
                print(f"   - 最终训练损失: {sklearn_clf.loss_curve_[-1]:.6f}")
            if hasattr(sklearn_clf, 'validation_scores_') and sklearn_clf.validation_scores_:
                print(f"   - 最终验证损失: {sklearn_clf.validation_scores_[-1]:.6f}")
            if sklearn_clf.n_iter_ < max_iter:
                print(f"   - 早停触发: 在{sklearn_clf.n_iter_}轮停止 (最大{max_iter}轮)")
        
        sklearn_pred_test = sklearn_clf.predict(X_test_scaled)
        sklearn_pred_val = sklearn_clf.predict(X_val_scaled)
        results['sklearn'] = {
            'test': evaluate_classification(y_test, sklearn_pred_test),
            'val': evaluate_classification(y_val, sklearn_pred_val)
        }
        
        # 2. PyTorch基线
        if verbose: print("训练 PyTorch基线...")
        pytorch_model = self.create_pytorch_model(n_features, n_classes, hidden_layer_sizes)
        pytorch_model = self.train_pytorch_model(
            pytorch_model, X_train_scaled, y_train, X_val_scaled, y_val,
            epochs=max_iter, lr=learning_rate, task='classification',
            patience=50, tol=1e-4)
        
        # 显示PyTorch训练信息
        if verbose:
            print(f"   PyTorch基线训练完成:")
            print(f"   - 实际训练轮数: {pytorch_model.n_iter_}")
            print(f"   - 最终验证损失: {pytorch_model.final_loss_:.6f}")
            if pytorch_model.n_iter_ < max_iter:
                print(f"   - 早停触发: 在{pytorch_model.n_iter_}轮停止 (最大{max_iter}轮)")
            print(f"   - 早停参数: patience=50, tol=1e-4")
        
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_outputs_test = pytorch_model(torch.FloatTensor(X_test_scaled))
            pytorch_pred_test = torch.argmax(pytorch_outputs_test, dim=1).numpy()
            pytorch_outputs_val = pytorch_model(torch.FloatTensor(X_val_scaled))
            pytorch_pred_val = torch.argmax(pytorch_outputs_val, dim=1).numpy()
        results['pytorch'] = {
            'test': evaluate_classification(y_test, pytorch_pred_test),
            'val': evaluate_classification(y_val, pytorch_pred_val)
        }
        
        # 3. CausalEngine deterministic模式
        if verbose: print("训练 CausalEngine (deterministic)...")
        causal_det = self.train_causal_engine(
            X_train_scaled, y_train, X_val_scaled, y_val,
            'classification', 'deterministic', hidden_layer_sizes, max_iter, learning_rate,
            patience, tol, gamma_init, b_noise_init, b_noise_trainable, ovr_threshold_init, verbose=verbose
        )
        
        if verbose:
            print(f"   Deterministic模式训练完成:")
            print(f"   - 实际训练轮数: [已完成]")
            print(f"   - 早停触发: [已完成] (最大{max_iter}轮)")
        
        # 预测
        device = next(causal_det.parameters()).device
        causal_det.eval()
        with torch.no_grad():
            X_test_torch = torch.FloatTensor(X_test_scaled).to(device)
            X_val_torch = torch.FloatTensor(X_val_scaled).to(device)
            causal_det_pred_test = causal_det.predict(X_test_torch, 'deterministic').cpu().numpy()
            causal_det_pred_val = causal_det.predict(X_val_torch, 'deterministic').cpu().numpy()
        
        results['deterministic'] = {
            'test': evaluate_classification(y_test, causal_det_pred_test),
            'val': evaluate_classification(y_val, causal_det_pred_val)
        }
        
        # 4. CausalEngine standard模式
        if verbose: print("训练 CausalEngine (standard)...")
        causal_std = self.train_causal_engine(
            X_train_scaled, y_train, X_val_scaled, y_val,
            'classification', 'standard', hidden_layer_sizes, max_iter, learning_rate,
            patience, tol, gamma_init, b_noise_init, b_noise_trainable, ovr_threshold_init, verbose=verbose
        )
        
        if verbose:
            print(f"   Standard模式训练完成:")
            print(f"   - 实际训练轮数: [已完成]")
            print(f"   - 早停触发: [已完成] (最大{max_iter}轮)")
            print(f"   - 耐心值设置: 50轮无改善后停止")
        
        # 预测
        causal_std.eval()
        with torch.no_grad():
            X_test_torch = torch.FloatTensor(X_test_scaled).to(device)
            X_val_torch = torch.FloatTensor(X_val_scaled).to(device)
            causal_std_pred_test = causal_std.predict(X_test_torch, 'standard').cpu().numpy()
            causal_std_pred_val = causal_std.predict(X_val_torch, 'standard').cpu().numpy()
        
        results['standard'] = {
            'test': evaluate_classification(y_test, causal_std_pred_test),
            'val': evaluate_classification(y_val, causal_std_pred_val)
        }
        
        # 显示结果
        if verbose:
            print(f"\n📊 {n_classes}分类结果对比:")
            print("=" * 120)
            print(f"{'方法':<15} {'验证集':<50} {'测试集':<50}")
            print(f"{'':15} {'Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
            print("-" * 120)
            for method, metrics in results.items():
                val_m = metrics['val']
                test_m = metrics['test']
                print(f"{method:<15} {val_m['Acc']:<10.4f} {val_m['Precision']:<10.4f} {val_m['Recall']:<10.4f} {val_m['F1']:<10.4f} "
                      f"{test_m['Acc']:<10.4f} {test_m['Precision']:<10.4f} {test_m['Recall']:<10.4f} {test_m['F1']:<10.4f}")
            print("=" * 120)
        
        self.results['classification'] = results
        return results


def main():
    """主测试函数 - 完整复现原始脚本逻辑"""
    print("🚀 CausalEngine快速测试脚本")
    print("=" * 50)
    
    print("\n1️⃣ 详细回归测试:")
    print("   数据: 2000样本, 12特征")
    print("   噪声: 数据噪声0.1, 标签噪声10.0%")
    print("   网络: (128, 64), causal_size=None")
    print("   参数: γ_init=1.0, b_noise_init=1.0, trainable=True")
    print("   训练: max_iter=5000, lr=0.01, early_stop=True, patience=500, tol=1e-08")
    
    tester = QuickTester()
    tester.test_regression()
    
    print("\n2️⃣ 详细分类测试:")
    print("   数据: 3000样本, 15特征, 3类别")
    print("   噪声: 标签噪声20.0%, 分离度0.3")
    print("   网络: (128, 64), causal_size=None")
    print("   参数: γ_init=1.0, b_noise_init=1.0, ovr_threshold=0.0")
    print("   训练: max_iter=5000, lr=0.01, early_stop=True, patience=500, tol=1e-08")
    
    tester.test_classification()


if __name__ == "__main__":
    main()