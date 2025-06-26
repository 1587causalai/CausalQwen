#!/usr/bin/env python3
"""
CausalEngine 快速测试脚本
简单灵活的端对端测试，支持回归和分类任务
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier

class QuickTester:
    """
    CausalEngine快速测试器
    
    使用方法:
    tester = QuickTester()
    tester.test_regression() 或 tester.test_classification()
    """
    
    def __init__(self):
        self.results = {}
    
    def add_label_anomalies(self, y, anomaly_ratio=0.1, anomaly_type='regression'):
        """
        给标签添加异常 - 更实用的异常模拟
        
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
    
    def create_pytorch_model(self, input_size, output_size, hidden_sizes, task='regression'):
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
        # patience 和 tol 从参数传入
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
                
                if val_loss < best_loss - tol:  # 使用tol参数
                    best_loss = val_loss
                    no_improve = 0
                    # 保存最佳模型
                    import pickle
                    with open(best_model_path, 'wb') as f:
                        pickle.dump(model.state_dict(), f)
                    if epoch == 0:  # 只在第一次保存时提示存储位置
                        print(f"   最佳模型临时存储: {best_model_path}")
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    break
        
        # 恢复最佳模型
        import pickle
        import os
        if os.path.exists(best_model_path):
            with open(best_model_path, 'rb') as f:
                model.load_state_dict(pickle.load(f))
            print(f"   已恢复最佳模型，删除临时文件: {best_model_path}")
            os.remove(best_model_path)  # 清理临时文件
        
        # 将实际训练轮数作为属性添加到模型
        model.n_iter_ = epoch + 1
        model.final_loss_ = best_loss
        return model
    
    def test_regression(self, 
                       # 数据设置
                       n_samples=1000, n_features=10, noise=0.1, random_state=42,
                       anomaly_ratio=0.0,
                       
                       # 网络结构
                       hidden_layer_sizes=(64, 32), causal_size=None,
                       
                       # CausalEngine参数
                       gamma_init=10.0, b_noise_init=0.1, b_noise_trainable=True,
                       
                       # 训练参数
                       max_iter=1000, learning_rate=0.001, early_stopping=True,
                       
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
        
        # 先分割数据：保持test set干净
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)
        
        # 只对训练数据添加标签异常（保持test set干净用于真实评估）
        if anomaly_ratio > 0:
            y_train = self.add_label_anomalies(y_train, anomaly_ratio, 'regression')
        
        # 分割训练数据为训练集和验证集（验证集也有异常）
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state)
        
        results = {}
        
        # 1. sklearn MLPRegressor
        if verbose: print("训练 sklearn MLPRegressor...")
        sklearn_reg = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate,
            early_stopping=early_stopping,
            validation_fraction=0.15,  # 统一验证集比例
            n_iter_no_change=50,  # 统一耐心值
            tol=1e-4,  # 统一容忍度
            random_state=random_state,
            alpha=0.0001
        )
        sklearn_reg.fit(X_train, y_train)
        sklearn_pred_test = sklearn_reg.predict(X_test)
        sklearn_pred_val = sklearn_reg.predict(X_val)
        
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
        pytorch_model = self.create_pytorch_model(n_features, 1, hidden_layer_sizes, 'regression')
        pytorch_model = self.train_pytorch_model(
            pytorch_model, X_train, y_train, X_val, y_val, 
            epochs=max_iter, lr=learning_rate, task='regression',
            patience=50, tol=1e-4)  # 统一的早停参数
        
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_pred_test = pytorch_model(torch.FloatTensor(X_test)).squeeze().numpy()
            pytorch_pred_val = pytorch_model(torch.FloatTensor(X_val)).squeeze().numpy()
        
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
        
        # 3. CausalEngine deterministic模式
        if verbose: print("训练 CausalEngine (deterministic)...")
        causal_det = MLPCausalRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            causal_size=causal_size,
            mode='deterministic',
            gamma_init=gamma_init,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable,
            max_iter=max_iter,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            n_iter_no_change=50,  # 统一耐心值
            tol=1e-4,  # 统一容忍度
            validation_fraction=0.15,  # 统一验证集比例
            random_state=random_state,
            verbose=verbose  # 传递verbose参数以显示训练信息
        )
        
        # b_noise_trainable现在在初始化时设置，无需后处理
            
        causal_det.fit(X_train, y_train)
        causal_det_result_test = causal_det.predict(X_test)
        causal_det_result_val = causal_det.predict(X_val)
        
        # 处理CausalEngine返回格式
        if isinstance(causal_det_result_test, dict):
            causal_det_pred_test = causal_det_result_test['predictions']
        else:
            causal_det_pred_test = causal_det_result_test
        
        if isinstance(causal_det_result_val, dict):
            causal_det_pred_val = causal_det_result_val['predictions']
        else:
            causal_det_pred_val = causal_det_result_val
        
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
        if verbose: print("训练 CausalEngine (standard)...")
        causal_std = MLPCausalRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            causal_size=causal_size,
            mode='standard',
            gamma_init=gamma_init,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable,
            max_iter=max_iter,
            learning_rate=learning_rate,  # 统一学习率
            early_stopping=early_stopping,  # 统一早停策略
            n_iter_no_change=50,  # 统一耐心值
            tol=1e-4,  # 统一容忍度
            validation_fraction=0.15,  # 统一验证集比例
            random_state=random_state,
            verbose=verbose  # 传递verbose参数以显示训练信息
        )
        causal_std.fit(X_train, y_train)
        causal_std_result_test = causal_std.predict(X_test)
        causal_std_result_val = causal_std.predict(X_val)
        
        # 处理CausalEngine返回格式
        if isinstance(causal_std_result_test, dict):
            causal_std_pred_test = causal_std_result_test['predictions']
        else:
            causal_std_pred_test = causal_std_result_test
        
        if isinstance(causal_std_result_val, dict):
            causal_std_pred_val = causal_std_result_val['predictions']
        else:
            causal_std_pred_val = causal_std_result_val
        
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
                           n_samples=1000, n_features=10, n_classes=2, n_informative=None,
                           class_sep=1.0, random_state=42,
                           label_noise_ratio=0.0, label_noise_type='flip',
                           
                           # 网络结构
                           hidden_layer_sizes=(64, 32), causal_size=None,
                           
                           # CausalEngine参数
                           gamma_init=10.0, b_noise_init=0.1, ovr_threshold_init=0.0,
                           b_noise_trainable=True,
                           
                           # 训练参数
                           max_iter=1000, learning_rate=0.001, early_stopping=True,
                           
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
        
        # 先分割数据：保持test set干净
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y)
        
        # 只对训练数据添加标签异常（保持test set干净用于真实评估）
        if label_noise_ratio > 0:
            y_train = self.add_label_anomalies(y_train, label_noise_ratio, 'classification')
        
        # 分割训练数据为训练集和验证集（验证集也有噪声）
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
        
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
            validation_fraction=0.15,  # 统一验证集比例
            n_iter_no_change=50,  # 统一耐心值
            tol=1e-4,  # 统一容忍度
            random_state=random_state,
            alpha=0.0001
        )
        sklearn_clf.fit(X_train, y_train)
        
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
            else:
                print(f"   - 训练完整: 完成所有{max_iter}轮训练")
        
        sklearn_pred_test = sklearn_clf.predict(X_test)
        sklearn_pred_val = sklearn_clf.predict(X_val)
        results['sklearn'] = {
            'test': evaluate_classification(y_test, sklearn_pred_test),
            'val': evaluate_classification(y_val, sklearn_pred_val)
        }
        
        # 2. PyTorch基线
        if verbose: print("训练 PyTorch基线...")
        pytorch_model = self.create_pytorch_model(n_features, n_classes, hidden_layer_sizes, 'classification')
        pytorch_model = self.train_pytorch_model(
            pytorch_model, X_train, y_train, X_val, y_val,
            epochs=max_iter, lr=learning_rate, task='classification',
            patience=50, tol=1e-4)  # 统一的早停参数
        
        # 显示PyTorch训练信息
        if verbose:
            print(f"   PyTorch基线训练完成:")
            print(f"   - 实际训练轮数: {pytorch_model.n_iter_}")
            print(f"   - 最终验证损失: {pytorch_model.final_loss_:.6f}")
            if pytorch_model.n_iter_ < max_iter:
                print(f"   - 早停触发: 在{pytorch_model.n_iter_}轮停止 (最大{max_iter}轮)")
            else:
                print(f"   - 训练完整: 完成所有{max_iter}轮训练")
            print(f"   - 早停参数: patience=50, tol=1e-4")
        
        pytorch_model.eval()
        with torch.no_grad():
            # 测试集预测
            pytorch_outputs_test = pytorch_model(torch.FloatTensor(X_test))
            pytorch_pred_test = torch.argmax(pytorch_outputs_test, dim=1).numpy()
            # 验证集预测
            pytorch_outputs_val = pytorch_model(torch.FloatTensor(X_val))
            pytorch_pred_val = torch.argmax(pytorch_outputs_val, dim=1).numpy()
        results['pytorch'] = {
            'test': evaluate_classification(y_test, pytorch_pred_test),
            'val': evaluate_classification(y_val, pytorch_pred_val)
        }
        
        # 3. CausalEngine deterministic模式
        if verbose: print("训练 CausalEngine (deterministic)...")
        causal_det = MLPCausalClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            causal_size=causal_size,
            mode='deterministic',
            gamma_init=gamma_init,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable,
            ovr_threshold_init=ovr_threshold_init,
            max_iter=max_iter,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            n_iter_no_change=50,  # 统一的耐心值
            tol=1e-4,  # 统一的容忍度
            validation_fraction=0.15,  # 统一验证集比例
            random_state=random_state,
            verbose=verbose  # 传递verbose参数以显示训练信息
        )
        causal_det.fit(X_train, y_train)
        
        # 显示训练信息
        if verbose:
            print(f"   Deterministic模式训练完成:")
            print(f"   - 实际训练轮数: {causal_det.n_iter_}")
            if hasattr(causal_det, 'validation_scores_') and causal_det.validation_scores_:
                print(f"   - 最终验证损失: {causal_det.validation_scores_[-1]:.6f}")
            if early_stopping and causal_det.n_iter_ < max_iter:
                print(f"   - 早停触发: 在{causal_det.n_iter_}轮停止 (最大{max_iter}轮)")
            else:
                print(f"   - 训练完整: 完成所有{max_iter}轮训练")
        
        # 测试集和验证集预测
        causal_det_result_test = causal_det.predict(X_test)
        causal_det_pred_test = causal_det_result_test['predictions'] if isinstance(causal_det_result_test, dict) else causal_det_result_test
        causal_det_result_val = causal_det.predict(X_val)
        causal_det_pred_val = causal_det_result_val['predictions'] if isinstance(causal_det_result_val, dict) else causal_det_result_val
        results['deterministic'] = {
            'test': evaluate_classification(y_test, causal_det_pred_test),
            'val': evaluate_classification(y_val, causal_det_pred_val)
        }
        
        # 4. CausalEngine standard模式
        if verbose: print("训练 CausalEngine (standard)...")
        causal_std = MLPCausalClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            causal_size=causal_size,
            mode='standard',
            gamma_init=gamma_init,
            b_noise_init=b_noise_init,
            b_noise_trainable=b_noise_trainable,
            ovr_threshold_init=ovr_threshold_init,
            max_iter=max_iter,
            learning_rate=learning_rate,  # 统一学习率
            early_stopping=early_stopping,  # 统一早停策略
            n_iter_no_change=50,  # 统一的耐心值
            tol=1e-4,  # 统一的容忍度
            validation_fraction=0.15,  # 统一验证集比例
            random_state=random_state,
            verbose=verbose  # 传递verbose参数以显示训练信息
        )
        causal_std.fit(X_train, y_train)
        
        # 显示训练信息
        if verbose:
            print(f"   Standard模式训练完成:")
            print(f"   - 实际训练轮数: {causal_std.n_iter_}")
            if hasattr(causal_std, 'validation_scores_') and causal_std.validation_scores_:
                print(f"   - 最终验证损失: {causal_std.validation_scores_[-1]:.6f}")
            if early_stopping and causal_std.n_iter_ < max_iter:
                print(f"   - 早停触发: 在{causal_std.n_iter_}轮停止 (最大{max_iter}轮)")
                print(f"   - 耐心值设置: 50轮无改善后停止")
            else:
                print(f"   - 训练完整: 完成所有{max_iter}轮训练")
        
        # 测试集和验证集预测
        causal_std_result_test = causal_std.predict(X_test)
        causal_std_pred_test = causal_std_result_test['predictions'] if isinstance(causal_std_result_test, dict) else causal_std_result_test
        causal_std_result_val = causal_std.predict(X_val)
        causal_std_pred_val = causal_std_result_val['predictions'] if isinstance(causal_std_result_val, dict) else causal_std_result_val
        results['standard'] = {
            'test': evaluate_classification(y_test, causal_std_pred_test),
            'val': evaluate_classification(y_val, causal_std_pred_val)
        }
        
        # 显示结果
        if verbose:
            print(f"\n📊 {n_classes}分类结果对比:")
            print("=" * 120)
            print(f"{'方法':<15} {'验证集':<50} {'测试集':<50}")
            print(f"{'':15} {'Acc':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
            print("-" * 120)
            for method, metrics in results.items():
                val_m = metrics['val']
                test_m = metrics['test']
                print(f"{method:<15} {val_m['Acc']:<10.4f} {val_m['Precision']:<12.4f} {val_m['Recall']:<10.4f} {val_m['F1']:<10.4f} "
                      f"{test_m['Acc']:<10.4f} {test_m['Precision']:<12.4f} {test_m['Recall']:<10.4f} {test_m['F1']:<10.4f}")
            print("=" * 120)
        
        self.results['classification'] = results
        return results

# 使用示例和快捷函数
def quick_regression_test(**kwargs):
    """快速回归测试"""
    tester = QuickTester()
    return tester.test_regression(**kwargs)

def quick_classification_test(**kwargs):
    """快速分类测试"""
    tester = QuickTester()
    return tester.test_classification(**kwargs)

if __name__ == "__main__":
    print("🚀 CausalEngine快速测试脚本")
    print("=" * 50)
    
    # =============================================================================
    # 🔧 参数配置区域 - 您可以自由调整以下所有参数
    # =============================================================================
    
    # 📊 数据参数 - 增大样本数确保充分训练
    REGRESSION_SAMPLES = 2000       # 回归样本数 (增大确保充分训练)
    REGRESSION_FEATURES = 12        # 回归特征数
    REGRESSION_NOISE = 0.1          # 回归数据噪声
    REGRESSION_LABEL_NOISE = 0.1    # 回归标签异常比例 (0.0-0.5, 复合异常)
    
    CLASSIFICATION_SAMPLES = 3000   # 分类样本数 (增大确保充分训练)
    CLASSIFICATION_FEATURES = 15    # 分类特征数
    CLASSIFICATION_CLASSES = 3      # 分类类别数
    CLASSIFICATION_SEPARATION = 0.3 # 类别分离度 (0.1-2.0，越大越容易)
    CLASSIFICATION_LABEL_NOISE = 0.2  # 分类标签异常比例 (0.0-0.5, 标签翻转)
    
    # 🏗️ 网络结构参数
    HIDDEN_LAYER_SIZES = (128, 64)  # MLP隐藏层结构
    CAUSAL_SIZE = None              # 因果表征维度 (None=自动设为最后隐藏层大小)
    
    # ⚙️ CausalEngine核心参数
    GAMMA_INIT_REGRESSION = 1.0     # 回归γ_U初始化值
    GAMMA_INIT_CLASSIFICATION = 1.0 # 分类γ_U初始化值
    B_NOISE_INIT = 1.0              # 外生噪声初始值 
    B_NOISE_TRAINABLE = False        # 外生噪声是否可训练
    OVR_THRESHOLD_INIT = 0.0        # OvR阈值初始化
    
    # 🎯 训练参数 - 早停策略配置
    MAX_ITER = 5000                 # 大训练轮数 (给standard模式足够时间收敛)
    LEARNING_RATE = 0.01            # 统一学习率 (所有方法使用相同学习率)
    EARLY_STOPPING = True           # 是否启用早停
    PATIENCE = 500                  # 早停耐心值 (n_iter_no_change)
    TOLERANCE = 1e-8                # 早停容忍度 (tol) - 验证损失改善的最小阈值
    VALIDATION_FRACTION = 0.15      # 验证集比例
    RANDOM_STATE = 42               # 随机种子
    VERBOSE = True                  # 是否显示详细信息
    
    # =============================================================================
    # 🧪 测试执行区域
    # =============================================================================
    
    # 示例1: 详细回归测试
    print("\n1️⃣ 详细回归测试:")
    print(f"   数据: {REGRESSION_SAMPLES}样本, {REGRESSION_FEATURES}特征")
    print(f"   噪声: 数据噪声{REGRESSION_NOISE}, 标签噪声{REGRESSION_LABEL_NOISE:.1%}")
    print(f"   网络: {HIDDEN_LAYER_SIZES}, causal_size={CAUSAL_SIZE}")
    print(f"   参数: γ_init={GAMMA_INIT_REGRESSION}, b_noise_init={B_NOISE_INIT}, trainable={B_NOISE_TRAINABLE}")
    print(f"   训练: max_iter={MAX_ITER}, lr={LEARNING_RATE}, early_stop={EARLY_STOPPING}, patience={PATIENCE}, tol={TOLERANCE}")
    
    quick_regression_test(
        # 数据设置
        n_samples=REGRESSION_SAMPLES,
        n_features=REGRESSION_FEATURES, 
        noise=REGRESSION_NOISE,
        random_state=RANDOM_STATE,
        anomaly_ratio=REGRESSION_LABEL_NOISE,
        
        # 网络结构
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        causal_size=CAUSAL_SIZE,
        
        # CausalEngine参数
        gamma_init=GAMMA_INIT_REGRESSION,
        b_noise_init=B_NOISE_INIT,
        b_noise_trainable=B_NOISE_TRAINABLE,
        
        # 训练参数
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        early_stopping=EARLY_STOPPING,
        
        # 显示设置
        verbose=VERBOSE
    )
    
    # 示例2: 详细分类测试
    print("\n2️⃣ 详细分类测试:")
    print(f"   数据: {CLASSIFICATION_SAMPLES}样本, {CLASSIFICATION_FEATURES}特征, {CLASSIFICATION_CLASSES}类别")
    print(f"   噪声: 标签噪声{CLASSIFICATION_LABEL_NOISE:.1%}, 分离度{CLASSIFICATION_SEPARATION}")
    print(f"   网络: {HIDDEN_LAYER_SIZES}, causal_size={CAUSAL_SIZE}")
    print(f"   参数: γ_init={GAMMA_INIT_CLASSIFICATION}, b_noise_init={B_NOISE_INIT}, ovr_threshold={OVR_THRESHOLD_INIT}")
    print(f"   训练: max_iter={MAX_ITER}, lr={LEARNING_RATE}, early_stop={EARLY_STOPPING}, patience={PATIENCE}, tol={TOLERANCE}")
    
    quick_classification_test(
        # 数据设置
        n_samples=CLASSIFICATION_SAMPLES,
        n_features=CLASSIFICATION_FEATURES,
        n_classes=CLASSIFICATION_CLASSES,
        n_informative=None,  # 自动设置
        class_sep=CLASSIFICATION_SEPARATION,
        random_state=RANDOM_STATE,
        label_noise_ratio=CLASSIFICATION_LABEL_NOISE,
        label_noise_type='flip',
        
        # 网络结构
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        causal_size=CAUSAL_SIZE,
        
        # CausalEngine参数
        gamma_init=GAMMA_INIT_CLASSIFICATION,
        b_noise_init=B_NOISE_INIT,
        b_noise_trainable=B_NOISE_TRAINABLE,
        ovr_threshold_init=OVR_THRESHOLD_INIT,
        
        # 训练参数
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        early_stopping=EARLY_STOPPING,
        
        # 显示设置
        verbose=VERBOSE
    )
    
    # # =============================================================================
    # # 💡 快速调参建议
    # # =============================================================================
    # print("\n💡 快速调参建议:")
    # print("   🔹 提高性能: 增大gamma_init (5.0→20.0), 调整网络大小")
    # print("   🔹 处理噪声: 开启b_noise_trainable, 调整b_noise_init (0.01→1.0)")
    # print("   🔹 分类调优: 调整ovr_threshold_init (-2.0→2.0)")
    # print("   🔹 训练稳定: 调整learning_rate, 增加max_iter")
    # print("   🔹 数据难度: 调整class_sep (分类), noise (回归)")
    
    # print("\n🎯 修改参数请编辑文件顶部的参数配置区域")
    # print("   或直接调用 quick_regression_test() / quick_classification_test() 函数")