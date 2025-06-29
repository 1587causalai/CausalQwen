#!/usr/bin/env python3
"""
sklearn vs PyTorch 深度对比分析脚本
===================================

专门用于找出sklearn MLPRegressor和手写PyTorch实现之间的性能差异根本原因。
逐一排查可能的差异点：网络结构、优化器、数据处理、训练过程等。

使用说明:
1. 运行 python scripts/sklearn_vs_pytorch_debug.py
2. 查看详细的对比分析结果
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import sys
import warnings

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_sklearn.regressor import MLPPytorchRegressor
from causal_sklearn.utils import causal_split

warnings.filterwarnings('ignore')

# =============================================================================
# 测试配置
# =============================================================================

TEST_CONFIG = {
    'n_samples': 1600,
    'n_features': 8,
    'noise': 0.2,
    'random_state': 42,
    'anomaly_ratio': 0.15,
    
    'hidden_layer_sizes': (64, 32),
    'max_iter': 1000,
    'learning_rate': 0.01,
    'validation_fraction': 0.2,
    'patience': 50,
    'tol': 1e-6,
    'alpha': 0.0
}

# =============================================================================
# sklearn参数详细检查
# =============================================================================

def analyze_sklearn_parameters():
    """详细分析sklearn MLPRegressor的所有参数"""
    print("🔍 sklearn MLPRegressor 参数详细分析")
    print("=" * 80)
    
    model = MLPRegressor()
    
    print("网络结构相关:")
    print(f"  hidden_layer_sizes: {model.hidden_layer_sizes}")
    print(f"  activation: {model.activation}")
    print(f"  solver: {model.solver}")
    print(f"  alpha (L2 penalty): {model.alpha}")
    
    print("\n训练参数:")
    print(f"  learning_rate: {model.learning_rate}")
    print(f"  learning_rate_init: {model.learning_rate_init}")
    print(f"  power_t: {model.power_t}")
    print(f"  max_iter: {model.max_iter}")
    print(f"  shuffle: {model.shuffle}")
    print(f"  random_state: {model.random_state}")
    print(f"  tol: {model.tol}")
    print(f"  verbose: {model.verbose}")
    print(f"  warm_start: {model.warm_start}")
    print(f"  momentum: {model.momentum}")
    print(f"  nesterovs_momentum: {model.nesterovs_momentum}")
    
    print("\n早停相关:")
    print(f"  early_stopping: {model.early_stopping}")
    print(f"  validation_fraction: {model.validation_fraction}")
    print(f"  beta_1: {model.beta_1}")
    print(f"  beta_2: {model.beta_2}")
    print(f"  epsilon: {model.epsilon}")
    print(f"  n_iter_no_change: {model.n_iter_no_change}")
    print(f"  max_fun: {model.max_fun}")
    
    print("\n批处理:")
    print(f"  batch_size: {model.batch_size}")
    
    return model

# =============================================================================
# 精确复现sklearn的PyTorch实现
# =============================================================================

class SklearnCompatibleMLP(nn.Module):
    """完全按照sklearn参数构建的PyTorch MLP"""
    
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 构建网络层
        prev_size = input_size
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # 输出层
        self.layers.append(nn.Linear(prev_size, output_size))
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'logistic':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # 输出层不加激活函数（回归任务）
        x = self.layers[-1](x)
        return x

class SklearnCompatibleRegressor:
    """完全按照sklearn参数实现的PyTorch回归器"""
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='constant',
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4, verbose=False,
                 warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        
        # 存储所有参数
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        
        self.model_ = None
        self.n_iter_ = None
        self.loss_ = None
    
    def fit(self, X, y):
        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # 数据分割
        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # 确定batch size
        n_samples = len(X_train)
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = self.batch_size if self.batch_size is not None else n_samples
        
        # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
        
        # 构建模型
        self.model_ = SklearnCompatibleMLP(
            input_size=X.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            output_size=1,
            activation=self.activation
        )
        
        # 设置优化器 - 精确匹配sklearn的Adam参数
        if self.solver == 'adam':
            optimizer = optim.Adam(
                self.model_.parameters(),
                lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
                weight_decay=self.alpha
            )
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")
        
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # 训练阶段
            self.model_.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # 数据shuffle
            if self.shuffle:
                indices = torch.randperm(n_samples)
                X_train_shuffled = X_train_tensor[indices]
                y_train_shuffled = y_train_tensor[indices]
            else:
                X_train_shuffled = X_train_tensor
                y_train_shuffled = y_train_tensor
            
            # Mini-batch训练
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_epoch_loss = epoch_loss / n_batches
            
            # 验证阶段
            if self.early_stopping and X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                    
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = self.model_.state_dict().copy()
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        if best_state_dict is not None:
                            self.model_.load_state_dict(best_state_dict)
                        break
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.6f}")
        
        self.n_iter_ = epoch + 1
        self.loss_ = avg_epoch_loss
        
        return self
    
    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X)
        
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            return outputs.squeeze().cpu().numpy()

# =============================================================================
# 权重初始化对比
# =============================================================================

def compare_weight_initialization():
    """对比权重初始化策略"""
    print("\n🎯 权重初始化对比")
    print("=" * 80)
    
    # sklearn模型
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        random_state=42,
        max_iter=1
    )
    
    # 生成简单数据用于初始化
    X_dummy = np.random.randn(100, 8)
    y_dummy = np.random.randn(100)
    sklearn_model.fit(X_dummy, y_dummy)
    
    print("sklearn权重统计:")
    for i, coef in enumerate(sklearn_model.coefs_):
        print(f"  Layer {i}: shape={coef.shape}, mean={coef.mean():.6f}, std={coef.std():.6f}, range=[{coef.min():.3f}, {coef.max():.3f}]")
    
    print("\nsklearn偏置统计:")
    for i, bias in enumerate(sklearn_model.intercepts_):
        print(f"  Layer {i}: shape={bias.shape}, mean={bias.mean():.6f}, std={bias.std():.6f}, range=[{bias.min():.3f}, {bias.max():.3f}]")
    
    # PyTorch模型
    torch.manual_seed(42)
    pytorch_model = SklearnCompatibleMLP(8, (64, 32), 1)
    
    print("\nPyTorch权重统计:")
    for i, layer in enumerate(pytorch_model.layers):
        weight = layer.weight.data
        print(f"  Layer {i}: shape={tuple(weight.shape)}, mean={weight.mean():.6f}, std={weight.std():.6f}, range=[{weight.min():.3f}, {weight.max():.3f}]")
    
    print("\nPyTorch偏置统计:")
    for i, layer in enumerate(pytorch_model.layers):
        bias = layer.bias.data
        print(f"  Layer {i}: shape={tuple(bias.shape)}, mean={bias.mean():.6f}, std={bias.std():.6f}, range=[{bias.min():.3f}, {bias.max():.3f}]")

# =============================================================================
# 训练过程详细对比
# =============================================================================

def detailed_training_comparison():
    """详细对比训练过程"""
    print("\n📈 详细训练过程对比")
    print("=" * 80)
    
    # 生成相同的数据
    np.random.seed(TEST_CONFIG['random_state'])
    X, y = make_regression(
        n_samples=TEST_CONFIG['n_samples'],
        n_features=TEST_CONFIG['n_features'],
        noise=TEST_CONFIG['noise'],
        random_state=TEST_CONFIG['random_state']
    )
    
    # 添加异常
    X_train, X_test, y_train, y_test = causal_split(
        X, y,
        test_size=0.2,
        random_state=TEST_CONFIG['random_state'],
        anomaly_ratio=TEST_CONFIG['anomaly_ratio'],
        anomaly_type='regression'
    )
    
    print(f"数据统计:")
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"  X范围: [{X_train.min():.3f}, {X_train.max():.3f}], std={X_train.std():.3f}")
    print(f"  y范围: [{y_train.min():.3f}, {y_train.max():.3f}], std={y_train.std():.3f}")
    
    # sklearn训练
    print(f"\n🔧 sklearn训练:")
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=TEST_CONFIG['hidden_layer_sizes'],
        max_iter=TEST_CONFIG['max_iter'],
        learning_rate_init=TEST_CONFIG['learning_rate'],
        early_stopping=True,
        validation_fraction=TEST_CONFIG['validation_fraction'],
        n_iter_no_change=TEST_CONFIG['patience'],
        tol=TEST_CONFIG['tol'],
        random_state=TEST_CONFIG['random_state'],
        alpha=TEST_CONFIG['alpha'],
        batch_size=len(X_train),  # Full batch
        verbose=True
    )
    
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    sklearn_mae = mean_absolute_error(y_test, sklearn_pred)
    
    print(f"  训练轮数: {sklearn_model.n_iter_}")
    print(f"  最终损失: {sklearn_model.loss_:.6f}")
    print(f"  测试R²: {sklearn_r2:.6f}")
    print(f"  测试MAE: {sklearn_mae:.6f}")
    
    # 完全兼容的PyTorch训练
    print(f"\n🔧 sklearn兼容PyTorch训练:")
    pytorch_model = SklearnCompatibleRegressor(
        hidden_layer_sizes=TEST_CONFIG['hidden_layer_sizes'],
        max_iter=TEST_CONFIG['max_iter'],
        learning_rate_init=TEST_CONFIG['learning_rate'],
        early_stopping=True,
        validation_fraction=TEST_CONFIG['validation_fraction'],
        n_iter_no_change=TEST_CONFIG['patience'],
        tol=TEST_CONFIG['tol'],
        random_state=TEST_CONFIG['random_state'],
        alpha=TEST_CONFIG['alpha'],
        batch_size=len(X_train),  # Full batch
        verbose=True
    )
    
    pytorch_model.fit(X_train, y_train)
    pytorch_pred = pytorch_model.predict(X_test)
    pytorch_r2 = r2_score(y_test, pytorch_pred)
    pytorch_mae = mean_absolute_error(y_test, pytorch_pred)
    
    print(f"  训练轮数: {pytorch_model.n_iter_}")
    print(f"  最终损失: {pytorch_model.loss_:.6f}")
    print(f"  测试R²: {pytorch_r2:.6f}")
    print(f"  测试MAE: {pytorch_mae:.6f}")
    
    # 我们原来的PyTorch实现
    print(f"\n🔧 原始PyTorch实现训练:")
    original_model = MLPPytorchRegressor(
        hidden_layer_sizes=TEST_CONFIG['hidden_layer_sizes'],
        max_iter=TEST_CONFIG['max_iter'],
        learning_rate=TEST_CONFIG['learning_rate'],
        early_stopping=True,
        validation_fraction=TEST_CONFIG['validation_fraction'],
        n_iter_no_change=TEST_CONFIG['patience'],
        tol=TEST_CONFIG['tol'],
        random_state=TEST_CONFIG['random_state'],
        alpha=TEST_CONFIG['alpha'],
        batch_size=None,  # Full batch
        verbose=True
    )
    
    original_model.fit(X_train, y_train)
    original_pred = original_model.predict(X_test)
    original_r2 = r2_score(y_test, original_pred)
    original_mae = mean_absolute_error(y_test, original_pred)
    
    print(f"  训练轮数: {original_model.n_iter_}")
    print(f"  最终损失: {original_model.loss_:.6f}")
    print(f"  测试R²: {original_r2:.6f}")
    print(f"  测试MAE: {original_mae:.6f}")
    
    # 性能差异总结
    print(f"\n📊 性能差异总结:")
    print(f"{'模型':<20} {'R²':<10} {'MAE':<10} {'轮数':<8} {'最终损失':<12}")
    print("-" * 60)
    print(f"{'sklearn':<20} {sklearn_r2:<10.6f} {sklearn_mae:<10.4f} {sklearn_model.n_iter_:<8} {sklearn_model.loss_:<12.6f}")
    print(f"{'sklearn兼容':<20} {pytorch_r2:<10.6f} {pytorch_mae:<10.4f} {pytorch_model.n_iter_:<8} {pytorch_model.loss_:<12.6f}")
    print(f"{'原始实现':<20} {original_r2:<10.6f} {original_mae:<10.4f} {original_model.n_iter_:<8} {original_model.loss_:<12.6f}")
    
    print(f"\n🔍 差异分析:")
    print(f"  sklearn vs sklearn兼容: ΔR² = {pytorch_r2 - sklearn_r2:+.6f}")
    print(f"  sklearn vs 原始实现: ΔR² = {original_r2 - sklearn_r2:+.6f}")
    print(f"  sklearn兼容 vs 原始实现: ΔR² = {original_r2 - pytorch_r2:+.6f}")

# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print("🚀 sklearn vs PyTorch 深度对比分析")
    print("=" * 80)
    
    # 1. 分析sklearn参数
    analyze_sklearn_parameters()
    
    # 2. 对比权重初始化
    compare_weight_initialization()
    
    # 3. 详细训练对比
    detailed_training_comparison()
    
    print(f"\n✅ 分析完成!")
    print(f"\n💡 结论:")
    print(f"   如果sklearn兼容实现和sklearn性能相近，说明我们找到了关键差异点")
    print(f"   如果原始实现和sklearn兼容实现差异很大，说明实现细节有问题")

if __name__ == "__main__":
    main()