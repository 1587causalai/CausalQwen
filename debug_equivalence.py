"""
调试数学等价性问题的实验脚本

找出冻结CausalEngine与传统MLP性能差异的根本原因
"""

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os

sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from causal_engine.sklearn import MLPCausalRegressor

def freeze_abduction_to_identity(model):
    """冻结AbductionNetwork为恒等映射"""
    abduction = model.causal_engine.abduction
    
    if hasattr(abduction, '_loc_is_identity_candidate') and abduction._loc_is_identity_candidate:
        with torch.no_grad():
            causal_size = abduction.causal_size
            abduction.loc_net.weight.copy_(torch.eye(causal_size))
            abduction.loc_net.bias.zero_()
            
        abduction.loc_net.weight.requires_grad = False
        abduction.loc_net.bias.requires_grad = False
        return True
    return False

def enable_traditional_loss_mode(model, task_type='regression'):
    """为冻结的模型启用传统损失函数模式"""
    if task_type == 'regression':
        def mse_loss(predictions, targets):
            if isinstance(predictions, dict):
                if 'activation_output' in predictions and 'regression_values' in predictions['activation_output']:
                    pred_values = predictions['activation_output']['regression_values'].squeeze()
                elif 'loc_S' in predictions:
                    pred_values = predictions['loc_S'].squeeze()
                else:
                    raise ValueError("Cannot extract predictions for MSE loss")
            else:
                pred_values = predictions.squeeze()
            
            targets = targets.squeeze()
            return torch.nn.functional.mse_loss(pred_values, targets)
        
        model._traditional_loss = mse_loss
        model._use_traditional_loss = True

def debug_equivalence():
    """调试等价性问题"""
    print("🔍 调试数学等价性问题")
    print("="*50)
    
    # 生成回归数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"数据维度: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # 方案1: 原始方法（有问题）
    print("\n1️⃣ 原始方法 - 分两步训练（有问题）:")
    frozen_reg_v1 = MLPCausalRegressor(
        hidden_layer_sizes=(64, 32),
        causal_size=32,
        max_iter=500,
        random_state=42,
        verbose=False
    )
    
    # 分两步训练
    frozen_reg_v1.fit(X_train[:50], y_train[:50])  # 小批量初始化
    freeze_abduction_to_identity(frozen_reg_v1)
    enable_traditional_loss_mode(frozen_reg_v1, 'regression')
    original_compute_loss = frozen_reg_v1._compute_loss
    frozen_reg_v1._compute_loss = lambda predictions, targets: frozen_reg_v1._traditional_loss(predictions, targets)
    frozen_reg_v1.fit(X_train, y_train)  # 重新训练
    
    pred_v1 = frozen_reg_v1.predict(X_test, mode='compatible')
    r2_v1 = r2_score(y_test, pred_v1)
    
    # 方案2: 改进方法 - 一次性训练 + L2正则化
    print("\n2️⃣ 改进方法 - 一次性训练:")
    frozen_reg_v2 = MLPCausalRegressor(
        hidden_layer_sizes=(64, 32),
        causal_size=32,
        max_iter=500,
        random_state=42,
        verbose=False
    )
    
    # 先冻结，再训练（一次性）
    # 首先进行最小初始化
    frozen_reg_v2._build_model(X_train.shape[1])
    freeze_abduction_to_identity(frozen_reg_v2)
    enable_traditional_loss_mode(frozen_reg_v2, 'regression')
    frozen_reg_v2._compute_loss = lambda predictions, targets: frozen_reg_v2._traditional_loss(predictions, targets)
    
    # 添加L2正则化 (模拟sklearn的alpha=0.0001)
    optimizer_v2 = torch.optim.Adam(frozen_reg_v2.model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # 手动训练过程
    frozen_reg_v2.model.train()
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    for epoch in range(500):
        optimizer_v2.zero_grad()
        predictions = frozen_reg_v2._forward(X_train_tensor)
        loss = frozen_reg_v2._compute_loss(predictions, y_train_tensor)
        loss.backward()
        optimizer_v2.step()
        
        if epoch % 100 == 0:
            print(f"    Epoch {epoch}: Loss = {loss.item():.6f}")
    
    pred_v2 = frozen_reg_v2.predict(X_test, mode='compatible')
    r2_v2 = r2_score(y_test, pred_v2)
    
    # 传统方法对比
    print("\n3️⃣ 传统sklearn方法:")
    traditional_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        max_iter=500, 
        random_state=42,
        alpha=0.0001  # L2正则化
    )
    traditional_reg.fit(X_train, y_train)
    pred_trad = traditional_reg.predict(X_test)
    r2_trad = r2_score(y_test, pred_trad)
    
    # 结果对比
    print("\n📊 结果对比:")
    print(f"传统sklearn:     R² = {r2_trad:.6f}")
    print(f"原始方法(分步):   R² = {r2_v1:.6f}, 差异 = {abs(r2_trad - r2_v1):.6f}")
    print(f"改进方法(一步):   R² = {r2_v2:.6f}, 差异 = {abs(r2_trad - r2_v2):.6f}")
    
    print("\n🎯 分析:")
    if abs(r2_trad - r2_v2) < abs(r2_trad - r2_v1):
        print("✅ 改进方法显著减少了差异!")
        if abs(r2_trad - r2_v2) < 0.001:
            print("🎉 几乎完全等价!")
        else:
            print("⚠️ 仍有差异，需要进一步调试")
    else:
        print("❌ 改进方法没有明显效果，需要其他解决方案")

if __name__ == "__main__":
    debug_equivalence()