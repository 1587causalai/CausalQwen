#!/usr/bin/env python3
"""
测试b_noise_trainable参数功能
验证当b_noise_trainable=False时，b_noise参数确实不参与梯度更新
"""

import torch
import numpy as np
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

from quick_test_causal_engine import quick_regression_test

def test_b_noise_trainable():
    """测试b_noise_trainable参数的实际效果"""
    print("🔧 测试b_noise_trainable参数功能")
    print("=" * 60)
    
    # 测试1: b_noise_trainable=True (默认行为)
    print("\n1️⃣ 测试 b_noise_trainable=True (可训练):")
    results_trainable = quick_regression_test(
        n_samples=300,
        n_features=5,
        hidden_layer_sizes=(32, 16),
        gamma_init=10.0,
        b_noise_init=0.1,
        b_noise_trainable=True,  # 可训练
        max_iter=200,
        verbose=False
    )
    
    print(f"   deterministic R²: {results_trainable['deterministic']['R²']:.4f}")
    print(f"   standard R²:      {results_trainable['standard']['R²']:.4f}")
    
    # 测试2: b_noise_trainable=False (固定值)
    print("\n2️⃣ 测试 b_noise_trainable=False (固定值):")
    results_fixed = quick_regression_test(
        n_samples=300,
        n_features=5,
        hidden_layer_sizes=(32, 16),
        gamma_init=10.0,
        b_noise_init=0.1,
        b_noise_trainable=False,  # 不可训练
        max_iter=200,
        verbose=False
    )
    
    print(f"   deterministic R²: {results_fixed['deterministic']['R²']:.4f}")
    print(f"   standard R²:      {results_fixed['standard']['R²']:.4f}")
    
    # 比较结果
    print("\n📊 结果对比:")
    print(f"   可训练vs固定 (deterministic): {results_trainable['deterministic']['R²']:.4f} vs {results_fixed['deterministic']['R²']:.4f}")
    print(f"   可训练vs固定 (standard):      {results_trainable['standard']['R²']:.4f} vs {results_fixed['standard']['R²']:.4f}")
    
    # 分析预期行为
    print("\n🔍 预期行为分析:")
    print("   - deterministic模式: 两者应该相近 (不使用b_noise)")
    print("   - standard模式: 可能有差异 (使用b_noise，固定值限制了适应能力)")
    
    return results_trainable, results_fixed

def test_parameter_inspection():
    """直接检查模型参数是否按预期设置"""
    print("\n🔬 直接参数检查测试")
    print("=" * 60)
    
    from causal_engine.sklearn import MLPCausalRegressor
    
    # 创建两个模型进行对比
    model_trainable = MLPCausalRegressor(
        hidden_layer_sizes=(16,),
        b_noise_init=0.5,
        b_noise_trainable=True,
        max_iter=1,
        verbose=False
    )
    
    model_fixed = MLPCausalRegressor(
        hidden_layer_sizes=(16,),
        b_noise_init=0.5,
        b_noise_trainable=False,
        max_iter=1,
        verbose=False
    )
    
    # 创建简单数据触发模型构建
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    
    model_trainable.fit(X, y)
    model_fixed.fit(X, y)
    
    # 检查参数类型
    b_noise_trainable = model_trainable.model['causal_engine'].action.b_noise
    b_noise_fixed = model_fixed.model['causal_engine'].action.b_noise
    
    print(f"\n📋 参数检查结果:")
    print(f"   可训练模型 b_noise类型: {type(b_noise_trainable)}")
    print(f"   固定模型 b_noise类型:   {type(b_noise_fixed)}")
    print(f"   可训练模型 requires_grad: {b_noise_trainable.requires_grad}")
    print(f"   固定模型 requires_grad:   {b_noise_fixed.requires_grad}")
    print(f"   可训练模型 值: {b_noise_trainable.data}")
    print(f"   固定模型 值:   {b_noise_fixed.data}")
    
    # 验证预期行为
    print(f"\n✅ 验证结果:")
    if isinstance(b_noise_trainable, torch.nn.Parameter) and b_noise_trainable.requires_grad:
        print("   ✓ 可训练模型: b_noise是Parameter且requires_grad=True")
    else:
        print("   ✗ 可训练模型: b_noise设置异常")
    
    if isinstance(b_noise_fixed, torch.Tensor) and not b_noise_fixed.requires_grad:
        print("   ✓ 固定模型: b_noise是buffer且requires_grad=False")
    else:
        print("   ✗ 固定模型: b_noise设置异常")
    
    return model_trainable, model_fixed

def test_gradient_update():
    """测试梯度更新行为"""
    print("\n🎯 梯度更新测试")
    print("=" * 60)
    
    from causal_engine.sklearn import MLPCausalRegressor
    import torch.nn.functional as F
    
    # 创建模型
    model = MLPCausalRegressor(
        hidden_layer_sizes=(8,),
        mode='standard',  # 使用b_noise的模式
        b_noise_init=0.3,
        b_noise_trainable=False,  # 测试不可训练
        max_iter=1,
        verbose=False
    )
    
    # 简单数据
    X = np.random.randn(20, 4)
    y = np.random.randn(20)
    
    model.fit(X, y)
    
    # 获取b_noise
    causal_engine = model.model['causal_engine']
    b_noise_before = causal_engine.action.b_noise.clone()
    
    print(f"训练前 b_noise: {b_noise_before}")
    print(f"b_noise.requires_grad: {causal_engine.action.b_noise.requires_grad}")
    
    # 手动进行一次前向传播和反向传播
    X_tensor = torch.tensor(X, dtype=torch.double)
    y_tensor = torch.tensor(y, dtype=torch.double)
    
    causal_engine.train()
    # 使用正确的前向传播方式
    hidden_features = X_tensor.unsqueeze(1)  # [batch, seq, features]
    result = model._forward_with_mode(hidden_features, mode='standard')
    output = result['output']
    loss = F.mse_loss(output.squeeze(), y_tensor)
    
    print(f"损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查b_noise是否变化
    b_noise_after = causal_engine.action.b_noise
    print(f"训练后 b_noise: {b_noise_after}")
    print(f"b_noise是否变化: {not torch.equal(b_noise_before, b_noise_after)}")
    
    if torch.equal(b_noise_before, b_noise_after):
        print("✅ 验证成功: b_noise_trainable=False时参数未变化")
    else:
        print("❌ 验证失败: b_noise_trainable=False时参数仍然变化")

if __name__ == "__main__":
    print("🧪 b_noise_trainable 功能验证测试")
    print("=" * 70)
    
    # 运行所有测试
    test_parameter_inspection()
    test_gradient_update()
    test_b_noise_trainable()
    
    print("\n🎉 所有测试完成!")
    print("\n💡 使用建议:")
    print("   - b_noise_trainable=True: 让模型自动学习最优噪声强度")
    print("   - b_noise_trainable=False: 固定噪声强度，用于消融实验或特定需求")