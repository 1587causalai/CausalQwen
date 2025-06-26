#!/usr/bin/env python3
"""
测试CausalEngine的默认初始化值
"""

import numpy as np
import torch
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

from causal_engine.sklearn import MLPCausalRegressor

def test_default_initialization():
    """测试默认初始化是否符合预期"""
    print("🔍 测试CausalEngine默认初始化值...")
    
    # 创建默认配置的回归器
    print("\n1. 创建默认回归器:")
    regressor = MLPCausalRegressor()
    
    print(f"  期望: b_noise_init=0.1, gamma_init=10.0, ovr_threshold_init=0.0")
    print(f"  实际: b_noise_init={regressor.b_noise_init}, gamma_init={regressor.gamma_init}, ovr_threshold_init={regressor.ovr_threshold_init}")
    
    # 验证默认值
    assert regressor.b_noise_init == 0.1, f"b_noise_init错误: 期望0.1, 实际{regressor.b_noise_init}"
    assert regressor.gamma_init == 10.0, f"gamma_init错误: 期望10.0, 实际{regressor.gamma_init}"
    assert regressor.ovr_threshold_init == 0.0, f"ovr_threshold_init错误: 期望0.0, 实际{regressor.ovr_threshold_init}"
    
    # 创建最小数据集来触发模型构建
    print("\n2. 构建模型验证参数传递:")
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    
    # 只训练1轮来触发模型构建
    regressor.max_iter = 1
    regressor.verbose = False
    regressor.fit(X, y)
    
    # 检查CausalEngine内部参数
    print("\n3. 检查CausalEngine内部参数:")
    causal_engine = regressor.model['causal_engine']
    
    # 检查b_noise
    b_noise = causal_engine.action.b_noise
    print(f"  b_noise形状: {b_noise.shape}, 值: {b_noise}")
    print(f"  b_noise平均值: {b_noise.mean().item():.6f} (期望约0.1)")
    
    # 检查gamma_U的初始化
    # 需要创建测试输入来获取gamma_U
    actual_causal_size = causal_engine.causal_size
    print(f"  实际causal_size: {actual_causal_size}")
    test_input = torch.randn(1, 1, actual_causal_size, dtype=torch.double)
    with torch.no_grad():
        loc_U, scale_U = causal_engine.abduction(test_input)
    
    gamma_U = scale_U.squeeze()
    print(f"  gamma_U形状: {gamma_U.shape}, 值: {gamma_U}")
    print(f"  gamma_U范围: [{gamma_U.min().item():.3f}, {gamma_U.max().item():.3f}]")
    print(f"  gamma_U平均值: {gamma_U.mean().item():.3f}")
    
    # 验证gamma_U是否在合理范围内
    # 由于gamma_init=10.0，但实际初始化使用linspace(1.0, 2.0)，我们需要看实际效果
    print(f"\n4. 分析gamma_U初始化:")
    print(f"  配置的gamma_init: {regressor.gamma_init}")
    print(f"  实际gamma_U范围: [{gamma_U.min().item():.3f}, {gamma_U.max().item():.3f}]")
    
    # 检查scale_net的bias来理解初始化
    abduction = causal_engine.abduction
    linear_modules = [m for m in abduction.scale_net.modules() if isinstance(m, torch.nn.Linear)]
    if linear_modules:
        last_layer = linear_modules[-1]
        bias_values = last_layer.bias.data
        print(f"  scale_net最后层bias: {bias_values}")
        print(f"  softplus(bias): {torch.nn.functional.softplus(bias_values)}")
    
    print(f"\n✅ 默认初始化验证完成!")
    return regressor

if __name__ == "__main__":
    regressor = test_default_initialization()