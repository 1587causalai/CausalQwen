#!/usr/bin/env python3
"""
测试CausalEngine分类器的默认初始化值
"""

import numpy as np
import torch
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

from causal_engine.sklearn import MLPCausalClassifier

def test_classifier_default_initialization():
    """测试分类器默认初始化是否符合预期"""
    print("🔍 测试CausalEngine分类器默认初始化值...")
    
    # 创建默认配置的分类器
    print("\n1. 创建默认分类器:")
    classifier = MLPCausalClassifier()
    
    print(f"  期望: b_noise_init=0.1, gamma_init=10.0, ovr_threshold_init=0.0")
    print(f"  实际: b_noise_init={classifier.b_noise_init}, gamma_init={classifier.gamma_init}, ovr_threshold_init={classifier.ovr_threshold_init}")
    
    # 验证默认值
    assert classifier.b_noise_init == 0.1, f"b_noise_init错误: 期望0.1, 实际{classifier.b_noise_init}"
    assert classifier.gamma_init == 10.0, f"gamma_init错误: 期望10.0, 实际{classifier.gamma_init}"
    assert classifier.ovr_threshold_init == 0.0, f"ovr_threshold_init错误: 期望0.0, 实际{classifier.ovr_threshold_init}"
    
    # 创建分类数据集来触发模型构建
    print("\n2. 构建模型验证参数传递:")
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)  # 3类分类
    
    # 只训练1轮来触发模型构建
    classifier.max_iter = 1
    classifier.verbose = False
    classifier.fit(X, y)
    
    # 检查CausalEngine内部参数
    print("\n3. 检查CausalEngine内部参数:")
    causal_engine = classifier.model['causal_engine']
    
    # 检查b_noise
    b_noise = causal_engine.action.b_noise
    print(f"  b_noise形状: {b_noise.shape}, 值: {b_noise}")
    print(f"  b_noise平均值: {b_noise.mean().item():.6f} (期望约0.1)")
    
    # 检查gamma_U的初始化
    actual_causal_size = causal_engine.causal_size
    print(f"  实际causal_size: {actual_causal_size}")
    test_input = torch.randn(1, 1, actual_causal_size, dtype=torch.double)
    with torch.no_grad():
        loc_U, scale_U = causal_engine.abduction(test_input)
    
    gamma_U = scale_U.squeeze()
    print(f"  gamma_U形状: {gamma_U.shape}, 值: {gamma_U}")
    print(f"  gamma_U范围: [{gamma_U.min().item():.3f}, {gamma_U.max().item():.3f}]")
    print(f"  gamma_U平均值: {gamma_U.mean().item():.3f}")
    
    # 检查ovr_threshold (如果存在)
    if hasattr(causal_engine, 'activation') and hasattr(causal_engine.activation, 'classification_thresholds'):
        ovr_thresholds = causal_engine.activation.classification_thresholds
        print(f"  ovr_threshold形状: {ovr_thresholds.shape}, 值: {ovr_thresholds}")
        print(f"  ovr_threshold平均值: {ovr_thresholds.mean().item():.6f} (期望约0.0)")
    
    # 验证gamma_U是否在合理范围内
    print(f"\n4. 分析gamma_U初始化:")
    print(f"  配置的gamma_init: {classifier.gamma_init}")
    print(f"  实际gamma_U范围: [{gamma_U.min().item():.3f}, {gamma_U.max().item():.3f}]")
    
    # 检查scale_net的bias来理解初始化
    abduction = causal_engine.abduction
    linear_modules = [m for m in abduction.scale_net.modules() if isinstance(m, torch.nn.Linear)]
    if linear_modules:
        last_layer = linear_modules[-1]
        bias_values = last_layer.bias.data
        print(f"  scale_net最后层bias: {bias_values}")
        print(f"  softplus(bias): {torch.nn.functional.softplus(bias_values)}")
    
    print(f"\n✅ 分类器默认初始化验证完成!")
    return classifier

def test_classification_prediction():
    """测试分类器的预测功能"""
    print("\n" + "="*60)
    print("🎯 测试分类器预测功能")
    
    # 创建分类器
    classifier = MLPCausalClassifier(max_iter=50, verbose=False)
    
    # 生成测试数据
    np.random.seed(42)
    X_train = np.random.randn(200, 8)
    y_train = np.random.randint(0, 3, 200)
    X_test = np.random.randn(50, 8)
    
    # 训练
    print("训练分类器...")
    classifier.fit(X_train, y_train)
    
    # 预测
    print("进行预测...")
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    
    # 处理预测结果 (可能是字典格式)
    if isinstance(predictions, dict):
        pred_array = predictions['predictions']
        print(f"预测结果格式: 字典")
        print(f"预测形状: {pred_array.shape}")
        print(f"类别范围: [{pred_array.min()}, {pred_array.max()}]")
    else:
        pred_array = predictions
        print(f"预测结果格式: 数组")
        print(f"预测形状: {pred_array.shape}")
        print(f"类别范围: [{pred_array.min()}, {pred_array.max()}]")
    
    print(f"概率形状: {probabilities.shape}")
    print(f"概率范围: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    
    # 验证预测一致性
    predicted_classes = np.argmax(probabilities, axis=1)
    consistency = np.mean(pred_array == predicted_classes)
    print(f"预测一致性: {consistency:.3f} (期望1.0)")
    
    print("✅ 分类器预测功能验证完成!")

if __name__ == "__main__":
    classifier = test_classifier_default_initialization()
    test_classification_prediction()