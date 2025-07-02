#!/usr/bin/env python3
"""
数据处理问题诊断脚本

专门用于诊断 quick_test_causal_engine.py 中的数据处理问题。
通过详细的打印语句展示问题的来龙去脉，帮助定位和解决问题。

问题现象：
- 验证集指标异常高（MAE 193-225，R² 为负值）
- 测试集指标正常（MAE 3-29，R² 接近1）
- 巨大的性能差异无法解释

目标：
- 重现问题现象
- 分析数据处理的每个步骤
- 找出问题根因
- 提供修复方案
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import warnings

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_sklearn.regressor import MLPCausalRegressor, MLPPytorchRegressor
from causal_sklearn.utils import causal_split

warnings.filterwarnings('ignore')

def print_separator(title, char="=", width=80):
    """打印分隔符"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection(title, char="-", width=60):
    """打印子节标题"""
    print(f"\n{char * width}")
    print(f"🔍 {title}")
    print(f"{char * width}")

def analyze_data_statistics(X, y, data_name):
    """详细分析数据统计信息"""
    print(f"\n📊 {data_name} 数据统计分析：")
    print(f"   数据形状: X={X.shape}, y={y.shape}")
    
    # X 统计信息
    print(f"\n   X (特征) 统计:")
    print(f"   - 各特征均值: {X.mean(axis=0)}")
    print(f"   - 各特征标准差: {X.std(axis=0)}")
    print(f"   - 各特征最小值: {X.min(axis=0)}")
    print(f"   - 各特征最大值: {X.max(axis=0)}")
    print(f"   - 整体数据范围: [{X.min():.2f}, {X.max():.2f}]")
    
    # y 统计信息
    print(f"\n   y (目标) 统计:")
    print(f"   - 均值: {y.mean():.4f}")
    print(f"   - 标准差: {y.std():.4f}")
    print(f"   - 最小值: {y.min():.4f}")
    print(f"   - 最大值: {y.max():.4f}")
    print(f"   - 范围: {y.max() - y.min():.4f}")
    
    # 数据尺度评估
    print(f"\n   🔍 数据尺度评估:")
    if X.std().max() > 10 or X.std().min() < 0.1:
        print(f"   ❌ 特征尺度不一致！最大标准差: {X.std().max():.2f}, 最小标准差: {X.std().min():.2f}")
    else:
        print(f"   ✅ 特征尺度相对一致")
    
    if abs(y).max() > 1000:
        print(f"   ❌ 目标变量尺度过大！最大绝对值: {abs(y).max():.2f}")
    else:
        print(f"   ✅ 目标变量尺度合理")

def reproduce_quick_test_problem():
    """重现 quick_test_causal_engine.py 中的问题"""
    print_separator("🚨 第一部分：重现 quick_test_causal_engine.py 的问题现象")
    
    print("\n🎯 使用与 quick_test_causal_engine.py 相同的配置:")
    config = {
        'n_samples': 4000,
        'n_features': 12,
        'noise': 1.0,
        'random_state': 42,
        'test_size': 0.1,
        'anomaly_ratio': 0.2,
        'validation_fraction': 0.2,
        'max_iter': 2000,
        'learning_rate': 0.01,
        'patience': 50,
        'alpha': 0.0001
    }
    
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 第一步：生成原始数据
    print_subsection("第1步：生成原始数据")
    X, y = make_regression(
        n_samples=config['n_samples'], 
        n_features=config['n_features'],
        noise=config['noise'], 
        random_state=config['random_state']
    )
    
    print(f"✅ make_regression 完成")
    analyze_data_statistics(X, y, "原始生成")
    
    # 第二步：causal_split 分割
    print_subsection("第2步：causal_split 数据分割和异常注入")
    print(f"📝 执行 causal_split:")
    print(f"   test_size={config['test_size']}")
    print(f"   anomaly_ratio={config['anomaly_ratio']}")
    print(f"   anomaly_type='regression'")
    
    X_train, X_test, y_train, y_test = causal_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        anomaly_ratio=config['anomaly_ratio'], 
        anomaly_type='regression'
    )
    
    print(f"\n✅ causal_split 完成")
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    analyze_data_statistics(X_train, y_train, "训练集（含异常）")
    analyze_data_statistics(X_test, y_test, "测试集（纯净）")
    
    # 第三步：未标准化直接训练（重现问题）
    print_subsection("第3步：未标准化训练（重现问题）")
    print("❌ 关键问题：quick_test_causal_engine.py 第142行注释'数据不再进行标准化'")
    print("❌ 所有模型接收的是未标准化的原始数据！")
    
    # 训练 sklearn 模型
    print(f"\n🔧 训练 sklearn MLPRegressor (未标准化数据)...")
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=config['max_iter'],
        learning_rate_init=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        random_state=config['random_state'],
        alpha=config['alpha']
    )
    
    sklearn_model.fit(X_train, y_train)
    print(f"   训练完成: {sklearn_model.n_iter_} epochs")
    
    # 第四步：错误的验证集评估（重现问题）
    print_subsection("第4步：错误的验证集评估方式")
    print("❌ 关键问题：quick_test_causal_engine.py 第391-395行重新分割验证集")
    print("❌ 评估时的验证集 ≠ 训练时sklearn内部使用的验证集！")
    
    # 重新分割验证集（错误做法）
    X_train_eval, X_val_eval, y_train_eval, y_val_eval = train_test_split(
        X_train, y_train,
        test_size=config['validation_fraction'],
        random_state=config['random_state']
    )
    
    print(f"\n📊 重新分割的'验证集'统计:")
    analyze_data_statistics(X_val_eval, y_val_eval, "外部重新分割的验证集")
    
    # 预测和评估
    test_pred = sklearn_model.predict(X_test)
    val_pred = sklearn_model.predict(X_val_eval)
    
    # 计算指标
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    val_mae = mean_absolute_error(y_val_eval, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_eval, val_pred))
    val_r2 = r2_score(y_val_eval, val_pred)
    
    print_subsection("第5步：问题现象重现")
    print("🚨 异常结果重现:")
    print(f"   验证集 MAE: {val_mae:.4f} ← 异常高！")
    print(f"   验证集 RMSE: {val_rmse:.4f}")
    print(f"   验证集 R²: {val_r2:.4f} ← 可能为负值！")
    print(f"\n   测试集 MAE: {test_mae:.4f} ← 正常范围")
    print(f"   测试集 RMSE: {test_rmse:.4f}")
    print(f"   测试集 R²: {test_r2:.4f} ← 接近1.0")
    
    print(f"\n💡 问题现象总结:")
    if val_mae > test_mae * 5:
        print(f"   ❌ 验证集 MAE 比测试集高 {val_mae/test_mae:.1f} 倍！")
    if val_r2 < 0:
        print(f"   ❌ 验证集 R² 为负值，模型比简单均值还差！")
    if test_r2 > 0.9:
        print(f"   ❌ 测试集 R² 很高，说明模型实际很好！")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_val_eval': X_val_eval, 'y_val_eval': y_val_eval,
        'sklearn_model': sklearn_model,
        'test_pred': test_pred, 'val_pred': val_pred,
        'config': config
    }

def analyze_sklearn_internal_validation(data):
    """分析 sklearn 内部验证集处理"""
    print_separator("🧠 第二部分：sklearn 内部验证集 vs 外部验证集对比")
    
    X_train, y_train = data['X_train'], data['y_train']
    config = data['config']
    
    print("🔍 sklearn MLPRegressor 内部验证集处理:")
    print(f"   设置 early_stopping=True")
    print(f"   设置 validation_fraction={config['validation_fraction']}")
    print(f"   sklearn 会内部分割 {config['validation_fraction']:.1%} 作为验证集")
    
    # 计算 sklearn 内部的分割
    train_size = len(X_train)
    internal_val_size = int(train_size * config['validation_fraction'])
    internal_train_size = train_size - internal_val_size
    
    print(f"\n📊 sklearn 内部分割计算:")
    print(f"   原始训练集大小: {train_size}")
    print(f"   sklearn 内部训练集大小: {internal_train_size}")
    print(f"   sklearn 内部验证集大小: {internal_val_size}")
    
    print(f"\n📊 外部重新分割（quick_test_causal_engine.py 的做法）:")
    print(f"   外部验证集大小: {len(data['X_val_eval'])}")
    print(f"   外部训练集大小: {len(data['X_train']) - len(data['X_val_eval'])}")
    
    # 尝试模拟 sklearn 内部分割（注意：这只是近似）
    print(f"\n🔍 验证集数据差异分析:")
    print(f"   sklearn 内部使用的验证集: sklearn 内部随机分割")
    print(f"   quick_test_causal_engine.py 使用的验证集: 外部 train_test_split 分割")
    print(f"   random_state 相同，但分割时机不同！")
    
    # 分析两个验证集的统计差异
    y_val_external = data['y_val_eval']
    print(f"\n📊 外部验证集 y 统计:")
    print(f"   均值: {y_val_external.mean():.4f}")
    print(f"   标准差: {y_val_external.std():.4f}")
    print(f"   范围: [{y_val_external.min():.4f}, {y_val_external.max():.4f}]")
    
    print(f"\n❌ 关键问题识别:")
    print(f"   1. sklearn 在训练时使用内部分割的验证集")
    print(f"   2. 评估时使用外部重新分割的验证集")
    print(f"   3. 两个验证集完全不同！")
    print(f"   4. 导致验证集性能指标不可信")

def analyze_standardization_impact(data):
    """分析标准化对结果的影响"""
    print_separator("🔬 第三部分：数据标准化影响分析")
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    X_val_eval, y_val_eval = data['X_val_eval'], data['y_val_eval']
    config = data['config']
    
    print("🔍 未标准化数据的问题分析:")
    print(f"   当前 quick_test_causal_engine.py 使用未标准化数据")
    
    # 分析数据尺度对神经网络的影响
    x_range = X_train.max() - X_train.min()
    x_std_max = X_train.std(axis=0).max()
    x_std_min = X_train.std(axis=0).min()
    
    print(f"\n📊 未标准化数据尺度:")
    print(f"   X 数据范围: {x_range:.2f}")
    print(f"   X 最大标准差: {x_std_max:.2f}")
    print(f"   X 最小标准差: {x_std_min:.2f}")
    print(f"   标准差比例: {x_std_max/x_std_min:.2f}")
    
    if x_range > 100:
        print(f"   ❌ 数据范围过大，可能导致梯度爆炸")
    if x_std_max/x_std_min > 10:
        print(f"   ❌ 特征尺度差异巨大，影响训练稳定性")
    
    # 执行正确的标准化
    print_subsection("正确的标准化处理")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_val_eval_scaled = scaler_X.transform(X_val_eval)
    
    print(f"✅ 执行 StandardScaler:")
    print(f"   在训练集上拟合: scaler_X.fit_transform(X_train)")
    print(f"   变换测试集: scaler_X.transform(X_test)")
    print(f"   变换验证集: scaler_X.transform(X_val_eval)")
    
    print(f"\n📊 标准化后数据统计:")
    print(f"   X_train_scaled 均值: {X_train_scaled.mean(axis=0)}")
    print(f"   X_train_scaled 标准差: {X_train_scaled.std(axis=0)}")
    print(f"   X_train_scaled 范围: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    # 用标准化数据重新训练
    print_subsection("使用标准化数据重新训练")
    sklearn_model_scaled = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=config['max_iter'],
        learning_rate_init=config['learning_rate'],
        early_stopping=True,
        validation_fraction=config['validation_fraction'],
        n_iter_no_change=config['patience'],
        random_state=config['random_state'],
        alpha=config['alpha']
    )
    
    sklearn_model_scaled.fit(X_train_scaled, y_train)
    print(f"   训练完成: {sklearn_model_scaled.n_iter_} epochs")
    
    # 预测和评估
    test_pred_scaled = sklearn_model_scaled.predict(X_test_scaled)
    val_pred_scaled = sklearn_model_scaled.predict(X_val_eval_scaled)
    
    # 计算指标
    test_mae_scaled = mean_absolute_error(y_test, test_pred_scaled)
    test_r2_scaled = r2_score(y_test, test_pred_scaled)
    val_mae_scaled = mean_absolute_error(y_val_eval, val_pred_scaled)
    val_r2_scaled = r2_score(y_val_eval, val_pred_scaled)
    
    print_subsection("标准化前后对比")
    print("📊 性能对比:")
    print(f"   未标准化 - 测试集 MAE: {mean_absolute_error(y_test, data['test_pred']):.4f}")
    print(f"   标准化后 - 测试集 MAE: {test_mae_scaled:.4f}")
    print(f"   未标准化 - 测试集 R²: {r2_score(y_test, data['test_pred']):.4f}")
    print(f"   标准化后 - 测试集 R²: {test_r2_scaled:.4f}")
    
    print(f"\n   未标准化 - 验证集 MAE: {mean_absolute_error(y_val_eval, data['val_pred']):.4f}")
    print(f"   标准化后 - 验证集 MAE: {val_mae_scaled:.4f}")
    print(f"   未标准化 - 验证集 R²: {r2_score(y_val_eval, data['val_pred']):.4f}")
    print(f"   标准化后 - 验证集 R²: {val_r2_scaled:.4f}")
    
    improvement = abs(val_mae_scaled - test_mae_scaled) / abs(mean_absolute_error(y_val_eval, data['val_pred']) - mean_absolute_error(y_test, data['test_pred']))
    if improvement < 0.5:
        print(f"   ✅ 标准化后验证集和测试集性能更一致")
    else:
        print(f"   ⚠️ 仍存在验证集分割不一致问题")

def analyze_prediction_details(data):
    """详细分析模型预测"""
    print_separator("🎯 第四部分：模型预测详细分析")
    
    test_pred = data['test_pred']
    val_pred = data['val_pred']
    y_test = data['y_test']
    y_val_eval = data['y_val_eval']
    
    print("🔍 预测值详细分析:")
    
    print(f"\n📊 测试集预测分析:")
    print(f"   真实值范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"   预测值范围: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
    print(f"   预测值均值: {test_pred.mean():.2f}")
    print(f"   真实值均值: {y_test.mean():.2f}")
    print(f"   预测偏差: {abs(test_pred.mean() - y_test.mean()):.2f}")
    
    print(f"\n📊 验证集预测分析:")
    print(f"   真实值范围: [{y_val_eval.min():.2f}, {y_val_eval.max():.2f}]")
    print(f"   预测值范围: [{val_pred.min():.2f}, {val_pred.max():.2f}]")
    print(f"   预测值均值: {val_pred.mean():.2f}")
    print(f"   真实值均值: {y_val_eval.mean():.2f}")
    print(f"   预测偏差: {abs(val_pred.mean() - y_val_eval.mean()):.2f}")
    
    # 分析预测误差分布
    test_errors = np.abs(test_pred - y_test)
    val_errors = np.abs(val_pred - y_val_eval)
    
    print(f"\n📊 预测误差分析:")
    print(f"   测试集误差均值: {test_errors.mean():.2f}")
    print(f"   测试集误差标准差: {test_errors.std():.2f}")
    print(f"   测试集误差中位数: {np.median(test_errors):.2f}")
    
    print(f"\n   验证集误差均值: {val_errors.mean():.2f}")
    print(f"   验证集误差标准差: {val_errors.std():.2f}")
    print(f"   验证集误差中位数: {np.median(val_errors):.2f}")
    
    # 异常值分析
    test_outliers = np.sum(test_errors > test_errors.mean() + 2*test_errors.std())
    val_outliers = np.sum(val_errors > val_errors.mean() + 2*val_errors.std())
    
    print(f"\n📊 异常预测分析:")
    print(f"   测试集异常预测数量: {test_outliers} / {len(test_errors)}")
    print(f"   验证集异常预测数量: {val_outliers} / {len(val_errors)}")
    
    if val_errors.mean() > test_errors.mean() * 3:
        print(f"\n❌ 关键发现:")
        print(f"   验证集预测误差比测试集高 {val_errors.mean()/test_errors.mean():.1f} 倍")
        print(f"   这不是正常的模型行为！")
        print(f"   通常验证集和测试集性能应该接近")

def summarize_root_causes():
    """总结问题根因"""
    print_separator("🚨 第五部分：问题根因总结和修复方案")
    
    print("🔍 问题根因分析:")
    
    print(f"\n❌ 问题1: 数据未标准化")
    print(f"   📍 位置: quick_test_causal_engine.py 第142行")
    print(f"   📝 代码: # 数据不再进行标准化")
    print(f"   💥 影响: 神经网络接收大尺度数据，训练不稳定")
    print(f"   🔢 证据: X 数据范围过大，特征尺度不一致")
    
    print(f"\n❌ 问题2: 验证集分割不一致")
    print(f"   📍 位置: quick_test_causal_engine.py 第391-395行")
    print(f"   📝 代码: train_test_split(data['X_train'], data['y_train'], ...)")
    print(f"   💥 影响: 评估用的验证集 ≠ 训练时的验证集")
    print(f"   🔢 证据: 验证集性能异常，与测试集差距巨大")
    
    print(f"\n❌ 问题3: 数据尺度不匹配")
    print(f"   📍 原因: make_regression 生成大尺度数据")
    print(f"   💥 影响: 不同数据集上的模型表现差异巨大")
    print(f"   🔢 证据: 验证集 MAE 193-225，测试集 MAE 3-29")
    
    print_subsection("修复方案")
    print("✅ 修复步骤:")
    print("   1. 对特征数据进行标准化")
    print("   2. 保持目标变量原始尺度")
    print("   3. 使用一致的验证集分割策略")
    print("   4. 避免在评估时重新分割验证集")
    
    print(f"\n💡 正确的数据处理流程:")
    print(f"   1. causal_split() 分割训练/测试集")
    print(f"   2. StandardScaler 标准化特征")
    print(f"   3. 模型内部自动处理验证集分割")
    print(f"   4. 在测试集上进行最终评估")
    
    print(f"\n🎯 预期修复效果:")
    print(f"   - 验证集和测试集性能指标应该接近")
    print(f"   - R² 值应该为正数且合理")
    print(f"   - MAE 值应该在相似范围内")
    print(f"   - 模型训练更稳定，收敛更快")

def main():
    """主程序"""
    print("🔬 数据处理问题诊断脚本")
    print("=" * 80)
    print("目标: 诊断 quick_test_causal_engine.py 中验证集指标异常的问题")
    print("现象: 验证集 MAE 193-225 (异常高), 测试集 MAE 3-29 (正常)")
    print("=" * 80)
    
    # 第一部分：重现问题
    data = reproduce_quick_test_problem()
    
    # 第二部分：分析验证集问题
    analyze_sklearn_internal_validation(data)
    
    # 第三部分：分析标准化影响
    analyze_standardization_impact(data)
    
    # 第四部分：详细预测分析
    analyze_prediction_details(data)
    
    # 第五部分：总结根因
    summarize_root_causes()
    
    print_separator("🎉 诊断完成", char="=", width=80)
    print("💡 建议: 修复 quick_test_causal_engine.py 中的数据处理问题")
    print("📝 重点: 1) 添加数据标准化 2) 修复验证集分割逻辑")

if __name__ == "__main__":
    main()