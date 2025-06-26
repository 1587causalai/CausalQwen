"""
CausalEngine Sklearn Interface Demo v2.0 - 五模式系统演示

展示MLPCausalRegressor和MLPCausalClassifier的五模式系统：
1. Deterministic: γ_U=0, b_noise=0 (等价sklearn)
2. Exogenous: γ_U=0, b_noise≠0 (外生噪声推理)
3. Endogenous: γ_U≠0, b_noise=0 (内生因果推理)
4. Standard: γ_U≠0, b_noise→scale (标准因果推理)
5. Sampling: γ_U≠0, b_noise→location (探索性因果推理)

主要升级:
- 完整的五模式系统演示
- 新的predict_dist()方法演示
- 模式间性能对比分析
- 不确定性量化展示
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import sys
import os
import torch

# 添加项目路径
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

# 导入CausalEngine sklearn接口
try:
    from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier
    print("✅ CausalEngine sklearn接口导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def demo_five_modes_regression():
    """演示五模式回归系统"""
    print("\n" + "="*60)
    print("🔧 五模式回归系统演示")
    print("="*60)
    
    # 生成回归数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"数据维度: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # 传统sklearn基线
    print("\n📊 传统sklearn基线:")
    sklearn_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        max_iter=200, 
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    sklearn_reg.fit(X_train, y_train)
    sklearn_pred = sklearn_reg.predict(X_test)
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    print(f"sklearn MLPRegressor - R²: {sklearn_r2:.4f}, MSE: {sklearn_mse:.4f}")
    
    # 五模式测试
    modes = ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    results = {}
    
    print(f"\n🚀 CausalEngine五模式测试:")
    
    for mode in modes:
        print(f"\n--- {mode.upper()} 模式 ---")
        
        # 创建对应模式的回归器
        causal_reg = MLPCausalRegressor(
            hidden_layer_sizes=(64, 32),
            mode=mode,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
        
        # 训练
        causal_reg.fit(X_train, y_train)
        
        # 预测
        causal_pred = causal_reg.predict(X_test)
        if isinstance(causal_pred, dict):
            causal_pred = causal_pred['predictions']
            
        # 计算指标
        causal_r2 = r2_score(y_test, causal_pred)
        causal_mse = mean_squared_error(y_test, causal_pred)
        
        results[mode] = {
            'r2': causal_r2,
            'mse': causal_mse,
            'improvement_r2': causal_r2 - sklearn_r2,
            'model': causal_reg
        }
        
        print(f"R²: {causal_r2:.4f} (vs sklearn: {causal_r2-sklearn_r2:+.4f})")
        print(f"MSE: {causal_mse:.4f}")
        
        # 演示分布预测（非deterministic模式）
        if mode != 'deterministic':
            dist_params = causal_reg.predict_dist(X_test[:3])
            print(f"分布参数形状: {dist_params.shape}")
            print(f"前3样本位置参数: {dist_params[:3, 0, 0]}")
            print(f"前3样本尺度参数: {dist_params[:3, 0, 1]}")
    
    # 模式对比分析
    print(f"\n📊 五模式性能对比:")
    print(f"{'模式':<12} {'R²':<8} {'MSE':<10} {'vs sklearn':<10}")
    print("-" * 45)
    for mode in modes:
        r2, mse, imp = results[mode]['r2'], results[mode]['mse'], results[mode]['improvement_r2']
        print(f"{mode:<12} {r2:<8.4f} {mse:<10.1f} {imp:+8.4f}")
    
    return results


def demo_five_modes_classification():
    """演示五模式分类系统"""
    print("\n" + "="*60)
    print("🎯 五模式分类系统演示")
    print("="*60)
    
    # 生成分类数据
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                              n_redundant=0, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"数据维度: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"类别数: {len(np.unique(y))}")
    
    # 传统sklearn基线
    print("\n📊 传统sklearn基线:")
    sklearn_clf = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=200, 
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    sklearn_clf.fit(X_train, y_train)
    sklearn_pred = sklearn_clf.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    sklearn_proba = sklearn_clf.predict_proba(X_test)
    print(f"sklearn MLPClassifier - 准确率: {sklearn_acc:.4f}")
    
    # 五模式测试
    modes = ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    results = {}
    
    print(f"\n🚀 CausalEngine五模式测试:")
    
    for mode in modes:
        print(f"\n--- {mode.upper()} 模式 ---")
        
        # 创建对应模式的分类器
        causal_clf = MLPCausalClassifier(
            hidden_layer_sizes=(64, 32),
            mode=mode,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
        
        # 训练
        causal_clf.fit(X_train, y_train)
        
        # 预测
        causal_pred = causal_clf.predict(X_test)
        if isinstance(causal_pred, dict):
            causal_pred = causal_pred['predictions']
            
        # 计算指标
        causal_acc = accuracy_score(y_test, causal_pred)
        
        results[mode] = {
            'accuracy': causal_acc,
            'improvement': causal_acc - sklearn_acc,
            'model': causal_clf
        }
        
        print(f"准确率: {causal_acc:.4f} (vs sklearn: {causal_acc-sklearn_acc:+.4f})")
        
        # 演示概率预测
        causal_proba = causal_clf.predict_proba(X_test[:3])
        print(f"概率预测形状: {causal_proba.shape}")
        print(f"前3样本概率分布:\n{causal_proba}")
        
        # 演示分布预测（非deterministic模式）
        if mode != 'deterministic':
            dist_proba = causal_clf.predict_dist(X_test[:3])
            print(f"OvR激活概率形状: {dist_proba.shape}")
    
    # 模式对比分析
    print(f"\n📊 五模式性能对比:")
    print(f"{'模式':<12} {'准确率':<8} {'vs sklearn':<10}")
    print("-" * 35)
    for mode in modes:
        acc, imp = results[mode]['accuracy'], results[mode]['improvement']
        print(f"{mode:<12} {acc:<8.4f} {imp:+8.4f}")
    
    return results


def demo_uncertainty_quantification():
    """演示不确定性量化能力"""
    print("\n" + "="*60)
    print("🌡️ 不确定性量化演示")
    print("="*60)
    
    # 生成有噪声的回归数据
    X, y = make_regression(n_samples=300, n_features=5, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("对比不同模式的不确定性量化能力:\n")
    
    # 测试不同模式的不确定性表达
    modes = ['deterministic', 'endogenous', 'standard', 'sampling']
    
    for mode in modes:
        print(f"--- {mode.upper()} 模式的不确定性 ---")
        
        reg = MLPCausalRegressor(
            hidden_layer_sizes=(32, 16),
            mode=mode,
            max_iter=150,
            random_state=42,
            verbose=False
        )
        
        reg.fit(X_train, y_train)
        
        if mode == 'deterministic':
            # Deterministic模式：只有点估计
            pred = reg.predict(X_test[:5])
            print(f"预测值: {pred[:5]}")
            print("不确定性: 无（确定性预测）")
        else:
            # 其他模式：完整分布信息
            dist_params = reg.predict_dist(X_test[:5])
            loc = dist_params[:, 0, 0]  # 位置参数
            scale = dist_params[:, 0, 1]  # 尺度参数
            
            print(f"预测值 (位置): {loc}")
            print(f"不确定性 (尺度): {scale}")
            print(f"平均不确定性: {np.mean(scale):.4f}")
        
        print()


def demo_noise_robustness_five_modes():
    """演示五模式的噪声鲁棒性"""
    print("\n" + "="*60)
    print("🛡️ 五模式噪声鲁棒性演示")
    print("="*60)
    
    # 生成带噪声的分类数据
    X, y = make_classification(n_samples=500, n_features=8, n_classes=2, 
                              n_informative=6, random_state=42)
    X_train, X_test, y_train_clean, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 添加20%标签噪声
    y_train_noisy = y_train_clean.copy()
    n_noise = int(0.2 * len(y_train_noisy))
    noise_indices = np.random.choice(len(y_train_noisy), n_noise, replace=False)
    y_train_noisy[noise_indices] = 1 - y_train_noisy[noise_indices]  # 翻转标签
    
    print(f"数据: {len(y_train_clean)} 训练样本, {len(y_test)} 测试样本")
    print(f"噪声: {n_noise} 标签翻转 ({n_noise/len(y_train_clean)*100:.1f}%)")
    
    # sklearn基线（带噪声训练）
    sklearn_clf = MLPClassifier(
        hidden_layer_sizes=(32, 16), 
        max_iter=150, 
        random_state=42
    )
    sklearn_clf.fit(X_train, y_train_noisy)
    sklearn_acc_noisy = accuracy_score(y_test, sklearn_clf.predict(X_test))
    
    print(f"\nsklearn基线 (噪声训练): {sklearn_acc_noisy:.4f}")
    
    # 五模式噪声鲁棒性测试
    print(f"\n🚀 五模式噪声鲁棒性:")
    modes = ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    results = {}
    
    for mode in modes:
        clf = MLPCausalClassifier(
            hidden_layer_sizes=(32, 16),
            mode=mode,
            max_iter=150,
            random_state=42,
            verbose=False
        )
        
        # 在噪声数据上训练
        clf.fit(X_train, y_train_noisy)
        
        # 在干净测试集上评估
        pred = clf.predict(X_test)
        if isinstance(pred, dict):
            pred = pred['predictions']
            
        acc = accuracy_score(y_test, pred)
        improvement = acc - sklearn_acc_noisy
        
        results[mode] = {
            'accuracy': acc,
            'improvement': improvement
        }
        
        print(f"{mode:<12}: {acc:.4f} (vs sklearn: {improvement:+.4f})")
    
    return results


def demo_mode_switching():
    """演示模式切换功能"""
    print("\n" + "="*60)
    print("🔄 模式动态切换演示")
    print("="*60)
    
    # 生成数据
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练一个标准模式的回归器
    reg = MLPCausalRegressor(
        hidden_layer_sizes=(32, 16),
        mode='standard',  # 默认标准模式
        max_iter=150,
        random_state=42,
        verbose=False
    )
    
    reg.fit(X_train, y_train)
    print("✅ 模型已在'standard'模式下训练完成")
    
    # 演示同一模型在不同模式下的预测
    print(f"\n🔀 同一模型，不同模式预测 (前3个样本):")
    modes = ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    
    for mode in modes:
        # 使用相同模型，不同模式预测
        pred = reg.predict(X_test[:3], mode=mode)
        
        if isinstance(pred, dict):
            pred_values = pred['predictions']
            print(f"{mode:<12}: {pred_values}")
        else:
            print(f"{mode:<12}: {pred}")
    
    print(f"\n📊 模式切换说明:")
    print("- deterministic: 确定性预测，无随机性")
    print("- exogenous: 加入外生噪声")  
    print("- endogenous: 使用内生不确定性")
    print("- standard: 标准因果推理（训练模式）")
    print("- sampling: 探索性随机采样")


def main():
    """主演示函数"""
    print("🚀 CausalEngine五模式系统演示 v2.0")
    print("="*70)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 五模式回归演示
        regression_results = demo_five_modes_regression()
        
        # 2. 五模式分类演示  
        classification_results = demo_five_modes_classification()
        
        # 3. 不确定性量化演示
        demo_uncertainty_quantification()
        
        # 4. 噪声鲁棒性演示
        noise_results = demo_noise_robustness_five_modes()
        
        # 5. 模式切换演示
        demo_mode_switching()
        
        # 总结
        print("\n" + "="*70)
        print("📊 五模式系统演示总结")
        print("="*70)
        
        print("🔧 回归任务表现:")
        for mode in ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']:
            r2 = regression_results[mode]['r2']
            imp = regression_results[mode]['improvement_r2']
            print(f"  {mode:<12}: R² {r2:.4f} (vs sklearn: {imp:+.4f})")
        
        print("\n🎯 分类任务表现:")
        for mode in ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']:
            acc = classification_results[mode]['accuracy']
            imp = classification_results[mode]['improvement']
            print(f"  {mode:<12}: 准确率 {acc:.4f} (vs sklearn: {imp:+.4f})")
        
        print("\n🛡️ 噪声鲁棒性表现:")
        for mode in ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']:
            acc = noise_results[mode]['accuracy']
            imp = noise_results[mode]['improvement']
            print(f"  {mode:<12}: 准确率 {acc:.4f} (vs sklearn: {imp:+.4f})")
        
        print("\n✅ 五模式系统演示完成！")
        print("🎉 CausalEngine为您提供了从确定性建模到因果推理的完整光谱！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)