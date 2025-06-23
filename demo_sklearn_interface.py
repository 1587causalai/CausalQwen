"""
CausalEngine Sklearn Interface Demo

演示MLPCausalRegressor和MLPCausalClassifier的基础功能
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import sys
import os

# 添加项目路径
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

# 导入CausalEngine sklearn接口
try:
    from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier
    print("✅ CausalEngine sklearn接口导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def demo_regression():
    """演示因果回归功能"""
    print("\\n" + "="*50)
    print("🔧 因果回归演示")
    print("="*50)
    
    # 生成回归数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"数据维度: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # 传统MLPRegressor
    print("\\n训练传统MLPRegressor...")
    traditional_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    traditional_reg.fit(X_train, y_train)
    trad_pred = traditional_reg.predict(X_test)
    trad_r2 = r2_score(y_test, trad_pred)
    trad_mse = mean_squared_error(y_test, trad_pred)
    
    print(f"传统方法 - R²: {trad_r2:.4f}, MSE: {trad_mse:.4f}")
    
    # CausalEngine回归器
    print("\\n训练MLPCausalRegressor...")
    causal_reg = MLPCausalRegressor(
        hidden_layer_sizes=(64, 32), 
        max_iter=500, 
        random_state=42,
        verbose=True
    )
    causal_reg.fit(X_train, y_train)
    
    # 兼容模式预测
    causal_pred = causal_reg.predict(X_test, mode='compatible')
    causal_r2 = r2_score(y_test, causal_pred) 
    causal_mse = mean_squared_error(y_test, causal_pred)
    
    print(f"因果方法 - R²: {causal_r2:.4f}, MSE: {causal_mse:.4f}")
    
    # 高级预测模式
    print("\\n🚀 高级预测模式演示:")
    
    # 标准模式：包含不确定性
    advanced_pred = causal_reg.predict(X_test[:5], mode='standard')
    print(f"标准模式输出类型: {type(advanced_pred)}")
    if isinstance(advanced_pred, dict):
        print(f"  - 预测值: {advanced_pred['predictions'][:3]}")
        print(f"  - 分布信息: {list(advanced_pred['distributions'].keys())}")
    
    # 因果模式
    causal_pure = causal_reg.predict(X_test[:5], mode='causal')
    print(f"因果模式输出: 已计算")
    
    print(f"\\n✅ 回归演示完成！")
    return {
        'traditional_r2': trad_r2,
        'causal_r2': causal_r2,
        'improvement': causal_r2 - trad_r2
    }


def demo_classification():
    """演示因果分类功能"""
    print("\\n" + "="*50)
    print("🎯 因果分类演示")
    print("="*50)
    
    # 生成分类数据
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                              n_redundant=0, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"数据维度: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"类别数: {len(np.unique(y))}")
    
    # 传统MLPClassifier
    print("\\n训练传统MLPClassifier...")
    traditional_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    traditional_clf.fit(X_train, y_train)
    trad_pred = traditional_clf.predict(X_test)
    trad_acc = accuracy_score(y_test, trad_pred)
    trad_proba = traditional_clf.predict_proba(X_test)
    
    print(f"传统方法 - 准确率: {trad_acc:.4f}")
    
    # CausalEngine分类器
    print("\\n训练MLPCausalClassifier...")
    causal_clf = MLPCausalClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        verbose=True
    )
    causal_clf.fit(X_train, y_train)
    
    # 兼容模式预测
    causal_pred = causal_clf.predict(X_test, mode='compatible')
    causal_acc = accuracy_score(y_test, causal_pred)
    
    print(f"因果方法 - 准确率: {causal_acc:.4f}")
    
    # 概率预测对比
    print("\\n🎲 概率预测演示:")
    
    # Softmax兼容概率
    causal_proba_compat = causal_clf.predict_proba(X_test[:3], mode='compatible')
    print(f"Softmax兼容概率 (前3样本):")
    for i, prob in enumerate(causal_proba_compat):
        print(f"  样本{i}: {prob}")
        
    # OvR原生概率
    causal_proba_ovr = causal_clf.predict_proba(X_test[:3], mode='standard')
    print(f"\\nOvR原生概率 (前3样本):")
    for i, prob in enumerate(causal_proba_ovr):
        print(f"  样本{i}: {prob}")
    
    # 高级预测模式
    print("\\n🚀 高级预测模式演示:")
    advanced_pred = causal_clf.predict(X_test[:5], mode='standard')
    print(f"标准模式输出类型: {type(advanced_pred)}")
    if isinstance(advanced_pred, dict):
        print(f"  - 预测类别: {advanced_pred['predictions']}")
        print(f"  - 激活概率形状: {advanced_pred['probabilities'].shape}")
    
    print(f"\\n✅ 分类演示完成！")
    return {
        'traditional_acc': trad_acc,
        'causal_acc': causal_acc,
        'improvement': causal_acc - trad_acc
    }


def demo_noise_robustness():
    """演示标签噪声鲁棒性"""
    print("\\n" + "="*50)
    print("🛡️ 标签噪声鲁棒性演示")
    print("="*50)
    
    # 生成干净数据
    X, y = make_classification(n_samples=800, n_features=10, n_classes=3, 
                              n_redundant=0, n_informative=8, random_state=42)
    X_train, X_test, y_train_clean, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 添加标签噪声 (20%随机翻转)
    y_train_noisy = y_train_clean.copy()
    n_noise = int(0.2 * len(y_train_noisy))
    noise_indices = np.random.choice(len(y_train_noisy), n_noise, replace=False)
    
    for idx in noise_indices:
        available_labels = [l for l in np.unique(y) if l != y_train_noisy[idx]]
        y_train_noisy[idx] = np.random.choice(available_labels)
    
    print(f"原始数据: {len(y_train_clean)} 样本")
    print(f"噪声标签: {n_noise} 样本 ({n_noise/len(y_train_clean)*100:.1f}%)")
    
    # 传统方法 vs 因果方法在噪声数据上的表现
    print("\\n在噪声数据上训练...")
    
    # 传统方法
    trad_clf_noisy = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    trad_clf_noisy.fit(X_train, y_train_noisy)
    trad_acc_noisy = accuracy_score(y_test, trad_clf_noisy.predict(X_test))
    
    # 因果方法
    causal_clf_noisy = MLPCausalClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    causal_clf_noisy.fit(X_train, y_train_noisy)
    causal_acc_noisy = accuracy_score(y_test, causal_clf_noisy.predict(X_test))
    
    print(f"\\n在干净测试集上的表现:")
    print(f"传统方法: {trad_acc_noisy:.4f}")
    print(f"因果方法: {causal_acc_noisy:.4f}")
    print(f"因果优势: +{(causal_acc_noisy - trad_acc_noisy)*100:.1f}%")
    
    return {
        'traditional_noisy': trad_acc_noisy,
        'causal_noisy': causal_acc_noisy,
        'robustness_advantage': causal_acc_noisy - trad_acc_noisy
    }


def main():
    """主演示函数"""
    print("🚀 CausalEngine Sklearn接口演示")
    print("="*60)
    
    results = {}
    
    try:
        # 回归演示
        results['regression'] = demo_regression()
        
        # 分类演示  
        results['classification'] = demo_classification()
        
        # 噪声鲁棒性演示
        results['noise_robustness'] = demo_noise_robustness()
        
        # 总结
        print("\\n" + "="*60)
        print("📊 演示总结")
        print("="*60)
        
        print(f"🔧 回归任务:")
        print(f"  传统R²: {results['regression']['traditional_r2']:.4f}")
        print(f"  因果R²: {results['regression']['causal_r2']:.4f}")
        print(f"  改进: {results['regression']['improvement']:+.4f}")
        
        print(f"\\n🎯 分类任务:")
        print(f"  传统准确率: {results['classification']['traditional_acc']:.4f}")
        print(f"  因果准确率: {results['classification']['causal_acc']:.4f}")
        print(f"  改进: {results['classification']['improvement']:+.4f}")
        
        print(f"\\n🛡️ 噪声鲁棒性:")
        print(f"  传统方法(噪声): {results['noise_robustness']['traditional_noisy']:.4f}")
        print(f"  因果方法(噪声): {results['noise_robustness']['causal_noisy']:.4f}")
        print(f"  鲁棒优势: +{results['noise_robustness']['robustness_advantage']*100:.1f}%")
        
        print("\\n✅ 所有演示完成！CausalEngine sklearn接口正常工作。")
        
    except Exception as e:
        print(f"\\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)