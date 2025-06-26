#!/usr/bin/env python3
"""
调试standard模式训练问题
检查是否是训练不充分导致的性能差异
"""

import numpy as np
import torch
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

from causal_engine.sklearn import MLPCausalClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def debug_standard_mode_training():
    """详细调试standard模式的训练过程"""
    print("🔍 调试CausalEngine standard模式训练")
    print("=" * 60)
    
    # 生成相同的数据（与快速测试一致）
    np.random.seed(42)
    X, y = make_classification(
        n_samples=800, n_features=15, n_classes=3,
        n_informative=7, n_redundant=0, n_clusters_per_class=1,
        class_sep=1.0, random_state=42
    )
    
    # 添加10%标签噪声
    n_flip = int(len(y) * 0.1)
    flip_indices = np.random.choice(len(y), n_flip, replace=False)
    unique_labels = np.unique(y)
    
    for idx in flip_indices:
        other_labels = unique_labels[unique_labels != y[idx]]
        if len(other_labels) > 0:
            y[idx] = np.random.choice(other_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"数据形状: 训练{X_train.shape}, 测试{X_test.shape}")
    print(f"类别分布: {np.bincount(y_train)}")
    
    # 测试不同的训练配置
    configs = [
        {"max_iter": 800, "learning_rate": 0.001, "description": "默认配置"},
        {"max_iter": 1500, "learning_rate": 0.001, "description": "更多轮数"},
        {"max_iter": 800, "learning_rate": 0.0005, "description": "更小学习率"},
        {"max_iter": 1500, "learning_rate": 0.0005, "description": "更多轮数+小学习率"},
        {"max_iter": 800, "learning_rate": 0.001, "early_stopping": False, "description": "关闭早停"},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{i+1}️⃣ 测试配置: {config['description']}")
        print(f"   参数: {config}")
        
        # deterministic模式作为对照
        clf_det = MLPCausalClassifier(
            hidden_layer_sizes=(128, 64),
            mode='deterministic',
            random_state=42,
            verbose=True,  # 开启详细输出
            **{k: v for k, v in config.items() if k != 'description'}
        )
        
        clf_det.fit(X_train, y_train)
        pred_det = clf_det.predict(X_test)
        acc_det = accuracy_score(y_test, pred_det['predictions'] if isinstance(pred_det, dict) else pred_det)
        
        # standard模式
        clf_std = MLPCausalClassifier(
            hidden_layer_sizes=(128, 64),
            mode='standard',
            gamma_init=15.0,
            b_noise_init=0.1,
            random_state=42,
            verbose=True,  # 开启详细输出
            **{k: v for k, v in config.items() if k != 'description'}
        )
        
        clf_std.fit(X_train, y_train)
        pred_std = clf_std.predict(X_test)
        acc_std = accuracy_score(y_test, pred_std['predictions'] if isinstance(pred_std, dict) else pred_std)
        
        print(f"   deterministic准确率: {acc_det:.4f}")
        print(f"   standard准确率:      {acc_std:.4f}")
        print(f"   差异: {abs(acc_det - acc_std):.4f}")
        
        results.append({
            'config': config['description'],
            'det_acc': acc_det,
            'std_acc': acc_std,
            'diff': abs(acc_det - acc_std)
        })
        
        # 如果standard模式性能太差，检查损失收敛
        if acc_std < 0.5:
            print(f"   ⚠️  standard模式性能异常低！")
            
            # 检查训练损失历史
            if hasattr(clf_std, 'loss_curve_') and clf_std.loss_curve_:
                final_loss = clf_std.loss_curve_[-1] if clf_std.loss_curve_ else "N/A"
                print(f"   最终训练损失: {final_loss}")
                print(f"   训练轮数: {len(clf_std.loss_curve_) if clf_std.loss_curve_ else 'N/A'}")
            
            # 检查模型参数
            causal_engine = clf_std.model['causal_engine']
            b_noise = causal_engine.action.b_noise
            print(f"   b_noise值: {b_noise.data[:5]}... (前5个)")
            print(f"   b_noise范围: [{b_noise.min().item():.4f}, {b_noise.max().item():.4f}]")
            
    # 总结结果
    print(f"\n📊 配置对比总结:")
    print(f"{'配置':<20} {'deterministic':<15} {'standard':<15} {'差异':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['config']:<20} {r['det_acc']:<15.4f} {r['std_acc']:<15.4f} {r['diff']:<10.4f}")
    
    return results

def test_loss_function():
    """测试不同模式的损失函数计算"""
    print(f"\n🔍 测试损失函数计算")
    print("=" * 40)
    
    from causal_engine.sklearn import MLPCausalClassifier
    
    # 创建简单测试案例
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 3, 50)
    
    clf = MLPCausalClassifier(
        hidden_layer_sizes=(16,),
        mode='standard',
        max_iter=5,  # 只训练几轮
        verbose=True
    )
    
    print("训练standard模式分类器...")
    clf.fit(X, y)
    
    # 检查训练过程
    if hasattr(clf, 'loss_curve_') and clf.loss_curve_:
        print(f"损失变化: {clf.loss_curve_}")
    
    # 测试预测
    pred = clf.predict(X[:10])
    print(f"前10个预测: {pred}")
    
    return clf

if __name__ == "__main__":
    print("🐛 CausalEngine Standard模式调试")
    print("=" * 70)
    
    # 运行调试
    results = debug_standard_mode_training()
    test_clf = test_loss_function()
    
    print(f"\n🎯 调试建议:")
    print(f"   1. 检查是否early stopping过早")
    print(f"   2. 尝试降低学习率") 
    print(f"   3. 增加训练轮数")
    print(f"   4. 检查损失函数计算是否正确")
    print(f"   5. 验证模式切换逻辑")