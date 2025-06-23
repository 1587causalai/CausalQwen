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


def freeze_abduction_to_identity(model):
    """
    冻结模型的AbductionNetwork为恒等映射
    
    Returns:
        bool: 是否成功冻结
    """
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
    """
    为冻结的模型启用传统损失函数模式
    
    Args:
        model: CausalRegressor 或 CausalClassifier
        task_type: 'regression' 或 'classification'
    """
    if task_type == 'regression':
        # 为回归任务替换损失函数为MSE
        def mse_loss(predictions, targets):
            # 提取位置参数作为预测值
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
        
    elif task_type == 'classification':
        # 为分类任务替换损失函数为CrossEntropy
        def crossentropy_loss(predictions, targets):
            # 提取logits
            if isinstance(predictions, dict) and 'loc_S' in predictions:
                logits = predictions['loc_S']  # [batch_size, seq_len, n_classes]
                if logits.dim() == 3:
                    logits = logits.squeeze(1)  # [batch_size, n_classes]
            else:
                raise ValueError("Cannot extract logits for CrossEntropy loss")
                
            targets = targets.long().squeeze()
            return torch.nn.functional.cross_entropy(logits, targets)
        
        model._traditional_loss = crossentropy_loss  
        model._use_traditional_loss = True


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
    
    # CausalEngine回归器(完整)
    print("\\n训练MLPCausalRegressor(完整)...")
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
    
    print(f"因果方法(完整) - R²: {causal_r2:.4f}, MSE: {causal_mse:.4f}")
    
    # CausalEngine回归器(冻结+传统损失) - 正确的数学等价性验证
    print("\\n训练MLPCausalRegressor(冻结+传统损失)...")
    frozen_reg = MLPCausalRegressor(
        hidden_layer_sizes=(64, 32),
        causal_size=32,  # 等于最后隐藏层大小，便于冻结
        max_iter=500,
        random_state=42,
        verbose=False
    )
    
    # 先初始化再冻结和切换损失函数
    frozen_reg.fit(X_train[:50], y_train[:50])  # 小批量初始化
    freeze_success = freeze_abduction_to_identity(frozen_reg)
    
    if freeze_success:
        print("✅ 成功冻结AbductionNetwork为恒等映射")
        
        # 关键：启用传统MSE损失函数
        enable_traditional_loss_mode(frozen_reg, 'regression')
        
        # 替换损失函数
        original_compute_loss = frozen_reg._compute_loss
        frozen_reg._compute_loss = lambda predictions, targets: frozen_reg._traditional_loss(predictions, targets)
        
        print("✅ 已切换到传统MSE损失函数")
        
        # 重新训练（使用MSE损失）
        frozen_reg.fit(X_train, y_train)
        frozen_pred = frozen_reg.predict(X_test, mode='compatible')
        frozen_r2 = r2_score(y_test, frozen_pred)
        frozen_mse = mean_squared_error(y_test, frozen_pred)
        print(f"因果方法(冻结+MSE) - R²: {frozen_r2:.4f}, MSE: {frozen_mse:.4f}")
        
        # 恢复原损失函数
        frozen_reg._compute_loss = original_compute_loss
    else:
        print("❌ 无法冻结AbductionNetwork")
        frozen_r2 = frozen_mse = 0
    
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
        'frozen_r2': frozen_r2 if freeze_success else 0,
        'improvement': causal_r2 - trad_r2,
        'frozen_improvement': frozen_r2 - trad_r2 if freeze_success else 0,
        'freeze_success': freeze_success
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
    
    # CausalEngine分类器(完整)
    print("\\n训练MLPCausalClassifier(完整)...")
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
    
    print(f"因果方法(完整) - 准确率: {causal_acc:.4f}")
    
    # CausalEngine分类器(冻结+传统损失) - 正确的数学等价性验证
    print("\\n训练MLPCausalClassifier(冻结+传统损失)...")
    frozen_clf = MLPCausalClassifier(
        hidden_layer_sizes=(64, 32),
        causal_size=32,  # 等于最后隐藏层大小，便于冻结
        max_iter=500,
        random_state=42,
        verbose=False
    )
    
    # 先初始化再冻结和切换损失函数
    frozen_clf.fit(X_train[:50], y_train[:50])  # 小批量初始化
    freeze_success_clf = freeze_abduction_to_identity(frozen_clf)
    
    if freeze_success_clf:
        print("✅ 成功冻结AbductionNetwork为恒等映射")
        
        # 关键：启用传统CrossEntropy损失函数
        enable_traditional_loss_mode(frozen_clf, 'classification')
        
        # 替换损失函数
        original_compute_loss_clf = frozen_clf._compute_loss
        frozen_clf._compute_loss = lambda predictions, targets: frozen_clf._traditional_loss(predictions, targets)
        
        print("✅ 已切换到传统CrossEntropy损失函数")
        
        # 重新训练（使用CrossEntropy损失）
        frozen_clf.fit(X_train, y_train)
        frozen_pred = frozen_clf.predict(X_test, mode='compatible')
        frozen_acc = accuracy_score(y_test, frozen_pred)
        print(f"因果方法(冻结+CrossEntropy) - 准确率: {frozen_acc:.4f}")
        
        # 恢复原损失函数
        frozen_clf._compute_loss = original_compute_loss_clf
    else:
        print("❌ 无法冻结AbductionNetwork")
        frozen_acc = 0
    
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
        'frozen_acc': frozen_acc if freeze_success_clf else 0,
        'improvement': causal_acc - trad_acc,
        'frozen_improvement': frozen_acc - trad_acc if freeze_success_clf else 0,
        'freeze_success': freeze_success_clf
    }


def demo_noise_robustness():
    """演示标签噪声鲁棒性"""
    print("\\n" + "="*50)
    print("🛡️ 标签噪声鲁棒性演示")
    print("="*50)
    
    results = {}
    
    # === 回归任务噪声鲁棒性 ===
    print("\\n📊 回归任务噪声鲁棒性:")
    X_reg, y_reg_clean = make_regression(n_samples=800, n_features=10, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg_clean, y_test_reg = train_test_split(
        X_reg, y_reg_clean, test_size=0.2, random_state=42
    )
    
    # 添加回归噪声 (15%异常值 - 5倍标准差的随机偏移)
    y_train_reg_noisy = y_train_reg_clean.copy()
    noise_std = np.std(y_train_reg_clean)
    n_noise_reg = int(0.15 * len(y_train_reg_noisy))
    noise_indices_reg = np.random.choice(len(y_train_reg_noisy), n_noise_reg, replace=False)
    
    for idx in noise_indices_reg:
        noise_magnitude = np.random.choice([-5, -3, 3, 5]) * noise_std
        y_train_reg_noisy[idx] += noise_magnitude
    
    print(f"回归数据: {len(y_train_reg_clean)} 样本")
    print(f"噪声样本: {n_noise_reg} 样本 ({n_noise_reg/len(y_train_reg_clean)*100:.1f}%)")
    
    # 三种方法在噪声回归数据上的表现对比
    print("\\n回归噪声测试:")
    
    # 传统回归方法
    trad_reg_noisy = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    trad_reg_noisy.fit(X_train_reg, y_train_reg_noisy)
    trad_pred_noisy = trad_reg_noisy.predict(X_test_reg)
    trad_r2_noisy = r2_score(y_test_reg, trad_pred_noisy)
    
    # 完整因果回归方法
    causal_reg_noisy = MLPCausalRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    causal_reg_noisy.fit(X_train_reg, y_train_reg_noisy)
    causal_pred_noisy = causal_reg_noisy.predict(X_test_reg, mode='compatible')
    causal_r2_noisy = r2_score(y_test_reg, causal_pred_noisy)
    
    # 冻结因果回归方法(数学等价性验证)
    frozen_reg_noisy = MLPCausalRegressor(
        hidden_layer_sizes=(64, 32),
        causal_size=32,
        max_iter=300,
        random_state=42
    )
    
    frozen_reg_noisy.fit(X_train_reg[:50], y_train_reg_noisy[:50])
    freeze_success_reg = freeze_abduction_to_identity(frozen_reg_noisy)
    
    if freeze_success_reg:
        enable_traditional_loss_mode(frozen_reg_noisy, 'regression')
        original_compute_loss_reg = frozen_reg_noisy._compute_loss
        frozen_reg_noisy._compute_loss = lambda predictions, targets: frozen_reg_noisy._traditional_loss(predictions, targets)
        
        frozen_reg_noisy.fit(X_train_reg, y_train_reg_noisy)
        frozen_pred_noisy = frozen_reg_noisy.predict(X_test_reg, mode='compatible')
        frozen_r2_noisy = r2_score(y_test_reg, frozen_pred_noisy)
        
        frozen_reg_noisy._compute_loss = original_compute_loss_reg
    else:
        frozen_r2_noisy = 0
    
    print(f"  传统MLP (MSE): {trad_r2_noisy:.4f}")
    print(f"  因果完整 (Cauchy): {causal_r2_noisy:.4f}")
    if freeze_success_reg:
        print(f"  因果冻结 (MSE): {frozen_r2_noisy:.4f}")
        print(f"  完整因果优势: +{(causal_r2_noisy - trad_r2_noisy)*100:.1f}%")
        print(f"  冻结因果优势: +{(frozen_r2_noisy - trad_r2_noisy)*100:.1f}%")
    else:
        print(f"  因果优势: +{(causal_r2_noisy - trad_r2_noisy)*100:.1f}%")
    
    results['regression'] = {
        'traditional_noisy': trad_r2_noisy,
        'causal_noisy': causal_r2_noisy,
        'frozen_noisy': frozen_r2_noisy if freeze_success_reg else 0,
        'robustness_advantage': causal_r2_noisy - trad_r2_noisy,
        'frozen_advantage': frozen_r2_noisy - trad_r2_noisy if freeze_success_reg else 0,
        'freeze_success': freeze_success_reg
    }
    
    # === 分类任务噪声鲁棒性 ===  
    print("\\n🎯 分类任务噪声鲁棒性:")
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
    
    print(f"分类数据: {len(y_train_clean)} 样本")
    print(f"噪声标签: {n_noise} 样本 ({n_noise/len(y_train_clean)*100:.1f}%)")
    
    # 三种方法在噪声数据上的表现对比
    print("\\n分类噪声测试:")
    
    # 传统方法
    trad_clf_noisy = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    trad_clf_noisy.fit(X_train, y_train_noisy)
    trad_acc_noisy = accuracy_score(y_test, trad_clf_noisy.predict(X_test))
    
    # 因果方法(完整)
    causal_clf_noisy = MLPCausalClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    causal_clf_noisy.fit(X_train, y_train_noisy)
    causal_acc_noisy = accuracy_score(y_test, causal_clf_noisy.predict(X_test))
    
    # 因果方法(冻结+传统损失) - 真正的数学等价性对比
    frozen_clf_noisy = MLPCausalClassifier(
        hidden_layer_sizes=(64, 32), 
        causal_size=32,
        max_iter=300, 
        random_state=42
    )
    
    # 先初始化再冻结和切换损失函数
    frozen_clf_noisy.fit(X_train[:50], y_train_noisy[:50])
    freeze_success_noise = freeze_abduction_to_identity(frozen_clf_noisy)
    
    if freeze_success_noise:
        # 启用传统CrossEntropy损失函数
        enable_traditional_loss_mode(frozen_clf_noisy, 'classification')
        original_compute_loss_noise = frozen_clf_noisy._compute_loss
        frozen_clf_noisy._compute_loss = lambda predictions, targets: frozen_clf_noisy._traditional_loss(predictions, targets)
        
        frozen_clf_noisy.fit(X_train, y_train_noisy)
        frozen_acc_noisy = accuracy_score(y_test, frozen_clf_noisy.predict(X_test))
        
        # 恢复原损失函数
        frozen_clf_noisy._compute_loss = original_compute_loss_noise
    else:
        frozen_acc_noisy = 0
    
    print(f"  传统MLP (CrossEntropy): {trad_acc_noisy:.4f}")
    print(f"  因果完整 (OvR-BCE): {causal_acc_noisy:.4f}")
    if freeze_success_noise:
        print(f"  因果冻结 (CrossEntropy): {frozen_acc_noisy:.4f}")
        print(f"  完整因果优势: +{(causal_acc_noisy - trad_acc_noisy)*100:.1f}%")
        print(f"  冻结因果优势: +{(frozen_acc_noisy - trad_acc_noisy)*100:.1f}%")
    else:
        print(f"  因果优势: +{(causal_acc_noisy - trad_acc_noisy)*100:.1f}%")
    
    results['classification'] = {
        'traditional_noisy': trad_acc_noisy,
        'causal_noisy': causal_acc_noisy,
        'frozen_noisy': frozen_acc_noisy if freeze_success_noise else 0,
        'robustness_advantage': causal_acc_noisy - trad_acc_noisy,
        'frozen_advantage': frozen_acc_noisy - trad_acc_noisy if freeze_success_noise else 0,
        'freeze_success': freeze_success_noise
    }
    
    # 总结噪声鲁棒性
    print(f"\\n📊 噪声鲁棒性总结:")
    print(f"回归任务: 完整因果优势 +{results['regression']['robustness_advantage']*100:.1f}%, 冻结因果优势 +{results['regression']['frozen_advantage']*100:.1f}%")
    print(f"分类任务: 完整因果优势 +{results['classification']['robustness_advantage']*100:.1f}%, 冻结因果优势 +{results['classification']['frozen_advantage']*100:.1f}%")
    
    return results


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
        print(f"  传统MLP (MSE): {results['regression']['traditional_r2']:.4f}")
        print(f"  因果完整 (Cauchy): {results['regression']['causal_r2']:.4f}")
        if results['regression']['freeze_success']:
            print(f"  因果冻结 (MSE): {results['regression']['frozen_r2']:.4f}")
            print(f"  完整 vs 传统: {results['regression']['improvement']:+.4f}")
            print(f"  冻结 vs 传统: {results['regression']['frozen_improvement']:+.4f} (数学等价性)")
        else:
            print(f"  改进: {results['regression']['improvement']:+.4f}")
        
        print(f"\\n🎯 分类任务:")
        print(f"  传统MLP (CrossEntropy): {results['classification']['traditional_acc']:.4f}")
        print(f"  因果完整 (OvR-BCE): {results['classification']['causal_acc']:.4f}")
        if results['classification']['freeze_success']:
            print(f"  因果冻结 (CrossEntropy): {results['classification']['frozen_acc']:.4f}")
            print(f"  完整 vs 传统: {results['classification']['improvement']:+.4f}")
            print(f"  冻结 vs 传统: {results['classification']['frozen_improvement']:+.4f} (数学等价性)")
        else:
            print(f"  改进: {results['classification']['improvement']:+.4f}")
        
        print(f"\\n🛡️ 噪声鲁棒性:")
        print(f"  回归任务:")
        print(f"    传统MLP (MSE): {results['noise_robustness']['regression']['traditional_noisy']:.4f}")
        print(f"    因果完整 (Cauchy): {results['noise_robustness']['regression']['causal_noisy']:.4f}")
        if results['noise_robustness']['regression']['freeze_success']:
            print(f"    因果冻结 (MSE): {results['noise_robustness']['regression']['frozen_noisy']:.4f}")
            print(f"    完整因果优势: +{results['noise_robustness']['regression']['robustness_advantage']*100:.1f}%")
            print(f"    冻结因果优势: +{results['noise_robustness']['regression']['frozen_advantage']*100:.1f}%")
        else:
            print(f"    因果优势: +{results['noise_robustness']['regression']['robustness_advantage']*100:.1f}%")
        
        print(f"  分类任务:")
        print(f"    传统MLP (CrossEntropy): {results['noise_robustness']['classification']['traditional_noisy']:.4f}")
        print(f"    因果完整 (OvR-BCE): {results['noise_robustness']['classification']['causal_noisy']:.4f}")
        if results['noise_robustness']['classification']['freeze_success']:
            print(f"    因果冻结 (CrossEntropy): {results['noise_robustness']['classification']['frozen_noisy']:.4f}")
            print(f"    完整因果优势: +{results['noise_robustness']['classification']['robustness_advantage']*100:.1f}%")
            print(f"    冻结因果优势: +{results['noise_robustness']['classification']['frozen_advantage']*100:.1f}%")
        else:
            print(f"    因果优势: +{results['noise_robustness']['classification']['robustness_advantage']*100:.1f}%")
        
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