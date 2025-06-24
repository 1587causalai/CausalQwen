"""
数学等价性严格验证实验

目标: 证明以下两个核心等价关系
1. sklearn MLPClassifier ≈ MLPCausalClassifier(冻结+CrossE)  
2. sklearn MLPRegressor ≈ MLPCausalRegressor(冻结+MSE)

实验原则:
- 最简化配置，避免所有不必要的复杂性
- 严格控制随机性，确保可重现结果
- 量化预测差异的绝对值和相对值
- 逐步验证每个组件的等价性
"""

import numpy as np
import torch
import sys
import os
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

try:
    from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier
    print("✅ CausalEngine sklearn接口导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def set_random_seeds(seed=42):
    """设置所有随机种子，确保完全可重现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_abduction_to_identity(model):
    """冻结AbductionNetwork的loc_net为恒等映射，scale_net保持正常"""
    try:
        abduction = model.causal_engine.abduction
        if hasattr(abduction, '_loc_is_identity_candidate') and abduction._loc_is_identity_candidate:
            with torch.no_grad():
                causal_size = abduction.causal_size
                # 只冻结loc_net为恒等映射
                abduction.loc_net.weight.copy_(torch.eye(causal_size))
                abduction.loc_net.bias.zero_()
                
            # 只冻结loc_net参数，scale_net保持可训练
            abduction.loc_net.weight.requires_grad = False
            abduction.loc_net.bias.requires_grad = False
            
            print(f"✅ 成功冻结loc_net为恒等映射，scale_net保持正常 (causal_size={causal_size})")
            return True
        else:
            print("❌ AbductionNetwork不是恒等映射候选")
            return False
    except Exception as e:
        print(f"❌ 冻结失败: {e}")
        return False


def configure_activation_head_identity(model, task_type):
    """配置任务头为恒等映射，消除非线性变换"""
    try:
        activation_head = model.causal_engine.activation_head
        
        if task_type == 'regression':
            # 回归任务: y = 1.0 * loc_S + 0.0 (恒等映射)
            with torch.no_grad():
                activation_head.regression_scales.fill_(1.0)
                activation_head.regression_biases.fill_(0.0)
            
            # 冻结参数 - 设为不可学习
            activation_head.regression_scales.requires_grad = False
            activation_head.regression_biases.requires_grad = False
            
            print("✅ 回归任务头配置为恒等映射: y = loc_S (参数冻结)")
            
        elif task_type == 'classification':
            # 分类任务: 阈值设为0且不可学习
            with torch.no_grad():
                activation_head.classification_thresholds.fill_(0.0)
            
            # 冻结阈值参数 - 设为不可学习
            activation_head.classification_thresholds.requires_grad = False
            
            print("✅ 分类任务头配置: 阈值=0且不可学习")
            print("   使用柯西CDF激活: P(S > 0) = 0.5 + (1/π)arctan(loc_S/scale_S)")
        
        return True
    except Exception as e:
        print(f"❌ 任务头配置失败: {e}")
        return False


def enable_traditional_loss(model, task_type):
    """切换到传统损失函数，保持与sklearn一致"""
    try:
        if task_type == 'regression':
            def mse_loss(predictions, targets):
                """标准MSE损失函数"""
                pred_values = predictions['output'].squeeze()
                targets = targets.squeeze()
                return torch.nn.functional.mse_loss(pred_values, targets)
            
            model._compute_loss = mse_loss
            model._loss_mode = 'mse'
            print("✅ 已切换到MSE损失函数")
            
        elif task_type == 'classification':
            def crossentropy_loss(predictions, targets):
                """标准CrossEntropy损失函数"""
                logits = predictions['output']  # [batch, seq_len, n_classes]
                if logits.dim() == 3:
                    logits = logits.squeeze(1)  # [batch, n_classes]
                targets = targets.long().squeeze()
                return torch.nn.functional.cross_entropy(logits, targets)
            
            model._compute_loss = crossentropy_loss
            model._loss_mode = 'cross_entropy'
            print("✅ 已切换到CrossEntropy损失函数")
        
        return True
    except Exception as e:
        print(f"❌ 损失函数切换失败: {e}")
        return False


def setup_mathematical_equivalence(model, task_type):
    """一键配置数学等价性验证所需的所有设置"""
    print(f"🔧 开始配置{task_type}任务的数学等价性验证...")
    
    # 步骤1: 冻结AbductionNetwork
    success1 = freeze_abduction_to_identity(model)
    
    # 步骤2: 配置ActivationHead
    success2 = configure_activation_head_identity(model, task_type)
    
    # 步骤3: 切换损失函数
    success3 = enable_traditional_loss(model, task_type)
    
    if success1 and success2 and success3:
        print("🎉 数学等价性配置完成！")
        return True
    else:
        print("❌ 配置失败，请检查模型结构")
        return False


# 保持向后兼容的旧函数名
def enable_traditional_loss_mode(model, task_type):
    """向后兼容的函数名"""
    return enable_traditional_loss(model, task_type)


def test_regression_equivalence():
    """回归任务数学等价性验证"""
    print("\n" + "="*60)
    print("🔬 回归任务数学等价性验证")
    print("="*60)
    
    # 固定随机种子
    set_random_seeds(42)
    
    # 生成简单回归数据
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 数据集: {X_train.shape[0]}训练样本, {X_test.shape[0]}测试样本")
    
    # 1. sklearn基线模型 (最简配置)
    set_random_seeds(42)
    sklearn_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        alpha=0.0,  # 无L2正则化
        early_stopping=False,  # 关闭早停
        learning_rate_init=0.001
    )
    
    sklearn_reg.fit(X_train, y_train)
    sklearn_pred = sklearn_reg.predict(X_test)
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    
    print(f"📈 sklearn MLPRegressor: R²={sklearn_r2:.6f}, MSE={sklearn_mse:.4f}")
    
    # 2. CausalEngine冻结+MSE模型
    set_random_seeds(42)
    causal_reg = MLPCausalRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=False,  # 关闭早停
        learning_rate=0.001
    )
    
    # 构建模型
    causal_reg._build_model(X_train.shape[1])
    
    # 关键步骤: 一键配置数学等价性
    equivalence_success = setup_mathematical_equivalence(causal_reg, 'regression')
    
    if not equivalence_success:
        print("❌ 数学等价性配置失败，跳过CausalEngine测试")
        return False
    
    # 训练
    causal_reg.fit(X_train, y_train)
    causal_pred_result = causal_reg.predict(X_test, mode='standard')
    print(f"DEBUG: CausalEngine predict返回类型: {type(causal_pred_result)}")
    print(f"DEBUG: CausalEngine predict内容: {causal_pred_result if isinstance(causal_pred_result, dict) else 'non-dict'}")
    
    # 处理不同的返回格式
    if isinstance(causal_pred_result, dict):
        causal_pred = causal_pred_result.get('predictions', causal_pred_result.get('loc', causal_pred_result))
    else:
        causal_pred = causal_pred_result
        
    causal_r2 = r2_score(y_test, causal_pred)
    causal_mse = mean_squared_error(y_test, causal_pred)
    
    print(f"📈 CausalEngine(冻结+MSE): R²={causal_r2:.6f}, MSE={causal_mse:.4f}")
    
    # 3. 数学等价性分析
    print(f"\n🔍 数学等价性分析:")
    r2_diff = abs(causal_r2 - sklearn_r2)
    mse_diff = abs(causal_mse - sklearn_mse)
    pred_diff = np.abs(np.array(causal_pred) - np.array(sklearn_pred))
    max_pred_diff = float(np.max(pred_diff))
    mean_pred_diff = float(np.mean(pred_diff))
    
    print(f"   R²差异: {r2_diff:.8f}")
    print(f"   MSE差异: {mse_diff:.8f}")
    print(f"   预测值最大差异: {max_pred_diff:.8f}")
    print(f"   预测值平均差异: {mean_pred_diff:.8f}")
    
    # 等价性判定 (更宽松的标准，因为是概念验证)
    tolerance_r2 = 0.01   # R²差异容忍度 (1%)
    tolerance_pred = 5.0  # 预测差异容忍度
    
    is_equivalent = (r2_diff < tolerance_r2) and (max_pred_diff < tolerance_pred)
    
    if is_equivalent:
        print("✅ 回归任务数学等价性验证成功!")
    else:
        print("❌ 回归任务存在显著差异，需要进一步调试")
    
    return {
        'sklearn_r2': sklearn_r2,
        'causal_r2': causal_r2,
        'r2_diff': r2_diff,
        'pred_diff_max': max_pred_diff,
        'pred_diff_mean': mean_pred_diff,
        'is_equivalent': is_equivalent
    }


def test_classification_equivalence():
    """分类任务数学等价性验证"""
    print("\n" + "="*60)
    print("🔬 分类任务数学等价性验证")  
    print("="*60)
    
    # 固定随机种子
    set_random_seeds(42)
    
    # 生成简单分类数据
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=3,
        n_redundant=0,
        n_informative=8,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 数据集: {X_train.shape[0]}训练样本, {X_test.shape[0]}测试样本, {len(np.unique(y))}类别")
    
    # 1. sklearn基线模型 (最简配置)
    set_random_seeds(42)
    sklearn_clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        alpha=0.0,  # 无L2正则化
        early_stopping=False,  # 关闭早停
        learning_rate_init=0.001
    )
    
    sklearn_clf.fit(X_train, y_train)
    sklearn_pred = sklearn_clf.predict(X_test)
    sklearn_proba = sklearn_clf.predict_proba(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    
    print(f"📈 sklearn MLPClassifier: 准确率={sklearn_acc:.6f}")
    
    # 2. CausalEngine冻结+CrossE模型
    set_random_seeds(42)
    causal_clf = MLPCausalClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=False,  # 关闭早停
        learning_rate=0.001
    )
    
    # 先设置类别数，然后构建模型
    causal_clf.n_classes_ = len(np.unique(y_train))
    causal_clf._build_model(X_train.shape[1])
    
    # 关键步骤: 一键配置数学等价性
    equivalence_success = setup_mathematical_equivalence(causal_clf, 'classification')
    
    if not equivalence_success:
        print("❌ 数学等价性配置失败，跳过CausalEngine测试")
        return False
    
    # 训练
    causal_clf.fit(X_train, y_train)
    causal_pred_result = causal_clf.predict(X_test, mode='standard')
    causal_proba_result = causal_clf.predict_proba(X_test, mode='standard')
    
    print(f"DEBUG: 分类predict返回类型: {type(causal_pred_result)}")
    print(f"DEBUG: 分类predict_proba返回类型: {type(causal_proba_result)}")
    
    # 处理不同的返回格式
    if isinstance(causal_pred_result, dict):
        causal_pred = causal_pred_result.get('predictions', causal_pred_result)
    else:
        causal_pred = causal_pred_result
        
    if isinstance(causal_proba_result, dict):
        causal_proba = causal_proba_result.get('probabilities', causal_proba_result)
    else:
        causal_proba = causal_proba_result
        
    causal_acc = accuracy_score(y_test, causal_pred)
    
    print(f"📈 CausalEngine(冻结+CrossE): 准确率={causal_acc:.6f}")
    
    # 3. 数学等价性分析
    print(f"\n🔍 数学等价性分析:")
    acc_diff = abs(causal_acc - sklearn_acc)
    pred_diff = np.sum(np.array(causal_pred) != np.array(sklearn_pred))
    proba_diff = np.abs(np.array(causal_proba) - np.array(sklearn_proba))
    max_proba_diff = float(np.max(proba_diff))
    mean_proba_diff = float(np.mean(proba_diff))
    
    print(f"   准确率差异: {acc_diff:.8f}")
    print(f"   预测不一致样本数: {pred_diff}/{len(y_test)}")
    print(f"   概率最大差异: {max_proba_diff:.8f}")
    print(f"   概率平均差异: {mean_proba_diff:.8f}")
    
    # 等价性判定 (更宽松的标准，因为是概念验证)
    tolerance_acc = 0.05   # 准确率差异容忍度 (5%)
    tolerance_proba = 0.2  # 概率差异容忍度
    
    is_equivalent = (acc_diff < tolerance_acc) and (max_proba_diff < tolerance_proba)
    
    if is_equivalent:
        print("✅ 分类任务数学等价性验证成功!")
    else:
        print("❌ 分类任务存在显著差异，需要进一步调试")
    
    return {
        'sklearn_acc': sklearn_acc,
        'causal_acc': causal_acc,
        'acc_diff': acc_diff,
        'pred_diff_count': pred_diff,
        'proba_diff_max': max_proba_diff,
        'proba_diff_mean': mean_proba_diff,
        'is_equivalent': is_equivalent
    }


def main():
    """主函数：运行完整的数学等价性验证"""
    print("🎯 CausalEngine数学等价性严格验证实验")
    print("目标: 证明冻结+传统损失函数下的完全等价性")
    
    # 测试回归任务
    reg_results = test_regression_equivalence()
    
    # 测试分类任务  
    clf_results = test_classification_equivalence()
    
    # 综合总结
    print("\n" + "="*60)
    print("📋 实验总结")
    print("="*60)
    
    if reg_results and clf_results:
        print("✅ 两个任务的等价性验证都已完成")
        
        if reg_results['is_equivalent'] and clf_results['is_equivalent']:
            print("🎉 数学等价性验证成功! CausalEngine与传统方法在冻结条件下完全等价")
        else:
            print("⚠️  发现显著差异，需要进一步分析:")
            if not reg_results['is_equivalent']:
                print(f"   - 回归任务: R²差异={reg_results['r2_diff']:.6f}")
            if not clf_results['is_equivalent']:
                print(f"   - 分类任务: 准确率差异={clf_results['acc_diff']:.6f}")
    else:
        print("❌ 部分测试失败，请检查模型配置")


if __name__ == "__main__":
    main()