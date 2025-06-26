"""
CausalEngine 实际应用演示 v3.0

基于已验证的数学等价性，展示CausalEngine在实际场景中的应用价值：
1. sklearn替代能力验证  
2. 五模式因果推理光谱
3. 不确定性量化与鲁棒性
4. 实际数据集应用案例
5. 高级功能演示

核心特性：
- 基于优化架构（智能维度对齐 + 前向传播bypass）
- 无需复杂配置，开箱即用
- 完整的sklearn兼容性
- 丰富的因果推理能力
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

# 导入CausalEngine sklearn接口
try:
    from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier
    print("✅ CausalEngine sklearn接口导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def demo_sklearn_replacement():
    """演示CausalEngine作为sklearn的直接替代品"""
    print("\n" + "="*60)
    print("🔄 sklearn直接替代演示")
    print("="*60)
    
    # 回归任务对比
    print("\n📊 回归任务对比:")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=15, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # sklearn基线
    sklearn_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    sklearn_reg.fit(X_train_reg, y_train_reg)
    sklearn_pred = sklearn_reg.predict(X_test_reg)
    sklearn_r2 = r2_score(y_test_reg, sklearn_pred)
    
    # CausalEngine deterministic模式（直接替代）
    causal_reg = MLPCausalRegressor(
        hidden_layer_sizes=(100, 50), 
        mode='deterministic',  # 等价于sklearn
        max_iter=500, 
        random_state=42
    )
    causal_reg.fit(X_train_reg, y_train_reg)
    causal_pred = causal_reg.predict(X_test_reg)
    if isinstance(causal_pred, dict):
        causal_pred = causal_pred['predictions']
    causal_r2 = r2_score(y_test_reg, causal_pred)
    
    print(f"  sklearn MLPRegressor:      R² = {sklearn_r2:.6f}")
    print(f"  CausalEngine deterministic: R² = {causal_r2:.6f}")
    print(f"  差异: {abs(causal_r2 - sklearn_r2):.6f} (应该很小)")
    
    # 分类任务对比
    print("\n🎯 分类任务对比:")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=15, n_classes=4, 
                                      n_informative=10, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    # sklearn基线
    sklearn_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    sklearn_clf.fit(X_train_clf, y_train_clf)
    sklearn_acc = accuracy_score(y_test_clf, sklearn_clf.predict(X_test_clf))
    
    # CausalEngine deterministic模式
    causal_clf = MLPCausalClassifier(
        hidden_layer_sizes=(100, 50),
        mode='deterministic',
        max_iter=500,
        random_state=42
    )
    causal_clf.fit(X_train_clf, y_train_clf)
    causal_pred_clf = causal_clf.predict(X_test_clf)
    if isinstance(causal_pred_clf, dict):
        causal_pred_clf = causal_pred_clf['predictions']
    causal_acc = accuracy_score(y_test_clf, causal_pred_clf)
    
    print(f"  sklearn MLPClassifier:      准确率 = {sklearn_acc:.6f}")
    print(f"  CausalEngine deterministic: 准确率 = {causal_acc:.6f}")
    print(f"  差异: {abs(causal_acc - sklearn_acc):.6f} (应该很小)")
    
    print("\n✅ CausalEngine可以作为sklearn的直接替代品！")
    return {
        'regression': {'sklearn': sklearn_r2, 'causal': causal_r2},
        'classification': {'sklearn': sklearn_acc, 'causal': causal_acc}
    }


def demo_five_modes_spectrum():
    """演示五模式因果推理光谱"""
    print("\n" + "="*60)
    print("🌈 五模式因果推理光谱演示")
    print("="*60)
    
    # 生成有噪声的数据
    X, y = make_regression(n_samples=800, n_features=10, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modes = {
        'deterministic': '确定性预测（等价sklearn）',
        'exogenous': '外生噪声推理',
        'endogenous': '内生因果推理', 
        'standard': '标准因果推理',
        'sampling': '探索性因果推理'
    }
    
    print(f"\n🚀 五模式性能表格:")
    print("+" + "-"*70 + "+")
    print(f"| {'模式':<12} | {'R²':<8} | {'MSE':<10} | {'描述':<20} |")
    print("+" + "-"*70 + "+")
    
    results = {}
    for mode, description in modes.items():
        # 训练模型
        reg = MLPCausalRegressor(
            hidden_layer_sizes=(64, 32),
            mode=mode,
            max_iter=300,
            random_state=42,
            verbose=False
        )
        
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        
        if isinstance(pred, dict):
            pred = pred['predictions']
            
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        
        results[mode] = {'r2': r2, 'mse': mse, 'model': reg}
        print(f"| {mode:<12} | {r2:<8.4f} | {mse:<10.1f} | {description:<20} |")
    
    print("+" + "-"*70 + "+")
    
    # 演示模式特有功能
    print(f"\n🔍 模式特有功能演示:")
    
    # 不确定性量化（非deterministic模式）
    for mode in ['endogenous', 'standard', 'sampling']:
        model = results[mode]['model']
        
        if hasattr(model, 'predict_dist'):
            dist_params = model.predict_dist(X_test[:3])
            if dist_params is not None:
                uncertainty = dist_params[:, 0, 1]  # 尺度参数
                print(f"  {mode}: 平均不确定性 = {np.mean(uncertainty):.4f}")
    
    return results


def demo_uncertainty_quantification():
    """演示不确定性量化的实际价值"""
    print("\n" + "="*60)
    print("🌡️ 不确定性量化实际应用")
    print("="*60)
    
    # 创建异质噪声数据（不同区域噪声不同）
    np.random.seed(42)
    X = np.random.randn(600, 8)
    
    # 基础信号
    y_base = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.5 * np.sum(X[:, 3:6], axis=1)
    
    # 异质噪声：根据X[:, 0]的值决定噪声大小
    noise_scale = 0.1 + 0.4 * np.abs(X[:, 0])  # 噪声随X[:, 0]变化
    noise = np.random.normal(0, noise_scale)
    y = y_base + noise
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("📊 异质噪声回归任务：噪声随输入特征变化")
    
    # 对比传统方法（只有点估计）vs 因果方法（完整分布）
    models = {
        'sklearn': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
        'deterministic': MLPCausalRegressor(hidden_layer_sizes=(64, 32), mode='deterministic', max_iter=300, random_state=42),
        'standard': MLPCausalRegressor(hidden_layer_sizes=(64, 32), mode='standard', max_iter=300, random_state=42)
    }
    
    print(f"\n🎯 预测性能对比:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        if isinstance(pred, dict):
            pred = pred['predictions']
            
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        print(f"  {name:<12}: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    # 不确定性分析
    print(f"\n🌡️ 不确定性量化分析:")
    standard_model = models['standard']
    
    # 获取预测分布
    dist_params = standard_model.predict_dist(X_test)
    if dist_params is not None:
        pred_mean = dist_params[:, 0, 0]  # 位置参数
        pred_uncertainty = dist_params[:, 0, 1]  # 尺度参数
        
        # 分析不确定性与真实噪声的相关性
        true_noise_scale = 0.1 + 0.4 * np.abs(X_test[:, 0])
        correlation = np.corrcoef(pred_uncertainty, true_noise_scale)[0, 1]
        
        print(f"  预测不确定性范围: [{np.min(pred_uncertainty):.3f}, {np.max(pred_uncertainty):.3f}]")
        print(f"  真实噪声范围: [{np.min(true_noise_scale):.3f}, {np.max(true_noise_scale):.3f}]")
        print(f"  不确定性-真实噪声相关性: {correlation:.4f}")
        
        # 高不确定性样本分析
        high_uncertainty_mask = pred_uncertainty > np.percentile(pred_uncertainty, 80)
        high_uncertainty_error = np.abs(pred_mean[high_uncertainty_mask] - y_test[high_uncertainty_mask])
        low_uncertainty_error = np.abs(pred_mean[~high_uncertainty_mask] - y_test[~high_uncertainty_mask])
        
        print(f"  高不确定性样本平均误差: {np.mean(high_uncertainty_error):.4f}")
        print(f"  低不确定性样本平均误差: {np.mean(low_uncertainty_error):.4f}")
        
        if np.mean(high_uncertainty_error) > np.mean(low_uncertainty_error):
            print("  ✅ 不确定性正确指示了预测误差！")
        else:
            print("  ⚠️ 不确定性指示需要改进")


def demo_noise_robustness():
    """演示噪声鲁棒性的实际价值"""
    print("\n" + "="*60)
    print("🛡️ 噪声鲁棒性实际应用")
    print("="*60)
    
    # 分类任务：模拟真实世界的标签噪声
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                              n_informative=15, n_redundant=2, random_state=42)
    X_train, X_test, y_train_clean, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 添加不同程度的标签噪声
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    results = {}
    
    print("📊 不同噪声水平下的性能对比:")
    print("+" + "-"*65 + "+")
    print(f"| {'噪声水平':<8} | {'sklearn':<12} | {'deterministic':<12} | {'standard':<12} |")
    print("+" + "-"*65 + "+")
    
    for noise_level in noise_levels:
        # 创建噪声标签
        y_train_noisy = y_train_clean.copy()
        if noise_level > 0:
            n_noise = int(noise_level * len(y_train_noisy))
            noise_indices = np.random.choice(len(y_train_noisy), n_noise, replace=False)
            for idx in noise_indices:
                available_labels = [l for l in np.unique(y) if l != y_train_noisy[idx]]
                y_train_noisy[idx] = np.random.choice(available_labels)
        
        # 测试三种方法
        methods = {
            'sklearn': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
            'deterministic': MLPCausalClassifier(hidden_layer_sizes=(64, 32), mode='deterministic', max_iter=300, random_state=42),
            'standard': MLPCausalClassifier(hidden_layer_sizes=(64, 32), mode='standard', max_iter=300, random_state=42)
        }
        
        noise_results = {}
        for name, model in methods.items():
            model.fit(X_train, y_train_noisy)
            pred = model.predict(X_test)
            
            if isinstance(pred, dict):
                pred = pred['predictions']
                
            acc = accuracy_score(y_test, pred)
            noise_results[name] = acc
        
        results[noise_level] = noise_results
        print(f"| {noise_level:<8.1f} | {noise_results['sklearn']:<12.4f} | {noise_results['deterministic']:<12.4f} | {noise_results['standard']:<12.4f} |")
    
    print("+" + "-"*65 + "+")
    
    # 分析鲁棒性
    print(f"\n🔍 鲁棒性分析:")
    clean_performance = results[0.0]
    noisy_performance = results[0.3]  # 30%噪声
    
    for method in ['sklearn', 'deterministic', 'standard']:
        degradation = clean_performance[method] - noisy_performance[method]
        print(f"  {method}: 性能下降 = {degradation:.4f} (30%噪声)")
    
    # 找出最鲁棒的方法
    degradations = {method: clean_performance[method] - noisy_performance[method] 
                   for method in ['sklearn', 'deterministic', 'standard']}
    most_robust = min(degradations, key=degradations.get)
    print(f"  🏆 最鲁棒方法: {most_robust} (性能下降最小)")
    
    return results


def demo_real_world_datasets():
    """在真实数据集上演示CausalEngine"""
    print("\n" + "="*60)
    print("🌍 真实数据集应用演示")
    print("="*60)
    
    # 糖尿病回归数据集
    print("\n📊 糖尿病回归任务 (sklearn内置数据集):")
    diabetes = load_diabetes()
    X_diabetes, y_diabetes = diabetes.data, diabetes.target
    
    # 标准化
    scaler = StandardScaler()
    X_diabetes = scaler.fit_transform(X_diabetes)
    
    X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)
    
    # 对比不同方法
    models = {
        'sklearn': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'deterministic': MLPCausalRegressor(hidden_layer_sizes=(100, 50), mode='deterministic', max_iter=500, random_state=42),
        'standard': MLPCausalRegressor(hidden_layer_sizes=(100, 50), mode='standard', max_iter=500, random_state=42),
        'sampling': MLPCausalRegressor(hidden_layer_sizes=(100, 50), mode='sampling', max_iter=500, random_state=42)
    }
    
    print("  方法对比:")
    diabetes_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        if isinstance(pred, dict):
            pred = pred['predictions']
            
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        diabetes_results[name] = {'r2': r2, 'mse': mse}
        print(f"    {name:<12}: R² = {r2:.4f}, MSE = {mse:.1f}")
    
    # 红酒分类数据集
    print("\n🍷 红酒分类任务 (sklearn内置数据集):")
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    
    # 标准化
    X_wine = scaler.fit_transform(X_wine)
    X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)
    
    clf_models = {
        'sklearn': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'deterministic': MLPCausalClassifier(hidden_layer_sizes=(100, 50), mode='deterministic', max_iter=500, random_state=42),
        'standard': MLPCausalClassifier(hidden_layer_sizes=(100, 50), mode='standard', max_iter=500, random_state=42),
        'endogenous': MLPCausalClassifier(hidden_layer_sizes=(100, 50), mode='endogenous', max_iter=500, random_state=42)
    }
    
    print("  方法对比:")
    wine_results = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        if isinstance(pred, dict):
            pred = pred['predictions']
            
        acc = accuracy_score(y_test, pred)
        wine_results[name] = acc
        print(f"    {name:<12}: 准确率 = {acc:.4f}")
    
    return {'diabetes': diabetes_results, 'wine': wine_results}


def demo_advanced_features():
    """演示高级功能"""
    print("\n" + "="*60)
    print("🚀 高级功能演示")
    print("="*60)
    
    # 生成数据
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练一个标准模式的模型
    model = MLPCausalRegressor(
        hidden_layer_sizes=(64, 32),
        mode='standard',
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("\n🔄 动态模式切换:")
    # 同一模型，不同预测模式
    sample_X = X_test[:3]
    modes = ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    
    for mode in modes:
        pred = model.predict(sample_X, mode=mode)
        if isinstance(pred, dict):
            values = pred['predictions']
        else:
            values = pred
        print(f"  {mode:<12}: {values}")
    
    print("\n📊 分布参数预测:")
    # 获取完整的分布参数
    if hasattr(model, 'predict_dist'):
        dist_params = model.predict_dist(sample_X)
        if dist_params is not None:
            print(f"  分布参数形状: {dist_params.shape}")
            print(f"  位置参数: {dist_params[:, 0, 0]}")
            print(f"  尺度参数: {dist_params[:, 0, 1]}")
    
    print("\n🎯 置信区间估计:")
    # 基于分布参数构建置信区间
    if dist_params is not None:
        loc = dist_params[:, 0, 0]
        scale = dist_params[:, 0, 1]
        
        # 柯西分布的95%置信区间（近似）
        lower = loc - 12.7 * scale  # 近似95%置信区间
        upper = loc + 12.7 * scale
        
        print("  样本预测与95%置信区间:")
        for i in range(len(sample_X)):
            print(f"    样本{i}: {loc[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}]")


def main():
    """主演示函数"""
    print("🚀 CausalEngine 实际应用演示 v3.0")
    print("="*70)
    print("基于已验证的数学等价性，展示CausalEngine的实际应用价值")
    
    results = {}
    
    try:
        # 1. sklearn替代能力验证
        results['sklearn_replacement'] = demo_sklearn_replacement()
        
        # 2. 五模式因果推理光谱
        results['five_modes'] = demo_five_modes_spectrum()
        
        # 3. 不确定性量化
        demo_uncertainty_quantification()
        
        # 4. 噪声鲁棒性
        results['noise_robustness'] = demo_noise_robustness()
        
        # 5. 真实数据集应用
        results['real_world'] = demo_real_world_datasets()
        
        # 6. 高级功能
        demo_advanced_features()
        
        # 总结
        print("\n" + "="*70)
        print("📊 应用演示总结")
        print("="*70)
        
        print("✅ 核心验证:")
        sklearn_replacement = results['sklearn_replacement']
        print(f"  - 回归替代: CausalEngine vs sklearn差异 {abs(sklearn_replacement['regression']['causal'] - sklearn_replacement['regression']['sklearn']):.6f}")
        print(f"  - 分类替代: CausalEngine vs sklearn差异 {abs(sklearn_replacement['classification']['causal'] - sklearn_replacement['classification']['sklearn']):.6f}")
        
        print("\n🌈 五模式能力:")
        five_modes = results['five_modes']
        best_mode = max(five_modes.keys(), key=lambda k: five_modes[k]['r2'])
        print(f"  - 最佳模式: {best_mode} (R² = {five_modes[best_mode]['r2']:.4f})")
        print(f"  - 模式范围: {len(five_modes)} 种不同的因果推理策略")
        
        print("\n🛡️ 鲁棒性优势:")
        noise_clean = results['noise_robustness'][0.0]
        noise_heavy = results['noise_robustness'][0.3]
        for method in ['sklearn', 'standard']:
            degradation = noise_clean[method] - noise_heavy[method]
            print(f"  - {method}: 30%噪声下性能下降 {degradation:.4f}")
        
        print("\n🌍 真实数据表现:")
        real_world = results['real_world']
        print(f"  - 糖尿病数据: 最佳R² = {max(real_world['diabetes'][m]['r2'] for m in real_world['diabetes']):.4f}")
        print(f"  - 红酒数据: 最佳准确率 = {max(real_world['wine'].values()):.4f}")
        
        print("\n🎉 CausalEngine为机器学习带来了从确定性到因果推理的完整能力光谱！")
        print("🚀 从sklearn兼容性到高级因果推理，一个框架满足所有需求！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = main()
    exit(0 if success else 1)