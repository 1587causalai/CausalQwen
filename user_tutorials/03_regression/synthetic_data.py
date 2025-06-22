"""
回归任务实战 - 使用合成数据
=========================

这个教程将深入演示如何使用 CausalQwen 处理回归任务。
我们将使用 scikit-learn 生成的合成数据，模拟真实的业务场景。

学习目标：
1. 理解回归任务的特点
2. 掌握数据预处理技巧
3. 学会调节模型参数
4. 理解不确定性量化
5. 掌握结果解释方法
"""

import sys
import os
import numpy as np
import matplotlib
# 如果是非交互模式，使用非交互后端
if len(sys.argv) > 1:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.simple_models import SimpleCausalRegressor, compare_with_sklearn
from utils.data_helpers import (
    generate_regression_data,
    prepare_data_for_training,
    explore_data,
    visualize_predictions,
    save_results
)


def main():
    """主函数：完整的回归任务流程"""
    
    print("📈 CausalQwen 回归任务实战教程")
    print("=" * 50)
    
    # 演示多个不同的回归场景
    scenarios = [
        {
            'name': '销售预测',
            'description': '根据市场指标预测销售额',
            'n_samples': 1000,
            'n_features': 12,
            'noise_level': 0.1,
            'difficulty': 'easy'
        },
        {
            'name': '能耗预测', 
            'description': '根据建筑特征预测能耗',
            'n_samples': 800,
            'n_features': 15,
            'noise_level': 0.2,
            'difficulty': 'medium'
        },
        {
            'name': '股价预测',
            'description': '根据技术指标预测股价变化',
            'n_samples': 600,
            'n_features': 20,
            'noise_level': 0.3,
            'difficulty': 'hard'
        }
    ]
    
    print("\\n可用的回归场景:")
    for i, scenario in enumerate(scenarios):
        print(f"{i+1}. {scenario['name']} - {scenario['description']}")
    
    # 让用户选择场景
    choice = None
    if len(sys.argv) > 1:
        try:
            arg = int(sys.argv[1]) - 1
            if 0 <= arg < len(scenarios):
                choice = arg
                print(f"\\n自动选择场景: {arg + 1}")
        except ValueError:
            pass
    
    if choice is None:
        while True:
            try:
                choice = int(input("\\n请选择一个场景 (1-3): ")) - 1
                if 0 <= choice < len(scenarios):
                    break
                else:
                    print("请输入有效的选择 (1-3)")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\\n默认选择第一个场景...")
                choice = 0
                break
    
    selected_scenario = scenarios[choice]
    print(f"\\n🎯 您选择了: {selected_scenario['name']}")
    print(f"场景描述: {selected_scenario['description']}")
    
    # 运行选择的场景
    run_regression_scenario(selected_scenario)
    
    print("\\n🎉 回归任务教程完成！")
    print("\\n📖 接下来您可以：")
    print("  - 尝试其他回归场景")
    print("  - 调节模型参数看效果变化")
    print("  - 使用您自己的数据")
    print("  - 查看 tips_and_tricks.py 学习进阶技巧")


def run_regression_scenario(scenario):
    """运行特定的回归场景"""
    
    print(f"\\n🚀 开始 {scenario['name']} 场景")
    print("-" * 40)
    
    # 1. 生成数据
    print("\\n📊 步骤 1: 生成数据")
    X, y, info = generate_regression_data(
        n_samples=scenario['n_samples'],
        n_features=scenario['n_features'],
        noise_level=scenario['noise_level'],
        difficulty=scenario['difficulty'],
        random_state=42
    )
    
    # 为了更好的演示，给特征添加有意义的名称
    feature_names = generate_feature_names(scenario['name'], scenario['n_features'])
    
    print(f"\\n特征说明（{scenario['name']}场景）:")
    for i, name in enumerate(feature_names[:5]):  # 只显示前5个
        print(f"  特征 {i+1}: {name}")
    if len(feature_names) > 5:
        print(f"  ... 还有 {len(feature_names)-5} 个特征")
    
    # 2. 数据探索
    print("\\n🔍 步骤 2: 数据探索")
    explore_data(X, y, info, show_plots=True)
    
    # 3. 数据准备
    print("\\n🔧 步骤 3: 数据准备")
    data = prepare_data_for_training(X, y, test_size=0.2, validation_size=0.2)
    
    # 4. 训练基础模型
    print("\\n🚀 步骤 4: 训练 CausalQwen 模型")
    print("使用默认参数训练...")
    
    model_basic = SimpleCausalRegressor(random_state=42)
    model_basic.fit(
        data['X_train'], data['y_train'], 
        epochs=40, 
        verbose=True
    )
    
    # 5. 训练优化模型
    print("\\n⚙️ 步骤 5: 参数优化")
    print("让我们尝试不同的训练参数...")
    
    model_optimized = SimpleCausalRegressor(random_state=42)
    model_optimized.fit(
        data['X_train'], data['y_train'],
        epochs=60,
        validation_split=0.25,
        verbose=True
    )
    
    # 6. 模型对比
    print("\\n📊 步骤 6: 模型性能对比")
    
    models = {
        'CausalQwen (基础)': model_basic,
        'CausalQwen (优化)': model_optimized
    }
    
    results = {}
    
    for name, model in models.items():
        # 预测
        pred = model.predict(data['X_test'])
        
        # 如果数据被标准化，需要反标准化
        if 'y' in data['scalers']:
            pred_original = data['scalers']['y'].inverse_transform(pred.reshape(-1, 1)).flatten()
            y_test_original = data['scalers']['y'].inverse_transform(data['y_test'].reshape(-1, 1)).flatten()
        else:
            pred_original = pred
            y_test_original = data['y_test']
        
        # 计算指标
        r2 = r2_score(y_test_original, pred_original)
        mae = mean_absolute_error(y_test_original, pred_original)
        mse = mean_squared_error(y_test_original, pred_original)
        rmse = np.sqrt(mse)
        
        results[name] = {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'predictions': pred_original,
            'actual': y_test_original
        }
        
        print(f"\\n{name}:")
        print(f"  R² 分数: {r2:.4f}")
        print(f"  平均绝对误差: {mae:.4f}")
        print(f"  均方根误差: {rmse:.4f}")
    
    # 7. 不确定性分析
    print("\\n🔍 步骤 7: 不确定性分析")
    print("CausalQwen 的独特优势：量化预测不确定性")
    
    # 选择最佳模型进行不确定性分析
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = models[best_model_name]
    
    print(f"使用最佳模型: {best_model_name}")
    
    # 获取带不确定性的预测
    test_sample_indices = np.random.choice(len(data['X_test']), 10, replace=False)
    X_sample = data['X_test'][test_sample_indices]
    y_sample = data['y_test'][test_sample_indices]
    
    pred_mean, pred_std = best_model.predict(X_sample, return_uncertainty=True)
    
    print("\\n样本预测结果 (带不确定性):")
    print("  样本  |   真实值   |   预测值   |  不确定性  |   状态")
    print("  -----|-----------|-----------|-----------|----------")
    
    for i in range(len(pred_mean)):
        true_val = y_sample[i]
        pred_val = pred_mean[i]
        uncertainty = pred_std[i]
        
        # 判断预测是否在不确定性范围内
        in_range = abs(true_val - pred_val) <= uncertainty
        status = "✅ 准确" if in_range else "⚠️ 偏差"
        
        print(f"   {i+1:2d}   | {true_val:8.3f}  | {pred_val:8.3f}  | ±{uncertainty:7.3f}  | {status}")
    
    # 8. 特征重要性分析
    print("\\n🧠 步骤 8: 预测解释")
    analyze_feature_importance(best_model, X_sample, feature_names, scenario['name'])
    
    # 9. 与传统方法对比
    print("\\n⚖️ 步骤 9: 与传统机器学习对比")
    
    # 使用原始数据（未标准化）进行公平对比
    comparison_results = compare_with_sklearn(X, y, task_type='regression')
    
    # 10. 结果可视化
    print("\\n📊 步骤 10: 结果可视化")
    
    # 使用最佳模型的预测结果
    best_pred = results[best_model_name]['predictions']
    best_actual = results[best_model_name]['actual']
    
    visualize_predictions(best_actual, best_pred, 'regression', f'{scenario["name"]} - CausalQwen 结果')
    
    # 绘制不确定性图
    plot_uncertainty_analysis(y_sample, pred_mean, pred_std, scenario['name'])
    
    # 11. 保存结果
    print("\\n💾 步骤 11: 保存结果")
    
    final_results = {
        'scenario': scenario,
        'model_comparison': results,
        'sklearn_comparison': comparison_results,
        'feature_names': feature_names,
        'best_model': best_model_name
    }
    
    filename = f"user_tutorials/results/{scenario['name']}_regression_results.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_results(final_results, filename)
    
    print(f"\\n✨ {scenario['name']} 场景完成！")
    print("\\n🎯 关键收获:")
    print(f"  1. 最佳 R² 分数: {results[best_model_name]['r2']:.4f}")
    print("  2. 成功量化了预测不确定性")
    print("  3. 理解了特征对预测的影响")
    print("  4. 验证了相比传统方法的优势")


def generate_feature_names(scenario_name, n_features):
    """为不同场景生成有意义的特征名称"""
    
    feature_sets = {
        '销售预测': [
            '广告投入', '季节指数', '竞争对手价格', '客户满意度', '产品评分',
            '库存水平', '促销活动', '经济指数', '天气因素', '节假日影响',
            '社交媒体热度', '客户回购率', '新客户获取', '价格敏感度', '渠道效率'
        ],
        '能耗预测': [
            '建筑面积', '楼层数量', '建造年份', '窗户面积', '保温等级',
            '供暖类型', '通风系统', '照明类型', '设备数量', '使用时间',
            '外部温度', '湿度水平', '建筑朝向', '绿化程度', '维护水平'
        ],
        '股价预测': [
            'RSI指标', 'MACD', '成交量', '市盈率', '市净率',
            '资产负债率', '净利润增长', '收入增长', '行业指数', '市场情绪',
            '宏观经济', '政策影响', '国际因素', '技术创新', '竞争地位',
            '管理层变动', '分析师评级', '机构持仓', '散户情绪', '媒体关注度'
        ]
    }
    
    base_features = feature_sets.get(scenario_name, [])
    
    # 如果需要更多特征，添加通用特征
    while len(base_features) < n_features:
        base_features.append(f'特征_{len(base_features)+1}')
    
    return base_features[:n_features]


def analyze_feature_importance(model, X_sample, feature_names, scenario_name):
    """分析特征重要性（简化版本）"""
    
    print(f"\\n分析 {scenario_name} 中最重要的特征:")
    
    # 这里使用一个简化的特征重要性分析
    # 在实际的CausalEngine中，会有更sophisticated的方法
    
    if hasattr(model, '_get_feature_importance'):
        importance = model._get_feature_importance()
        if importance is not None:
            # 获取最重要的5个特征
            top_indices = np.argsort(importance)[-5:][::-1]
            
            print("\\n最重要的5个特征:")
            for i, idx in enumerate(top_indices):
                feature_name = feature_names[idx] if idx < len(feature_names) else f"特征_{idx}"
                importance_score = importance[idx]
                print(f"  {i+1}. {feature_name}: {importance_score:.4f}")
        else:
            print("  特征重要性信息暂不可用")
    else:
        print("  特征重要性分析功能正在开发中")
    
    # 展示几个样本的预测解释
    print("\\n样本预测解释:")
    if hasattr(model, 'predict_with_explanation'):
        try:
            explanations = model.predict_with_explanation(X_sample[:3], feature_names)
            
            for i, exp in enumerate(explanations):
                print(f"\\n  样本 {i+1}:")
                print(f"    预测值: {exp.get('prediction', 'N/A'):.3f}")
                print(f"    置信度: {exp.get('confidence', 0):.3f}")
                
                top_features = exp.get('top_features', [])
                if top_features:
                    print("    关键影响因素:")
                    for j, feature in enumerate(top_features):
                        print(f"      {j+1}. {feature['feature']}: {feature['value']:.3f}")
        except Exception as e:
            print(f"  预测解释功能遇到问题: {e}")
    else:
        print("  详细的预测解释功能正在开发中")


def plot_uncertainty_analysis(y_true, y_pred, y_std, scenario_name):
    """绘制不确定性分析图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 预测值 vs 真实值，带误差棒
    ax1.errorbar(range(len(y_true)), y_pred, yerr=y_std, fmt='o', capsize=5, alpha=0.7, label='预测±不确定性')
    ax1.scatter(range(len(y_true)), y_true, color='red', alpha=0.8, label='真实值')
    ax1.set_xlabel('样本索引')
    ax1.set_ylabel('值')
    ax1.set_title(f'{scenario_name} - 预测不确定性分析')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 不确定性分布
    ax2.hist(y_std, bins=10, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(y_std), color='red', linestyle='--', label=f'平均不确定性: {np.mean(y_std):.3f}')
    ax2.set_xlabel('不确定性')
    ax2.set_ylabel('频次')
    ax2.set_title('不确定性分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\n📊 不确定性统计:")
    print(f"  平均不确定性: {np.mean(y_std):.4f}")
    print(f"  不确定性范围: [{np.min(y_std):.4f}, {np.max(y_std):.4f}]")
    print(f"  不确定性标准差: {np.std(y_std):.4f}")


if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建结果目录
    os.makedirs("user_tutorials/results", exist_ok=True)
    
    # 运行主程序
    main()