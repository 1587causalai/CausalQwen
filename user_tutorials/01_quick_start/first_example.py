"""
CausalEngine 第一个完整示例
===========================

这是您使用 CausalEngine 的第一个示例！
只需要5分钟，就能看到因果推理的强大效果。

本示例将演示：
1. 如何生成或加载数据
2. 如何训练因果推理模型
3. 如何进行预测
4. 如何与传统方法对比
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加路径以便导入我们的工具
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.simple_models import SimpleCausalClassifier, SimpleCausalRegressor, compare_with_sklearn
from utils.data_helpers import (
    generate_classification_data, 
    generate_regression_data,
    explore_data,
    visualize_predictions
)

def main():
    """主函数：运行完整的演示"""
    
    print("🌟 欢迎使用 CausalEngine!")
    print("这是您的第一个因果推理示例")
    print("=" * 50)
    
    # 让用户选择任务类型
    print("\\n请选择您想要尝试的任务类型:")
    print("1. 分类任务 (例如：客户分类、疾病诊断)")
    print("2. 回归任务 (例如：价格预测、销量预测)")
    
    # 检查是否有命令行参数
    choice = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ['1', '2', 'classification', 'regression']:
            choice = '1' if arg in ['1', 'classification'] else '2'
            print(f"\\n自动选择: {choice}")
    
    if choice is None:
        while True:
            try:
                choice = input("\\n请输入您的选择 (1 或 2): ").strip()
                if choice in ['1', '2']:
                    break
                print("请输入有效的选择 (1 或 2)")
            except (EOFError, KeyboardInterrupt):
                print("\\n默认运行分类演示...")
                choice = '1'
                break
    
    if choice == '1':
        run_classification_demo()
    else:
        run_regression_demo()
    
    print("\\n🎉 恭喜！您已经成功运行了第一个 CausalEngine 示例！")
    print("\\n📖 接下来您可以：")
    print("  - 尝试 02_classification/ 目录下的更多分类示例")
    print("  - 尝试 03_regression/ 目录下的更多回归示例")
    print("  - 使用您自己的数据运行模型")
    print("  - 查看 04_real_world_examples/ 中的实际应用")


def run_classification_demo():
    """运行分类任务演示"""
    
    print("\\n🎯 分类任务演示")
    print("我们将创建一个客户分类的场景")
    print("-" * 30)
    
    # 1. 生成数据 (模拟客户数据)
    print("\\n📊 步骤 1: 生成客户数据")
    print("假设我们要根据客户的行为特征，将客户分为3类：")
    print("- 类别 0: 低价值客户")
    print("- 类别 1: 中价值客户") 
    print("- 类别 2: 高价值客户")
    
    X, y, info = generate_classification_data(
        n_samples=800,      # 800个客户
        n_features=12,      # 12个行为特征
        n_classes=3,        # 3个客户类别
        difficulty='medium', # 中等难度
        random_state=42
    )
    
    # 2. 数据探索
    print("\\n🔍 步骤 2: 探索数据")
    explore_data(X, y, info, show_plots=True)
    
    # 3. 训练 CausalQwen 模型
    print("\\n🚀 步骤 3: 训练 CausalEngine 分类器")
    print("正在训练中，请稍候...")
    
    model = SimpleCausalClassifier(random_state=42)
    model.fit(X, y, epochs=30, verbose=True)
    
    # 4. 进行预测
    print("\\n🔮 步骤 4: 进行预测")
    
    # 使用一部分数据进行预测演示
    test_indices = np.random.choice(len(X), 10, replace=False)
    X_demo = X[test_indices]
    y_demo = y[test_indices]
    
    # 不同温度的预测
    print("\\n不同推理模式的预测结果:")
    
    for temp, mode_name in [(0, "确定性因果推理"), (1.0, "标准推理"), (1.5, "探索性推理")]:
        pred_labels, pred_probs = model.predict(X_demo, return_probabilities=True, temperature=temp)
        
        print(f"\\n{mode_name} (温度={temp}):")
        for i in range(5):  # 只显示前5个样本
            true_label = y_demo[i]
            pred_label = pred_labels[i]
            confidence = np.max(pred_probs[i])
            
            status = "✅" if pred_label == true_label else "❌"
            print(f"  样本{i+1}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.3f} {status}")
    
    # 5. 获取预测解释
    print("\\n🔍 步骤 5: 预测解释")
    print("让我们看看模型是如何做出决策的...")
    
    explanations = model.predict_with_explanation(X_demo[:3])
    
    for i, exp in enumerate(explanations):
        print(f"\\n客户 {i+1}:")
        print(f"  预测类别: {exp['prediction']}")
        print(f"  预测置信度: {exp['confidence']:.3f}")
        print(f"  最重要的特征:")
        for j, feature in enumerate(exp['top_features']):
            print(f"    {j+1}. {feature['feature']}: {feature['value']:.3f} (重要性: {feature['importance']:.3f})")
    
    # 6. 与传统方法对比
    print("\\n⚖️ 步骤 6: 与传统机器学习对比")
    print("让我们看看 CausalQwen 相比传统方法的优势...")
    
    comparison_results = compare_with_sklearn(X, y, task_type='classification')
    
    # 7. 可视化结果
    print("\\n📊 步骤 7: 可视化结果")
    all_predictions = model.predict(X)
    visualize_predictions(y, all_predictions, 'classification', 'CausalQwen 分类结果')
    
    print("\\n✨ 分类演示完成！")
    print("\\n🎯 关键收获:")
    print("  1. CausalQwen 能够理解数据背后的因果关系")
    print("  2. 不同的推理温度提供不同的预测策略")
    print("  3. 模型提供清晰的预测解释")
    print("  4. 通常比传统方法有更好的泛化能力")


def run_regression_demo():
    """运行回归任务演示"""
    
    print("\\n📈 回归任务演示") 
    print("我们将创建一个房价预测的场景")
    print("-" * 30)
    
    # 1. 生成数据 (模拟房价数据)
    print("\\n📊 步骤 1: 生成房价数据")
    print("假设我们要根据房屋特征预测房价")
    print("特征包括：面积、位置、装修等级、建造年份等")
    
    X, y, info = generate_regression_data(
        n_samples=600,      # 600套房屋
        n_features=10,      # 10个房屋特征
        noise_level=0.15,   # 一些市场噪声
        difficulty='medium', # 中等难度
        random_state=42
    )
    
    # 2. 数据探索
    print("\\n🔍 步骤 2: 探索数据")
    explore_data(X, y, info, show_plots=True)
    
    # 3. 训练 CausalQwen 模型
    print("\\n🚀 步骤 3: 训练 CausalQwen 回归器")
    print("正在训练中，请稍候...")
    
    model = SimpleCausalRegressor(random_state=42)
    model.fit(X, y, epochs=30, verbose=True)
    
    # 4. 进行预测
    print("\\n🔮 步骤 4: 进行预测")
    
    # 使用一部分数据进行预测演示
    test_indices = np.random.choice(len(X), 8, replace=False)
    X_demo = X[test_indices]
    y_demo = y[test_indices]
    
    # 带不确定性的预测
    predictions, uncertainties = model.predict(X_demo, return_uncertainty=True)
    
    print("\\n房价预测结果:")
    print("  房屋  |  真实价格  |  预测价格  |  不确定性  |  误差")
    print("  -----|-----------|-----------|-----------|-------")
    
    for i in range(len(predictions)):
        true_price = y_demo[i]
        pred_price = predictions[i]
        uncertainty = uncertainties[i]
        error = abs(true_price - pred_price)
        
        print(f"   {i+1:2d}   | {true_price:8.2f}  | {pred_price:8.2f}  | ±{uncertainty:7.2f}  | {error:6.2f}")
    
    # 5. 不确定性分析
    print("\\n🔍 步骤 5: 不确定性分析")
    print("CausalQwen 的一个重要优势是能够量化预测的不确定性")
    
    avg_uncertainty = np.mean(uncertainties)
    print(f"平均不确定性: ±{avg_uncertainty:.2f}")
    print("\\n不确定性解释:")
    print("- 不确定性低: 模型对此房屋的价格很有把握")
    print("- 不确定性高: 该房屋可能有特殊情况，需要更多信息")
    
    # 6. 与传统方法对比
    print("\\n⚖️ 步骤 6: 与传统机器学习对比")
    print("让我们看看 CausalQwen 相比传统方法的优势...")
    
    comparison_results = compare_with_sklearn(X, y, task_type='regression')
    
    # 7. 可视化结果
    print("\\n📊 步骤 7: 可视化结果")
    all_predictions = model.predict(X)
    visualize_predictions(y, all_predictions, 'regression', 'CausalQwen 回归结果')
    
    print("\\n✨ 回归演示完成！")
    print("\\n🎯 关键收获:")
    print("  1. CausalQwen 能够准确预测连续数值")
    print("  2. 提供有意义的不确定性估计")
    print("  3. 在数据分布变化时更加稳定")
    print("  4. 适合各种回归任务")


if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    main()