"""
真实数据集示例 - 鸢尾花分类
===========================

这个教程使用经典的鸢尾花数据集，演示如何用 CausalQwen 处理真实数据。
鸢尾花数据集是机器学习领域最著名的数据集之一，包含3种鸢尾花的测量数据。

学习目标：
1. 学会处理真实数据集
2. 理解数据预处理的重要性
3. 掌握小数据集的处理技巧
4. 学会解释现实世界的预测结果
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.simple_models import SimpleCausalClassifier, compare_with_sklearn
from utils.data_helpers import visualize_predictions


def main():
    """主函数：鸢尾花分类完整流程"""
    
    print("🌸 CausalQwen 真实数据示例 - 鸢尾花分类")
    print("=" * 50)
    
    print("\\n📚 关于鸢尾花数据集:")
    print("这是1936年由生物学家Edgar Anderson收集的数据，")
    print("包含了3种鸢尾花（山鸢尾、变色鸢尾、维吉尼亚鸢尾）的形态测量数据。")
    print("每种花有50个样本，共150个样本，4个特征。")
    
    # 1. 加载数据
    print("\\n📊 步骤 1: 加载鸢尾花数据集")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    print(f"✅ 数据加载成功:")
    print(f"   样本数量: {X.shape[0]}")
    print(f"   特征数量: {X.shape[1]}")
    print(f"   类别数量: {len(class_names)}")
    
    print(f"\\n🏷️ 特征说明:")
    for i, name in enumerate(feature_names):
        print(f"   {i+1}. {name}")
    
    print(f"\\n🌸 鸢尾花种类:")
    for i, name in enumerate(class_names):
        print(f"   {i}. {name} ({name})")
    
    # 2. 数据探索
    print("\\n🔍 步骤 2: 数据探索")
    explore_iris_data(X, y, feature_names, class_names)
    
    # 3. 数据预处理
    print("\\n🔧 步骤 3: 数据预处理")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    print(f"   特征已标准化")
    
    # 4. 训练 CausalQwen 模型
    print("\\n🚀 步骤 4: 训练 CausalQwen 分类器")
    
    model = SimpleCausalClassifier(random_state=42)
    model.fit(X_train_scaled, y_train, epochs=100, verbose=True)
    
    # 5. 模型评估
    print("\\n📊 步骤 5: 模型评估")
    
    # 预测
    predictions = model.predict(X_test_scaled)
    pred_probs = model.predict(X_test_scaled, return_probabilities=True)[1]
    
    # 计算指标
    accuracy = accuracy_score(y_test, predictions)
    print(f"\\n测试集准确率: {accuracy:.4f}")
    
    # 详细分类报告
    print("\\n详细分类报告:")
    report = classification_report(y_test, predictions, target_names=class_names)
    print(report)
    
    # 6. 不同推理模式对比
    print("\\n🌡️ 步骤 6: 不同推理模式对比")
    compare_inference_modes(model, X_test_scaled, y_test, class_names)
    
    # 7. 预测解释
    print("\\n🧠 步骤 7: 预测解释分析")
    analyze_predictions(model, X_test_scaled, y_test, feature_names, class_names)
    
    # 8. 特征重要性
    print("\\n📈 步骤 8: 特征重要性分析")
    analyze_feature_importance(model, X_test_scaled, feature_names, class_names)
    
    # 9. 错误案例分析
    print("\\n🔍 步骤 9: 错误案例分析")
    analyze_errors(X_test_scaled, y_test, predictions, feature_names, class_names, scaler)
    
    # 10. 与传统方法对比
    print("\\n⚖️ 步骤 10: 与传统机器学习对比")
    comparison_results = compare_with_sklearn(X, y, task_type='classification')
    
    # 11. 实际应用演示
    print("\\n🌸 步骤 11: 实际应用演示")
    demo_real_world_prediction(model, scaler, feature_names, class_names)
    
    # 12. 结果可视化
    print("\\n📊 步骤 12: 结果可视化")
    visualize_predictions(y_test, predictions, 'classification', '鸢尾花分类 - CausalQwen 结果')
    
    # 绘制特征分布图
    plot_feature_distributions(X, y, feature_names, class_names)
    
    print("\\n🎉 鸢尾花分类教程完成！")
    print("\\n🎯 关键收获:")
    print(f"   1. 在经典数据集上达到了 {accuracy:.4f} 的准确率")
    print("   2. 学会了处理真实世界的小数据集")
    print("   3. 理解了特征的实际含义和重要性")
    print("   4. 掌握了模型解释和错误分析方法")


def explore_iris_data(X, y, feature_names, class_names):
    """探索鸢尾花数据"""
    
    # 创建DataFrame便于分析
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [class_names[i] for i in y]
    
    print("\\n基本统计信息:")
    print(df.describe())
    
    print("\\n各类别样本数:")
    print(df['species'].value_counts())
    
    # 计算各类别各特征的均值
    print("\\n各类别特征均值:")
    class_means = df.groupby('species')[feature_names].mean()
    print(class_means)
    
    # 寻找最具区分性的特征
    print("\\n特征区分度分析:")
    for feature in feature_names:
        feature_values = df.groupby('species')[feature].mean()
        max_diff = feature_values.max() - feature_values.min()
        print(f"   {feature}: 最大差异 = {max_diff:.3f}")


def compare_inference_modes(model, X_test, y_test, class_names):
    """对比不同推理模式"""
    
    modes = [
        (0, False, "确定性因果推理"),
        (1.0, False, "标准推理"),
        (0.8, True, "探索性推理")
    ]
    
    results = {}
    
    for temp, do_sample, mode_name in modes:
        try:
            if do_sample:
                # 采样模式多次预测取众数
                predictions_list = []
                for _ in range(5):
                    pred = model.predict(X_test, temperature=temp)
                    predictions_list.append(pred)
                
                # 计算众数
                predictions_array = np.array(predictions_list)
                final_pred = []
                for i in range(len(X_test)):
                    unique, counts = np.unique(predictions_array[:, i], return_counts=True)
                    final_pred.append(unique[np.argmax(counts)])
                final_pred = np.array(final_pred)
            else:
                final_pred = model.predict(X_test, temperature=temp)
            
            accuracy = accuracy_score(y_test, final_pred)
            results[mode_name] = accuracy
            
        except Exception as e:
            print(f"   {mode_name} 遇到问题: {e}")
            continue
    
    print("\\n不同推理模式准确率对比:")
    for mode_name, accuracy in results.items():
        print(f"   {mode_name}: {accuracy:.4f}")


def analyze_predictions(model, X_test, y_test, feature_names, class_names):
    """分析预测结果"""
    
    # 获取带概率的预测
    pred_labels, pred_probs = model.predict(X_test, return_probabilities=True)
    
    print("\\n预测置信度分析:")
    max_probs = np.max(pred_probs, axis=1)
    print(f"   平均置信度: {max_probs.mean():.4f}")
    print(f"   最高置信度: {max_probs.max():.4f}")
    print(f"   最低置信度: {max_probs.min():.4f}")
    
    # 找到几个有趣的预测案例
    print("\\n代表性预测案例:")
    
    # 高置信度正确预测
    correct_mask = pred_labels == y_test
    high_conf_correct = np.where(correct_mask & (max_probs > 0.9))[0]
    if len(high_conf_correct) > 0:
        idx = high_conf_correct[0]
        print(f"\\n  🎯 高置信度正确预测 (样本 {idx}):")
        print(f"     真实类别: {class_names[y_test[idx]]}")
        print(f"     预测类别: {class_names[pred_labels[idx]]}")
        print(f"     置信度: {max_probs[idx]:.4f}")
        show_sample_features(X_test[idx], feature_names)
    
    # 低置信度预测
    low_conf_indices = np.where(max_probs < 0.7)[0]
    if len(low_conf_indices) > 0:
        idx = low_conf_indices[0]
        print(f"\\n  🤔 低置信度预测 (样本 {idx}):")
        print(f"     真实类别: {class_names[y_test[idx]]}")
        print(f"     预测类别: {class_names[pred_labels[idx]]}")
        print(f"     置信度: {max_probs[idx]:.4f}")
        print(f"     各类别概率: {[f'{p:.3f}' for p in pred_probs[idx]]}")
        show_sample_features(X_test[idx], feature_names)
    
    # 错误预测
    error_indices = np.where(~correct_mask)[0]
    if len(error_indices) > 0:
        idx = error_indices[0]
        print(f"\\n  ❌ 错误预测案例 (样本 {idx}):")
        print(f"     真实类别: {class_names[y_test[idx]]}")
        print(f"     预测类别: {class_names[pred_labels[idx]]}")
        print(f"     置信度: {max_probs[idx]:.4f}")
        show_sample_features(X_test[idx], feature_names)


def show_sample_features(sample, feature_names):
    """显示样本的特征值"""
    print("     特征值:")
    for i, (name, value) in enumerate(zip(feature_names, sample)):
        print(f"       {name}: {value:.3f}")


def analyze_feature_importance(model, X_test, feature_names, class_names):
    """分析特征重要性"""
    
    if hasattr(model, '_get_feature_importance'):
        importance = model._get_feature_importance()
        if importance is not None:
            print("\\n特征重要性排序:")
            
            # 按重要性排序
            sorted_indices = np.argsort(importance)[::-1]
            
            for i, idx in enumerate(sorted_indices):
                print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
            
            # 解释最重要的特征
            most_important = feature_names[sorted_indices[0]]
            print(f"\\n💡 最重要的特征是 '{most_important}'")
            print("   这个特征对区分不同鸢尾花种类最有帮助。")
        else:
            print("\\n特征重要性信息暂不可用")
    
    # 基于域知识的特征解释
    print("\\n🌸 鸢尾花特征的生物学意义:")
    feature_meanings = {
        'sepal length (cm)': '花萼长度 - 保护花朵的外层结构',
        'sepal width (cm)': '花萼宽度 - 影响花朵的整体形状',
        'petal length (cm)': '花瓣长度 - 最显著的区分特征',
        'petal width (cm)': '花瓣宽度 - 花朵的视觉特征'
    }
    
    for feature, meaning in feature_meanings.items():
        print(f"   • {feature}: {meaning}")


def analyze_errors(X_test, y_test, predictions, feature_names, class_names, scaler):
    """分析错误预测"""
    
    error_mask = y_test != predictions
    error_count = error_mask.sum()
    
    if error_count == 0:
        print("\\n🎉 完美预测！没有错误案例。")
        return
    
    print(f"\\n错误预测分析 (共 {error_count} 个错误):")
    
    error_indices = np.where(error_mask)[0]
    
    for i, idx in enumerate(error_indices):
        if i >= 3:  # 只显示前3个错误案例
            print(f"   ... 还有 {len(error_indices) - 3} 个错误案例")
            break
        
        true_class = class_names[y_test[idx]]
        pred_class = class_names[predictions[idx]]
        
        print(f"\\n   错误案例 {i+1}:")
        print(f"     真实类别: {true_class}")
        print(f"     预测类别: {pred_class}")
        
        # 反标准化特征值以便理解
        original_features = scaler.inverse_transform([X_test[idx]])[0]
        print(f"     特征值:")
        for name, value in zip(feature_names, original_features):
            print(f"       {name}: {value:.2f} cm")
        
        # 分析为什么会误分类
        print(f"     可能原因: {true_class} 和 {pred_class} 在某些特征上相似")


def demo_real_world_prediction(model, scaler, feature_names, class_names):
    """演示真实世界预测"""
    
    print("\\n假设您在野外发现了一朵鸢尾花，测量了以下数据:")
    
    # 创建一个假设的新样本
    new_flower = np.array([[5.8, 3.2, 4.5, 1.4]])  # 一个中等大小的鸢尾花
    
    print("\\n🌸 新发现的鸢尾花特征:")
    for name, value in zip(feature_names, new_flower[0]):
        print(f"   {name}: {value:.1f} cm")
    
    # 标准化
    new_flower_scaled = scaler.transform(new_flower)
    
    # 预测
    pred_label, pred_probs = model.predict(new_flower_scaled, return_probabilities=True)
    pred_label = pred_label[0]
    pred_probs = pred_probs[0]
    
    print(f"\\n🔮 CausalQwen 的预测结果:")
    print(f"   最可能的种类: {class_names[pred_label]}")
    print(f"   置信度: {pred_probs[pred_label]:.4f}")
    
    print(f"\\n📊 各种类的可能性:")
    for i, (class_name, prob) in enumerate(zip(class_names, pred_probs)):
        bar = "█" * int(prob * 20)  # 简单的条形图
        print(f"   {class_name:15}: {prob:.4f} {bar}")
    
    # 给出建议
    confidence = pred_probs[pred_label]
    if confidence > 0.8:
        print(f"\\n💡 建议: 这很可能是 {class_names[pred_label]}，置信度很高。")
    elif confidence > 0.6:
        print(f"\\n💡 建议: 可能是 {class_names[pred_label]}，但建议再测量几个样本确认。")
    else:
        print(f"\\n💡 建议: 分类不确定，建议寻求专家帮助或使用其他特征。")


def plot_feature_distributions(X, y, feature_names, class_names):
    """绘制特征分布图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    colors = ['red', 'green', 'blue']
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        for j, class_name in enumerate(class_names):
            mask = y == j
            ax.hist(X[mask, i], alpha=0.6, label=class_name, color=colors[j], bins=15)
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel('频次')
        ax.set_title(f'{feature_name} 的分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\\n📊 特征分布图说明:")
    print("   - 不同颜色代表不同的鸢尾花种类")
    print("   - 重叠少的特征区分性更好")
    print("   - 花瓣长度和宽度通常最具区分性")


if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    main()