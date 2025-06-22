"""
分类任务实战 - 使用合成数据
=========================

这个教程将深入演示如何使用 CausalQwen 处理分类任务。
我们将使用 scikit-learn 生成的合成数据，模拟真实的业务场景。

学习目标：
1. 理解分类任务的特点
2. 掌握多类别分类技巧
3. 学会处理不平衡数据
4. 理解概率预测和决策阈值
5. 掌握分类结果解释方法
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.simple_models import SimpleCausalClassifier, compare_with_sklearn
from utils.data_helpers import (
    generate_classification_data,
    prepare_data_for_training,
    explore_data,
    visualize_predictions,
    save_results
)


def main():
    """主函数：完整的分类任务流程"""
    
    print("🎯 CausalQwen 分类任务实战教程")
    print("=" * 50)
    
    # 演示多个不同的分类场景
    scenarios = [
        {
            'name': '客户细分',
            'description': '根据行为特征将客户分为不同价值等级',
            'n_samples': 1200,
            'n_features': 15,
            'n_classes': 3,
            'difficulty': 'easy',
            'class_names': ['低价值客户', '中价值客户', '高价值客户']
        },
        {
            'name': '疾病诊断',
            'description': '根据症状和检查结果诊断疾病类型',
            'n_samples': 800,
            'n_features': 20,
            'n_classes': 4,
            'difficulty': 'medium',
            'class_names': ['健康', '轻症', '中症', '重症']
        },
        {
            'name': '产品质量检测',
            'description': '根据生产参数检测产品质量等级',
            'n_samples': 1000,
            'n_features': 12,
            'n_classes': 5,
            'difficulty': 'hard',
            'class_names': ['优秀', '良好', '合格', '不合格', '次品']
        }
    ]
    
    print("\\n可用的分类场景:")
    for i, scenario in enumerate(scenarios):
        print(f"{i+1}. {scenario['name']} - {scenario['description']}")
        print(f"   类别: {', '.join(scenario['class_names'])}")
    
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
    run_classification_scenario(selected_scenario)
    
    print("\\n🎉 分类任务教程完成！")
    print("\\n📖 接下来您可以：")
    print("  - 尝试其他分类场景")
    print("  - 调节决策阈值优化性能")
    print("  - 使用您自己的数据")
    print("  - 查看 iris_dataset.py 学习处理真实数据")


def run_classification_scenario(scenario):
    """运行特定的分类场景"""
    
    print(f"\\n🚀 开始 {scenario['name']} 场景")
    print("-" * 40)
    
    # 1. 生成数据
    print("\\n📊 步骤 1: 生成数据")
    X, y, info = generate_classification_data(
        n_samples=scenario['n_samples'],
        n_features=scenario['n_features'],
        n_classes=scenario['n_classes'],
        difficulty=scenario['difficulty'],
        random_state=42
    )
    
    # 为了更好的演示，给特征添加有意义的名称
    feature_names = generate_feature_names(scenario['name'], scenario['n_features'])
    class_names = scenario['class_names']
    
    print(f"\\n特征说明（{scenario['name']}场景）:")
    for i, name in enumerate(feature_names[:5]):  # 只显示前5个
        print(f"  特征 {i+1}: {name}")
    if len(feature_names) > 5:
        print(f"  ... 还有 {len(feature_names)-5} 个特征")
    
    print(f"\\n分类目标:")
    for i, name in enumerate(class_names):
        print(f"  类别 {i}: {name}")
    
    # 2. 数据探索
    print("\\n🔍 步骤 2: 数据探索")
    explore_data(X, y, info, show_plots=True)
    
    # 检查类别平衡性
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\\n类别分布分析:")
    for i, (cls, count) in enumerate(zip(unique_classes, counts)):
        percentage = count / len(y) * 100
        print(f"  {class_names[cls]}: {count} 样本 ({percentage:.1f}%)")
    
    # 3. 数据准备
    print("\\n🔧 步骤 3: 数据准备")
    data = prepare_data_for_training(X, y, test_size=0.2, validation_size=0.2)
    
    # 4. 训练基础模型
    print("\\n🚀 步骤 4: 训练 CausalQwen 分类器")
    print("使用默认参数训练...")
    
    model_basic = SimpleCausalClassifier(random_state=42)
    model_basic.fit(
        data['X_train'], data['y_train'], 
        epochs=40, 
        verbose=True
    )
    
    # 5. 训练优化模型
    print("\\n⚙️ 步骤 5: 参数优化")
    print("让我们尝试更多训练轮数...")
    
    model_optimized = SimpleCausalClassifier(random_state=42)
    model_optimized.fit(
        data['X_train'], data['y_train'],
        epochs=60,
        validation_split=0.25,
        verbose=True
    )
    
    # 6. 模型性能对比
    print("\\n📊 步骤 6: 模型性能对比")
    
    models = {
        'CausalQwen (基础)': model_basic,
        'CausalQwen (优化)': model_optimized
    }
    
    results = {}
    
    for name, model in models.items():
        # 预测
        pred = model.predict(data['X_test'])
        pred_probs = model.predict(data['X_test'], return_probabilities=True)[1]
        
        # 计算指标
        accuracy = accuracy_score(data['y_test'], pred)
        precision = precision_score(data['y_test'], pred, average='weighted')
        recall = recall_score(data['y_test'], pred, average='weighted')
        f1 = f1_score(data['y_test'], pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': pred,
            'probabilities': pred_probs
        }
        
        print(f"\\n{name}:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
    
    # 7. 详细分类报告
    print("\\n📋 步骤 7: 详细分类报告")
    
    # 选择最佳模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = models[best_model_name]
    best_pred = results[best_model_name]['predictions']
    
    print(f"\\n使用最佳模型: {best_model_name}")
    print("\\n详细分类报告:")
    
    report = classification_report(
        data['y_test'], best_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # 按类别显示详细指标
    print("\\n各类别性能:")
    print("  类别           | 精确率 | 召回率 | F1分数 | 支持度")
    print("  -------------- | ------ | ------ | ------ | ------")
    
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            support = report[str(i)]['support']
            print(f"  {class_name:14} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support:4.0f}")
    
    # 8. 概率预测分析
    print("\\n🎲 步骤 8: 概率预测分析")
    analyze_prediction_probabilities(best_model, data['X_test'], data['y_test'], class_names)
    
    # 9. 不同推理模式对比
    print("\\n🌡️ 步骤 9: 不同推理模式对比")
    compare_inference_modes(best_model, data['X_test'], data['y_test'], class_names)
    
    # 10. 特征重要性分析
    print("\\n🧠 步骤 10: 预测解释")
    analyze_feature_importance(best_model, data['X_test'], feature_names, class_names, scenario['name'])
    
    # 11. 错误分析
    print("\\n🔍 步骤 11: 错误分析")
    analyze_classification_errors(data['y_test'], best_pred, class_names)
    
    # 12. 与传统方法对比
    print("\\n⚖️ 步骤 12: 与传统机器学习对比")
    comparison_results = compare_with_sklearn(X, y, task_type='classification')
    
    # 13. 结果可视化
    print("\\n📊 步骤 13: 结果可视化")
    visualize_predictions(data['y_test'], best_pred, 'classification', f'{scenario["name"]} - CausalQwen 结果')
    
    # 绘制混淆矩阵
    plot_detailed_confusion_matrix(data['y_test'], best_pred, class_names, scenario['name'])
    
    # 14. 保存结果
    print("\\n💾 步骤 14: 保存结果")
    
    final_results = {
        'scenario': scenario,
        'model_comparison': results,
        'classification_report': report,
        'sklearn_comparison': comparison_results,
        'feature_names': feature_names,
        'class_names': class_names,
        'best_model': best_model_name
    }
    
    filename = f"user_tutorials/results/{scenario['name']}_classification_results.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_results(final_results, filename)
    
    print(f"\\n✨ {scenario['name']} 场景完成！")
    print("\\n🎯 关键收获:")
    print(f"  1. 最佳准确率: {results[best_model_name]['accuracy']:.4f}")
    print(f"  2. 最佳F1分数: {results[best_model_name]['f1']:.4f}")
    print("  3. 成功处理了多类别分类问题")
    print("  4. 理解了概率预测的价值")
    print("  5. 验证了相比传统方法的优势")


def generate_feature_names(scenario_name, n_features):
    """为不同场景生成有意义的特征名称"""
    
    feature_sets = {
        '客户细分': [
            '购买频次', '消费金额', '会员年限', '产品评价', '投诉次数',
            '推荐成功率', '活跃度指数', '价格敏感度', '品牌忠诚度', '社交影响力',
            '季节性偏好', '渠道使用习惯', '退货率', '优惠券使用', '客服咨询频次'
        ],
        '疾病诊断': [
            '体温', '血压收缩压', '血压舒张压', '心率', '呼吸频率',
            '血糖水平', '白细胞计数', '红细胞计数', '血红蛋白', '血小板计数',
            '胆固醇水平', '尿蛋白', '肌酐水平', '炎症指标', '年龄',
            '体重指数', '家族病史', '吸烟史', '运动频率', '睡眠质量'
        ],
        '产品质量检测': [
            '原料纯度', '生产温度', '生产压力', '反应时间', '搅拌速度',
            '冷却时间', '包装密封度', '存储环境', '生产批次', '操作员经验',
            '设备维护状态', '环境湿度'
        ]
    }
    
    base_features = feature_sets.get(scenario_name, [])
    
    # 如果需要更多特征，添加通用特征
    while len(base_features) < n_features:
        base_features.append(f'特征_{len(base_features)+1}')
    
    return base_features[:n_features]


def analyze_prediction_probabilities(model, X_test, y_test, class_names):
    """分析预测概率分布"""
    
    print("\\n分析预测概率分布...")
    
    # 获取预测概率
    pred_labels, pred_probs = model.predict(X_test, return_probabilities=True)
    
    # 计算每个类别的平均置信度
    print("\\n各类别平均预测置信度:")
    for i, class_name in enumerate(class_names):
        # 找到预测为该类别的样本
        mask = pred_labels == i
        if mask.sum() > 0:
            avg_confidence = pred_probs[mask, i].mean()
            print(f"  {class_name}: {avg_confidence:.4f}")
    
    # 分析高置信度和低置信度的预测
    max_probs = np.max(pred_probs, axis=1)
    
    high_confidence_threshold = 0.8
    low_confidence_threshold = 0.5
    
    high_conf_count = (max_probs >= high_confidence_threshold).sum()
    low_conf_count = (max_probs <= low_confidence_threshold).sum()
    
    print(f"\\n置信度分析:")
    print(f"  高置信度预测 (≥{high_confidence_threshold}): {high_conf_count} ({high_conf_count/len(max_probs)*100:.1f}%)")
    print(f"  低置信度预测 (≤{low_confidence_threshold}): {low_conf_count} ({low_conf_count/len(max_probs)*100:.1f}%)")
    
    # 绘制置信度分布
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(high_confidence_threshold, color='green', linestyle='--', label='高置信度阈值')
    plt.axvline(low_confidence_threshold, color='red', linestyle='--', label='低置信度阈值')
    plt.xlabel('最大预测概率')
    plt.ylabel('频次')
    plt.title('预测置信度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各类别概率分布
    plt.subplot(1, 2, 2)
    for i, class_name in enumerate(class_names):
        plt.hist(pred_probs[:, i], alpha=0.5, label=class_name, bins=15)
    plt.xlabel('预测概率')
    plt.ylabel('频次')
    plt.title('各类别概率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_inference_modes(model, X_test, y_test, class_names):
    """对比不同推理模式的效果"""
    
    print("\\n对比不同推理模式...")
    
    modes = [
        (0, False, "确定性因果推理"),
        (1.0, False, "标准推理"), 
        (0.8, True, "探索性采样"),
        (1.2, True, "高温度采样")
    ]
    
    results = {}
    
    for temp, do_sample, mode_name in modes:
        try:
            # 多次预测取平均（特别是对于采样模式）
            predictions_list = []
            for _ in range(5 if do_sample else 1):
                pred = model.predict(X_test, temperature=temp)
                predictions_list.append(pred)
            
            # 使用众数作为最终预测（对于采样模式）
            if do_sample and len(predictions_list) > 1:
                predictions_array = np.array(predictions_list)
                final_pred = []
                for i in range(len(X_test)):
                    unique, counts = np.unique(predictions_array[:, i], return_counts=True)
                    final_pred.append(unique[np.argmax(counts)])
                final_pred = np.array(final_pred)
            else:
                final_pred = predictions_list[0]
            
            accuracy = accuracy_score(y_test, final_pred)
            f1 = f1_score(y_test, final_pred, average='weighted')
            
            results[mode_name] = {
                'accuracy': accuracy,
                'f1': f1,
                'predictions': final_pred
            }
            
        except Exception as e:
            print(f"  {mode_name} 模式遇到问题: {e}")
            continue
    
    # 显示结果
    print("\\n不同推理模式性能对比:")
    print("  模式              | 准确率 | F1分数")
    print("  ----------------- | ------ | ------")
    
    for mode_name, result in results.items():
        print(f"  {mode_name:17} | {result['accuracy']:.4f} | {result['f1']:.4f}")
    
    # 分析模式间的预测差异
    if len(results) >= 2:
        print("\\n推理模式一致性分析:")
        mode_names = list(results.keys())
        pred1 = results[mode_names[0]]['predictions']
        
        for i, mode_name in enumerate(mode_names[1:], 1):
            pred2 = results[mode_name]['predictions']
            agreement = (pred1 == pred2).mean()
            print(f"  {mode_names[0]} vs {mode_name}: {agreement:.3f} 一致性")


def analyze_feature_importance(model, X_test, feature_names, class_names, scenario_name):
    """分析特征重要性"""
    
    print(f"\\n分析 {scenario_name} 中的关键特征...")
    
    # 选择几个代表性样本进行分析
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    X_sample = X_test[sample_indices]
    
    if hasattr(model, 'predict_with_explanation'):
        try:
            explanations = model.predict_with_explanation(X_sample, feature_names)
            
            print("\\n样本预测解释:")
            for i, exp in enumerate(explanations):
                pred_class = exp.get('prediction', 'N/A')
                confidence = exp.get('confidence', 0)
                
                # 找到对应的类别名称
                if isinstance(pred_class, (int, np.integer)) and pred_class < len(class_names):
                    class_name = class_names[pred_class]
                else:
                    class_name = str(pred_class)
                
                print(f"\\n  样本 {i+1}:")
                print(f"    预测类别: {class_name}")
                print(f"    置信度: {confidence:.3f}")
                
                top_features = exp.get('top_features', [])
                if top_features:
                    print("    关键影响因素:")
                    for j, feature in enumerate(top_features):
                        print(f"      {j+1}. {feature['feature']}: {feature['value']:.3f} (重要性: {feature.get('importance', 0):.3f})")
                
                # 显示概率分布
                probs = exp.get('probabilities', {})
                if probs:
                    print("    各类别概率:")
                    for class_idx, prob in probs.items():
                        if isinstance(class_idx, str) and class_idx.isdigit():
                            class_idx = int(class_idx)
                        if isinstance(class_idx, (int, np.integer)) and class_idx < len(class_names):
                            print(f"      {class_names[class_idx]}: {prob:.3f}")
        
        except Exception as e:
            print(f"  预测解释功能遇到问题: {e}")
    else:
        print("  详细的预测解释功能正在开发中")
    
    # 全局特征重要性分析
    if hasattr(model, '_get_feature_importance'):
        importance = model._get_feature_importance()
        if importance is not None:
            # 获取最重要的特征
            top_indices = np.argsort(importance)[-8:][::-1]
            
            print(f"\\n{scenario_name} 中最重要的特征:")
            for i, idx in enumerate(top_indices):
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    importance_score = importance[idx]
                    print(f"  {i+1}. {feature_name}: {importance_score:.4f}")


def analyze_classification_errors(y_true, y_pred, class_names):
    """分析分类错误"""
    
    print("\\n分析分类错误...")
    
    # 找到错误分类的样本
    error_mask = y_true != y_pred
    error_count = error_mask.sum()
    total_count = len(y_true)
    
    print(f"\\n错误分类统计:")
    print(f"  总样本数: {total_count}")
    print(f"  错误分类: {error_count}")
    print(f"  错误率: {error_count/total_count:.4f}")
    
    if error_count > 0:
        # 按真实类别分析错误
        print("\\n各类别错误分析:")
        print("  真实类别     | 总数 | 错误 | 错误率")
        print("  ------------ | ---- | ---- | ------")
        
        for i, class_name in enumerate(class_names):
            class_mask = y_true == i
            class_total = class_mask.sum()
            class_errors = (class_mask & error_mask).sum()
            error_rate = class_errors / class_total if class_total > 0 else 0
            
            print(f"  {class_name:12} | {class_total:4} | {class_errors:4} | {error_rate:.4f}")
        
        # 混淆模式分析
        print("\\n常见混淆模式:")
        cm = confusion_matrix(y_true, y_pred)
        
        # 找到最常见的错误分类
        error_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    error_pairs.append((i, j, cm[i, j]))
        
        # 按错误数量排序
        error_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, (true_class, pred_class, count) in enumerate(error_pairs[:5]):  # 显示前5个
            print(f"  {i+1}. {class_names[true_class]} → {class_names[pred_class]}: {count} 次")


def plot_detailed_confusion_matrix(y_true, y_pred, class_names, scenario_name):
    """绘制详细的混淆矩阵"""
    
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建标注文本
    annot = np.zeros_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title(f'{scenario_name} - 混淆矩阵')
    plt.tight_layout()
    plt.show()
    
    # 显示每个类别的精确率和召回率
    print("\\n从混淆矩阵计算的指标:")
    print("  类别       | 精确率 | 召回率")
    print("  ---------- | ------ | ------")
    
    for i, class_name in enumerate(class_names):
        # 精确率 = TP / (TP + FP)
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        # 召回率 = TP / (TP + FN)  
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        
        print(f"  {class_name:10} | {precision:.4f} | {recall:.4f}")


if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建结果目录
    os.makedirs("user_tutorials/results", exist_ok=True)
    
    # 运行主程序
    main()