"""
数据处理辅助函数
================

这个文件包含了常用的数据处理和可视化函数，
让用户能够轻松地准备数据和查看结果。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    difficulty: str = 'medium',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    生成分类数据集
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        difficulty: 数据难度 ('easy', 'medium', 'hard')
        random_state: 随机种子
    
    Returns:
        X, y, info: 特征矩阵，标签向量，数据集信息
    """
    print(f"📊 生成分类数据集...")
    
    # 根据难度设置参数
    if difficulty == 'easy':
        n_informative = int(n_features * 0.9)
        n_redundant = int(n_features * 0.1)
        n_clusters_per_class = 1
        class_sep = 1.5
    elif difficulty == 'medium':
        n_informative = int(n_features * 0.7)
        n_redundant = int(n_features * 0.2)
        n_clusters_per_class = 2
        class_sep = 1.0
    else:  # hard
        n_informative = int(n_features * 0.5)
        n_redundant = int(n_features * 0.3)
        n_clusters_per_class = 3
        class_sep = 0.7
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=random_state
    )
    
    # 数据集信息
    info = {
        'task_type': 'classification',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'difficulty': difficulty,
        'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
    }
    
    print(f"✅ 生成完成: {n_samples} 样本, {n_features} 特征, {n_classes} 类别")
    print(f"   难度: {difficulty}")
    print(f"   类别分布: {info['class_distribution']}")
    
    return X, y, info


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 15,
    noise_level: float = 0.1,
    difficulty: str = 'medium',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    生成回归数据集
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        noise_level: 噪声水平 (0-1)
        difficulty: 数据难度 ('easy', 'medium', 'hard')
        random_state: 随机种子
    
    Returns:
        X, y, info: 特征矩阵，目标向量，数据集信息
    """
    print(f"📊 生成回归数据集...")
    
    # 根据难度设置参数
    if difficulty == 'easy':
        n_informative = int(n_features * 0.9)
        noise = noise_level * 0.5
    elif difficulty == 'medium':
        n_informative = int(n_features * 0.7)
        noise = noise_level
    else:  # hard
        n_informative = int(n_features * 0.5)
        noise = noise_level * 2
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    
    # 数据集信息
    info = {
        'task_type': 'regression',
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_level': noise,
        'difficulty': difficulty,
        'target_range': (y.min(), y.max()),
        'target_std': y.std()
    }
    
    print(f"✅ 生成完成: {n_samples} 样本, {n_features} 特征")
    print(f"   难度: {difficulty}, 噪声水平: {noise:.3f}")
    print(f"   目标范围: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, info


def load_sample_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    加载示例数据集
    
    Args:
        dataset_name: 数据集名称 ('iris', 'wine', 'boston', 'diabetes')
    
    Returns:
        X, y, info: 特征矩阵，标签/目标向量，数据集信息
    """
    print(f"📂 加载数据集: {dataset_name}")
    
    if dataset_name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        info = {
            'task_type': 'classification',
            'description': '鸢尾花分类：根据花瓣和花萼尺寸预测鸢尾花品种',
            'features': list(data.feature_names),
            'classes': list(data.target_names)
        }
        
    elif dataset_name == 'wine':
        from sklearn.datasets import load_wine
        data = load_wine()
        info = {
            'task_type': 'classification',
            'description': '葡萄酒分类：根据化学成分预测葡萄酒类型',
            'features': list(data.feature_names),
            'classes': list(data.target_names)
        }
        
    elif dataset_name == 'boston':
        # 使用加利福尼亚房价数据替代波士顿房价
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        info = {
            'task_type': 'regression',
            'description': '加利福尼亚房价预测：根据地理和房屋信息预测房价',
            'features': list(data.feature_names),
            'target_name': 'house_value'
        }
        
    elif dataset_name == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        info = {
            'task_type': 'regression',
            'description': '糖尿病进展预测：根据基础指标预测疾病进展',
            'features': list(data.feature_names),
            'target_name': 'disease_progression'
        }
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    print(f"✅ 加载完成: {data.data.shape[0]} 样本, {data.data.shape[1]} 特征")
    print(f"   描述: {info['description']}")
    
    return data.data, data.target, info


def explore_data(X, y, info: Dict[str, Any], show_plots: bool = True):
    """
    数据探索和可视化
    
    Args:
        X: 特征矩阵
        y: 标签/目标向量
        info: 数据集信息
        show_plots: 是否显示图表
    """
    print("\n🔍 数据探索分析")
    print("=" * 40)
    
    # 基本统计信息
    print(f"📊 基本信息:")
    print(f"   数据形状: {X.shape}")
    print(f"   任务类型: {info['task_type']}")
    print(f"   缺失值: {np.isnan(X).sum()}")
    
    if info['task_type'] == 'classification':
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"   类别数量: {len(unique_classes)}")
        print(f"   类别分布: {dict(zip(unique_classes, counts))}")
        
        if show_plots:
            # 类别分布图
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.bar(range(len(unique_classes)), counts)
            plt.xlabel('类别')
            plt.ylabel('样本数量')
            plt.title('类别分布')
            plt.xticks(range(len(unique_classes)), unique_classes)
            
            # 特征分布（选择前几个特征）
            plt.subplot(1, 2, 2)
            n_features_to_show = min(5, X.shape[1])
            for i in range(n_features_to_show):
                plt.hist(X[:, i], alpha=0.5, label=f'特征{i+1}')
            plt.xlabel('特征值')
            plt.ylabel('频次')
            plt.title('特征分布')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    else:  # regression
        print(f"   目标范围: [{y.min():.3f}, {y.max():.3f}]")
        print(f"   目标均值: {y.mean():.3f}")
        print(f"   目标标准差: {y.std():.3f}")
        
        if show_plots:
            plt.figure(figsize=(12, 4))
            
            # 目标分布
            plt.subplot(1, 3, 1)
            plt.hist(y, bins=30, alpha=0.7)
            plt.xlabel('目标值')
            plt.ylabel('频次')
            plt.title('目标值分布')
            
            # 特征分布
            plt.subplot(1, 3, 2)
            n_features_to_show = min(5, X.shape[1])
            for i in range(n_features_to_show):
                plt.hist(X[:, i], alpha=0.5, label=f'特征{i+1}')
            plt.xlabel('特征值')
            plt.ylabel('频次')
            plt.title('特征分布')
            plt.legend()
            
            # 特征与目标的相关性（选择第一个特征）
            plt.subplot(1, 3, 3)
            plt.scatter(X[:, 0], y, alpha=0.5)
            plt.xlabel('特征1')
            plt.ylabel('目标值')
            plt.title('特征1 vs 目标值')
            
            plt.tight_layout()
            plt.show()


def prepare_data_for_training(
    X, y, 
    test_size: float = 0.2,
    validation_size: float = 0.2,
    scale_features: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    为训练准备数据
    
    Args:
        X: 特征矩阵
        y: 标签/目标向量
        test_size: 测试集比例
        validation_size: 验证集比例（从训练集中划分）
        scale_features: 是否标准化特征
        random_state: 随机种子
    
    Returns:
        准备好的数据字典
    """
    print("🔧 准备训练数据...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(np.unique(y)) < 20 else None  # 分类任务时使用分层采样
    )
    
    # 从训练集中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=random_state,
        stratify=y_train if len(np.unique(y_train)) < 20 else None
    )
    
    # 特征标准化
    scalers = {}
    if scale_features:
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        scalers['X'] = scaler_X
        
        # 如果是回归任务，也标准化目标变量
        if len(np.unique(y)) > 20:  # 假设超过20个不同值就是回归
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            scalers['y'] = scaler_y
    
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scalers': scalers
    }
    
    print(f"✅ 数据准备完成:")
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   验证集: {X_val.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    print(f"   特征标准化: {'是' if scale_features else '否'}")
    
    return data


def visualize_predictions(y_true, y_pred, task_type: str, title: str = "预测结果"):
    """
    可视化预测结果
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        task_type: 任务类型 ('classification' 或 'regression')
        title: 图表标题
    """
    plt.figure(figsize=(10, 4))
    
    if task_type == 'classification':
        from sklearn.metrics import confusion_matrix, classification_report
        
        # 混淆矩阵
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title(f'{title} - 混淆矩阵')
        
        # 预测准确性
        plt.subplot(1, 2, 2)
        accuracy = (y_true == y_pred).mean()
        
        # 按类别统计准确率
        unique_classes = np.unique(y_true)
        class_accuracies = []
        for cls in unique_classes:
            mask = y_true == cls
            if mask.sum() > 0:
                acc = (y_true[mask] == y_pred[mask]).mean()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0)
        
        plt.bar(range(len(unique_classes)), class_accuracies)
        plt.xlabel('类别')
        plt.ylabel('准确率')
        plt.title(f'{title} - 各类别准确率')
        plt.xticks(range(len(unique_classes)), unique_classes)
        plt.ylim(0, 1)
        
        # 在控制台显示详细报告
        print(f"\n📊 {title} - 分类报告:")
        print(classification_report(y_true, y_pred))
        
    else:  # regression
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        # 预测 vs 真实值散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # 理想预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{title} - 预测 vs 真实')
        
        # 残差图
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('预测值')
        plt.ylabel('残差 (真实值 - 预测值)')
        plt.title(f'{title} - 残差分析')
        
        # 在控制台显示指标
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\n📊 {title} - 回归指标:")
        print(f"   R² 分数: {r2:.4f}")
        print(f"   平均绝对误差 (MAE): {mae:.4f}")
        print(f"   均方误差 (MSE): {mse:.4f}")
        print(f"   均方根误差 (RMSE): {rmse:.4f}")
    
    plt.tight_layout()
    plt.show()


def save_results(results: Dict[str, Any], filename: str):
    """
    保存实验结果
    
    Args:
        results: 结果字典
        filename: 保存文件名
    """
    import json
    
    # 转换numpy数组为列表（JSON序列化）
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    results_json = convert_numpy(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    print(f"💾 结果已保存到: {filename}")


def create_quick_demo():
    """
    创建一个完整的快速演示
    """
    print("🚀 快速演示：CausalQwen 的威力")
    print("=" * 50)
    
    # 生成数据
    X, y, info = generate_classification_data(
        n_samples=500, 
        n_features=10, 
        difficulty='medium'
    )
    
    # 探索数据
    explore_data(X, y, info, show_plots=False)
    
    # 准备数据
    data = prepare_data_for_training(X, y)
    
    # 与sklearn对比
    try:
        from simple_models import compare_with_sklearn
        results = compare_with_sklearn(X, y, task_type='classification')
    except ImportError:
        print("⚠️ 模型对比功能需要 simple_models 模块")
        results = {}
    
    print("\n🎉 快速演示完成！")
    print("这就是 CausalQwen 的强大之处 - 简单易用，效果出色！")
    
    return results


if __name__ == "__main__":
    # 运行快速演示
    create_quick_demo()