#!/usr/bin/env python3
"""
分析分类数据的详细情况
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from quick_test_causal_engine import QuickTester

def analyze_classification_data():
    # 生成与测试相同的数据
    np.random.seed(42)
    X, y = make_classification(
        n_samples=2000, n_features=15, n_classes=3,
        n_informative=7, n_redundant=0, n_clusters_per_class=1,
        class_sep=1.0, random_state=42
    )

    print('🔍 分类数据分析报告')
    print('=' * 50)
    print(f'数据形状: {X.shape}')
    print(f'特征范围: [{X.min():.2f}, {X.max():.2f}]')
    print(f'特征均值: {X.mean():.2f}, 标准差: {X.std():.2f}')
    print()

    # 原始类别分布
    unique, counts = np.unique(y, return_counts=True)
    print('📊 原始类别分布:')
    for label, count in zip(unique, counts):
        print(f'   类别{label}: {count}个样本 ({count/len(y)*100:.1f}%)')
    print()

    # 分割数据模拟训练过程
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')

    # 训练集类别分布
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print('📊 训练集类别分布:')
    for label, count in zip(unique_train, counts_train):
        print(f'   类别{label}: {count}个样本 ({count/len(y_train)*100:.1f}%)')
    print()

    # 模拟40%标签异常
    tester = QuickTester()
    y_train_noisy = tester.add_label_anomalies(y_train, 0.4, 'classification')

    print('📊 添加40%异常后的训练集分布:')
    unique_noisy, counts_noisy = np.unique(y_train_noisy, return_counts=True)
    for label, count in zip(unique_noisy, counts_noisy):
        print(f'   类别{label}: {count}个样本 ({count/len(y_train_noisy)*100:.1f}%)')
    print()

    # 分析异常影响
    changed_indices = y_train != y_train_noisy
    print(f'🎯 异常统计:')
    print(f'   异常样本数: {np.sum(changed_indices)} / {len(y_train)} ({np.sum(changed_indices)/len(y_train)*100:.1f}%)')
    print()

    # 异常转换矩阵
    print('🔄 标签转换矩阵 (原始→异常):')
    for orig in unique:
        for new in unique:
            count = np.sum((y_train == orig) & (y_train_noisy == new))
            if count > 0 and orig != new:
                print(f'   {orig}→{new}: {count}个')
    print()

    # 数据可分离性分析
    print('📈 数据可分离性分析:')
    
    # 在干净数据上的基准性能
    clf_clean = LogisticRegression(random_state=42, max_iter=1000)
    clf_clean.fit(X_train, y_train)
    acc_clean = accuracy_score(y_test, clf_clean.predict(X_test))

    # 在异常数据上的性能
    clf_noisy = LogisticRegression(random_state=42, max_iter=1000)
    clf_noisy.fit(X_train, y_train_noisy)
    acc_noisy = accuracy_score(y_test, clf_noisy.predict(X_test))

    print(f'   Logistic回归 - 干净数据训练: {acc_clean:.3f}')
    print(f'   Logistic回归 - 40%异常训练: {acc_noisy:.3f}')
    print(f'   性能下降: {(acc_clean-acc_noisy)*100:.1f}个百分点')
    print()
    
    # 类别分离度分析
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    # 使用KMeans评估可分离性
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, cluster_labels)
    
    print(f'🎯 数据特征分析:')
    print(f'   轮廓系数 (数据可分离性): {sil_score:.3f}')
    print(f'   类别分离度设置: 1.0 (中等难度)')
    print(f'   特征信息量: 7/{15} (约47%有效特征)')
    print()
    
    # 分析为什么异常影响可能较小
    print('🤔 可能的异常影响较小原因:')
    print('   1. 数据本身较易分类 (class_sep=1.0)')
    print('   2. 3分类任务，标签翻转后仍有33%概率正确')
    print('   3. 特征维度较高(15维)，冗余信息可能帮助抵抗异常')
    print('   4. 样本量充足(2000个)，异常影响被稀释')

if __name__ == "__main__":
    analyze_classification_data()