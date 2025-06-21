#!/usr/bin/env python3
"""
CausalEngine 表格数据快速测试

使用几个经典的小型数据集快速测试 CausalEngine 的性能
数据集：Iris、Wine、Breast Cancer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# 导入 CausalEngine
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from causal_engine import CausalEngine, ActivationMode


class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CausalClassifier(nn.Module):
    """CausalEngine 分类器"""
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        # 简单的特征编码
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        self.causal_engine = CausalEngine(
            hidden_size=32,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
    
    def forward(self, x, temperature=1.0):
        features = self.encoder(x)
        output = self.causal_engine(
            features.unsqueeze(1),
            temperature=temperature,
            do_sample=False,
            return_dict=True
        )
        return output['output'].squeeze(1)


def load_datasets():
    """加载多个经典数据集"""
    
    datasets_info = []
    
    # 1. Iris（鸢尾花）- 150样本，4特征，3类
    print("\n🌸 加载 Iris 数据集...")
    iris = datasets.load_iris()
    datasets_info.append({
        'name': 'Iris',
        'X': iris.data,
        'y': iris.target,
        'num_classes': 3,
        'num_features': 4,
        'num_samples': len(iris.data)
    })
    
    # 2. Wine（葡萄酒）- 178样本，13特征，3类
    print("🍷 加载 Wine 数据集...")
    wine = datasets.load_wine()
    datasets_info.append({
        'name': 'Wine',
        'X': wine.data,
        'y': wine.target,
        'num_classes': 3,
        'num_features': 13,
        'num_samples': len(wine.data)
    })
    
    # 3. Breast Cancer（乳腺癌）- 569样本，30特征，2类
    print("🏥 加载 Breast Cancer 数据集...")
    cancer = datasets.load_breast_cancer()
    datasets_info.append({
        'name': 'Breast Cancer',
        'X': cancer.data,
        'y': cancer.target,
        'num_classes': 2,
        'num_features': 30,
        'num_samples': len(cancer.data)
    })
    
    # 4. Digits（手写数字简化版）- 1797样本，64特征，10类
    print("🔢 加载 Digits 数据集...")
    digits = datasets.load_digits()
    # 采样500个样本以加快速度
    indices = np.random.choice(len(digits.data), 500, replace=False)
    datasets_info.append({
        'name': 'Digits (500)',
        'X': digits.data[indices],
        'y': digits.target[indices],
        'num_classes': 10,
        'num_features': 64,
        'num_samples': 500
    })
    
    # 5. 生成一个合成数据集 - 1000样本，20特征，4类
    print("🎲 生成合成数据集...")
    X_synthetic, y_synthetic = datasets.make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=42
    )
    datasets_info.append({
        'name': 'Synthetic',
        'X': X_synthetic,
        'y': y_synthetic,
        'num_classes': 4,
        'num_features': 20,
        'num_samples': 1000
    })
    
    return datasets_info


def quick_train_causal(model, train_loader, val_loader, epochs=20):
    """快速训练 CausalEngine"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    outputs = model(features, temperature=0.0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
    
    return best_val_acc


def evaluate_on_dataset(dataset_info):
    """在单个数据集上评估"""
    
    print(f"\n{'='*50}")
    print(f"📊 评估数据集: {dataset_info['name']}")
    print(f"   样本数: {dataset_info['num_samples']}")
    print(f"   特征数: {dataset_info['num_features']}")
    print(f"   类别数: {dataset_info['num_classes']}")
    print(f"{'='*50}")
    
    X = dataset_info['X']
    y = dataset_info['y']
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 如果训练集太小，使用更多数据
    if len(X_train) < 100:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    
    results = {}
    
    # 1. CausalEngine
    print("\n🔧 训练 CausalEngine...")
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    causal_model = CausalClassifier(X_train.shape[1], dataset_info['num_classes'])
    val_acc = quick_train_causal(causal_model, train_loader, train_loader, epochs=30)
    
    # 测试
    causal_model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in test_loader:
            outputs = causal_model(features, temperature=0.0)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
    
    causal_acc = accuracy_score(y_test, predictions)
    results['CausalEngine'] = causal_acc
    print(f"   测试准确率: {causal_acc:.4f}")
    
    # 2. 逻辑回归
    print("\n📊 训练逻辑回归...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    results['Logistic Regression'] = lr_acc
    print(f"   测试准确率: {lr_acc:.4f}")
    
    # 3. 随机森林
    print("\n🌲 训练随机森林...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    results['Random Forest'] = rf_acc
    print(f"   测试准确率: {rf_acc:.4f}")
    
    # 4. SVM
    print("\n🎯 训练 SVM...")
    svm = SVC(random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = svm.score(X_test, y_test)
    results['SVM'] = svm_acc
    print(f"   测试准确率: {svm_acc:.4f}")
    
    return results


def visualize_results(all_results):
    """可视化所有结果"""
    
    print("\n📈 生成可视化...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # 1. 准确率对比条形图
    ax = axes[0]
    datasets = list(all_results.keys())
    methods = ['CausalEngine', 'Logistic Regression', 'Random Forest', 'SVM']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, method in enumerate(methods):
        accuracies = [all_results[d][method] for d in datasets]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=method, alpha=0.8)
        
        # 突出CausalEngine
        if method == 'CausalEngine':
            for bar in bars:
                bar.set_color('red')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 2. 方法平均性能
    ax = axes[1]
    method_avg = {method: np.mean([all_results[d][method] for d in datasets]) 
                  for method in methods}
    
    bars = ax.bar(methods, method_avg.values(), alpha=0.7)
    bars[0].set_color('red')  # CausalEngine
    
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Average Performance Across All Datasets')
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, (method, acc) in zip(bars, method_avg.items()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 3. CausalEngine相对性能
    ax = axes[2]
    relative_perfs = []
    labels = []
    
    for dataset in datasets:
        causal_acc = all_results[dataset]['CausalEngine']
        best_other = max(all_results[dataset][m] for m in methods[1:])
        relative_perf = (causal_acc - best_other) / best_other * 100
        relative_perfs.append(relative_perf)
        labels.append(dataset)
    
    colors = ['green' if rp > 0 else 'red' for rp in relative_perfs]
    bars = ax.bar(range(len(labels)), relative_perfs, color=colors, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Relative Performance (%)')
    ax.set_title('CausalEngine vs Best Traditional Method')
    
    # 4. 数据集特征 vs CausalEngine性能
    ax = axes[3]
    
    # 提取数据集特征
    num_samples = []
    num_features = []
    causal_accs = []
    dataset_names = []
    
    for dataset_name in datasets:
        # 从全局变量中获取数据集信息（这里简化处理）
        if 'Iris' in dataset_name:
            num_samples.append(150)
            num_features.append(4)
        elif 'Wine' in dataset_name:
            num_samples.append(178)
            num_features.append(13)
        elif 'Breast' in dataset_name:
            num_samples.append(569)
            num_features.append(30)
        elif 'Digits' in dataset_name:
            num_samples.append(500)
            num_features.append(64)
        elif 'Synthetic' in dataset_name:
            num_samples.append(1000)
            num_features.append(20)
        
        causal_accs.append(all_results[dataset_name]['CausalEngine'])
        dataset_names.append(dataset_name)
    
    # 用特征数作为x轴，准确率作为y轴，点大小表示样本数
    scatter = ax.scatter(num_features, causal_accs, 
                        s=[n/5 for n in num_samples],  # 缩放点大小
                        alpha=0.6, c=range(len(dataset_names)), cmap='viridis')
    
    # 添加标签
    for i, name in enumerate(dataset_names):
        ax.annotate(name, (num_features[i], causal_accs[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('CausalEngine Accuracy')
    ax.set_title('Dataset Complexity vs CausalEngine Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tabular_quick_test_results.png', dpi=150, bbox_inches='tight')
    print("   结果已保存到 tabular_quick_test_results.png")
    plt.close()


def print_summary(all_results):
    """打印总结"""
    
    print("\n" + "="*60)
    print("📊 快速测试总结")
    print("="*60)
    
    # 计算统计
    methods = ['CausalEngine', 'Logistic Regression', 'Random Forest', 'SVM']
    method_wins = {m: 0 for m in methods}
    
    for dataset, results in all_results.items():
        best_method = max(results.items(), key=lambda x: x[1])[0]
        method_wins[best_method] += 1
        
        print(f"\n{dataset}:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (method, acc) in enumerate(sorted_results):
            marker = "🏆" if i == 0 else "  "
            print(f"  {marker} {method:20s}: {acc:.4f}")
    
    print("\n🏆 获胜统计:")
    for method, wins in sorted(method_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"   {method}: {wins} 次")
    
    # CausalEngine 分析
    print("\n🔍 CausalEngine 分析:")
    causal_ranks = []
    for dataset, results in all_results.items():
        sorted_methods = sorted(results.items(), key=lambda x: x[1], reverse=True)
        rank = next(i for i, (m, _) in enumerate(sorted_methods) if m == 'CausalEngine') + 1
        causal_ranks.append(rank)
    
    print(f"   平均排名: {np.mean(causal_ranks):.1f}/4")
    print(f"   最好排名: {min(causal_ranks)}")
    print(f"   最差排名: {max(causal_ranks)}")


def main():
    """主函数"""
    
    print("="*60)
    print("🚀 CausalEngine 表格数据快速测试")
    print("   使用经典小数据集进行快速对比")
    print("="*60)
    
    # 加载数据集
    datasets_info = load_datasets()
    
    # 在每个数据集上评估
    all_results = {}
    
    for dataset_info in datasets_info:
        results = evaluate_on_dataset(dataset_info)
        all_results[dataset_info['name']] = results
    
    # 可视化
    visualize_results(all_results)
    
    # 打印总结
    print_summary(all_results)
    
    print("\n✅ 快速测试完成！")
    print("📊 请查看 tabular_quick_test_results.png")


if __name__ == "__main__":
    main() 