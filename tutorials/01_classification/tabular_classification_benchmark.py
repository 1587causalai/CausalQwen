#!/usr/bin/env python3
"""
CausalEngine 表格数据分类基准测试

在多个经典表格数据集上对比 CausalEngine 和传统方法
数据集包括：
1. Wine Quality（葡萄酒质量）- 多分类
2. Adult Income（成人收入）- 二分类
3. Cover Type（森林覆盖类型）- 多分类
4. Bank Marketing（银行营销）- 二分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# import xgboost as xgb  # 移除 XGBoost 避免 macOS segfault
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# 导入 CausalEngine
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from causal_engine import CausalEngine, ActivationMode

# 数据下载
from sklearn.datasets import fetch_openml
import urllib.request
import zipfile
import os


class TabularDataset(Dataset):
    """通用表格数据集"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TabularFeatureExtractor(nn.Module):
    """表格数据特征提取器"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        self.net = nn.Sequential(*layers)
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x):
        return self.net(x)


class CausalTabularClassifier(nn.Module):
    """CausalEngine 表格数据分类器"""
    
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128, 64]):
        super().__init__()
        
        self.feature_extractor = TabularFeatureExtractor(input_size, hidden_sizes)
        self.causal_engine = CausalEngine(
            hidden_size=self.feature_extractor.output_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
    
    def forward(self, x, temperature=1.0, do_sample=False, return_details=False):
        features = self.feature_extractor(x)
        output = self.causal_engine(
            features.unsqueeze(1),
            temperature=temperature,
            do_sample=do_sample,
            return_dict=True
        )
        
        if return_details:
            return output
        else:
            return output['output'].squeeze(1)


class SimpleNNClassifier(nn.Module):
    """简单神经网络分类器（对比基线）"""
    
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128, 64]):
        super().__init__()
        
        self.feature_extractor = TabularFeatureExtractor(input_size, hidden_sizes)
        self.classifier = nn.Linear(self.feature_extractor.output_size, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


def load_wine_quality_data():
    """加载葡萄酒质量数据集"""
    
    print("\n🍷 加载葡萄酒质量数据集...")
    
    # 尝试从 OpenML 加载
    try:
        # 加载红酒数据
        red_wine = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
        red_df = red_wine.frame
        red_df['wine_type'] = 0  # 红酒
        
        # 加载白酒数据
        white_wine = fetch_openml('wine-quality-white', version=1, as_frame=True, parser='auto')
        white_df = white_wine.frame
        white_df['wine_type'] = 1  # 白酒
        
        # 合并数据
        df = pd.concat([red_df, white_df], ignore_index=True)
        
        # 提取特征和标签
        feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                       'pH', 'sulphates', 'alcohol', 'wine_type']
        X = df[feature_cols].values.astype(float)
        y = df['quality'].values.astype(int) - 3  # 质量分数3-9映射到0-6
        
        print(f"   样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)}")
        
        return X, y, len(np.unique(y))
        
    except Exception as e:
        print(f"   下载失败: {e}")
        print("   生成模拟数据...")
        # 生成模拟数据
        n_samples = 6000
        n_features = 12
        n_classes = 7
        
        X = np.random.randn(n_samples, n_features)
        # 创建有意义的标签（基于特征的线性组合）
        weights = np.random.randn(n_features)
        scores = X @ weights + np.random.randn(n_samples) * 0.5
        y = np.digitize(scores, np.percentile(scores, np.linspace(0, 100, n_classes + 1)[1:-1]))
        
        print(f"   模拟样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)}")
        
        return X, y, n_classes


def load_adult_income_data():
    """加载成人收入数据集"""
    
    print("\n💰 加载成人收入数据集...")
    
    try:
        # 从 OpenML 加载
        adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        df = adult.frame
        
        # 处理分类特征
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # 提取特征和标签
        X = df_encoded.drop('class', axis=1).values.astype(float)
        y = LabelEncoder().fit_transform(df['class'])
        
        print(f"   样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)} (<=50K, >50K)")
        
        return X, y, 2
        
    except Exception as e:
        print(f"   下载失败: {e}")
        print("   生成模拟数据...")
        # 生成模拟二分类数据
        n_samples = 30000
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        # 创建不平衡的二分类
        weights = np.random.randn(n_features)
        scores = X @ weights + np.random.randn(n_samples)
        y = (scores > np.percentile(scores, 75)).astype(int)  # 约25%正样本
        
        print(f"   模拟样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)}")
        
        return X, y, 2


def load_covertype_data(sample_size=10000):
    """加载森林覆盖类型数据集（采样版）"""
    
    print("\n🌲 加载森林覆盖类型数据集...")
    
    try:
        # 从 OpenML 加载
        covertype = fetch_openml('covertype', version=3, as_frame=True, parser='auto')
        df = covertype.frame
        
        # 采样以减少计算时间
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # 提取特征和标签
        X = df.drop('class', axis=1).values.astype(float)
        y = LabelEncoder().fit_transform(df['class'])
        
        print(f"   样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)}")
        
        return X, y, len(np.unique(y))
        
    except Exception as e:
        print(f"   下载失败: {e}")
        print("   生成模拟数据...")
        # 生成模拟多分类数据
        n_samples = sample_size
        n_features = 54
        n_classes = 7
        
        X = np.random.randn(n_samples, n_features)
        # 创建有结构的多分类
        centers = np.random.randn(n_classes, n_features) * 2
        y = []
        for i in range(n_samples):
            distances = [np.linalg.norm(X[i] - center) for center in centers]
            y.append(np.argmin(distances))
        y = np.array(y)
        
        print(f"   模拟样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)}")
        
        return X, y, n_classes


def load_bank_marketing_data():
    """加载银行营销数据集"""
    
    print("\n🏦 加载银行营销数据集...")
    
    try:
        # 从 OpenML 加载
        bank = fetch_openml('bank-marketing', version=1, as_frame=True, parser='auto')
        df = bank.frame
        
        # 处理分类特征
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols.difference(['class']), drop_first=True)
        
        # 提取特征和标签
        X = df_encoded.drop('class', axis=1).values.astype(float)
        y = LabelEncoder().fit_transform(df['class'])
        
        print(f"   样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)} (no, yes)")
        
        return X, y, 2
        
    except Exception as e:
        print(f"   下载失败: {e}")
        print("   生成模拟数据...")
        # 生成模拟不平衡二分类数据
        n_samples = 40000
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        # 创建高度不平衡的二分类（模拟营销响应）
        weights = np.random.randn(n_features)
        scores = X @ weights + np.random.randn(n_samples) * 2
        y = (scores > np.percentile(scores, 88)).astype(int)  # 约12%正样本
        
        print(f"   模拟样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   类别分布: {np.bincount(y)}")
        
        return X, y, 2


def train_causal_model(model, train_loader, val_loader, num_classes, max_epochs=50):
    """训练 CausalEngine 模型"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features, temperature=1.0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features, temperature=0.0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.numpy())
                val_labels.extend(labels.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_acc


def train_nn_model(model, train_loader, val_loader, num_classes, max_epochs=50):
    """训练简单神经网络模型"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.numpy())
                val_labels.extend(labels.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_acc


def evaluate_model(model, test_loader, model_type='nn'):
    """评估模型"""
    
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            if model_type == 'causal':
                outputs = model(features, temperature=0.0)
                probs = outputs
            else:  # nn
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(probs, 1)
            predictions.extend(predicted.numpy())
            probabilities.append(probs.numpy())
            true_labels.extend(labels.numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.vstack(probabilities)
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    
    # 对于二分类，计算AUC
    if len(np.unique(true_labels)) == 2:
        auc = roc_auc_score(true_labels, probabilities[:, 1])
    else:
        auc = None
    
    # F1 分数
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels
    }


def run_benchmark_on_dataset(dataset_name, X, y, num_classes):
    """在单个数据集上运行基准测试"""
    
    print(f"\n{'='*60}")
    print(f"🏃 运行基准测试: {dataset_name}")
    print(f"{'='*60}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    results = {}
    
    # 1. CausalEngine
    print("\n🔧 训练 CausalEngine...")
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    causal_model = CausalTabularClassifier(X_train.shape[1], num_classes)
    val_acc = train_causal_model(causal_model, train_loader, val_loader, num_classes)
    test_result = evaluate_model(causal_model, test_loader, 'causal')
    results['CausalEngine'] = test_result
    print(f"   验证准确率: {val_acc:.4f}, 测试准确率: {test_result['accuracy']:.4f}")
    
    # 2. 简单神经网络
    print("\n🧠 训练简单神经网络...")
    nn_model = SimpleNNClassifier(X_train.shape[1], num_classes)
    val_acc = train_nn_model(nn_model, train_loader, val_loader, num_classes)
    test_result = evaluate_model(nn_model, test_loader, 'nn')
    results['Neural Network'] = test_result
    print(f"   验证准确率: {val_acc:.4f}, 测试准确率: {test_result['accuracy']:.4f}")
    
    # 3. 随机森林
    print("\n🌲 训练随机森林...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['Random Forest'] = {
        'accuracy': rf_acc,
        'predictions': rf_pred,
        'probabilities': rf_prob,
        'auc': roc_auc_score(y_test, rf_prob[:, 1]) if num_classes == 2 else None,
        'f1': f1_score(y_test, rf_pred, average='weighted')
    }
    print(f"   测试准确率: {rf_acc:.4f}")
    
    # 4. Gradient Boosting (替代 XGBoost 避免 macOS segfault)
    print("\n🚀 训练梯度提升...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_prob = gb_model.predict_proba(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    results['Gradient Boosting'] = {
        'accuracy': gb_acc,
        'predictions': gb_pred,
        'probabilities': gb_prob,
        'auc': roc_auc_score(y_test, gb_prob[:, 1]) if num_classes == 2 else None,
        'f1': f1_score(y_test, gb_pred, average='weighted')
    }
    print(f"   测试准确率: {gb_acc:.4f}")
    
    # 5. 逻辑回归（作为简单基线）
    print("\n📊 训练逻辑回归...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    results['Logistic Regression'] = {
        'accuracy': lr_acc,
        'predictions': lr_pred,
        'probabilities': lr_prob,
        'auc': roc_auc_score(y_test, lr_prob[:, 1]) if num_classes == 2 else None,
        'f1': f1_score(y_test, lr_pred, average='weighted')
    }
    print(f"   测试准确率: {lr_acc:.4f}")
    
    return results, y_test


def visualize_benchmark_results(all_results):
    """可视化所有基准测试结果"""
    
    print("\n📈 生成基准测试可视化...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(20, 12))
    
    # 整理数据
    datasets = list(all_results.keys())
    methods = ['CausalEngine', 'Neural Network', 'Random Forest', 'Gradient Boosting', 'Logistic Regression']
    
    # 1. 准确率对比热图
    ax1 = plt.subplot(2, 3, 1)
    accuracy_matrix = []
    for dataset in datasets:
        row = []
        for method in methods:
            if method in all_results[dataset]:
                row.append(all_results[dataset][method]['accuracy'])
            else:
                row.append(0)
        accuracy_matrix.append(row)
    
    accuracy_matrix = np.array(accuracy_matrix)
    
    sns.heatmap(accuracy_matrix, 
                xticklabels=[m.replace(' ', '\n') for m in methods],
                yticklabels=datasets,
                annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Accuracy Comparison Across Datasets')
    
    # 2. 方法平均性能
    ax2 = plt.subplot(2, 3, 2)
    method_avg_acc = []
    method_std_acc = []
    
    for method in methods:
        accs = []
        for dataset in datasets:
            if method in all_results[dataset]:
                accs.append(all_results[dataset][method]['accuracy'])
        if accs:
            method_avg_acc.append(np.mean(accs))
            method_std_acc.append(np.std(accs))
        else:
            method_avg_acc.append(0)
            method_std_acc.append(0)
    
    bars = ax2.bar(range(len(methods)), method_avg_acc, yerr=method_std_acc, 
                   capsize=5, alpha=0.7)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace(' ', '\n') for m in methods], rotation=45)
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Average Performance Across Datasets')
    
    # 为CausalEngine使用不同颜色
    bars[0].set_color('red')
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, method_avg_acc)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 3. 数据集难度分析
    ax3 = plt.subplot(2, 3, 3)
    dataset_avg_acc = []
    
    for dataset in datasets:
        accs = []
        for method in methods:
            if method in all_results[dataset]:
                accs.append(all_results[dataset][method]['accuracy'])
        dataset_avg_acc.append(np.mean(accs))
    
    bars = ax3.bar(range(len(datasets)), dataset_avg_acc, alpha=0.7, color='green')
    ax3.set_xticks(range(len(datasets)))
    ax3.set_xticklabels([d.replace(' ', '\n') for d in datasets], rotation=45)
    ax3.set_ylabel('Average Accuracy')
    ax3.set_title('Dataset Difficulty (Average Accuracy)')
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, dataset_avg_acc)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 4. CausalEngine vs 最佳方法
    ax4 = plt.subplot(2, 3, 4)
    causal_accs = []
    best_other_accs = []
    dataset_names = []
    
    for dataset in datasets:
        if 'CausalEngine' in all_results[dataset]:
            causal_acc = all_results[dataset]['CausalEngine']['accuracy']
            
            # 找到除CausalEngine外的最佳方法
            best_acc = 0
            for method in methods[1:]:  # 跳过CausalEngine
                if method in all_results[dataset]:
                    acc = all_results[dataset][method]['accuracy']
                    if acc > best_acc:
                        best_acc = acc
            
            causal_accs.append(causal_acc)
            best_other_accs.append(best_acc)
            dataset_names.append(dataset)
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, causal_accs, width, label='CausalEngine', alpha=0.8, color='red')
    bars2 = ax4.bar(x + width/2, best_other_accs, width, label='Best Other', alpha=0.8, color='blue')
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('CausalEngine vs Best Traditional Method')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.replace(' ', '\n') for d in dataset_names], rotation=45)
    ax4.legend()
    
    # 5. F1分数比较
    ax5 = plt.subplot(2, 3, 5)
    f1_matrix = []
    for dataset in datasets:
        row = []
        for method in methods:
            if method in all_results[dataset] and 'f1' in all_results[dataset][method]:
                row.append(all_results[dataset][method]['f1'])
            else:
                row.append(0)
        f1_matrix.append(row)
    
    f1_matrix = np.array(f1_matrix)
    
    sns.heatmap(f1_matrix, 
                xticklabels=[m.replace(' ', '\n') for m in methods],
                yticklabels=datasets,
                annot=True, fmt='.3f', cmap='YlGnBu', ax=ax5)
    ax5.set_title('F1 Score Comparison')
    
    # 6. 性能提升/下降分析
    ax6 = plt.subplot(2, 3, 6)
    improvements = []
    labels = []
    
    for dataset in datasets:
        if 'CausalEngine' in all_results[dataset]:
            causal_acc = all_results[dataset]['CausalEngine']['accuracy']
            
            # 与逻辑回归比较（作为基线）
            if 'Logistic Regression' in all_results[dataset]:
                lr_acc = all_results[dataset]['Logistic Regression']['accuracy']
                improvement = (causal_acc - lr_acc) / lr_acc * 100
                improvements.append(improvement)
                labels.append(dataset)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax6.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
    ax6.set_xticks(range(len(labels)))
    ax6.set_xticklabels([l.replace(' ', '\n') for l in labels], rotation=45)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_ylabel('Improvement over Logistic Regression (%)')
    ax6.set_title('CausalEngine Performance Gain/Loss')
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, 
                height + (1 if height > 0 else -1),
                f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('tabular_benchmark_results.png', dpi=150, bbox_inches='tight')
    print("   基准测试结果已保存到 tabular_benchmark_results.png")
    plt.close()


def print_summary_report(all_results):
    """打印总结报告"""
    
    print("\n" + "="*70)
    print("📊 基准测试总结报告")
    print("="*70)
    
    # 计算每个方法的平均性能
    methods = ['CausalEngine', 'Neural Network', 'Random Forest', 'Gradient Boosting', 'Logistic Regression']
    method_stats = {}
    
    for method in methods:
        accs = []
        for dataset in all_results:
            if method in all_results[dataset]:
                accs.append(all_results[dataset][method]['accuracy'])
        
        if accs:
            method_stats[method] = {
                'mean': np.mean(accs),
                'std': np.std(accs),
                'min': np.min(accs),
                'max': np.max(accs)
            }
    
    # 排序并打印
    print("\n🏆 方法排名（按平均准确率）:")
    print("-" * 50)
    sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for rank, (method, stats) in enumerate(sorted_methods, 1):
        print(f"{rank}. {method:20s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"   范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # CausalEngine 详细分析
    print("\n🔍 CausalEngine 详细分析:")
    print("-" * 50)
    
    causal_wins = 0
    causal_top2 = 0
    total_datasets = 0
    
    for dataset in all_results:
        if 'CausalEngine' in all_results[dataset]:
            total_datasets += 1
            
            # 排序该数据集上的所有方法
            dataset_results = [(m, r['accuracy']) for m, r in all_results[dataset].items()]
            dataset_results.sort(key=lambda x: x[1], reverse=True)
            
            # 检查CausalEngine的排名
            causal_rank = next(i for i, (m, _) in enumerate(dataset_results) if m == 'CausalEngine') + 1
            
            if causal_rank == 1:
                causal_wins += 1
            if causal_rank <= 2:
                causal_top2 += 1
            
            print(f"   {dataset}: 排名 {causal_rank}/{len(dataset_results)}")
    
    print(f"\n   获胜次数: {causal_wins}/{total_datasets}")
    print(f"   前二名次数: {causal_top2}/{total_datasets}")
    
    # 数据集特征分析
    print("\n📈 数据集特征与CausalEngine性能:")
    print("-" * 50)
    
    for dataset in all_results:
        if 'CausalEngine' in all_results[dataset]:
            causal_acc = all_results[dataset]['CausalEngine']['accuracy']
            best_acc = max(r['accuracy'] for r in all_results[dataset].values())
            gap = best_acc - causal_acc
            
            print(f"   {dataset}:")
            print(f"      CausalEngine: {causal_acc:.3f}")
            print(f"      最佳方法: {best_acc:.3f}")
            print(f"      差距: {gap:.3f} ({gap/best_acc*100:.1f}%)")
    
    # 关键洞察
    print("\n💡 关键洞察:")
    print("-" * 50)
    
    # 检查CausalEngine是否在某类数据集上表现更好
    binary_accs = []
    multi_accs = []
    
    for dataset, results in all_results.items():
        if 'CausalEngine' in results:
            if '二分类' in dataset or 'Income' in dataset or 'Marketing' in dataset:
                binary_accs.append(results['CausalEngine']['accuracy'])
            else:
                multi_accs.append(results['CausalEngine']['accuracy'])
    
    if binary_accs and multi_accs:
        print(f"   二分类平均准确率: {np.mean(binary_accs):.3f}")
        print(f"   多分类平均准确率: {np.mean(multi_accs):.3f}")
        
        if np.mean(binary_accs) > np.mean(multi_accs):
            print("   → CausalEngine 在二分类任务上表现更好")
        else:
            print("   → CausalEngine 在多分类任务上表现更好")


def main():
    """主函数"""
    
    print("="*70)
    print("🏁 CausalEngine 表格数据分类基准测试")
    print("   在多个经典数据集上全面评估性能")
    print("="*70)
    
    all_results = {}
    
    # 1. 葡萄酒质量数据集
    X, y, num_classes = load_wine_quality_data()
    results, _ = run_benchmark_on_dataset("Wine Quality", X, y, num_classes)
    all_results["Wine Quality"] = results
    
    # 2. 成人收入数据集
    X, y, num_classes = load_adult_income_data()
    results, _ = run_benchmark_on_dataset("Adult Income", X, y, num_classes)
    all_results["Adult Income"] = results
    
    # 3. 森林覆盖类型数据集
    X, y, num_classes = load_covertype_data(sample_size=10000)
    results, _ = run_benchmark_on_dataset("Cover Type", X, y, num_classes)
    all_results["Cover Type"] = results
    
    # 4. 银行营销数据集
    X, y, num_classes = load_bank_marketing_data()
    results, _ = run_benchmark_on_dataset("Bank Marketing", X, y, num_classes)
    all_results["Bank Marketing"] = results
    
    # 可视化结果
    visualize_benchmark_results(all_results)
    
    # 打印总结报告
    print_summary_report(all_results)
    
    print("\n✅ 基准测试完成！")
    print("📊 请查看 tabular_benchmark_results.png 了解详细对比")


if __name__ == "__main__":
    main() 