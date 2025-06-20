#!/usr/bin/env python3
"""
CausalEngine 数字识别消融实验

消融实验设计：
1. 特征网络复杂度：简单MLP vs 复杂MLP vs CNN-like
2. 分类器类型：普通Softmax vs CausalEngine OvR vs CausalEngine 有序分类
3. 系统性分析哪个组件影响性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

# 导入 CausalEngine
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from causal_engine import CausalEngine, ActivationMode


class DigitsDataset(Dataset):
    """手写数字数据集"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============ 特征提取器变体 ============

class SimpleFeatureExtractor(nn.Module):
    """简单特征提取器：单层MLP"""
    
    def __init__(self, input_size=64, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)


class DeepFeatureExtractor(nn.Module):
    """深层特征提取器：多层MLP"""
    
    def __init__(self, input_size=64, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)


class CNNLikeFeatureExtractor(nn.Module):
    """CNN风格特征提取器：将64维重塑为8x8，使用卷积"""
    
    def __init__(self, input_size=64, hidden_size=128):
        super().__init__()
        # 8x8 输入 → 卷积特征 → MLP
        self.conv_layers = nn.Sequential(
            # 输入: (batch, 1, 8, 8)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # → (batch, 16, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (batch, 16, 4, 4)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (batch, 32, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (batch, 32, 2, 2)
            
            nn.Flatten(),  # → (batch, 32*2*2=128)
        )
        
        # 确保输出维度为 hidden_size
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        # 重塑 64维 → 8x8
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 8, 8)
        
        # 卷积特征提取
        x = self.conv_layers(x)
        
        # 最终特征
        x = self.fc(x)
        return x


# ============ 分类器变体 ============

class SoftmaxClassifier(nn.Module):
    """普通 Softmax 分类器"""
    
    def __init__(self, feature_extractor, hidden_size=128, num_classes=10):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class CausalOvRClassifier(nn.Module):
    """CausalEngine OvR 分类器"""
    
    def __init__(self, feature_extractor, hidden_size=128, num_classes=10):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.causal_engine = CausalEngine(
            hidden_size=hidden_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.05,
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


class CausalOrdinalClassifier(nn.Module):
    """CausalEngine 有序分类器"""
    
    def __init__(self, feature_extractor, hidden_size=128, num_classes=10):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.causal_engine = CausalEngine(
            hidden_size=hidden_size,
            vocab_size=1,
            activation_modes="ordinal",
            ordinal_num_classes=num_classes,
            ordinal_threshold_init=1.0,
            b_noise_init=0.05,
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
            return output['output'].squeeze()


def prepare_data():
    """准备数据"""
    
    print("🔢 加载数据...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"   训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 创建数据加载器
    train_dataset = DigitsDataset(X_train, y_train)
    val_dataset = DigitsDataset(X_val, y_val)
    test_dataset = DigitsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_test


def train_softmax_model(model, train_loader, val_loader, max_epochs=80, patience=10):
    """训练 Softmax 模型"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    return best_val_acc


def train_causal_ovr_model(model, train_loader, val_loader, max_epochs=80, patience=10):
    """训练 CausalEngine OvR 模型"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features, temperature=1.0, do_sample=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    return best_val_acc


def train_causal_ordinal_model(model, train_loader, val_loader, max_epochs=80, patience=10):
    """训练 CausalEngine 有序分类模型"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            output = model(features, temperature=1.0, return_details=True)
            decision_scores = output['loc_S'].squeeze()
            loss = criterion(decision_scores, labels.float())
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for features, labels in val_loader:
                preds = model(features, temperature=0.0).long()
                predictions.extend(preds.numpy())
                targets.extend(labels.numpy())
        
        val_acc = accuracy_score(targets, predictions)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    return best_val_acc


def evaluate_model(model, test_loader, model_type='softmax'):
    """评估模型"""
    
    model.eval()
    predictions = []
    true_labels = []
    uncertainties = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            if model_type == 'softmax':
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.numpy())
                # Softmax 没有不确定性
                uncertainties.extend([0.0] * len(labels))
                
            elif model_type == 'causal_ovr':
                output = model(features, temperature=0.0, return_details=True)
                probs = output['output'].squeeze(1)
                _, predicted = torch.max(probs, 1)
                predictions.extend(predicted.numpy())
                # OvR 不确定性：平均尺度参数
                scale_S = output['scale_S'].squeeze(1)
                avg_uncertainty = scale_S.mean(dim=1)
                uncertainties.extend(avg_uncertainty.numpy())
                
            elif model_type == 'causal_ordinal':
                preds = model(features, temperature=0.0).long()
                predictions.extend(preds.numpy())
                # 有序分类不确定性
                output = model(features, temperature=0.0, return_details=True)
                scale_S = output['scale_S'].squeeze()
                uncertainties.extend(scale_S.numpy())
            
            true_labels.extend(labels.numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    uncertainties = np.array(uncertainties)
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    adjacent_accuracy = np.mean(np.abs(true_labels - predictions) <= 1)
    mae = mean_absolute_error(true_labels, predictions)
    tau, _ = kendalltau(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'adjacent_accuracy': adjacent_accuracy,
        'mae': mae,
        'kendall_tau': tau,
        'predictions': predictions,
        'true_labels': true_labels,
        'uncertainties': uncertainties
    }


def run_ablation_study():
    """运行完整的消融实验"""
    
    print("🧪 开始消融实验...")
    
    # 准备数据
    train_loader, val_loader, test_loader, y_test = prepare_data()
    
    # 特征提取器变体
    feature_extractors = {
        'Simple MLP': SimpleFeatureExtractor,
        'Deep MLP': DeepFeatureExtractor,
        'CNN-like': CNNLikeFeatureExtractor
    }
    
    # 分类器变体
    classifier_types = ['Softmax', 'CausalEngine OvR', 'CausalEngine Ordinal']
    
    results = {}
    
    print("\n" + "="*60)
    print("🏃‍♂️ 运行所有组合...")
    
    for feat_name, feat_class in feature_extractors.items():
        print(f"\n📊 特征提取器: {feat_name}")
        print("-" * 40)
        
        for classifier_type in classifier_types:
            print(f"   🔧 分类器: {classifier_type}")
            
            # 创建特征提取器
            feature_extractor = feat_class(input_size=64, hidden_size=128)
            
            # 创建完整模型
            if classifier_type == 'Softmax':
                model = SoftmaxClassifier(feature_extractor, hidden_size=128, num_classes=10)
                val_acc = train_softmax_model(model, train_loader, val_loader)
                test_result = evaluate_model(model, test_loader, 'softmax')
                
            elif classifier_type == 'CausalEngine OvR':
                model = CausalOvRClassifier(feature_extractor, hidden_size=128, num_classes=10)
                val_acc = train_causal_ovr_model(model, train_loader, val_loader)
                test_result = evaluate_model(model, test_loader, 'causal_ovr')
                
            elif classifier_type == 'CausalEngine Ordinal':
                model = CausalOrdinalClassifier(feature_extractor, hidden_size=128, num_classes=10)
                val_acc = train_causal_ordinal_model(model, train_loader, val_loader)
                test_result = evaluate_model(model, test_loader, 'causal_ordinal')
            
            # 保存结果
            key = f"{feat_name} + {classifier_type}"
            results[key] = {
                'val_accuracy': val_acc,
                'test_accuracy': test_result['accuracy'],
                'test_mae': test_result['mae'],
                'test_kendall_tau': test_result['kendall_tau'],
                'feature_extractor': feat_name,
                'classifier': classifier_type,
                **test_result
            }
            
            print(f"      验证准确率: {val_acc:.4f}")
            print(f"      测试准确率: {test_result['accuracy']:.4f}")
            print(f"      测试MAE: {test_result['mae']:.4f}")
    
    return results


def visualize_ablation_results(results):
    """可视化消融实验结果"""
    
    print("\n📈 生成消融实验可视化...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(20, 12))
    
    # 整理数据
    feature_types = ['Simple MLP', 'Deep MLP', 'CNN-like']
    classifier_types = ['Softmax', 'CausalEngine OvR', 'CausalEngine Ordinal']
    
    # 创建性能矩阵
    accuracy_matrix = np.zeros((len(feature_types), len(classifier_types)))
    mae_matrix = np.zeros((len(feature_types), len(classifier_types)))
    
    for i, feat in enumerate(feature_types):
        for j, clf in enumerate(classifier_types):
            key = f"{feat} + {clf}"
            if key in results:
                accuracy_matrix[i, j] = results[key]['test_accuracy']
                mae_matrix[i, j] = results[key]['test_mae']
    
    # 1. 准确率热图
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(accuracy_matrix, 
                xticklabels=[c.replace('CausalEngine ', '') for c in classifier_types],
                yticklabels=feature_types,
                annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Test Accuracy Matrix')
    ax1.set_xlabel('Classifier Type')
    ax1.set_ylabel('Feature Extractor')
    
    # 2. MAE热图
    ax2 = plt.subplot(3, 3, 2)
    sns.heatmap(mae_matrix, 
                xticklabels=[c.replace('CausalEngine ', '') for c in classifier_types],
                yticklabels=feature_types,
                annot=True, fmt='.3f', cmap='YlGnBu_r', ax=ax2)
    ax2.set_title('Test MAE Matrix (Lower Better)')
    ax2.set_xlabel('Classifier Type')
    ax2.set_ylabel('Feature Extractor')
    
    # 3. 特征提取器影响
    ax3 = plt.subplot(3, 3, 3)
    feat_avg_acc = []
    feat_std_acc = []
    
    for feat in feature_types:
        accs = []
        for clf in classifier_types:
            key = f"{feat} + {clf}"
            if key in results:
                accs.append(results[key]['test_accuracy'])
        feat_avg_acc.append(np.mean(accs))
        feat_std_acc.append(np.std(accs))
    
    bars = ax3.bar(range(len(feature_types)), feat_avg_acc, 
                   yerr=feat_std_acc, alpha=0.7, capsize=5)
    ax3.set_xlabel('Feature Extractor')
    ax3.set_ylabel('Average Test Accuracy')
    ax3.set_title('Feature Extractor Impact')
    ax3.set_xticks(range(len(feature_types)))
    ax3.set_xticklabels(feature_types, rotation=45)
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, feat_avg_acc)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 4. 分类器影响
    ax4 = plt.subplot(3, 3, 4)
    clf_avg_acc = []
    clf_std_acc = []
    
    for clf in classifier_types:
        accs = []
        for feat in feature_types:
            key = f"{feat} + {clf}"
            if key in results:
                accs.append(results[key]['test_accuracy'])
        clf_avg_acc.append(np.mean(accs))
        clf_std_acc.append(np.std(accs))
    
    bars = ax4.bar(range(len(classifier_types)), clf_avg_acc, 
                   yerr=clf_std_acc, alpha=0.7, capsize=5,
                   color=['blue', 'orange', 'green'])
    ax4.set_xlabel('Classifier Type')
    ax4.set_ylabel('Average Test Accuracy')
    ax4.set_title('Classifier Impact')
    ax4.set_xticks(range(len(classifier_types)))
    ax4.set_xticklabels([c.replace('CausalEngine ', '') for c in classifier_types], rotation=45)
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, clf_avg_acc)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 5. 最佳组合详细对比
    ax5 = plt.subplot(3, 3, 5)
    
    # 找到每种分类器的最佳特征提取器组合
    best_combos = {}
    for clf in classifier_types:
        best_acc = 0
        best_feat = None
        for feat in feature_types:
            key = f"{feat} + {clf}"
            if key in results and results[key]['test_accuracy'] > best_acc:
                best_acc = results[key]['test_accuracy']
                best_feat = feat
        best_combos[clf] = (best_feat, best_acc)
    
    clf_names = [c.replace('CausalEngine ', '') for c in classifier_types]
    best_accs = [best_combos[clf][1] for clf in classifier_types]
    colors = ['blue', 'orange', 'green']
    
    bars = ax5.bar(range(len(classifier_types)), best_accs, 
                   color=colors, alpha=0.7)
    ax5.set_xlabel('Classifier Type')
    ax5.set_ylabel('Best Test Accuracy')
    ax5.set_title('Best Performance per Classifier')
    ax5.set_xticks(range(len(classifier_types)))
    ax5.set_xticklabels(clf_names, rotation=45)
    
    # 添加最佳特征提取器信息
    for i, (bar, clf) in enumerate(zip(bars, classifier_types)):
        feat, acc = best_combos[clf]
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}\n({feat})', ha='center', va='bottom', fontsize=8)
    
    # 6. 不确定性分析（仅CausalEngine）
    ax6 = plt.subplot(3, 3, 6)
    
    causal_methods = ['CausalEngine OvR', 'CausalEngine Ordinal']
    uncertainty_data = []
    uncertainty_labels = []
    
    for feat in feature_types:
        for method in causal_methods:
            key = f"{feat} + {method}"
            if key in results:
                uncertainties = results[key]['uncertainties']
                if len(uncertainties) > 0 and np.mean(uncertainties) > 0:  # 排除全零的情况
                    uncertainty_data.append(uncertainties)
                    uncertainty_labels.append(f"{feat}\n{method.replace('CausalEngine ', '')}")
    
    if uncertainty_data:
        ax6.boxplot(uncertainty_data, labels=uncertainty_labels)
        ax6.set_ylabel('Uncertainty')
        ax6.set_title('Uncertainty Distribution')
        ax6.tick_params(axis='x', rotation=45)
    else:
        ax6.text(0.5, 0.5, 'No uncertainty data\navailable', 
                ha='center', va='center', transform=ax6.transAxes)
    
    # 7. 性能 vs 复杂度权衡
    ax7 = plt.subplot(3, 3, 7)
    
    # 估计模型复杂度（参数数量的相对比较）
    complexity_scores = {
        'Simple MLP': 1,
        'Deep MLP': 3,
        'CNN-like': 2
    }
    
    # 对每种分类器绘制 复杂度 vs 性能
    for i, clf in enumerate(classifier_types):
        complexities = []
        accuracies = []
        
        for feat in feature_types:
            key = f"{feat} + {clf}"
            if key in results:
                complexities.append(complexity_scores[feat])
                accuracies.append(results[key]['test_accuracy'])
        
        if complexities:
            ax7.plot(complexities, accuracies, 'o-', label=clf.replace('CausalEngine ', ''), 
                    color=colors[i], linewidth=2, markersize=8)
    
    ax7.set_xlabel('Feature Extractor Complexity')
    ax7.set_ylabel('Test Accuracy')
    ax7.set_title('Performance vs Complexity Trade-off')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks([1, 2, 3])
    ax7.set_xticklabels(['Simple', 'CNN-like', 'Deep'])
    
    # 8. 错误类型分析（仅显示最佳组合）
    ax8 = plt.subplot(3, 3, 8)
    
    # 找到整体最佳的方法
    best_key = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_result = results[best_key]
    
    # 计算错误距离分布
    error_distances = np.abs(best_result['true_labels'] - best_result['predictions'])
    error_counts = np.bincount(error_distances, minlength=10)
    
    bars = ax8.bar(range(10), error_counts, alpha=0.7, color='red')
    ax8.set_xlabel('Error Distance')
    ax8.set_ylabel('Count')
    ax8.set_title(f'Error Distance Distribution\n(Best: {best_key})')
    ax8.set_xticks(range(10))
    
    # 9. 总结表格
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 创建总结表格
    summary_data = []
    for key, result in results.items():
        summary_data.append([
            key.replace(' + ', '\n'),
            f"{result['test_accuracy']:.3f}",
            f"{result['test_mae']:.3f}",
            f"{result['test_kendall_tau']:.3f}"
        ])
    
    # 按准确率排序
    summary_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    # 只显示前6个
    if len(summary_data) > 6:
        summary_data = summary_data[:6]
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Method', 'Accuracy', 'MAE', 'Kendall τ'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax9.set_title('Top 6 Method Rankings', y=0.9)
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=150, bbox_inches='tight')
    print("   消融实验结果已保存到 ablation_study_results.png")
    plt.close()


def print_ablation_summary(results):
    """打印消融实验总结"""
    
    print("\n" + "="*60)
    print("🏆 消融实验总结")
    print("="*60)
    
    # 按准确率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    print("\n📊 性能排名 (按测试准确率):")
    print("-" * 60)
    for i, (method, result) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {method:35s} - 准确率: {result['test_accuracy']:.4f}")
    
    # 特征提取器影响分析
    print(f"\n🔧 特征提取器影响分析:")
    print("-" * 40)
    
    feature_avg = {}
    for feat_type in ['Simple MLP', 'Deep MLP', 'CNN-like']:
        accs = []
        for method, result in results.items():
            if feat_type in method:
                accs.append(result['test_accuracy'])
        if accs:
            feature_avg[feat_type] = np.mean(accs)
    
    for feat, avg_acc in sorted(feature_avg.items(), key=lambda x: x[1], reverse=True):
        print(f"   {feat:15s}: 平均准确率 {avg_acc:.4f}")
    
    # 分类器影响分析  
    print(f"\n🎯 分类器影响分析:")
    print("-" * 40)
    
    classifier_avg = {}
    for clf_type in ['Softmax', 'CausalEngine OvR', 'CausalEngine Ordinal']:
        accs = []
        for method, result in results.items():
            if clf_type in method:
                accs.append(result['test_accuracy'])
        if accs:
            classifier_avg[clf_type] = np.mean(accs)
    
    for clf, avg_acc in sorted(classifier_avg.items(), key=lambda x: x[1], reverse=True):
        print(f"   {clf:20s}: 平均准确率 {avg_acc:.4f}")
    
    # 关键发现
    print(f"\n💡 关键发现:")
    print("-" * 40)
    
    best_method, best_result = sorted_results[0]
    worst_method, worst_result = sorted_results[-1]
    
    print(f"✅ 最佳组合: {best_method}")
    print(f"   测试准确率: {best_result['test_accuracy']:.4f}")
    print(f"   测试MAE: {best_result['test_mae']:.4f}")
    
    print(f"\n❌ 最差组合: {worst_method}")
    print(f"   测试准确率: {worst_result['test_accuracy']:.4f}")
    print(f"   测试MAE: {worst_result['test_mae']:.4f}")
    
    print(f"\n📈 性能提升: {best_result['test_accuracy'] - worst_result['test_accuracy']:.4f}")
    
    # 特定洞察
    simple_softmax = results.get('Simple MLP + Softmax', {}).get('test_accuracy', 0)
    deep_softmax = results.get('Deep MLP + Softmax', {}).get('test_accuracy', 0)
    cnn_softmax = results.get('CNN-like + Softmax', {}).get('test_accuracy', 0)
    
    if all([simple_softmax, deep_softmax, cnn_softmax]):
        if max(deep_softmax, cnn_softmax) > simple_softmax:
            print(f"\n🧠 特征网络复杂度确实很重要！")
            print(f"   简单MLP: {simple_softmax:.4f}")
            print(f"   深层MLP: {deep_softmax:.4f}") 
            print(f"   CNN风格: {cnn_softmax:.4f}")


def main():
    """主函数"""
    
    print("="*70)
    print("🧪 CausalEngine 数字识别消融实验")
    print("   系统性分析特征网络复杂度 & 分类器类型的影响")
    print("="*70)
    
    # 运行消融实验
    results = run_ablation_study()
    
    # 可视化结果
    visualize_ablation_results(results)
    
    # 打印总结
    print_ablation_summary(results)
    
    print("\n✅ 消融实验完成！")
    print("📊 请查看 ablation_study_results.png 了解详细分析")


if __name__ == "__main__":
    main() 