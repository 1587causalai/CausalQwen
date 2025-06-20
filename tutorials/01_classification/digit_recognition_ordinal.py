#!/usr/bin/env python3
"""
CausalEngine 手写数字识别教程 - 有序分类版

对比 OvR 多分类 vs 有序分类两种方法：
1. OvR多分类：将0-9视为10个独立类别
2. 有序分类：利用数字的自然顺序关系 0<1<2<...<9
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


class CausalDigitClassifier_OvR(nn.Module):
    """基于 CausalEngine 的数字分类器 - OvR多分类版"""
    
    def __init__(self, input_size=64, hidden_size=128, num_classes=10):
        super().__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # OvR多分类：10个独立类别
        self.causal_engine = CausalEngine(
            hidden_size=hidden_size,
            vocab_size=num_classes,  # 10个独立类别
            activation_modes="classification",  # 忽略顺序关系
            b_noise_init=0.05,
            gamma_init=1.0
        )
    
    def forward(self, x, temperature=1.0, do_sample=False, return_details=False):
        hidden_states = self.feature_encoder(x)
        output = self.causal_engine(
            hidden_states.unsqueeze(1),
            temperature=temperature,
            do_sample=do_sample,
            return_dict=True
        )
        
        if return_details:
            return output
        else:
            return output['output'].squeeze(1)


class CausalDigitClassifier_Ordinal(nn.Module):
    """基于 CausalEngine 的数字分类器 - 有序分类版"""
    
    def __init__(self, input_size=64, hidden_size=128, num_classes=10):
        super().__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 有序分类：利用0<1<2<...<9的顺序关系
        self.causal_engine = CausalEngine(
            hidden_size=hidden_size,
            vocab_size=1,  # 单一输出维度
            activation_modes="ordinal",  # 有序分类激活
            ordinal_num_classes=num_classes,  # 10个有序类别：0-9
            ordinal_threshold_init=1.0,
            b_noise_init=0.05,
            gamma_init=1.0
        )
    
    def forward(self, x, temperature=1.0, do_sample=False, return_details=False):
        hidden_states = self.feature_encoder(x)
        output = self.causal_engine(
            hidden_states.unsqueeze(1),
            temperature=temperature,
            do_sample=do_sample,
            return_dict=True
        )
        
        if return_details:
            return output
        else:
            return output['output'].squeeze()


def prepare_digits_data():
    """准备手写数字数据"""
    
    print("🔢 加载手写数字数据集...")
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"   样本数: {X.shape[0]}")
    print(f"   特征维度: {X.shape[1]} (8x8 图像展平)")
    print(f"   类别数: {len(np.unique(y))} (数字 0-9)")
    print(f"   类别分布: {np.bincount(y)}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"✅ 数据划分完成:")
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   验证集: {X_val.shape[0]} 样本") 
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_ordinal_metrics(y_true, y_pred):
    """计算有序分类的各种指标"""
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 精确准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 相邻准确率 (允许误差为1)
    adjacent_correct = np.abs(y_true - y_pred) <= 1
    adjacent_accuracy = np.mean(adjacent_correct)
    
    # 平均绝对误差 (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 肯德尔相关系数 (衡量排序一致性)
    tau, _ = kendalltau(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'adjacent_accuracy': adjacent_accuracy,
        'mae': mae,
        'kendall_tau': tau
    }


def train_ovr_model(model, train_loader, val_loader, max_epochs=100, patience=10):
    """训练 OvR 多分类模型"""
    
    print("\n🚀 训练 OvR 多分类模型...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features, temperature=1.0, do_sample=False)
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
                outputs = model(features, temperature=0.0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    print(f"✅ OvR模型训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    
    return train_losses, val_accuracies


def train_ordinal_model(model, train_loader, val_loader, max_epochs=100, patience=10):
    """训练有序分类模型"""
    
    print("\n🚀 训练有序分类模型...")
    
    # 使用 MSE 损失，将决策分数推向目标类别索引
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_accuracies = []
    val_mae_scores = []
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            # 获取决策分数
            output = model(features, temperature=1.0, return_details=True)
            decision_scores = output['loc_S'].squeeze()
            
            # MSE损失：决策分数 → 目标数字
            loss = criterion(decision_scores, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                predictions = model(features, temperature=0.0).long()
                val_predictions.extend(predictions.numpy())
                val_targets.extend(labels.numpy())
        
        # 计算有序指标
        metrics = calculate_ordinal_metrics(val_targets, val_predictions)
        val_acc = metrics['accuracy']
        val_mae = metrics['mae']
        
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_acc)
        val_mae_scores.append(val_mae)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}, Val MAE: {val_mae:.4f}")
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    print(f"✅ 有序模型训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    
    return train_losses, val_accuracies, val_mae_scores


def evaluate_models(ovr_model, ordinal_model, test_loader):
    """评估两种模型"""
    
    print("\n📊 评估两种方法...")
    
    results = {}
    
    # 评估 OvR 模型
    ovr_model.eval()
    ovr_predictions = []
    ovr_uncertainties = []
    true_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            output = ovr_model(features, temperature=0.0, return_details=True)
            probs = output['output'].squeeze(1)
            _, predicted = torch.max(probs, 1)
            ovr_predictions.extend(predicted.numpy())
            
            # OvR的不确定性：所有类别的平均尺度参数
            scale_S = output['scale_S'].squeeze(1)
            avg_uncertainty = scale_S.mean(dim=1)
            ovr_uncertainties.extend(avg_uncertainty.numpy())
            
            true_labels.extend(labels.numpy())
    
    ovr_metrics = calculate_ordinal_metrics(true_labels, ovr_predictions)
    ovr_metrics['predictions'] = np.array(ovr_predictions)
    ovr_metrics['uncertainties'] = np.array(ovr_uncertainties)
    results['OvR Classification'] = ovr_metrics
    
    # 评估有序模型
    ordinal_model.eval()
    ordinal_predictions = []
    ordinal_uncertainties = []
    decision_scores = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            output = ordinal_model(features, temperature=0.0, return_details=True)
            predictions = output['output'].squeeze().long()
            ordinal_predictions.extend(predictions.numpy())
            
            # 有序分类的不确定性
            scale_S = output['scale_S'].squeeze()
            ordinal_uncertainties.extend(scale_S.numpy())
            
            # 决策分数
            loc_S = output['loc_S'].squeeze()
            decision_scores.extend(loc_S.numpy())
    
    ordinal_metrics = calculate_ordinal_metrics(true_labels, ordinal_predictions)
    ordinal_metrics['predictions'] = np.array(ordinal_predictions)
    ordinal_metrics['uncertainties'] = np.array(ordinal_uncertainties)
    ordinal_metrics['decision_scores'] = np.array(decision_scores)
    results['Ordinal Classification'] = ordinal_metrics
    
    # 打印结果
    print("测试集结果对比:")
    print("-" * 50)
    for method, metrics in results.items():
        print(f"{method}:")
        print(f"  精确准确率: {metrics['accuracy']:.4f}")
        print(f"  相邻准确率: {metrics['adjacent_accuracy']:.4f}")
        print(f"  平均绝对误差: {metrics['mae']:.4f}")
        print(f"  肯德尔Tau: {metrics['kendall_tau']:.4f}")
        print(f"  平均不确定性: {np.mean(metrics['uncertainties']):.4f}")
        print()
    
    return results, np.array(true_labels)


def analyze_ordinal_properties(ordinal_model, test_loader):
    """分析有序分类的特殊性质"""
    
    print("\n🎯 分析有序分类特殊性质...")
    
    ordinal_model.eval()
    
    # 收集所有决策分数
    all_decision_scores = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            output = ordinal_model(features, temperature=0.0, return_details=True)
            decision_scores = output['loc_S'].squeeze()
            all_decision_scores.extend(decision_scores.numpy())
            all_labels.extend(labels.numpy())
    
    all_decision_scores = np.array(all_decision_scores)
    all_labels = np.array(all_labels)
    
    # 分析决策分数的分布
    decision_scores_by_digit = {}
    for digit in range(10):
        mask = all_labels == digit
        scores = all_decision_scores[mask]
        decision_scores_by_digit[digit] = scores
        print(f"   数字{digit}的决策得分: 均值={scores.mean():.3f}, 标准差={scores.std():.3f}")
    
    # 检查单调性：更大的数字应该有更高的决策得分
    monotonicity_violations = 0
    total_comparisons = 0
    
    for i in range(10):
        for j in range(i+1, 10):
            mean_i = decision_scores_by_digit[i].mean()
            mean_j = decision_scores_by_digit[j].mean()
            
            if mean_i > mean_j:  # 违反单调性
                monotonicity_violations += 1
            total_comparisons += 1
    
    monotonicity_preservation = 1 - (monotonicity_violations / total_comparisons)
    print(f"   单调性保持度: {monotonicity_preservation:.3f} (1.0为完全单调)")
    
    return {
        'decision_scores_by_digit': decision_scores_by_digit,
        'monotonicity_preservation': monotonicity_preservation
    }


def visualize_comparison(results, true_labels, ordinal_analysis):
    """可视化 OvR vs 有序分类对比"""
    
    print("\n📈 生成对比可视化...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 准确率对比
    ax1 = plt.subplot(3, 3, 1)
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    adj_accuracies = [results[m]['adjacent_accuracy'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Exact Accuracy', alpha=0.8)
    ax1.bar(x + width/2, adj_accuracies, width, label='Adjacent Accuracy', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    
    # 添加数值标签
    for i, (acc, adj_acc) in enumerate(zip(accuracies, adj_accuracies)):
        ax1.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, adj_acc + 0.01, f'{adj_acc:.3f}', ha='center', va='bottom')
    
    # 2. MAE和Kendall Tau对比
    ax2 = plt.subplot(3, 3, 2)
    mae_scores = [results[m]['mae'] for m in methods]
    tau_scores = [results[m]['kendall_tau'] for m in methods]
    
    ax2.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8, color='red')
    ax2.set_ylabel('MAE', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, tau_scores, width, label='Kendall Tau', alpha=0.8, color='blue')
    ax2_twin.set_ylabel('Kendall Tau', color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')
    
    ax2.set_title('MAE vs Kendall Tau')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45)
    
    # 3. 混淆矩阵对比 - OvR
    ax3 = plt.subplot(3, 3, 3)
    ovr_cm = confusion_matrix(true_labels, results['OvR Classification']['predictions'])
    sns.heatmap(ovr_cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('OvR Classification Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # 4. 混淆矩阵对比 - 有序
    ax4 = plt.subplot(3, 3, 4)
    ordinal_cm = confusion_matrix(true_labels, results['Ordinal Classification']['predictions'])
    sns.heatmap(ordinal_cm, annot=True, fmt='d', cmap='Greens', ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Ordinal Classification Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    
    # 5. 决策分数分布（仅有序）
    ax5 = plt.subplot(3, 3, 5)
    decision_scores_by_digit = ordinal_analysis['decision_scores_by_digit']
    
    for digit in range(10):
        scores = decision_scores_by_digit[digit]
        ax5.scatter([digit] * len(scores), scores, alpha=0.6, s=10, label=f'Digit {digit}')
    
    # 添加平均值连线
    means = [decision_scores_by_digit[digit].mean() for digit in range(10)]
    ax5.plot(range(10), means, 'r-', linewidth=2, label='Mean Score')
    
    ax5.set_xlabel('True Digit')
    ax5.set_ylabel('Decision Score')
    ax5.set_title('Decision Score Distribution (Ordinal)')
    ax5.set_xticks(range(10))
    ax5.grid(True, alpha=0.3)
    
    # 6. 错误类型分析
    ax6 = plt.subplot(3, 3, 6)
    
    # 分析两种方法的错误距离
    ovr_errors = np.abs(true_labels - results['OvR Classification']['predictions'])
    ordinal_errors = np.abs(true_labels - results['Ordinal Classification']['predictions'])
    
    error_distances = np.arange(10)
    ovr_error_counts = np.bincount(ovr_errors, minlength=10)
    ordinal_error_counts = np.bincount(ordinal_errors, minlength=10)
    
    ax6.bar(error_distances - 0.2, ovr_error_counts, 0.4, label='OvR', alpha=0.8)
    ax6.bar(error_distances + 0.2, ordinal_error_counts, 0.4, label='Ordinal', alpha=0.8)
    ax6.set_xlabel('Error Distance')
    ax6.set_ylabel('Count')
    ax6.set_title('Error Distance Distribution')
    ax6.legend()
    ax6.set_xticks(error_distances)
    
    # 7. 不确定性对比
    ax7 = plt.subplot(3, 3, 7)
    
    ovr_uncertainties = results['OvR Classification']['uncertainties']
    ordinal_uncertainties = results['Ordinal Classification']['uncertainties']
    
    ax7.hist(ovr_uncertainties, bins=30, alpha=0.6, label='OvR', density=True)
    ax7.hist(ordinal_uncertainties, bins=30, alpha=0.6, label='Ordinal', density=True)
    ax7.set_xlabel('Uncertainty')
    ax7.set_ylabel('Density')
    ax7.set_title('Uncertainty Distribution')
    ax7.legend()
    
    # 8. 正确性vs不确定性
    ax8 = plt.subplot(3, 3, 8)
    
    ovr_correct = (results['OvR Classification']['predictions'] == true_labels)
    ordinal_correct = (results['Ordinal Classification']['predictions'] == true_labels)
    
    ax8.scatter(ovr_uncertainties[ovr_correct], [1] * ovr_correct.sum(), 
                alpha=0.6, color='blue', s=10, label='OvR Correct')
    ax8.scatter(ovr_uncertainties[~ovr_correct], [0.8] * (~ovr_correct).sum(), 
                alpha=0.6, color='blue', s=10, label='OvR Incorrect', marker='x')
    
    ax8.scatter(ordinal_uncertainties[ordinal_correct], [0.6] * ordinal_correct.sum(), 
                alpha=0.6, color='green', s=10, label='Ordinal Correct')
    ax8.scatter(ordinal_uncertainties[~ordinal_correct], [0.4] * (~ordinal_correct).sum(), 
                alpha=0.6, color='green', s=10, label='Ordinal Incorrect', marker='x')
    
    ax8.set_xlabel('Uncertainty')
    ax8.set_ylabel('Model & Correctness')
    ax8.set_title('Uncertainty vs Correctness')
    ax8.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax8.set_yticklabels(['Ord-Wrong', 'Ord-Right', 'OvR-Wrong', 'OvR-Right'])
    ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 9. 性能总结雷达图
    ax9 = plt.subplot(3, 3, 9, projection='polar')
    
    metrics = ['Accuracy', 'Adj Accuracy', 'Low MAE', 'Kendall Tau', 'Monotonicity']
    
    # 标准化指标到0-1
    ovr_values = [
        results['OvR Classification']['accuracy'],
        results['OvR Classification']['adjacent_accuracy'],
        1 - results['OvR Classification']['mae'] / 10,  # 反转MAE
        results['OvR Classification']['kendall_tau'],
        0.8  # OvR没有单调性概念，给个中等分数
    ]
    
    ordinal_values = [
        results['Ordinal Classification']['accuracy'],
        results['Ordinal Classification']['adjacent_accuracy'],
        1 - results['Ordinal Classification']['mae'] / 10,  # 反转MAE
        results['Ordinal Classification']['kendall_tau'],
        ordinal_analysis['monotonicity_preservation']
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    ovr_values += ovr_values[:1]
    ordinal_values += ordinal_values[:1]
    
    ax9.plot(angles, ovr_values, 'o-', linewidth=2, label='OvR', color='blue')
    ax9.fill(angles, ovr_values, alpha=0.25, color='blue')
    
    ax9.plot(angles, ordinal_values, 'o-', linewidth=2, label='Ordinal', color='green')
    ax9.fill(angles, ordinal_values, alpha=0.25, color='green')
    
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(metrics)
    ax9.set_ylim(0, 1)
    ax9.set_title('Overall Performance Comparison', y=1.08)
    ax9.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax9.grid(True)
    
    plt.tight_layout()
    plt.savefig('ovr_vs_ordinal_comparison.png', dpi=150, bbox_inches='tight')
    print("   对比分析已保存到 ovr_vs_ordinal_comparison.png")
    plt.close()


def main():
    """主函数"""
    
    print("="*70)
    print("🎯 CausalEngine 数字识别：OvR vs 有序分类对比")
    print("   探索利用数字顺序关系的优势")
    print("="*70)
    
    # 1. 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_digits_data()
    
    # 创建数据加载器
    train_dataset = DigitsDataset(X_train, y_train)
    val_dataset = DigitsDataset(X_val, y_val)
    test_dataset = DigitsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. 训练 OvR 模型
    ovr_model = CausalDigitClassifier_OvR(input_size=64, hidden_size=128, num_classes=10)
    ovr_train_losses, ovr_val_accs = train_ovr_model(ovr_model, train_loader, val_loader)
    
    # 3. 训练有序模型
    ordinal_model = CausalDigitClassifier_Ordinal(input_size=64, hidden_size=128, num_classes=10)
    ordinal_train_losses, ordinal_val_accs, ordinal_mae = train_ordinal_model(ordinal_model, train_loader, val_loader)
    
    # 4. 评估两种模型
    results, true_labels = evaluate_models(ovr_model, ordinal_model, test_loader)
    
    # 5. 分析有序分类的特殊性质
    ordinal_analysis = analyze_ordinal_properties(ordinal_model, test_loader)
    
    # 6. 可视化对比
    visualize_comparison(results, true_labels, ordinal_analysis)
    
    # 7. 总结
    print("\n💡 关键发现:")
    print("-" * 50)
    print("✨ 有序分类 vs OvR多分类:")
    
    ovr_acc = results['OvR Classification']['accuracy']
    ordinal_acc = results['Ordinal Classification']['accuracy']
    ovr_mae = results['OvR Classification']['mae']
    ordinal_mae = results['Ordinal Classification']['mae']
    
    print(f"📊 精确准确率: OvR={ovr_acc:.4f} vs 有序={ordinal_acc:.4f}")
    print(f"📊 平均绝对误差: OvR={ovr_mae:.4f} vs 有序={ordinal_mae:.4f}")
    print(f"📊 单调性保持: {ordinal_analysis['monotonicity_preservation']:.3f}")
    
    if ordinal_acc > ovr_acc:
        print("🎯 有序分类在准确率上有优势！")
    else:
        print("🎯 OvR多分类在准确率上略胜一筹")
    
    if ordinal_mae < ovr_mae:
        print("🎯 有序分类在误差控制上更好（更小的MAE）")
    else:
        print("🎯 OvR多分类在误差控制上略好")
    
    print("\n🔍 数字识别任务特点:")
    print("• 数据量较小（~1800样本）时，两种方法性能接近")
    print("• 有序分类更适合'相邻误差可接受'的场景")
    print("• OvR分类更适合'类别完全独立'的场景") 
    print("• 有序分类提供额外的单调性约束，增强可解释性")
    
    print("\n✅ 对比教程完成！")
    print("📊 请查看 ovr_vs_ordinal_comparison.png 了解详细对比")


if __name__ == "__main__":
    main() 