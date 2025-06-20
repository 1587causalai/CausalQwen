#!/usr/bin/env python3
"""
CausalEngine 手写数字识别教程

这个教程展示如何使用 CausalEngine 进行手写数字识别（多分类任务）。
我们将使用 sklearn 的 Digits 数据集，对比传统方法和 CausalEngine 的优势。

重点展示：
1. CausalEngine 的多分类激活函数
2. 不确定性量化能力  
3. OvR (One-vs-Rest) 概率计算
4. 三种推理模式的实际效果
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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


class CausalDigitClassifier(nn.Module):
    """基于 CausalEngine 的数字分类器"""
    
    def __init__(self, input_size=64, hidden_size=128, num_classes=10):
        super().__init__()
        
        # 特征编码层
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # CausalEngine 核心
        self.causal_engine = CausalEngine(
            hidden_size=hidden_size,
            vocab_size=num_classes,  # 10个数字类别
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
    
    def forward(self, x, temperature=1.0, do_sample=False, return_details=False):
        # 特征编码
        hidden_states = self.feature_encoder(x)
        
        # CausalEngine 推理
        output = self.causal_engine(
            hidden_states.unsqueeze(1),  # 添加序列维度
            temperature=temperature,
            do_sample=do_sample,
            return_dict=True
        )
        
        if return_details:
            return output
        else:
            return output['output'].squeeze(1)  # 移除序列维度


def prepare_digits_data():
    """准备手写数字数据"""
    
    print("🔢 加载手写数字数据集...")
    
    # 加载数据
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 数据信息
    print(f"   样本数: {X.shape[0]}")
    print(f"   特征维度: {X.shape[1]} (8x8 图像展平)")
    print(f"   类别数: {len(np.unique(y))} (数字 0-9)")
    print(f"   类别分布: {np.bincount(y)}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集 (60% 训练, 20% 验证, 20% 测试)
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def visualize_sample_digits(X_test, y_test, n_samples=10):
    """可视化一些样本数字"""
    
    print("\n👁️ 可视化样本数字...")
    
    # Set matplotlib to use English
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # 随机选择样本
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # 重塑为8x8图像
        image = X_test[idx].reshape(8, 8)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {y_test[idx]}')
        axes[i].axis('off')
    
    plt.suptitle('Sample Digits from Test Set')
    plt.tight_layout()
    plt.savefig('digit_samples.png', dpi=150, bbox_inches='tight')
    print("   样本图像已保存到 digit_samples.png")
    plt.close()


def train_causal_model(model, train_loader, val_loader, epochs=30):
    """训练 CausalEngine 模型"""
    
    print("\n🚀 开始训练 CausalEngine 模型...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            # 前向传播（使用标准模式）
            outputs = model(features, temperature=1.0, do_sample=False)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features, temperature=0.0)  # 纯因果模式
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {accuracy:.2f}%")
    
    print("✅ 训练完成!")
    return train_losses, val_accuracies


def train_baseline_models(X_train, y_train, X_val, y_val):
    """训练传统基线模型"""
    
    print("\n📊 训练传统基线模型...")
    
    baselines = {}
    
    # 1. 逻辑回归
    lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_acc = accuracy_score(y_val, lr_pred)
    baselines['Logistic Regression'] = {
        'model': lr, 
        'accuracy': lr_acc,
        'predictions': lr_pred
    }
    print(f"   Logistic Regression 验证准确率: {lr_acc:.3f}")
    
    # 2. SVM
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_val)
    svm_acc = accuracy_score(y_val, svm_pred)
    baselines['SVM (RBF)'] = {
        'model': svm,
        'accuracy': svm_acc,
        'predictions': svm_pred
    }
    print(f"   SVM (RBF) 验证准确率: {svm_acc:.3f}")
    
    # 3. 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_pred)
    baselines['Random Forest'] = {
        'model': rf,
        'accuracy': rf_acc,
        'predictions': rf_pred
    }
    print(f"   Random Forest 验证准确率: {rf_acc:.3f}")
    
    return baselines


def evaluate_inference_modes(model, test_loader):
    """评估 CausalEngine 的三种推理模式"""
    
    print("\n🔍 评估三种推理模式...")
    
    model.eval()
    results = {}
    
    modes = [
        ("纯因果模式", {"temperature": 0.0, "do_sample": False}),
        ("标准模式", {"temperature": 1.0, "do_sample": False}), 
        ("采样模式", {"temperature": 0.8, "do_sample": True})
    ]
    
    for mode_name, params in modes:
        all_predictions = []
        all_probabilities = []
        all_uncertainties = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                # 获取详细输出
                output = model(features, return_details=True, **params)
                
                # 提取概率
                probs = output['output'].squeeze(1)
                all_probabilities.append(probs.numpy())
                
                # 预测类别
                _, pred = torch.max(probs, 1)
                all_predictions.extend(pred.numpy())
                
                # 不确定性（所有类别的平均尺度参数）
                scale_S = output['scale_S'].squeeze(1)
                avg_uncertainty = scale_S.mean(dim=1)
                all_uncertainties.extend(avg_uncertainty.numpy())
                
                all_labels.extend(labels.numpy())
        
        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.vstack(all_probabilities)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_uncertainty = np.mean(all_uncertainties)
        
        # 每个类别的准确率
        class_accuracies = []
        for i in range(10):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = (all_predictions[mask] == i).mean()
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        results[mode_name] = {
            'accuracy': accuracy,
            'avg_uncertainty': avg_uncertainty,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'uncertainties': all_uncertainties,
            'true_labels': all_labels,
            'class_accuracies': class_accuracies
        }
        
        print(f"   {mode_name}: 准确率={accuracy:.3f}, 平均不确定性={avg_uncertainty:.3f}")
    
    return results


def visualize_results(train_losses, val_accuracies, inference_results, baseline_results):
    """可视化分析结果"""
    
    print("\n📈 生成可视化分析...")
    
    # Set matplotlib to use English
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 训练曲线
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('CausalEngine Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('CausalEngine Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 2. 推理模式对比
    ax3 = plt.subplot(3, 3, 3)
    mode_mapping = {
        '纯因果模式': 'Pure Causal',
        '标准模式': 'Standard',
        '采样模式': 'Sampling'
    }
    
    modes_en = [mode_mapping.get(mode, mode) for mode in inference_results.keys()]
    accuracies = [inference_results[mode]['accuracy'] for mode in inference_results.keys()]
    uncertainties = [inference_results[mode]['avg_uncertainty'] for mode in inference_results.keys()]
    
    x = np.arange(len(modes_en))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, uncertainties, width, label='Uncertainty', 
                         color='orange', alpha=0.8)
    
    ax3.set_xlabel('Inference Mode')
    ax3.set_ylabel('Accuracy', color='blue')
    ax3_twin.set_ylabel('Uncertainty', color='orange')
    ax3.set_title('Accuracy and Uncertainty across Inference Modes')
    ax3.set_xticks(x)
    ax3.set_xticklabels(modes_en)
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 3. 与传统方法对比
    ax4 = plt.subplot(3, 3, 4)
    all_methods = list(baseline_results.keys()) + ['CausalEngine']
    all_accuracies = [baseline_results[method]['accuracy'] for method in baseline_results.keys()]
    all_accuracies.append(inference_results['纯因果模式']['accuracy'])
    
    colors = ['gray'] * len(baseline_results) + ['red']
    bars = ax4.bar(all_methods, all_accuracies, alpha=0.7, color=colors)
    ax4.set_title('CausalEngine vs Traditional Methods')
    ax4.set_ylabel('Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, acc in zip(bars, all_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 4. 混淆矩阵 (纯因果模式)
    ax5 = plt.subplot(3, 3, 5)
    mode_data = inference_results['纯因果模式']
    cm = confusion_matrix(mode_data['true_labels'], mode_data['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
    ax5.set_title('Confusion Matrix (Pure Causal Mode)')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('True')
    
    # 5. 每个类别的准确率
    ax6 = plt.subplot(3, 3, 6)
    class_accs = mode_data['class_accuracies']
    bars = ax6.bar(range(10), class_accs, alpha=0.7, color='green')
    ax6.set_title('Per-Class Accuracy (Pure Causal Mode)')
    ax6.set_xlabel('Digit Class')
    ax6.set_ylabel('Accuracy')
    ax6.set_xticks(range(10))
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, class_accs)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}', ha='center', va='bottom')
    
    # 6. 不确定性与预测正确性的关系
    ax7 = plt.subplot(3, 3, 7)
    predictions = mode_data['predictions']
    true_labels = mode_data['true_labels']
    uncertainties_array = np.array(mode_data['uncertainties'])
    
    correct_mask = predictions == true_labels
    incorrect_mask = ~correct_mask
    
    ax7.hist(uncertainties_array[correct_mask], bins=30, alpha=0.6, 
             label='Correct', color='green', density=True)
    ax7.hist(uncertainties_array[incorrect_mask], bins=30, alpha=0.6,
             label='Incorrect', color='red', density=True)
    ax7.set_xlabel('Uncertainty')
    ax7.set_ylabel('Density')
    ax7.set_title('Uncertainty Distribution by Prediction Correctness')
    ax7.legend()
    
    # 7. 概率分布热图
    ax8 = plt.subplot(3, 3, 8)
    # 对每个真实类别，显示平均预测概率分布
    prob_matrix = np.zeros((10, 10))
    for true_class in range(10):
        mask = true_labels == true_class
        if mask.sum() > 0:
            prob_matrix[true_class] = mode_data['probabilities'][mask].mean(axis=0)
    
    sns.heatmap(prob_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax8)
    ax8.set_title('Average Prediction Probabilities')
    ax8.set_xlabel('Predicted Class')
    ax8.set_ylabel('True Class')
    
    # 8. 错误案例分析
    ax9 = plt.subplot(3, 3, 9)
    # 找出预测错误的样本
    error_indices = np.where(incorrect_mask)[0]
    if len(error_indices) > 0:
        # 统计错误类型
        error_pairs = []
        for idx in error_indices:
            true = true_labels[idx]
            pred = predictions[idx]
            error_pairs.append(f'{true}→{pred}')
        
        from collections import Counter
        error_counts = Counter(error_pairs)
        top_errors = error_counts.most_common(10)
        
        if top_errors:
            error_types, counts = zip(*top_errors)
            y_pos = np.arange(len(error_types))
            ax9.barh(y_pos, counts, alpha=0.7, color='red')
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels(error_types)
            ax9.set_xlabel('Count')
            ax9.set_title('Top 10 Error Types (True→Predicted)')
            ax9.invert_yaxis()
    else:
        ax9.text(0.5, 0.5, 'No Errors!', ha='center', va='center', 
                transform=ax9.transAxes, fontsize=20)
    
    plt.tight_layout()
    plt.savefig('causal_digit_analysis.png', dpi=150, bbox_inches='tight')
    print("   分析图表已保存到 causal_digit_analysis.png")
    plt.close()


def analyze_uncertainty_quality(model, test_loader):
    """深入分析不确定性的质量"""
    
    print("\n🔬 分析不确定性质量...")
    
    model.eval()
    
    # 收集所有预测结果
    all_probs = []
    all_uncertainties = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            output = model(features, temperature=1.0, return_details=True)
            
            probs = output['output'].squeeze(1)
            scale_S = output['scale_S'].squeeze(1)
            
            # 预测概率和类别
            max_probs, predictions = torch.max(probs, 1)
            
            # 预测类别的不确定性
            pred_uncertainties = scale_S[range(len(predictions)), predictions]
            
            all_probs.extend(max_probs.numpy())
            all_uncertainties.extend(pred_uncertainties.numpy())
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_uncertainties = np.array(all_uncertainties)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算校准误差
    correct_mask = all_predictions == all_labels
    
    # 按不确定性分组
    n_bins = 10
    uncertainty_bins = np.percentile(all_uncertainties, np.linspace(0, 100, n_bins + 1))
    
    calibration_data = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (all_uncertainties >= uncertainty_bins[i])
        else:
            mask = (all_uncertainties >= uncertainty_bins[i]) & (all_uncertainties < uncertainty_bins[i+1])
        
        if mask.sum() > 0:
            bin_accuracy = correct_mask[mask].mean()
            bin_confidence = all_probs[mask].mean()
            bin_uncertainty = all_uncertainties[mask].mean()
            calibration_data.append({
                'bin': i,
                'accuracy': bin_accuracy,
                'confidence': bin_confidence,
                'uncertainty': bin_uncertainty,
                'count': mask.sum()
            })
    
    # 可视化校准结果
    plt.figure(figsize=(12, 5))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 子图1: 准确率 vs 不确定性
    plt.subplot(1, 2, 1)
    bins_centers = [d['uncertainty'] for d in calibration_data]
    accuracies = [d['accuracy'] for d in calibration_data]
    plt.plot(bins_centers, accuracies, 'o-', markersize=8)
    plt.xlabel('Average Uncertainty')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Uncertainty Calibration')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 样本分布
    plt.subplot(1, 2, 2)
    plt.hist(all_uncertainties[correct_mask], bins=30, alpha=0.6, 
             label='Correct', color='green', density=True)
    plt.hist(all_uncertainties[~correct_mask], bins=30, alpha=0.6,
             label='Incorrect', color='red', density=True)
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('uncertainty_calibration.png', dpi=150, bbox_inches='tight')
    print("   不确定性校准分析已保存到 uncertainty_calibration.png")
    plt.close()
    
    # 打印统计信息
    print(f"   正确预测的平均不确定性: {all_uncertainties[correct_mask].mean():.3f}")
    print(f"   错误预测的平均不确定性: {all_uncertainties[~correct_mask].mean():.3f}")
    print(f"   不确定性与准确率的相关系数: {np.corrcoef(all_uncertainties, correct_mask.astype(float))[0,1]:.3f}")


def main():
    """主函数"""
    
    print("="*70)
    print("🎯 CausalEngine 手写数字识别教程")
    print("   展示多分类任务中的因果推理能力")
    print("="*70)
    
    # 1. 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_digits_data()
    
    # 2. 可视化样本
    visualize_sample_digits(X_test, y_test)
    
    # 3. 创建数据加载器
    train_dataset = DigitsDataset(X_train, y_train)
    val_dataset = DigitsDataset(X_val, y_val)
    test_dataset = DigitsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 创建并训练 CausalEngine 模型
    model = CausalDigitClassifier(input_size=64, hidden_size=128, num_classes=10)
    train_losses, val_accuracies = train_causal_model(model, train_loader, val_loader, epochs=30)
    
    # 5. 训练基线模型
    baseline_results = train_baseline_models(X_train, y_train, X_val, y_val)
    
    # 6. 评估不同推理模式
    inference_results = evaluate_inference_modes(model, test_loader)
    
    # 7. 可视化结果
    visualize_results(train_losses, val_accuracies, inference_results, baseline_results)
    
    # 8. 深入分析不确定性
    analyze_uncertainty_quality(model, test_loader)
    
    # 9. 在测试集上最终评估
    print("\n📊 最终测试集评估:")
    
    # CausalEngine (纯因果模式)
    model.eval()
    with torch.no_grad():
        test_preds = []
        for features, _ in test_loader:
            outputs = model(features, temperature=0.0)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.numpy())
    
    print("\nCausalEngine (纯因果模式):")
    print(classification_report(y_test, test_preds, digits=3))
    
    # 基线模型
    for name, baseline in baseline_results.items():
        test_pred = baseline['model'].predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"\n{name} 测试准确率: {test_acc:.3f}")
    
    print("\n✅ 教程完成！")
    print("📊 请查看生成的图片文件了解详细分析结果：")
    print("   - digit_samples.png: 数据集样本展示")
    print("   - causal_digit_analysis.png: 完整的模型分析")
    print("   - uncertainty_calibration.png: 不确定性校准分析")


if __name__ == "__main__":
    main() 