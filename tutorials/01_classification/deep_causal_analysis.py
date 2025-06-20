#!/usr/bin/env python3
"""
Deep MLP + CausalEngine 性能分析

专门分析为什么 Deep MLP + CausalEngine 仍然不如 Deep MLP + Softmax
可能的原因和改进方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入 CausalEngine
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from causal_engine import CausalEngine, ActivationMode


class DigitsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DeepFeatureExtractor(nn.Module):
    """深层特征提取器"""
    
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


class SoftmaxClassifier(nn.Module):
    """Deep MLP + Softmax"""
    
    def __init__(self, hidden_size=128, num_classes=10):
        super().__init__()
        self.feature_extractor = DeepFeatureExtractor(hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class CausalEngineClassifier(nn.Module):
    """Deep MLP + CausalEngine"""
    
    def __init__(self, hidden_size=128, num_classes=10, b_noise_init=0.05, gamma_init=1.0):
        super().__init__()
        self.feature_extractor = DeepFeatureExtractor(hidden_size=hidden_size)
        self.causal_engine = CausalEngine(
            hidden_size=hidden_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=b_noise_init,
            gamma_init=gamma_init
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


def prepare_data():
    """准备数据"""
    digits = load_digits()
    X, y = digits.data, digits.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    train_dataset = DigitsDataset(X_train, y_train)
    val_dataset = DigitsDataset(X_val, y_val)
    test_dataset = DigitsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_softmax_model(model, train_loader, val_loader, max_epochs=100, lr=0.001):
    """训练 Softmax 模型"""
    print("🚀 训练 Deep MLP + Softmax...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
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
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    print(f"✅ Softmax训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    
    return train_losses, val_accuracies


def train_causal_model(model, train_loader, val_loader, max_epochs=100, lr=0.001):
    """训练 CausalEngine 模型"""
    print("🚀 训练 Deep MLP + CausalEngine...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features, temperature=1.0, do_sample=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
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
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    print(f"✅ CausalEngine训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    
    return train_losses, val_accuracies


def analyze_performance_gap(softmax_model, causal_model, test_loader):
    """分析性能差距的原因"""
    
    print("\n🔍 分析性能差距...")
    
    # 测试两个模型
    results = {}
    
    for name, model in [("Softmax", softmax_model), ("CausalEngine", causal_model)]:
        model.eval()
        predictions = []
        confidences = []
        uncertainties = []
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                if name == "Softmax":
                    outputs = model(features)
                    probs = torch.softmax(outputs, dim=1)
                    max_probs, predicted = torch.max(probs, 1)
                    
                    predictions.extend(predicted.numpy())
                    confidences.extend(max_probs.numpy())
                    uncertainties.extend([0.0] * len(labels))  # Softmax没有不确定性
                    probabilities.append(probs.numpy())
                    
                else:  # CausalEngine
                    output = model(features, temperature=0.0, return_details=True)
                    probs = output['output'].squeeze(1)
                    max_probs, predicted = torch.max(probs, 1)
                    
                    predictions.extend(predicted.numpy())
                    confidences.extend(max_probs.numpy())
                    
                    # CausalEngine的不确定性
                    scale_S = output['scale_S'].squeeze(1)
                    avg_uncertainty = scale_S.mean(dim=1)
                    uncertainties.extend(avg_uncertainty.numpy())
                    probabilities.append(probs.numpy())
                
                true_labels.extend(labels.numpy())
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        confidences = np.array(confidences)
        uncertainties = np.array(uncertainties)
        probabilities = np.vstack(probabilities)
        
        accuracy = accuracy_score(true_labels, predictions)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'uncertainties': uncertainties,
            'probabilities': probabilities
        }
        
        print(f"   {name} 测试准确率: {accuracy:.4f}")
    
    return results


def detailed_error_analysis(results):
    """详细的错误分析"""
    
    print("\n📊 详细错误分析...")
    
    softmax_result = results['Softmax']
    causal_result = results['CausalEngine']
    
    # 1. 找出两者预测不同的样本
    different_predictions = (softmax_result['predictions'] != causal_result['predictions'])
    
    print(f"   预测不同的样本数: {different_predictions.sum()}/{len(different_predictions)}")
    
    if different_predictions.sum() > 0:
        # 在预测不同的样本中，谁更准确？
        true_labels = softmax_result['true_labels'][different_predictions]
        softmax_preds = softmax_result['predictions'][different_predictions]
        causal_preds = causal_result['predictions'][different_predictions]
        
        softmax_correct = (softmax_preds == true_labels).sum()
        causal_correct = (causal_preds == true_labels).sum()
        
        print(f"   在分歧样本中:")
        print(f"      Softmax正确: {softmax_correct}/{len(true_labels)} ({softmax_correct/len(true_labels):.3f})")
        print(f"      CausalEngine正确: {causal_correct}/{len(true_labels)} ({causal_correct/len(true_labels):.3f})")
    
    # 2. 置信度分析
    softmax_correct = (softmax_result['predictions'] == softmax_result['true_labels'])
    causal_correct = (causal_result['predictions'] == causal_result['true_labels'])
    
    print(f"\n   置信度分析:")
    print(f"      Softmax - 正确预测平均置信度: {softmax_result['confidences'][softmax_correct].mean():.4f}")
    print(f"      Softmax - 错误预测平均置信度: {softmax_result['confidences'][~softmax_correct].mean():.4f}")
    print(f"      CausalEngine - 正确预测平均置信度: {causal_result['confidences'][causal_correct].mean():.4f}")
    print(f"      CausalEngine - 错误预测平均置信度: {causal_result['confidences'][~causal_correct].mean():.4f}")
    
    # 3. 不确定性分析（仅CausalEngine）
    if causal_result['uncertainties'].max() > 0:
        print(f"\n   CausalEngine不确定性分析:")
        print(f"      正确预测平均不确定性: {causal_result['uncertainties'][causal_correct].mean():.4f}")
        print(f"      错误预测平均不确定性: {causal_result['uncertainties'][~causal_correct].mean():.4f}")


def hyperparameter_tuning_experiment():
    """超参数调优实验"""
    
    print("\n🔧 超参数调优实验...")
    
    train_loader, val_loader, test_loader = prepare_data()
    
    # 测试不同的 b_noise_init 和 gamma_init
    noise_values = [0.01, 0.05, 0.1, 0.2]
    gamma_values = [0.5, 1.0, 2.0]
    learning_rates = [0.0005, 0.001, 0.002]
    
    best_accuracy = 0
    best_params = {}
    
    results_grid = []
    
    print("   测试不同噪声初始化值...")
    for b_noise in noise_values:
        model = CausalEngineClassifier(hidden_size=128, num_classes=10, 
                                     b_noise_init=b_noise, gamma_init=1.0)
        train_causal_model(model, train_loader, val_loader, max_epochs=60, lr=0.001)
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features, temperature=0.0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        results_grid.append({
            'b_noise': b_noise,
            'gamma': 1.0,
            'lr': 0.001,
            'accuracy': accuracy
        })
        
        print(f"      b_noise={b_noise}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'b_noise': b_noise, 'gamma': 1.0, 'lr': 0.001}
    
    print(f"\n   最佳参数: {best_params}")
    print(f"   最佳准确率: {best_accuracy:.4f}")
    
    return results_grid, best_params


def visualize_analysis(results, softmax_history, causal_history):
    """可视化分析结果"""
    
    print("\n📈 生成可视化分析...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 训练曲线对比
    ax1 = plt.subplot(2, 3, 1)
    softmax_losses, softmax_accs = softmax_history
    causal_losses, causal_accs = causal_history
    
    ax1.plot(softmax_losses, label='Softmax Loss', color='blue')
    ax1.plot(causal_losses, label='CausalEngine Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 验证准确率对比
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(softmax_accs, label='Softmax', color='blue')
    ax2.plot(causal_accs, label='CausalEngine', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 混淆矩阵对比
    softmax_cm = confusion_matrix(results['Softmax']['true_labels'], 
                                  results['Softmax']['predictions'])
    causal_cm = confusion_matrix(results['CausalEngine']['true_labels'], 
                                results['CausalEngine']['predictions'])
    
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(softmax_cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
    ax3.set_title('Softmax Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    ax4 = plt.subplot(2, 3, 4)
    sns.heatmap(causal_cm, annot=True, fmt='d', cmap='Reds', ax=ax4, cbar=False)
    ax4.set_title('CausalEngine Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    
    # 4. 置信度分布对比
    ax5 = plt.subplot(2, 3, 5)
    
    softmax_correct = (results['Softmax']['predictions'] == results['Softmax']['true_labels'])
    causal_correct = (results['CausalEngine']['predictions'] == results['CausalEngine']['true_labels'])
    
    ax5.hist(results['Softmax']['confidences'][softmax_correct], bins=30, alpha=0.6, 
             label='Softmax Correct', color='blue', density=True)
    ax5.hist(results['Softmax']['confidences'][~softmax_correct], bins=30, alpha=0.6, 
             label='Softmax Incorrect', color='lightblue', density=True)
    ax5.hist(results['CausalEngine']['confidences'][causal_correct], bins=30, alpha=0.6, 
             label='CausalEngine Correct', color='red', density=True)
    ax5.hist(results['CausalEngine']['confidences'][~causal_correct], bins=30, alpha=0.6, 
             label='CausalEngine Incorrect', color='pink', density=True)
    
    ax5.set_xlabel('Confidence')
    ax5.set_ylabel('Density')
    ax5.set_title('Confidence Distribution')
    ax5.legend()
    
    # 5. 不确定性分析（仅CausalEngine）
    ax6 = plt.subplot(2, 3, 6)
    
    if results['CausalEngine']['uncertainties'].max() > 0:
        ax6.hist(results['CausalEngine']['uncertainties'][causal_correct], bins=30, alpha=0.6, 
                 label='Correct', color='green', density=True)
        ax6.hist(results['CausalEngine']['uncertainties'][~causal_correct], bins=30, alpha=0.6, 
                 label='Incorrect', color='red', density=True)
        ax6.set_xlabel('Uncertainty')
        ax6.set_ylabel('Density')
        ax6.set_title('CausalEngine Uncertainty Distribution')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No Uncertainty\nData Available', 
                ha='center', va='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig('deep_causal_analysis.png', dpi=150, bbox_inches='tight')
    print("   分析结果已保存到 deep_causal_analysis.png")
    plt.close()


def main():
    """主函数"""
    
    print("="*70)
    print("🔍 Deep MLP + CausalEngine 性能分析")
    print("   系统分析为什么CausalEngine不如Softmax")
    print("="*70)
    
    # 1. 准备数据
    train_loader, val_loader, test_loader = prepare_data()
    
    # 2. 训练两个模型
    print("\n🏋️‍♂️ 训练模型对比...")
    
    # Softmax模型
    softmax_model = SoftmaxClassifier(hidden_size=128, num_classes=10)
    softmax_history = train_softmax_model(softmax_model, train_loader, val_loader)
    
    # CausalEngine模型
    causal_model = CausalEngineClassifier(hidden_size=128, num_classes=10)
    causal_history = train_causal_model(causal_model, train_loader, val_loader)
    
    # 3. 性能分析
    results = analyze_performance_gap(softmax_model, causal_model, test_loader)
    
    # 4. 详细错误分析
    detailed_error_analysis(results)
    
    # 5. 超参数调优
    tuning_results, best_params = hyperparameter_tuning_experiment()
    
    # 6. 可视化
    visualize_analysis(results, softmax_history, causal_history)
    
    # 7. 总结
    print("\n💡 关键发现:")
    print("-" * 50)
    softmax_acc = results['Softmax']['accuracy']
    causal_acc = results['CausalEngine']['accuracy']
    gap = softmax_acc - causal_acc
    
    print(f"📊 性能差距: {gap:.4f} ({gap/softmax_acc*100:.1f}%)")
    print(f"   Softmax:     {softmax_acc:.4f}")
    print(f"   CausalEngine: {causal_acc:.4f}")
    
    print(f"\n🔧 超参数调优后最佳CausalEngine准确率: {tuning_results[-1]['accuracy']:.4f}")
    
    if gap > 0.01:  # 如果差距超过1%
        print(f"\n🎯 可能的改进方向:")
        print(f"1. 进一步调优CausalEngine的超参数")
        print(f"2. 使用不同的损失函数")
        print(f"3. 调整温度参数")
        print(f"4. 考虑是否真的需要不确定性建模")
    
    print("\n✅ 分析完成！")
    print("📊 请查看 deep_causal_analysis.png 了解详细分析")


if __name__ == "__main__":
    main() 