#!/usr/bin/env python3
"""
CausalEngine 表格数据快速测试 - 修复版

修复了关键问题：CausalEngine 使用 OvR 分类，需要二元交叉熵损失！
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


class FixedCausalClassifier(nn.Module):
    """修复的 CausalEngine 分类器 - 正确处理 OvR"""
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        # 根据输入大小调整网络深度
        if input_size <= 4:
            hidden_sizes = [32, 16]
        elif input_size <= 20:
            hidden_sizes = [64, 32]
        else:
            hidden_sizes = [128, 64, 32]
        
        # 特征编码器 - 移除 BatchNorm 避免小批次问题
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),  # 使用 LayerNorm 替代 BatchNorm
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.encoder = nn.Sequential(*layers)
        
        # CausalEngine - 注意这里使用 OvR 分类
        self.causal_engine = CausalEngine(
            hidden_size=prev_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
        
        self.num_classes = num_classes
    
    def forward(self, x, temperature=1.0):
        features = self.encoder(x)
        output = self.causal_engine(
            features.unsqueeze(1),
            temperature=temperature,
            do_sample=False,
            return_dict=True
        )
        # 输出已经是概率了，每个类别独立的概率 [0, 1]
        return output['output'].squeeze(1)


def ovr_loss(probs, labels, num_classes):
    """
    One-vs-Rest 损失函数
    
    Args:
        probs: [batch_size, num_classes] 每个类别的独立概率
        labels: [batch_size] 真实标签 (整数)
        num_classes: 类别数
    
    Returns:
        loss: 二元交叉熵损失
    """
    # 将标签转换为 one-hot 编码
    targets = F.one_hot(labels, num_classes=num_classes).float()
    
    # 计算二元交叉熵损失
    # 注意：probs 已经是概率，不是 logits
    loss = F.binary_cross_entropy(probs, targets, reduction='mean')
    
    return loss


def ovr_predict(probs):
    """
    OvR 预测：选择概率最高的类别
    
    Args:
        probs: [batch_size, num_classes] 每个类别的独立概率
    
    Returns:
        predictions: [batch_size] 预测标签
    """
    return torch.argmax(probs, dim=1)


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


def train_causal_ovr(model, train_loader, val_loader, num_classes, epochs=50):
    """使用正确的 OvR 损失训练 CausalEngine"""
    
    # 使用 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    print(f"      🔧 使用 OvR 二元交叉熵损失训练...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            # 动态温度调整
            if epoch < epochs // 3:
                temperature = 1.2
            elif epoch < 2 * epochs // 3:
                temperature = 1.0
            else:
                temperature = 0.8
            
            # 前向传播
            probs = model(features, temperature=temperature)
            
            # 关键修复：使用 OvR 损失函数
            loss = ovr_loss(probs, labels, num_classes)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # 验证阶段
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    probs = model(features, temperature=0.1)
                    predictions = ovr_predict(probs)  # 使用 OvR 预测
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
            
            val_acc = correct / total
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"      Epoch {epoch:3d}: Loss={total_loss/len(train_loader):.4f}, Val_Acc={val_acc:.4f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"      提前停止在第 {epoch} 轮，最佳验证准确率: {best_val_acc:.4f}")
                break
    
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
    
    # 数据划分
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"   训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    results = {}
    
    # 1. 修复后的 CausalEngine (OvR)
    print("\n🔧 训练修复版 CausalEngine (OvR)...")
    
    # 调整批次大小
    batch_size = min(32, len(X_train) // 4) if len(X_train) > 32 else 8
    
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    test_dataset = SimpleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    causal_model = FixedCausalClassifier(X_train.shape[1], dataset_info['num_classes'])
    
    val_acc = train_causal_ovr(
        causal_model, train_loader, val_loader, 
        dataset_info['num_classes'], epochs=60
    )
    
    # 测试评估
    causal_model.eval()
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for features, _ in test_loader:
            probs = causal_model(features, temperature=0.1)
            preds = ovr_predict(probs)  # 使用 OvR 预测
            max_probs = torch.max(probs, dim=1)[0]  # 最大概率作为置信度
            
            predictions.extend(preds.numpy())
            confidences.extend(max_probs.numpy())
    
    causal_acc = accuracy_score(y_test, predictions)
    avg_confidence = np.mean(confidences)
    
    results['CausalEngine (OvR)'] = causal_acc
    print(f"   验证准确率: {val_acc:.4f}")
    print(f"   测试准确率: {causal_acc:.4f}")
    print(f"   平均最大概率: {avg_confidence:.4f}")
    
    # 2. 逻辑回归
    print("\n📊 训练逻辑回归...")
    lr = LogisticRegression(random_state=42, max_iter=2000, C=1.0)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    results['Logistic Regression'] = lr_acc
    print(f"   测试准确率: {lr_acc:.4f}")
    
    # 3. 随机森林
    print("\n🌲 训练随机森林...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    results['Random Forest'] = rf_acc
    print(f"   测试准确率: {rf_acc:.4f}")
    
    # 4. SVM
    print("\n🎯 训练 SVM...")
    svm = SVC(random_state=42, C=1.0)
    svm.fit(X_train, y_train)
    svm_acc = svm.score(X_test, y_test)
    results['SVM'] = svm_acc
    print(f"   测试准确率: {svm_acc:.4f}")
    
    return results


def visualize_results(all_results):
    """可视化修复后的结果"""
    
    print("\n📈 生成修复后结果可视化...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    datasets = list(all_results.keys())
    methods = ['CausalEngine (OvR)', 'Logistic Regression', 'Random Forest', 'SVM']
    
    # 1. 准确率对比
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, method in enumerate(methods):
        accuracies = [all_results[d][method] for d in datasets]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=method, alpha=0.8)
        
        if 'CausalEngine' in method:
            for bar in bars:
                bar.set_color('red')
                bar.set_alpha(0.9)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Fixed CausalEngine (OvR) vs Traditional Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 2. 修复前后对比
    ax = axes[1]
    
    # 假设的修复前性能
    old_perfs = {
        'Iris': 0.8444,
        'Wine': 0.9630,
        'Breast Cancer': 0.9649,
        'Digits (500)': 0.8733,
        'Synthetic': 0.6967
    }
    
    improvements = []
    labels = []
    
    for dataset in datasets:
        if dataset in old_perfs:
            old = old_perfs[dataset]
            new = all_results[dataset]['CausalEngine (OvR)']
            improvement = (new - old) / old * 100
            improvements.append(improvement)
            labels.append(dataset)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(range(len(labels)), improvements, color=colors, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement After OvR Fix')
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, 
                height + (1 if height > 0 else -1),
                f'{imp:.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top', weight='bold')
    
    # 3. 方法平均性能
    ax = axes[2]
    method_avg = {method: np.mean([all_results[d][method] for d in datasets]) 
                  for method in methods}
    
    bars = ax.bar(methods, method_avg.values(), alpha=0.7)
    bars[0].set_color('red')  # CausalEngine
    
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Average Performance - Fixed Version')
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, (method, acc) in zip(bars, method_avg.items()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', weight='bold')
    
    # 4. 排名分析
    ax = axes[3]
    causal_ranks = []
    
    for dataset in datasets:
        results_list = [(method, acc) for method, acc in all_results[dataset].items()]
        results_list.sort(key=lambda x: x[1], reverse=True)
        rank = next(i for i, (m, _) in enumerate(results_list) if 'CausalEngine' in m) + 1
        causal_ranks.append(rank)
    
    bars = ax.bar(range(len(datasets)), causal_ranks, alpha=0.7)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45)
    ax.set_ylabel('Rank (1=Best)')
    ax.set_title('CausalEngine (OvR) Ranking per Dataset')
    ax.set_ylim(0.5, 4.5)
    ax.invert_yaxis()
    
    # 添加颜色编码
    for i, (bar, rank) in enumerate(zip(bars, causal_ranks)):
        ax.text(bar.get_x() + bar.get_width()/2, rank,
                str(rank), ha='center', va='center', weight='bold')
        
        if rank == 1:
            bar.set_color('gold')
        elif rank == 2:
            bar.set_color('silver')
        elif rank == 3:
            bar.set_color('#CD7F32')
        else:
            bar.set_color('gray')
    
    # 5. 问题修复总结
    ax = axes[4]
    wins = sum(1 for rank in causal_ranks if rank == 1)
    top2 = sum(1 for rank in causal_ranks if rank <= 2)
    
    ax.text(0.5, 0.7, '🔧 关键修复:', ha='center', fontsize=14, weight='bold')
    ax.text(0.5, 0.6, '✅ 使用 OvR 二元交叉熵损失', ha='center', fontsize=11)
    ax.text(0.5, 0.55, '✅ 正确的 OvR 预测方法', ha='center', fontsize=11)
    ax.text(0.5, 0.5, '✅ 不再使用错误的 softmax', ha='center', fontsize=11)
    
    ax.text(0.5, 0.35, f'📊 修复后效果:', ha='center', fontsize=14, weight='bold')
    ax.text(0.5, 0.25, f'获胜次数: {wins}/{len(datasets)}', ha='center', fontsize=12)
    ax.text(0.5, 0.2, f'前二名次数: {top2}/{len(datasets)}', ha='center', fontsize=12)
    ax.text(0.5, 0.15, f'平均排名: {np.mean(causal_ranks):.1f}/4', ha='center', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('OvR Fix Impact Summary')
    
    # 6. 修复前后对比表
    ax = axes[5]
    ax.text(0.5, 0.9, 'Before vs After Fix', ha='center', fontsize=14, weight='bold')
    
    table_data = []
    headers = ['Dataset', 'Before', 'After', 'Δ%']
    
    for dataset in datasets:
        if dataset in old_perfs:
            old = old_perfs[dataset]
            new = all_results[dataset]['CausalEngine (OvR)']
            delta = (new - old) / old * 100
            table_data.append([dataset, f'{old:.3f}', f'{new:.3f}', f'{delta:+.1f}%'])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('tabular_quick_test_fixed_results.png', dpi=150, bbox_inches='tight')
    print("   修复结果已保存到 tabular_quick_test_fixed_results.png")
    plt.close()


def print_summary(all_results):
    """打印修复后的总结"""
    
    print("\n" + "="*60)
    print("📊 修复版测试总结 - 正确的 OvR 损失函数")
    print("="*60)
    
    methods = ['CausalEngine (OvR)', 'Logistic Regression', 'Random Forest', 'SVM']
    method_wins = {m: 0 for m in methods}
    
    for dataset, results in all_results.items():
        best_method = max(results.items(), key=lambda x: x[1])[0]
        method_wins[best_method] += 1
        
        print(f"\n{dataset}:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (method, acc) in enumerate(sorted_results):
            if i == 0:
                marker = "🥇"
            elif i == 1:
                marker = "🥈" 
            elif i == 2:
                marker = "🥉"
            else:
                marker = "  "
            
            if 'CausalEngine' in method:
                print(f"  {marker} {method:20s}: {acc:.4f} ⭐")
            else:
                print(f"  {marker} {method:20s}: {acc:.4f}")
    
    print("\n🏆 获胜统计:")
    for method, wins in sorted(method_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"   {method}: {wins} 次")
    
    # CausalEngine 详细分析
    print("\n🔍 CausalEngine 修复效果分析:")
    causal_ranks = []
    causal_accs = []
    
    for dataset, results in all_results.items():
        sorted_methods = sorted(results.items(), key=lambda x: x[1], reverse=True)
        rank = next(i for i, (m, _) in enumerate(sorted_methods) if 'CausalEngine' in m) + 1
        causal_ranks.append(rank)
        causal_accs.append(results['CausalEngine (OvR)'])
    
    wins = sum(1 for rank in causal_ranks if rank == 1)
    top2 = sum(1 for rank in causal_ranks if rank <= 2)
    
    print(f"   平均排名: {np.mean(causal_ranks):.1f}/4 (修复前: 3.6)")
    print(f"   获胜次数: {wins}/{len(causal_ranks)} (修复前: 0)")
    print(f"   前二名次数: {top2}/{len(causal_ranks)}")
    print(f"   平均准确率: {np.mean(causal_accs):.4f}")
    
    print("\n🔧 关键修复内容:")
    print("   ✅ 使用二元交叉熵损失 (BCE) 替代交叉熵")
    print("   ✅ 正确处理 OvR 输出格式")
    print("   ✅ 使用 argmax 而非 softmax")
    print("   ✅ 将标签转换为 one-hot 编码")
    print("   ✅ 理解每个类别独立概率的含义")


def main():
    """主函数"""
    
    print("="*60)
    print("🚀 CausalEngine 表格数据快速测试 - 修复版")
    print("   修复关键问题：使用正确的 OvR 损失函数！")
    print("="*60)
    
    print("\n🔧 关键修复:")
    print("   • CausalEngine 使用 OvR (One-vs-Rest) 分类")
    print("   • 每个类别输出独立概率，不使用 softmax")
    print("   • 损失函数：二元交叉熵 (BCE) 而非交叉熵")
    print("   • 预测方法：argmax 选择最大概率类别")
    
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
    
    print("\n✅ 修复版测试完成！")
    print("📊 请查看 tabular_quick_test_fixed_results.png")
    print("\n💡 如果效果显著改善，说明之前的问题确实是损失函数错误！")


if __name__ == "__main__":
    main() 