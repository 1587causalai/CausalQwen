#!/usr/bin/env python3
"""
CausalEngine 消融实验

完美的对照实验：
- A组：完整 CausalEngine (loc + scale + OvR + BCE)  
- B组：简化版本 (仅loc + Softmax + CrossEntropy)
- 网络结构完全相同，唯一差异是是否使用因果推理的不确定性建模

🔧 修复了关键训练问题：
1. ✅ 保存和恢复最佳模型权重  
2. ✅ 每个epoch都验证
3. ✅ 固定温度避免训练不稳定
4. ✅ 详细训练日志
5. ✅ 增加patience提高训练充分性
6. ✅ 添加verbose模式显示网络结构和配置
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy
import warnings
warnings.filterwarnings('ignore')

# 导入 CausalEngine
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')
from causal_engine import CausalEngine


class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SharedEncoder(nn.Module):
    """共享的特征编码器 - 确保完全相同的网络结构"""
    
    def __init__(self, input_size, output_size=32, verbose=False):
        super().__init__()
        
        # 根据输入大小调整网络深度
        if input_size <= 4:
            hidden_sizes = [32, 16]
        elif input_size <= 20:
            hidden_sizes = [64, 32]
        else:
            hidden_sizes = [128, 64, 32]
        
        # 特征编码器
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # 最终输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.encoder = nn.Sequential(*layers)
        self.output_size = output_size
        
        if verbose:
            print(f"      📐 SharedEncoder Architecture:")
            print(f"         输入维度: {input_size}")
            print(f"         隐藏层维度: {hidden_sizes}")
            print(f"         输出维度: {output_size}")
            print(f"         总参数数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        return self.encoder(x)


class FullCausalClassifier(nn.Module):
    """完整的 CausalEngine 分类器 (A组) - loc + scale + OvR + BCE"""
    
    def __init__(self, input_size, num_classes, verbose=False):
        super().__init__()
        
        self.encoder = SharedEncoder(input_size, output_size=32, verbose=verbose)
        
        # 完整的 CausalEngine
        self.causal_engine = CausalEngine(
            hidden_size=self.encoder.output_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
        
        self.num_classes = num_classes
        
        if verbose:
            print(f"      🔧 FullCausalClassifier Configuration:")
            print(f"         输入维度: {input_size}")
            print(f"         类别数: {num_classes}")
            print(f"         编码器输出维度: {self.encoder.output_size}")
            print(f"         CausalEngine 配置:")
            print(f"           - hidden_size: {self.encoder.output_size}")
            print(f"           - vocab_size: {num_classes}")
            print(f"           - activation_modes: classification")
            print(f"           - b_noise_init: 0.1")
            print(f"           - gamma_init: 1.0")
            total_params = sum(p.numel() for p in self.parameters())
            encoder_params = sum(p.numel() for p in self.encoder.parameters())
            causal_params = sum(p.numel() for p in self.causal_engine.parameters())
            print(f"         参数统计:")
            print(f"           - 编码器参数: {encoder_params:,}")
            print(f"           - CausalEngine参数: {causal_params:,}")
            print(f"           - 总参数: {total_params:,}")
    
    def forward(self, x, temperature=1.0, return_components=False):
        features = self.encoder(x)
        output = self.causal_engine(
            features.unsqueeze(1),
            temperature=temperature,
            do_sample=False,
            return_dict=True
        )
        
        if return_components:
            return {
                'probs': output['output'].squeeze(1),
                'loc_S': output['loc_S'].squeeze(1),
                'scale_S': output['scale_S'].squeeze(1)
            }
        else:
            return output['output'].squeeze(1)


class SimplifiedCausalClassifier(nn.Module):
    """简化的 CausalEngine 分类器 (B组) - 仅loc + Softmax + CrossEntropy"""
    
    def __init__(self, input_size, num_classes, verbose=False):
        super().__init__()
        
        # 使用完全相同的编码器！
        self.encoder = SharedEncoder(input_size, output_size=32, verbose=verbose)
        
        # 创建 CausalEngine 但只使用 loc 部分
        self.causal_engine = CausalEngine(
            hidden_size=self.encoder.output_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
        
        self.num_classes = num_classes
        
        if verbose:
            print(f"      📊 SimplifiedCausalClassifier Configuration:")
            print(f"         输入维度: {input_size}")
            print(f"         类别数: {num_classes}")
            print(f"         编码器输出维度: {self.encoder.output_size}")
            print(f"         关键差异: 仅使用 loc_S，忽略 scale_S")
            print(f"         输出方式: Softmax 归一化")
            print(f"         损失函数: 标准交叉熵")
            total_params = sum(p.numel() for p in self.parameters())
            print(f"         总参数: {total_params:,} (与完整版相同)")
    
    def forward(self, x, temperature=1.0, return_components=False):
        features = self.encoder(x)
        output = self.causal_engine(
            features.unsqueeze(1),
            temperature=temperature,
            do_sample=False,
            return_dict=True
        )
        
        # 关键：只使用 loc_S，忽略 scale_S！
        loc_S = output['loc_S'].squeeze(1)  # [batch_size, num_classes]
        
        # 应用 softmax 得到传统的概率分布
        logits = loc_S  # 将 loc 当作 logits
        probs = F.softmax(logits, dim=1)
        
        if return_components:
            return {
                'probs': probs,
                'logits': logits,
                'loc_S': loc_S,
                'scale_S': output['scale_S'].squeeze(1)
            }
        else:
            return probs


def full_causal_loss(probs, labels, num_classes):
    """完整 CausalEngine 的 OvR 损失函数"""
    targets = F.one_hot(labels, num_classes=num_classes).float()
    return F.binary_cross_entropy(probs, targets, reduction='mean')


def simplified_causal_loss(probs, labels):
    """简化版本的标准交叉熵损失"""
    return F.cross_entropy(torch.log(probs + 1e-8), labels)


def train_model_fixed(model, train_loader, val_loader, loss_fn, model_name, epochs=100, verbose=False):
    """🔧 修复的训练函数：保存最佳模型权重 + 每轮验证"""
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=12
    )
    
    best_val_acc = 0
    best_model_state = None  # 🔧 关键修复：保存最佳模型状态
    patience = 25  # 🔧 增加patience确保训练充分
    patience_counter = 0
    
    # 🔧 修复：使用固定温度提高训练稳定性
    fixed_temperature = 1.0
    
    if verbose:
        print(f"      🚀 训练配置详情:")
        print(f"         优化器: AdamW (lr=0.001, weight_decay=1e-4)")
        print(f"         学习率调度: ReduceLROnPlateau (patience=12)")
        print(f"         早停: patience={patience}")
        print(f"         最大轮数: {epochs}")
        print(f"         固定温度: {fixed_temperature}")
        print(f"         梯度裁剪: max_norm=1.0")
    
    print(f"      🔧 Training {model_name} (FIXED - best model saving)...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            # 🔧 修复：使用固定温度，避免训练不稳定
            if 'Full' in model_name:
                probs = model(features, temperature=fixed_temperature)
                loss = loss_fn(probs, labels, model.num_classes)
            else:
                probs = model(features, temperature=fixed_temperature)
                loss = loss_fn(probs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # 🔧 关键修复：每个epoch都验证，精确早停
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                probs = model(features, temperature=fixed_temperature)
                predictions = torch.argmax(probs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        
        val_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # 🔧 关键修复：当验证准确率提升时保存最佳模型状态
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())  # 🔧 保存权重！
            patience_counter = 0
            improvement_marker = "⭐ NEW BEST"
        else:
            patience_counter += 1
            improvement_marker = ""
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 详细日志
        if verbose or epoch % 10 == 0 or epoch == epochs - 1 or improvement_marker:
            lr_info = f", LR: {new_lr:.6f}" if old_lr != new_lr else ""
            print(f"      Epoch {epoch:3d}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}{lr_info} {improvement_marker}")
        
        # 早停
        if patience_counter >= patience:
            print(f"      Early stop at epoch {epoch}, best val acc: {best_val_acc:.4f}")
            break
    
    # 🔧 关键修复：恢复最佳模型权重！
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"      ✅ Restored BEST model weights (val_acc={best_val_acc:.4f})")
    else:
        print(f"      ⚠️ No improvement found, using final weights")
    
    return best_val_acc


def run_ablation_experiment(verbose=False):
    """运行完整的消融实验"""
    
    print("="*70)
    print("🔬 CausalEngine 消融实验 - 最终版")
    print("   🔧 修复了所有训练问题：")
    print("   ✅ 保存和恢复最佳模型权重")
    print("   ✅ 每个epoch验证，精确早停")  
    print("   ✅ 固定温度提高训练稳定性")
    print("   ✅ 增加patience确保训练充分")
    if verbose:
        print("   ✅ 详细输出网络结构和配置")
    print("="*70)
    
    # 加载数据集
    datasets_info = [
        {
            'name': 'Iris',
            'loader': lambda: datasets.load_iris(),
            'classes': 3
        },
        {
            'name': 'Wine', 
            'loader': lambda: datasets.load_wine(),
            'classes': 3
        },
        {
            'name': 'Breast Cancer',
            'loader': lambda: datasets.load_breast_cancer(),
            'classes': 2
        },
        {
            'name': 'Digits',
            'loader': lambda: datasets.load_digits(),
            'classes': 10
        }
    ]
    
    results = []
    
    for dataset_info in datasets_info:
        print(f"\n{'='*60}")
        print(f"📊 数据集: {dataset_info['name']}")
        print(f"{'='*60}")
        
        # 加载数据
        data = dataset_info['loader']()
        X, y = data.data, data.target
        
        # 对于 Digits，采样以加快速度
        if dataset_info['name'] == 'Digits':
            indices = np.random.choice(len(X), 800, replace=False)
            X, y = X[indices], y[indices]
        
        print(f"   样本数: {len(X)}, 特征数: {X.shape[1]}, 类别数: {dataset_info['classes']}")
        
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
        
        # 创建数据加载器
        batch_size = min(32, len(X_train) // 4) if len(X_train) > 32 else 8
        
        train_dataset = SimpleDataset(X_train, y_train)
        val_dataset = SimpleDataset(X_val, y_val)
        test_dataset = SimpleDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        if verbose:
            print(f"   数据划分详情:")
            print(f"     训练集: {len(X_train)} 样本")
            print(f"     验证集: {len(X_val)} 样本") 
            print(f"     测试集: {len(X_test)} 样本")
            print(f"     批次大小: {batch_size}")
        
        # 创建模型
        print("\n🔧 创建模型...")
        full_model = FullCausalClassifier(X_train.shape[1], dataset_info['classes'], verbose=verbose)
        print("   " + "="*50)
        simplified_model = SimplifiedCausalClassifier(X_train.shape[1], dataset_info['classes'], verbose=verbose)
        
        print("   ✅ 网络结构完全相同")
        print("   🔧 使用修复的训练函数")
        
        # 🔧 使用修复的训练函数
        print("\n🚀 开始修复版训练...")
        
        full_val_acc = train_model_fixed(
            full_model, train_loader, val_loader, full_causal_loss, 
            "Full CausalEngine", epochs=100, verbose=verbose
        )
        
        simplified_val_acc = train_model_fixed(
            simplified_model, train_loader, val_loader, simplified_causal_loss,
            "Simplified Version", epochs=100, verbose=verbose
        )
        
        # 测试评估
        print(f"\n📊 测试集评估 - {dataset_info['name']}:")
        
        # 完整模型测试
        full_model.eval()
        full_predictions = []
        full_confidences = []
        with torch.no_grad():
            for features, _ in test_loader:
                probs = full_model(features, temperature=1.0)
                preds = torch.argmax(probs, dim=1)
                confs = torch.max(probs, dim=1)[0]
                full_predictions.extend(preds.numpy())
                full_confidences.extend(confs.numpy())
        
        full_test_acc = accuracy_score(y_test, full_predictions)
        
        # 简化模型测试
        simplified_model.eval()
        simp_predictions = []
        simp_confidences = []
        with torch.no_grad():
            for features, _ in test_loader:
                probs = simplified_model(features, temperature=1.0)
                preds = torch.argmax(probs, dim=1)
                confs = torch.max(probs, dim=1)[0]
                simp_predictions.extend(preds.numpy())
                simp_confidences.extend(confs.numpy())
        
        simp_test_acc = accuracy_score(y_test, simp_predictions)
        
        # 计算一致性
        agreement = np.mean(np.array(full_predictions) == np.array(simp_predictions))
        
        print(f"   完整版 - 验证: {full_val_acc:.4f}, 测试: {full_test_acc:.4f}, 置信度: {np.mean(full_confidences):.4f}")
        print(f"   简化版 - 验证: {simplified_val_acc:.4f}, 测试: {simp_test_acc:.4f}, 置信度: {np.mean(simp_confidences):.4f}")
        print(f"   测试集差异: {full_test_acc - simp_test_acc:+.4f}")
        print(f"   预测一致性: {agreement:.4f}")
        
        results.append({
            'dataset': dataset_info['name'],
            'full_val_acc': full_val_acc,
            'simplified_val_acc': simplified_val_acc,
            'full_test_acc': full_test_acc,
            'simplified_test_acc': simp_test_acc,
            'test_difference': full_test_acc - simp_test_acc,
            'full_confidence': np.mean(full_confidences),
            'simplified_confidence': np.mean(simp_confidences),
            'agreement': agreement
        })
    
    return results


def visualize_ablation_results(results):
    """可视化消融实验结果"""
    
    print("\n📈 生成消融实验可视化...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    datasets = [r['dataset'] for r in results]
    full_accs = [r['full_test_acc'] for r in results]
    simplified_accs = [r['simplified_test_acc'] for r in results]
    differences = [r['test_difference'] for r in results]
    
    # 1. 准确率对比
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, full_accs, width, label='Full CausalEngine', 
                   color='red', alpha=0.8)
    bars2 = ax.bar(x + width/2, simplified_accs, width, label='Simplified (loc+softmax)',
                   color='blue', alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Ablation: Full vs Simplified CausalEngine')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 性能差异
    ax = axes[0, 1]
    colors = ['green' if diff > 0 else 'red' for diff in differences]
    bars = ax.bar(datasets, differences, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Test Accuracy Difference (Full - Simplified)')
    ax.set_title('Performance Gain from Full CausalEngine')
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, 
                height + (0.005 if height > 0 else -0.005),
                f'{diff:+.3f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=10, weight='bold')
    
    # 3. 置信度对比
    ax = axes[1, 0]
    full_confs = [r['full_confidence'] for r in results]
    simp_confs = [r['simplified_confidence'] for r in results]
    
    x = np.arange(len(datasets))
    bars1 = ax.bar(x - width/2, full_confs, width, label='Full CausalEngine', 
                   color='red', alpha=0.8)
    bars2 = ax.bar(x + width/2, simp_confs, width, label='Simplified',
                   color='blue', alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Confidence')
    ax.set_title('Confidence Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. 实验总结
    ax = axes[1, 1]
    
    avg_improvement = np.mean(differences)
    positive_improvements = sum(1 for d in differences if d > 0)
    avg_full_acc = np.mean(full_accs)
    avg_simp_acc = np.mean(simplified_accs)
    
    summary_text = f"""Final Ablation Study Summary

Average Test Performance:
• Full Version: {avg_full_acc:.3f}
• Simplified: {avg_simp_acc:.3f}
• Avg Improvement: {avg_improvement:+.3f}

Win Statistics:
• Full Version Wins: {positive_improvements}/{len(results)}
• Simplified Wins: {len(results)-positive_improvements}/{len(results)}

Key Findings:
{"✅ Causal uncertainty modeling effective" if avg_improvement > 0.01 else "⚠️ Simplified version competitive"}
{"✅ OvR strategy > Softmax" if positive_improvements > len(results)/2 else "⚠️ Softmax competitive"}

Core Insight:
CausalEngine improvement comes
{"mainly from causal reasoning" if avg_improvement > 0.02 else "partly from architecture"}"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('causal_ablation_final_results.png', dpi=150, bbox_inches='tight')
    print("   最终消融实验结果已保存到 causal_ablation_final_results.png")
    plt.close()


def print_final_conclusions(results):
    """打印最终实验结论"""
    
    print("\n" + "="*70)
    print("🎯 最终消融实验结论")
    print("="*70)
    
    test_differences = [r['test_difference'] for r in results]
    avg_improvement = np.mean(test_differences)
    positive_improvements = sum(1 for d in test_differences if d > 0)
    
    print(f"\n📊 最终整体结果:")
    print(f"   平均测试集性能提升: {avg_improvement:+.4f}")
    print(f"   完整版获胜数据集: {positive_improvements}/{len(results)}")
    
    print(f"\n📈 各数据集详细对比:")
    for result in results:
        status = "🟢" if result['test_difference'] > 0 else "🔴"
        print(f"   {status} {result['dataset']:15s}: {result['test_difference']:+.4f}")
        print(f"       验证集 - 完整: {result['full_val_acc']:.4f}, 简化: {result['simplified_val_acc']:.4f}")
        print(f"       测试集 - 完整: {result['full_test_acc']:.4f}, 简化: {result['simplified_test_acc']:.4f}")
        print(f"       预测一致性: {result['agreement']:.4f}")
        print()
    
    print(f"\n🔍 最终科学结论:")
    if avg_improvement > 0.02:
        print("   ✅ 因果推理的不确定性建模显著有效")
        print("   ✅ OvR + scale_S 策略明显优于简化版本")
        print("   ✅ CausalEngine 的性能提升主要来自因果推理机制")
    elif avg_improvement > 0.005:
        print("   ⚠️ 因果推理的不确定性建模轻微有效")
        print("   ⚠️ 性能提升部分来自因果推理机制")
        print("   ⚠️ 网络架构也有贡献")
    else:
        print("   ❌ 简化版本表现相当或更好")
        print("   ❌ 因果推理的不确定性建模在这些任务上效果有限")
        print("   ❌ 可能需要调整参数或适用场景")
    
    print(f"\n🔧 实验验证:")
    print("   ✅ 最佳模型权重正确保存和恢复")
    print("   ✅ 每轮验证确保精确早停")
    print("   ✅ 固定温度提高训练稳定性")
    print("   ✅ 增加patience确保训练充分")
    print("   ✅ 严格控制网络架构变量")
    print("   ✅ 清晰量化因果推理机制的贡献")


def main(verbose=False):
    """主函数"""
    
    # 运行消融实验
    results = run_ablation_experiment(verbose=verbose)
    
    # 可视化结果
    visualize_ablation_results(results)
    
    # 打印结论
    print_final_conclusions(results)
    
    print("\n✅ 最终消融实验完成！")
    print("📊 请查看 causal_ablation_final_results.png 了解详细对比")


if __name__ == "__main__":
    # 可以通过命令行参数或直接修改这里来启用verbose模式
    import sys
    verbose = len(sys.argv) > 1 and sys.argv[1].lower() == 'verbose'
    main(verbose=verbose) 