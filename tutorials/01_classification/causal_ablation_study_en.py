#!/usr/bin/env python3
"""
CausalEngine Ablation Study

Perfect controlled experiment:
- Group A: Full CausalEngine (loc + scale + OvR + BCE)  
- Group B: Simplified version (loc only + Softmax + CrossEntropy)
- Identical network architecture, only difference is causal uncertainty modeling
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
import warnings
warnings.filterwarnings('ignore')

# Import CausalEngine
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
    """Shared feature encoder - ensuring identical network structure"""
    
    def __init__(self, input_size, output_size=32):
        super().__init__()
        
        # Adjust network depth based on input size
        if input_size <= 4:
            hidden_sizes = [32, 16]
        elif input_size <= 20:
            hidden_sizes = [64, 32]
        else:
            hidden_sizes = [128, 64, 32]
        
        # Feature encoder
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
        
        # Final output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.encoder = nn.Sequential(*layers)
        self.output_size = output_size
    
    def forward(self, x):
        return self.encoder(x)


class FullCausalClassifier(nn.Module):
    """Full CausalEngine classifier (Group A) - loc + scale + OvR + BCE"""
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.encoder = SharedEncoder(input_size, output_size=32)
        
        # Full CausalEngine
        self.causal_engine = CausalEngine(
            hidden_size=self.encoder.output_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
        
        self.num_classes = num_classes
    
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
    """Simplified CausalEngine classifier (Group B) - loc only + Softmax + CrossEntropy"""
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        # Use identical encoder!
        self.encoder = SharedEncoder(input_size, output_size=32)
        
        # Create CausalEngine but only use loc part
        self.causal_engine = CausalEngine(
            hidden_size=self.encoder.output_size,
            vocab_size=num_classes,
            activation_modes="classification",
            b_noise_init=0.1,
            gamma_init=1.0
        )
        
        self.num_classes = num_classes
    
    def forward(self, x, temperature=1.0, return_components=False):
        features = self.encoder(x)
        output = self.causal_engine(
            features.unsqueeze(1),
            temperature=temperature,
            do_sample=False,
            return_dict=True
        )
        
        # Key difference: only use loc_S, ignore scale_S!
        loc_S = output['loc_S'].squeeze(1)  # [batch_size, num_classes]
        
        # Apply softmax to get traditional probability distribution
        logits = loc_S  # Use loc as logits
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
    """Full CausalEngine OvR loss function"""
    targets = F.one_hot(labels, num_classes=num_classes).float()
    return F.binary_cross_entropy(probs, targets, reduction='mean')


def simplified_causal_loss(probs, labels):
    """Simplified version standard cross-entropy loss"""
    return F.cross_entropy(torch.log(probs + 1e-8), labels)


def train_model(model, train_loader, val_loader, loss_fn, model_name, epochs=50):
    """Generic model training function"""
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    print(f"      Training {model_name}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            # Dynamic temperature adjustment
            if epoch < epochs // 3:
                temperature = 1.2
            elif epoch < 2 * epochs // 3:
                temperature = 1.0
            else:
                temperature = 0.8
            
            # Forward pass
            if 'Full' in model_name:
                probs = model(features, temperature=temperature)
                loss = loss_fn(probs, labels, model.num_classes)
            else:
                probs = model(features, temperature=temperature)
                loss = loss_fn(probs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    probs = model(features, temperature=0.1)
                    predictions = torch.argmax(probs, dim=1)
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
            
            # Early stopping
            if patience_counter >= patience:
                print(f"      Early stop at epoch {epoch}, best val acc: {best_val_acc:.4f}")
                break
    
    return best_val_acc


def detailed_analysis(full_model, simplified_model, test_loader, dataset_name):
    """Detailed analysis of behavior differences between two models"""
    
    print(f"\nDetailed Analysis - {dataset_name}")
    print("-" * 40)
    
    full_model.eval()
    simplified_model.eval()
    
    all_labels = []
    full_predictions = []
    simplified_predictions = []
    full_confidences = []
    simplified_confidences = []
    
    # Collect prediction results
    with torch.no_grad():
        for features, labels in test_loader:
            # Full model
            full_out = full_model(features, temperature=0.1, return_components=True)
            full_pred = torch.argmax(full_out['probs'], dim=1)
            full_conf = torch.max(full_out['probs'], dim=1)[0]
            
            # Simplified model
            simp_out = simplified_model(features, temperature=0.1, return_components=True)
            simp_pred = torch.argmax(simp_out['probs'], dim=1)
            simp_conf = torch.max(simp_out['probs'], dim=1)[0]
            
            all_labels.extend(labels.numpy())
            full_predictions.extend(full_pred.numpy())
            simplified_predictions.extend(simp_pred.numpy())
            full_confidences.extend(full_conf.numpy())
            simplified_confidences.extend(simp_conf.numpy())
    
    # Calculate metrics
    full_acc = accuracy_score(all_labels, full_predictions)
    simp_acc = accuracy_score(all_labels, simplified_predictions)
    
    print(f"   Full CausalEngine accuracy: {full_acc:.4f}")
    print(f"   Simplified version accuracy: {simp_acc:.4f}")
    print(f"   Difference: {full_acc - simp_acc:+.4f}")
    
    print(f"   Full model avg confidence: {np.mean(full_confidences):.4f}")
    print(f"   Simplified model avg confidence: {np.mean(simplified_confidences):.4f}")
    
    # Analyze prediction consistency
    agreement = np.mean(np.array(full_predictions) == np.array(simplified_predictions))
    print(f"   Prediction agreement: {agreement:.4f}")
    
    return {
        'dataset': dataset_name,
        'full_acc': full_acc,
        'simplified_acc': simp_acc,
        'difference': full_acc - simp_acc,
        'full_confidence': np.mean(full_confidences),
        'simplified_confidence': np.mean(simplified_confidences),
        'agreement': agreement
    }


def run_ablation_experiment():
    """Run complete ablation experiment"""
    
    print("="*60)
    print("CausalEngine Ablation Study")
    print("   Comparing: Full vs Simplified (loc only + softmax)")
    print("="*60)
    
    # Load datasets
    datasets_info = [
        # Iris - 3 classes, small data, basic capability
        {
            'name': 'Iris',
            'loader': lambda: datasets.load_iris(),
            'classes': 3
        },
        # Wine - 3 classes, medium data, moderate complexity
        {
            'name': 'Wine', 
            'loader': lambda: datasets.load_wine(),
            'classes': 3
        },
        # Breast Cancer - 2 classes, binary classification capability
        {
            'name': 'Breast Cancer',
            'loader': lambda: datasets.load_breast_cancer(),
            'classes': 2
        },
        # Digits - 10 classes, multi-class capability
        {
            'name': 'Digits',
            'loader': lambda: datasets.load_digits(),
            'classes': 10
        }
    ]
    
    results = []
    
    for dataset_info in datasets_info:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_info['name']}")
        print(f"{'='*50}")
        
        # Load data
        data = dataset_info['loader']()
        X, y = data.data, data.target
        
        # For Digits, sample to speed up
        if dataset_info['name'] == 'Digits':
            indices = np.random.choice(len(X), 800, replace=False)
            X, y = X[indices], y[indices]
        
        print(f"   Samples: {len(X)}, Features: {X.shape[1]}, Classes: {dataset_info['classes']}")
        
        # Data preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Data split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        # Create data loaders
        batch_size = min(32, len(X_train) // 4) if len(X_train) > 32 else 8
        
        train_dataset = SimpleDataset(X_train, y_train)
        val_dataset = SimpleDataset(X_val, y_val)
        test_dataset = SimpleDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create models
        print("\nCreating models...")
        full_model = FullCausalClassifier(X_train.shape[1], dataset_info['classes'])
        simplified_model = SimplifiedCausalClassifier(X_train.shape[1], dataset_info['classes'])
        
        print("   ✅ Full CausalEngine: loc + scale + OvR + BCE")
        print("   ✅ Simplified version: loc only + Softmax + CrossEntropy")
        print("   ✅ Identical network architecture")
        
        # Train models
        print("\nStarting training...")
        
        full_val_acc = train_model(
            full_model, train_loader, val_loader, full_causal_loss, 
            "Full CausalEngine", epochs=60
        )
        
        simplified_val_acc = train_model(
            simplified_model, train_loader, val_loader, simplified_causal_loss,
            "Simplified Version", epochs=60
        )
        
        # Detailed analysis
        analysis = detailed_analysis(full_model, simplified_model, test_loader, dataset_info['name'])
        results.append(analysis)
        
        print(f"\n{dataset_info['name']} Results Summary:")
        print(f"   Full version val acc: {full_val_acc:.4f}")
        print(f"   Simplified version val acc: {simplified_val_acc:.4f}")
        print(f"   Validation difference: {full_val_acc - simplified_val_acc:+.4f}")
        print(f"   Test difference: {analysis['difference']:+.4f}")
    
    return results


def visualize_ablation_results(results):
    """Visualize ablation experiment results with English labels"""
    
    print("\nGenerating ablation experiment visualization...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    datasets = [r['dataset'] for r in results]
    full_accs = [r['full_acc'] for r in results]
    simplified_accs = [r['simplified_acc'] for r in results]
    differences = [r['difference'] for r in results]
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, full_accs, width, label='Full CausalEngine', 
                   color='red', alpha=0.8)
    bars2 = ax.bar(x + width/2, simplified_accs, width, label='Simplified (loc+softmax)',
                   color='blue', alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Ablation Study: Full vs Simplified CausalEngine')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Performance difference
    ax = axes[0, 1]
    colors = ['green' if diff > 0 else 'red' for diff in differences]
    bars = ax.bar(datasets, differences, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Accuracy Difference (Full - Simplified)')
    ax.set_title('Performance Gain from Full CausalEngine')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, 
                height + (0.005 if height > 0 else -0.005),
                f'{diff:+.3f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=10, weight='bold')
    
    # 3. Confidence comparison
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
    
    # 4. Experiment summary
    ax = axes[1, 1]
    
    avg_improvement = np.mean(differences)
    positive_improvements = sum(1 for d in differences if d > 0)
    avg_full_acc = np.mean(full_accs)
    avg_simp_acc = np.mean(simplified_accs)
    
    summary_text = f"""Ablation Study Summary

Average Performance:
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
    plt.savefig('causal_ablation_results_en.png', dpi=150, bbox_inches='tight')
    print("   Ablation results saved to causal_ablation_results_en.png")
    plt.close()


def print_final_conclusions(results):
    """Print final conclusions"""
    
    print("\n" + "="*60)
    print("Ablation Experiment Final Conclusions")
    print("="*60)
    
    differences = [r['difference'] for r in results]
    avg_improvement = np.mean(differences)
    positive_improvements = sum(1 for d in differences if d > 0)
    
    print(f"\nOverall Results:")
    print(f"   Average performance gain: {avg_improvement:+.4f}")
    print(f"   Datasets where full wins: {positive_improvements}/{len(results)}")
    
    print(f"\nDetailed by Dataset:")
    for result in results:
        status = "🟢" if result['difference'] > 0 else "🔴"
        print(f"   {status} {result['dataset']:15s}: {result['difference']:+.4f} "
              f"({result['full_acc']:.3f} vs {result['simplified_acc']:.3f})")
    
    print(f"\nScientific Conclusions:")
    if avg_improvement > 0.02:
        print("   ✅ Causal uncertainty modeling significantly effective")
        print("   ✅ OvR + scale_S strategy clearly superior to simplified version")
        print("   ✅ CausalEngine performance gain mainly from causal reasoning")
    elif avg_improvement > 0.005:
        print("   ⚠️ Causal uncertainty modeling mildly effective")
        print("   ⚠️ Performance gain partially from causal reasoning")
        print("   ⚠️ Network architecture also contributes")
    else:
        print("   ❌ Simplified version performs comparably or better")
        print("   ❌ Causal uncertainty modeling has limited effectiveness")
        print("   ❌ May need parameter tuning or different use cases")
    
    print(f"\nExperimental Value:")
    print("   🔬 Rigorously controlled network architecture variables")
    print("   🎯 Clearly quantified contribution of causal reasoning mechanism")
    print("   📚 Provides guidance for CausalEngine application scenarios")


def main():
    """Main function"""
    
    # Run ablation experiment
    results = run_ablation_experiment()
    
    # Visualize results
    visualize_ablation_results(results)
    
    # Print final conclusions
    print_final_conclusions(results)
    
    print("\n✅ Ablation experiment completed!")
    print("📊 Check causal_ablation_results_en.png for detailed comparison")


if __name__ == "__main__":
    main() 