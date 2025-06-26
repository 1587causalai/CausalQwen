#!/usr/bin/env python3
"""
Deterministic模式优势分析脚本

通过控制变量实验，逐一验证可能导致deterministic模式优越性的因素：
1. 数值精度 (float32 vs float64)
2. 优化器选择 (Adam vs SGD+momentum)
3. 初始化策略
4. 梯度裁剪
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

from causal_engine.sklearn import MLPCausalClassifier

class ControlledPyTorchModel(nn.Module):
    """受控PyTorch模型，用于逐一测试差异因素"""
    
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def init_xavier_uniform(self):
        """使用Xavier均匀初始化（与CausalEngine一致）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def init_identity_first_layer(self):
        """第一层使用恒等初始化（模拟CausalEngine的AbductionNetwork）"""
        first_layer = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                first_layer = m
                break
        
        if first_layer and first_layer.in_features == first_layer.out_features:
            with torch.no_grad():
                first_layer.weight.copy_(torch.eye(first_layer.in_features, dtype=first_layer.weight.dtype))
                first_layer.bias.zero_()


def train_controlled_pytorch(model, X_train, y_train, X_val, y_val, 
                            use_float64=False, optimizer_type='adam', use_grad_clip=False,
                            epochs=1000, lr=0.001):
    """受控训练PyTorch模型"""
    
    # 数据类型控制
    if use_float64:
        model = model.double()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float64)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    else:
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
    
    criterion = nn.CrossEntropyLoss()
    
    # 优化器控制
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
    
    # 早停设置
    best_loss = float('inf')
    patience = 50
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        
        # 梯度裁剪控制
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    model.n_iter_ = epoch + 1
    return model


def run_controlled_experiment():
    """运行控制变量实验"""
    print("🔬 Deterministic模式优势分析实验")
    print("=" * 80)
    
    # 生成数据
    n_samples, n_features, n_classes = 1000, 10, 3
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_classes=n_classes, n_informative=7, 
                              class_sep=0.8, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    hidden_sizes = (64, 32)
    lr = 0.001
    max_iter = 1000
    
    results = {}
    
    print(f"数据: {n_samples}样本, {n_features}特征, {n_classes}类别")
    print(f"网络: {hidden_sizes}, lr={lr}, max_iter={max_iter}")
    print()
    
    # 1. 基线: 标准PyTorch (float32 + Adam)
    print("1️⃣ 基线PyTorch (float32 + Adam + 默认初始化)")
    model1 = ControlledPyTorchModel(n_features, n_classes, hidden_sizes)
    model1 = train_controlled_pytorch(model1, X_train, y_train, X_val, y_val,
                                    use_float64=False, optimizer_type='adam', use_grad_clip=False)
    model1.eval()
    with torch.no_grad():
        pred1 = torch.argmax(model1(torch.FloatTensor(X_test)), dim=1).numpy()
    acc1 = accuracy_score(y_test, pred1)
    results['基线PyTorch'] = acc1
    print(f"   准确率: {acc1:.4f}, 训练轮数: {model1.n_iter_}")
    
    # 2. float64精度测试
    print("2️⃣ 高精度PyTorch (float64 + Adam + 默认初始化)")
    model2 = ControlledPyTorchModel(n_features, n_classes, hidden_sizes)
    model2 = train_controlled_pytorch(model2, X_train, y_train, X_val, y_val,
                                    use_float64=True, optimizer_type='adam', use_grad_clip=False)
    model2.eval()
    with torch.no_grad():
        pred2 = torch.argmax(model2(torch.tensor(X_test, dtype=torch.float64)), dim=1).numpy()
    acc2 = accuracy_score(y_test, pred2)
    results['高精度PyTorch'] = acc2
    print(f"   准确率: {acc2:.4f}, 训练轮数: {model2.n_iter_}")
    
    # 3. 优化器测试
    print("3️⃣ 优化器PyTorch (float64 + SGD+momentum + 默认初始化)")
    model3 = ControlledPyTorchModel(n_features, n_classes, hidden_sizes)
    model3 = train_controlled_pytorch(model3, X_train, y_train, X_val, y_val,
                                    use_float64=True, optimizer_type='sgd_momentum', use_grad_clip=False)
    model3.eval()
    with torch.no_grad():
        pred3 = torch.argmax(model3(torch.tensor(X_test, dtype=torch.float64)), dim=1).numpy()
    acc3 = accuracy_score(y_test, pred3)
    results['优化器PyTorch'] = acc3
    print(f"   准确率: {acc3:.4f}, 训练轮数: {model3.n_iter_}")
    
    # 4. 初始化测试
    print("4️⃣ 初始化PyTorch (float64 + SGD+momentum + Xavier初始化)")
    model4 = ControlledPyTorchModel(n_features, n_classes, hidden_sizes)
    model4.init_xavier_uniform()  # Xavier初始化
    model4 = train_controlled_pytorch(model4, X_train, y_train, X_val, y_val,
                                    use_float64=True, optimizer_type='sgd_momentum', use_grad_clip=False)
    model4.eval()
    with torch.no_grad():
        pred4 = torch.argmax(model4(torch.tensor(X_test, dtype=torch.float64)), dim=1).numpy()
    acc4 = accuracy_score(y_test, pred4)
    results['初始化PyTorch'] = acc4
    print(f"   准确率: {acc4:.4f}, 训练轮数: {model4.n_iter_}")
    
    # 5. 梯度裁剪测试
    print("5️⃣ 完整PyTorch (float64 + SGD+momentum + Xavier初始化 + 梯度裁剪)")
    model5 = ControlledPyTorchModel(n_features, n_classes, hidden_sizes)
    model5.init_xavier_uniform()
    model5 = train_controlled_pytorch(model5, X_train, y_train, X_val, y_val,
                                    use_float64=True, optimizer_type='sgd_momentum', use_grad_clip=True)
    model5.eval()
    with torch.no_grad():
        pred5 = torch.argmax(model5(torch.tensor(X_test, dtype=torch.float64)), dim=1).numpy()
    acc5 = accuracy_score(y_test, pred5)
    results['完整PyTorch'] = acc5
    print(f"   准确率: {acc5:.4f}, 训练轮数: {model5.n_iter_}")
    
    # 6. CausalEngine deterministic模式
    print("6️⃣ CausalEngine (deterministic模式)")
    causal_det = MLPCausalClassifier(
        hidden_layer_sizes=hidden_sizes,
        mode='deterministic',
        max_iter=max_iter,
        learning_rate=lr,
        early_stopping=True,
        random_state=42,
        verbose=False
    )
    causal_det.fit(X_train, y_train)
    pred6 = causal_det.predict(X_test)
    if isinstance(pred6, dict):
        pred6 = pred6['predictions']
    acc6 = accuracy_score(y_test, pred6)
    results['CausalEngine'] = acc6
    print(f"   准确率: {acc6:.4f}, 训练轮数: {causal_det.n_iter_}")
    
    # 7. sklearn基线
    print("7️⃣ sklearn MLPClassifier")
    sklearn_clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        max_iter=max_iter,
        learning_rate_init=lr,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    sklearn_clf.fit(X_train, y_train)
    pred7 = sklearn_clf.predict(X_test)
    acc7 = accuracy_score(y_test, pred7)
    results['sklearn'] = acc7
    print(f"   准确率: {acc7:.4f}, 训练轮数: {sklearn_clf.n_iter_}")
    
    # 结果分析
    print("\n📊 实验结果分析:")
    print("=" * 80)
    print(f"{'方法':<20} {'准确率':<10} {'提升':<10}")
    print("-" * 40)
    
    baseline_acc = results['基线PyTorch']
    for method, acc in results.items():
        improvement = acc - baseline_acc
        print(f"{method:<20} {acc:<10.4f} {improvement:+.4f}")
    
    print("\n🔍 差异因素影响分析:")
    factors = [
        ('数值精度', results['高精度PyTorch'] - results['基线PyTorch']),
        ('优化器', results['优化器PyTorch'] - results['高精度PyTorch']),
        ('初始化', results['初始化PyTorch'] - results['优化器PyTorch']),
        ('梯度裁剪', results['完整PyTorch'] - results['初始化PyTorch']),
    ]
    
    for factor, improvement in factors:
        print(f"   {factor}: {improvement:+.4f}")
    
    total_pytorch_improvement = results['完整PyTorch'] - results['基线PyTorch']
    causal_advantage = results['CausalEngine'] - results['完整PyTorch']
    
    print(f"\n总PyTorch改进: {total_pytorch_improvement:+.4f}")
    print(f"剩余CausalEngine优势: {causal_advantage:+.4f}")
    
    if causal_advantage > 0.01:
        print("🎯 CausalEngine仍有显著优势，可能来自：")
        print("   - AbductionNetwork的恒等初始化策略")
        print("   - 更精细的早停策略")
        print("   - 架构上的微妙差异")
    else:
        print("✅ 主要差异已被解释，deterministic模式的优势主要来自上述因素")


if __name__ == "__main__":
    run_controlled_experiment()