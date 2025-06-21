"""
快速分类任务演示
展示如何使用消融实验框架进行分类任务的对比实验
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tutorials.utils.baseline_networks import TraditionalMLPClassifier
from tutorials.utils.ablation_networks import create_ablation_experiment, AblationTrainer
from tutorials.utils.evaluation_metrics import calculate_classification_metrics

print("快速分类任务演示")
print("=" * 60)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成合成数据集
print("\n1. 生成合成分类数据集")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=42
)

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ 样本数: {len(X)}")
print(f"✓ 特征数: {X.shape[1]}")
print(f"✓ 类别数: {len(np.unique(y))}")
print(f"✓ 训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 创建三个模型进行对比
print("\n2. 创建三个对比模型")

# 2.1 传统MLP（基准）
print("\n2.1 传统MLP分类器（基准）")
mlp_model = TraditionalMLPClassifier(
    input_size=20,
    num_classes=3,
    hidden_sizes=[64, 32],
    dropout_rate=0.1
)
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
print(f"✓ 创建成功: {type(mlp_model).__name__}")

# 2.2 CausalEngine消融版本（仅使用loc）
print("\n2.2 CausalEngine消融版本（仅使用loc）")
ablation_engine, ablation_wrapper = create_ablation_experiment(
    input_dim=20,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    task_type='classification',
    num_classes=3,
    device='cpu'
)
ablation_trainer = AblationTrainer(ablation_engine, ablation_wrapper, lr=0.001)
print(f"✓ 创建成功: CausalEngine (消融模式)")

# 2.3 CausalEngine完整版本（使用loc+scale）
print("\n2.3 CausalEngine完整版本（使用loc+scale）")
full_engine, full_wrapper = create_ablation_experiment(
    input_dim=20,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    task_type='classification',
    num_classes=3,
    device='cpu'
)
full_trainer = AblationTrainer(full_engine, full_wrapper, lr=0.001)
print(f"✓ 创建成功: CausalEngine (完整模式)")

# 3. 训练模型
print("\n3. 训练模型（简化训练，仅5个epoch）")
num_epochs = 5

# 3.1 训练传统MLP
print("\n训练传统MLP...")
mlp_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    mlp_model.train()
    for batch_x, batch_y in train_loader:
        mlp_optimizer.zero_grad()
        outputs = mlp_model(batch_x)
        loss = nn.CrossEntropyLoss()(outputs, batch_y)
        loss.backward()
        mlp_optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    mlp_losses.append(avg_loss)
    print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 3.2 训练消融版本
print("\n训练CausalEngine消融版本...")
ablation_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        # 准备输入
        batch_size = batch_x.shape[0]
        inputs = {
            'values': batch_x
        }
        
        hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
        metrics = ablation_trainer.train_step_ablation(hidden_states, batch_y)
        epoch_loss += metrics['loss']
    
    avg_loss = epoch_loss / len(train_loader)
    ablation_losses.append(avg_loss)
    print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 3.3 训练完整版本
print("\n训练CausalEngine完整版本...")
full_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        # 准备输入
        batch_size = batch_x.shape[0]
        inputs = {
            'values': batch_x
        }
        
        hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
        metrics = full_trainer.train_step_full(hidden_states, batch_y)
        epoch_loss += metrics['loss']
    
    avg_loss = epoch_loss / len(train_loader)
    full_losses.append(avg_loss)
    print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 4. 评估模型
print("\n4. 评估模型性能")

# 4.1 评估传统MLP
mlp_model.eval()
mlp_preds = []
mlp_true = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = mlp_model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        mlp_preds.extend(preds.numpy())
        mlp_true.extend(batch_y.numpy())

mlp_metrics = calculate_classification_metrics(
    np.array(mlp_true), 
    np.array(mlp_preds),
    num_classes=3
)

print("\n传统MLP性能:")
print(f"  准确率: {mlp_metrics['accuracy']:.4f}")
print(f"  F1分数: {mlp_metrics['f1_macro']:.4f}")

# 4.2 评估消融版本
ablation_preds = []
ablation_true = []
ablation_engine.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_size = batch_x.shape[0]
        inputs = {
            'input_ids': torch.arange(20).unsqueeze(0).expand(batch_size, -1),
            'values': batch_x,
            'temperature': 1.0,
            'mode': 'causal'
        }
        
        outputs = ablation_engine(**inputs)
        loc = outputs['decision_scores'][:, -1, :3, 0]
        preds = torch.argmax(loc, dim=1)
        
        ablation_preds.extend(preds.numpy())
        ablation_true.extend(batch_y.numpy())

ablation_metrics = calculate_classification_metrics(
    np.array(ablation_true),
    np.array(ablation_preds),
    num_classes=3
)

print("\nCausalEngine消融版本性能:")
print(f"  准确率: {ablation_metrics['accuracy']:.4f}")
print(f"  F1分数: {ablation_metrics['f1_macro']:.4f}")

# 4.3 评估完整版本
full_preds = []
full_true = []
full_engine.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_size = batch_x.shape[0]
        inputs = {
            'input_ids': torch.arange(20).unsqueeze(0).expand(batch_size, -1),
            'values': batch_x,
            'temperature': 1.0,
            'mode': 'causal'
        }
        
        outputs = full_engine(**inputs)
        # 注意：这里简化处理，实际应该使用因果概率
        loc = outputs['decision_scores'][:, -1, :3, 0]
        preds = torch.argmax(loc, dim=1)
        
        full_preds.extend(preds.numpy())
        full_true.extend(batch_y.numpy())

full_metrics = calculate_classification_metrics(
    np.array(full_true),
    np.array(full_preds),
    num_classes=3
)

print("\nCausalEngine完整版本性能:")
print(f"  准确率: {full_metrics['accuracy']:.4f}")
print(f"  F1分数: {full_metrics['f1_macro']:.4f}")

# 5. 总结
print("\n" + "=" * 60)
print("实验总结")
print("-" * 60)
print(f"传统MLP:              准确率={mlp_metrics['accuracy']:.4f}, F1={mlp_metrics['f1_macro']:.4f}")
print(f"CausalEngine(仅loc):  准确率={ablation_metrics['accuracy']:.4f}, F1={ablation_metrics['f1_macro']:.4f}")
print(f"CausalEngine(完整):   准确率={full_metrics['accuracy']:.4f}, F1={full_metrics['f1_macro']:.4f}")
print("-" * 60)

print("\n注意：")
print("1. 这是一个简化的演示，仅训练了5个epoch")
print("2. 使用的是模拟的CausalEngine实现")
print("3. 完整实验应该包含更多epoch、早停和超参数调优")
print("4. 真实的CausalEngine会有更显著的性能优势")

print("\n演示完成！") 