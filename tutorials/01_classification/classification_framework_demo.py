"""
分类任务框架演示（无训练版本）
展示消融实验框架的架构和评估流程
"""
import sys
import os
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tutorials.utils.baseline_networks import TraditionalMLPClassifier
from tutorials.utils.ablation_networks import create_ablation_experiment
from tutorials.utils.evaluation_metrics import calculate_classification_metrics

print("分类任务框架演示")
print("=" * 60)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 准备数据
print("\n1. 准备示例数据")
X, y = make_classification(
    n_samples=200,  # 小数据集用于演示
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✓ 训练集: {len(X_train)} 样本")
print(f"✓ 测试集: {len(X_test)} 样本")
print(f"✓ 特征数: {X.shape[1]}")
print(f"✓ 类别数: {len(np.unique(y))}")

# 转换为张量
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 2. 创建模型
print("\n2. 创建三个对比模型")

# 2.1 传统MLP
print("\n传统MLP分类器:")
mlp_model = TraditionalMLPClassifier(
    input_size=20,
    num_classes=3,
    hidden_sizes=[64, 32]
)
print(f"  ✓ 参数量: {sum(p.numel() for p in mlp_model.parameters()):,}")
print(f"  ✓ 网络结构: MLP -> Softmax -> 预测")

# 2.2 CausalEngine (用于消融和完整版本)
print("\nCausalEngine架构:")
engine, wrapper = create_ablation_experiment(
    input_dim=20,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    task_type='classification',
    num_classes=3,
    device='cpu'
)
print(f"  ✓ 网络结构: CausalEngine -> 决策得分(loc, scale)")
print(f"  ✓ 消融版本: 仅使用loc进行预测")
print(f"  ✓ 完整版本: 使用loc+scale进行因果推理")

# 3. 展示推理过程
print("\n3. 展示推理过程（使用测试集的前5个样本）")

# 准备小批量数据
sample_size = 5
X_sample = X_test_tensor[:sample_size]
y_sample = y_test_tensor[:sample_size]

print(f"\n样本标签: {y_sample.numpy()}")

# 3.1 传统MLP推理
print("\n传统MLP推理:")
mlp_model.eval()
with torch.no_grad():
    mlp_logits = mlp_model(X_sample)
    mlp_probs = torch.softmax(mlp_logits, dim=-1)
    mlp_preds = torch.argmax(mlp_logits, dim=-1)
    
print(f"  Logits形状: {mlp_logits.shape}")
print(f"  预测类别: {mlp_preds.numpy()}")
print(f"  预测概率:")
for i in range(sample_size):
    print(f"    样本{i}: [{mlp_probs[i,0]:.3f}, {mlp_probs[i,1]:.3f}, {mlp_probs[i,2]:.3f}]")

# 3.2 CausalEngine推理
print("\nCausalEngine推理:")
engine.eval()
with torch.no_grad():
    # 准备输入
    inputs = {
        'input_ids': torch.arange(20).unsqueeze(0).expand(sample_size, -1),
        'values': X_sample,
        'temperature': 1.0,
        'mode': 'causal'
    }
    
    # 前向传播
    outputs = engine(**inputs)
    decision_scores = outputs['decision_scores'][:, -1, :3, :]  # [batch, 3, 2]
    
    # 提取loc和scale
    loc = decision_scores[:, :, 0]  # [batch, 3]
    scale = decision_scores[:, :, 1]  # [batch, 3]
    
    # 消融版本：仅使用loc
    ablation_probs = torch.softmax(loc, dim=-1)
    ablation_preds = torch.argmax(loc, dim=-1)
    
print(f"  决策得分形状: {decision_scores.shape}")
print(f"  Loc形状: {loc.shape}, Scale形状: {scale.shape}")
print(f"\n消融版本（仅loc）:")
print(f"  预测类别: {ablation_preds.numpy()}")
print(f"  预测概率:")
for i in range(sample_size):
    print(f"    样本{i}: [{ablation_probs[i,0]:.3f}, {ablation_probs[i,1]:.3f}, {ablation_probs[i,2]:.3f}]")

print(f"\n完整版本（loc+scale）:")
print(f"  说明: 完整版本会使用柯西分布U~Cauchy(loc,scale)进行因果推理")
print(f"  Scale均值: {scale.mean():.4f} (反映不确定性)")

# 4. 评估对比
print("\n4. 模型评估对比（完整测试集）")

# 评估函数
def evaluate_model(model_name, predictions, true_labels):
    metrics = calculate_classification_metrics(
        true_labels.numpy(),
        predictions.numpy(),
        num_classes=3
    )
    print(f"\n{model_name}:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数(宏平均): {metrics['f1_macro']:.4f}")
    print(f"  精确率(宏平均): {metrics['precision_macro']:.4f}")
    print(f"  召回率(宏平均): {metrics['recall_macro']:.4f}")
    return metrics

# 在完整测试集上评估
all_mlp_preds = []
all_ablation_preds = []

with torch.no_grad():
    # MLP预测
    mlp_outputs = mlp_model(X_test_tensor)
    all_mlp_preds = torch.argmax(mlp_outputs, dim=-1)
    
    # CausalEngine预测
    batch_size = X_test_tensor.shape[0]
    ce_inputs = {
        'input_ids': torch.arange(20).unsqueeze(0).expand(batch_size, -1),
        'values': X_test_tensor,
        'temperature': 1.0,
        'mode': 'causal'
    }
    ce_outputs = engine(**ce_inputs)
    ce_loc = ce_outputs['decision_scores'][:, -1, :3, 0]
    all_ablation_preds = torch.argmax(ce_loc, dim=-1)

# 计算指标
mlp_metrics = evaluate_model("传统MLP", all_mlp_preds, y_test_tensor)
ablation_metrics = evaluate_model("CausalEngine(消融版本)", all_ablation_preds, y_test_tensor)

# 5. 总结
print("\n" + "=" * 60)
print("实验框架总结")
print("-" * 60)
print("三层对比设计:")
print("1. 传统神经网络（MLP）: 标准前馈网络基准")
print("2. CausalEngine消融版本: 同样的网络，但仅使用loc")
print("3. CausalEngine完整版本: 利用loc+scale进行因果推理")
print("-" * 60)
print("关键洞察:")
print("• 网络架构完全相同，确保公平对比")
print("• 唯一差异在于损失函数和推理方式")
print("• 通过对比可以量化因果机制的价值")
print("-" * 60)
print("\n注意: 本演示使用模拟的CausalEngine，主要展示框架设计")
print("真实实验需要完整的CausalEngine实现和充分的训练")

print("\n演示完成！") 