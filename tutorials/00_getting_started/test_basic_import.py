"""
基础导入测试
测试教程框架的基本功能是否正常工作
"""
import sys
import os
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print("-" * 50)

# 测试导入
try:
    from tutorials.utils import baseline_networks
    print("✓ 成功导入 baseline_networks")
except ImportError as e:
    print(f"✗ 导入 baseline_networks 失败: {e}")

try:
    from tutorials.utils import ablation_networks
    print("✓ 成功导入 ablation_networks")
except ImportError as e:
    print(f"✗ 导入 ablation_networks 失败: {e}")

try:
    from tutorials.utils import data_loaders
    print("✓ 成功导入 data_loaders")
except ImportError as e:
    print(f"✗ 导入 data_loaders 失败: {e}")

try:
    from tutorials.utils import evaluation_metrics
    print("✓ 成功导入 evaluation_metrics")
except ImportError as e:
    print(f"✗ 导入 evaluation_metrics 失败: {e}")

print("-" * 50)

# 测试基础网络创建
print("\n测试基础网络创建:")
try:
    # 创建传统MLP分类器
    mlp_classifier = baseline_networks.TraditionalMLPClassifier(
        input_size=10,
        num_classes=3,
        hidden_sizes=[64, 32]
    )
    print(f"✓ 创建MLP分类器成功: {mlp_classifier}")
    
    # 测试前向传播
    x = torch.randn(4, 10)  # batch_size=4, input_dim=10
    output = mlp_classifier(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
except Exception as e:
    print(f"✗ 创建MLP分类器失败: {e}")
    import traceback
    traceback.print_exc()

print("-" * 50)

# 测试消融网络创建
print("\n测试消融网络创建:")
try:
    # 创建消融实验组件
    engine, wrapper = ablation_networks.create_ablation_experiment(
        input_dim=10,
        hidden_dim=64,
        num_layers=2,
        task_type='classification',
        num_classes=3,
        device='cpu'  # 使用CPU以确保兼容性
    )
    print(f"✓ 创建消融实验组件成功")
    print(f"  Engine: {engine}")
    print(f"  Wrapper: {wrapper}")
    
    # 测试损失计算
    batch_size = 4
    seq_len = 10
    inputs = {
        'input_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        'values': torch.randn(batch_size, seq_len),
        'temperature': 1.0,
        'mode': 'causal'
    }
    targets = torch.randint(0, 3, (batch_size,))
    
    # 测试消融损失
    ablation_loss = wrapper.compute_loss_ablation(inputs.copy(), targets)
    print(f"  消融损失: {ablation_loss.item():.4f}")
    
except Exception as e:
    print(f"✗ 创建消融网络失败: {e}")
    import traceback
    traceback.print_exc()

print("-" * 50)

# 测试兼容性包装器
print("\n测试兼容性包装器:")
try:
    # 使用旧API创建分类器
    classifier = ablation_networks.create_ablated_classifier(
        input_size=10,
        num_classes=3,
        causal_size=64,
        device='cpu'
    )
    print(f"✓ 创建兼容性分类器成功: {classifier}")
    
    # 测试预测
    x = torch.randn(4, 10)
    predictions = classifier.predict(x)
    print(f"  预测结果: {predictions}")
    
    # 测试概率预测
    probs = classifier.predict_proba(x)
    print(f"  概率形状: {probs.shape}")
    print(f"  概率和: {probs.sum(dim=1)}")
    
except Exception as e:
    print(f"✗ 兼容性包装器测试失败: {e}")
    import traceback
    traceback.print_exc()

print("-" * 50)
print("\n基础测试完成！") 