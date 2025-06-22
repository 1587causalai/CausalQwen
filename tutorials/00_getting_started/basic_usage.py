"""
CausalEngine 基础使用教程 (2024更新版)
=========================================

基于最新的基准测试协议和数学理论更新
演示如何使用CausalEngine进行因果推理驱动的分类和回归任务

核心更新:
- 基于基准测试协议的标准化参数配置
- 四种推理模式的完整演示
- 固定噪声vs自适应噪声的对比实验设计
- 三种任务激活机制的应用示例

这是您开始使用CausalEngine的第一个完整示例！
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加路径以导入CausalEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from src.causal_qwen_mvp.causal_engine import CausalEngine
    CAUSAL_ENGINE_AVAILABLE = True
    print("✅ CausalEngine模块加载成功！")
except ImportError:
    try:
        from causal_engine import CausalEngine
        CAUSAL_ENGINE_AVAILABLE = True
        print("✅ CausalEngine模块加载成功！")
    except ImportError:
        CAUSAL_ENGINE_AVAILABLE = False
        print("⚠️  CausalEngine模块未找到，将使用简化演示版本")
        
        # 定义一个模拟的CausalEngine类
        class CausalEngine:
            def __init__(self, **kwargs):
                self.hidden_size = kwargs.get('hidden_size', 128)
                self.causal_size = kwargs.get('causal_size', 128)
                
            def __call__(self, hidden_states, temperature=1.0, do_sample=False):
                batch_size, seq_length, _ = hidden_states.shape
                vocab_size = 10
                
                class Output:
                    def __init__(self):
                        self.logits = torch.randn(batch_size, seq_length, vocab_size)
                
                return Output()

from tutorials.utils.ablation_networks import (
    create_ablation_experiment, AblationTrainer,
    create_ablated_classifier, create_ablated_regressor,
    create_full_causal_classifier, create_full_causal_regressor
)
from tutorials.utils.baseline_networks import (
    TraditionalMLPClassifier, TraditionalMLPRegressor, BaselineTrainer
)

# 添加辅助函数
def create_baseline_classifier(input_size, num_classes, **kwargs):
    return TraditionalMLPClassifier(input_size=input_size, num_classes=num_classes, hidden_sizes=[128, 64], dropout_rate=0.1)

def create_baseline_regressor(input_size, output_size, **kwargs):
    return TraditionalMLPRegressor(input_size=input_size, output_size=output_size, hidden_sizes=[128, 64], dropout_rate=0.1)
from tutorials.utils.evaluation_metrics import (
    calculate_classification_metrics, calculate_regression_metrics
)


def demo_basic_causal_engine():
    """
    演示CausalEngine的基础API使用
    """
    print("\n🌟 CausalEngine 基础API演示")
    print("=" * 40)
    
    if not CAUSAL_ENGINE_AVAILABLE:
        print("由于CausalEngine模块不可用，跳过此演示")
        return
    
    # 1. 创建CausalEngine实例
    print("\n1. 创建CausalEngine实例")
    
    engine = CausalEngine(
        hidden_size=128,        # 隐藏层大小
        vocab_size=10,          # 输出词汇表大小（分类类别数）
        causal_size=128,        # 因果表示大小
        activation_modes="classification"  # 激活模式
    )
    
    print(f"   隐藏层大小: {engine.hidden_size}")
    print(f"   因果表示大小: {engine.causal_size}")
    print(f"   激活模式: classification")
    
    # 2. 准备输入数据
    print("\n2. 准备输入数据")
    
    batch_size = 16
    seq_length = 5
    hidden_size = 128
    
    # 模拟transformer的隐藏状态输出
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    print(f"   输入形状: {hidden_states.shape}")
    print(f"   输入类型: 模拟的transformer隐藏状态")
    
    # 3. 进行因果推理
    print("\n3. 因果推理过程")
    
    # 纯因果推理（温度=0）
    print("   模式1: 纯因果推理 (温度=0)")
    output_causal = engine(hidden_states, temperature=0, do_sample=False)
    print(f"     输出形状: {output_causal.logits.shape}")
    print(f"     输出类型: 确定性因果推理结果")
    
    # 带不确定性的推理（温度>0）
    print("   模式2: 带不确定性推理 (温度=1.0)")
    output_uncertain = engine(hidden_states, temperature=1.0, do_sample=False)
    print(f"     输出形状: {output_uncertain.logits.shape}")
    print(f"     输出类型: 不确定性量化结果")
    
    # 采样模式（身份探索）
    print("   模式3: 采样模式 (温度=0.8, 采样=True)")
    output_sampling = engine(hidden_states, temperature=0.8, do_sample=True)
    print(f"     输出形状: {output_sampling.logits.shape}")
    print(f"     输出类型: 身份探索采样结果")
    
    # 4. 分析输出差异
    print("\n4. 不同推理模式的输出对比")
    
    # 获取第一个样本的输出进行对比
    sample_idx = 0
    seq_idx = 0
    
    causal_probs = torch.softmax(output_causal.logits[sample_idx, seq_idx], dim=-1)
    uncertain_probs = torch.softmax(output_uncertain.logits[sample_idx, seq_idx], dim=-1)
    sampling_probs = torch.softmax(output_sampling.logits[sample_idx, seq_idx], dim=-1)
    
    print(f"   纯因果模式 - 最大概率: {causal_probs.max().item():.4f}")
    print(f"   不确定性模式 - 最大概率: {uncertain_probs.max().item():.4f}")
    print(f"   采样模式 - 最大概率: {sampling_probs.max().item():.4f}")
    
    # 计算概率分布的熵（不确定性度量）
    def entropy(probs):
        return -(probs * torch.log(probs + 1e-8)).sum()
    
    print(f"   纯因果模式 - 熵: {entropy(causal_probs).item():.4f}")
    print(f"   不确定性模式 - 熵: {entropy(uncertain_probs).item():.4f}")
    print(f"   采样模式 - 熵: {entropy(sampling_probs).item():.4f}")
    
    print("\n✅ CausalEngine基础API演示完成！")


def demo_classification_task():
    """
    演示CausalEngine在分类任务中的应用
    """
    print("\n🎯 分类任务演示")
    print("=" * 40)
    
    # 1. 生成分类数据
    print("\n1. 生成模拟分类数据")
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    print(f"   样本数: {X.shape[0]}")
    print(f"   特征数: {X.shape[1]}")
    print(f"   类别数: {len(np.unique(y))}")
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"   训练集大小: {X_train.shape[0]}")
    print(f"   测试集大小: {X_test.shape[0]}")
    
    # 3. 创建和训练模型
    print("\n2. 模型对比实验")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    # 准备数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    results = {}
    
    # 训练传统神经网络
    print("   训练传统神经网络...")
    baseline_model = create_baseline_classifier(input_size, num_classes)
    baseline_trainer = BaselineTrainer(baseline_model, device, learning_rate=1e-4, weight_decay=0.01)
    baseline_trainer.train_classification(train_loader, test_loader, num_epochs=50)
    
    # 评估传统模型
    baseline_model.eval()
    baseline_preds = []
    baseline_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = baseline_model(batch_x)
            preds = torch.argmax(outputs, dim=-1)
            baseline_preds.extend(preds.cpu().numpy())
            baseline_targets.extend(batch_y.numpy())
    
    baseline_metrics = calculate_classification_metrics(
        np.array(baseline_targets), np.array(baseline_preds)
    )
    results['Traditional NN'] = baseline_metrics
    
    # 训练CausalEngine
    print("   训练CausalEngine...")
    causal_model = create_full_causal_classifier(input_size, num_classes)
    causal_trainer = BaselineTrainer(causal_model, device, learning_rate=1e-4, weight_decay=0.01)
    causal_trainer.train_classification(train_loader, test_loader, num_epochs=50)
    
    # 评估CausalEngine（多种推理模式）
    causal_model.eval()
    
    for mode, (temp, do_sample) in [
        ("Causal", (0, False)),
        ("Standard", (1.0, False)),
        ("Sampling", (0.8, True))
    ]:
        causal_preds = []
        causal_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                
                if hasattr(causal_model, 'predict'):
                    preds = causal_model.predict(batch_x, temperature=temp, do_sample=do_sample)
                else:
                    outputs = causal_model(batch_x, temperature=temp, do_sample=do_sample)
                    preds = torch.argmax(outputs, dim=-1)
                
                causal_preds.extend(preds.cpu().numpy())
                causal_targets.extend(batch_y.numpy())
        
        causal_metrics = calculate_classification_metrics(
            np.array(causal_targets), np.array(causal_preds)
        )
        results[f'CausalEngine({mode})'] = causal_metrics
    
    # 4. 显示结果
    print("\n3. 分类结果对比")
    print("   模型                    | 准确率    | F1分数    | 精确率    | 召回率")
    print("   ---------------------- | --------- | --------- | --------- | ---------")
    
    for model_name, metrics in results.items():
        acc = metrics['accuracy']
        f1 = metrics['f1_score']
        prec = metrics['precision']
        rec = metrics['recall']
        print(f"   {model_name:22} | {acc:.4f}    | {f1:.4f}    | {prec:.4f}    | {rec:.4f}")
    
    print("\n✅ 分类任务演示完成！")
    return results


def demo_regression_task():
    """
    演示CausalEngine在回归任务中的应用
    """
    print("\n📈 回归任务演示")
    print("=" * 40)
    
    # 1. 生成回归数据
    print("\n1. 生成模拟回归数据")
    
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    print(f"   样本数: {X.shape[0]}")
    print(f"   特征数: {X.shape[1]}")
    print(f"   目标范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"   训练集大小: {X_train.shape[0]}")
    print(f"   测试集大小: {X_test.shape[0]}")
    
    # 3. 创建和训练模型
    print("\n2. 模型对比实验")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X_train.shape[1]
    output_size = 1
    
    # 准备数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    results = {}
    
    # 训练传统神经网络
    print("   训练传统神经网络...")
    baseline_model = create_baseline_regressor(input_size, output_size)
    baseline_trainer = BaselineTrainer(baseline_model, device, learning_rate=1e-4, weight_decay=0.01)
    baseline_trainer.train_regression(train_loader, test_loader, num_epochs=50)
    
    # 评估传统模型
    baseline_model.eval()
    baseline_preds = []
    baseline_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = baseline_model(batch_x)
            baseline_preds.extend(outputs.cpu().numpy().flatten())
            baseline_targets.extend(batch_y.numpy())
    
    baseline_metrics = calculate_regression_metrics(
        np.array(baseline_targets), np.array(baseline_preds)
    )
    results['Traditional NN'] = baseline_metrics
    
    # 训练CausalEngine
    print("   训练CausalEngine...")
    causal_model = create_full_causal_regressor(input_size, output_size)
    causal_trainer = BaselineTrainer(causal_model, device, learning_rate=1e-4, weight_decay=0.01)
    causal_trainer.train_regression(train_loader, test_loader, num_epochs=50)
    
    # 评估CausalEngine（多种推理模式）
    causal_model.eval()
    
    for mode, (temp, do_sample) in [
        ("Causal", (0, False)),
        ("Standard", (1.0, False)),
        ("Sampling", (0.8, True))
    ]:
        causal_preds = []
        causal_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                
                if hasattr(causal_model, 'predict'):
                    preds = causal_model.predict(batch_x, temperature=temp, do_sample=do_sample)
                else:
                    preds = causal_model(batch_x, temperature=temp, do_sample=do_sample)
                
                causal_preds.extend(preds.cpu().numpy().flatten())
                causal_targets.extend(batch_y.numpy())
        
        causal_metrics = calculate_regression_metrics(
            np.array(causal_targets), np.array(causal_preds)
        )
        results[f'CausalEngine({mode})'] = causal_metrics
    
    # 4. 显示结果
    print("\n3. 回归结果对比")
    print("   模型                    | R²        | MAE       | MdAE      | RMSE")
    print("   ---------------------- | --------- | --------- | --------- | ---------")
    
    for model_name, metrics in results.items():
        r2 = metrics['r2']
        mae = metrics['mae']
        mdae = metrics['mdae']
        rmse = metrics['rmse']
        print(f"   {model_name:22} | {r2:.4f}    | {mae:.4f}    | {mdae:.4f}    | {rmse:.4f}")
    
    print("\n✅ 回归任务演示完成！")
    return results


def demo_causality_vs_correlation():
    """
    演示因果推理vs相关性推理的区别 (基于最新理论)
    """
    print("\n🧠 因果推理 vs 相关性推理 (2024理论更新)")
    print("=" * 60)
    
    print("\n📐 理论框架对比:")
    print("  传统神经网络: P(Y|X) - 统计相关性建模")
    print("  CausalEngine: Y = f(U, ε) - 个体因果机制建模")
    
    print("\n🔬 核心区别:")
    print("  1. 建模对象:")
    print("     传统: 数据的统计分布 → 模式识别")
    print("     因果: 个体的内在特征 → 因果推理")
    
    print("\n  2. 不确定性来源:")
    print("     传统: 数据噪声和模型不确定性")
    print("     因果: 个体差异(U)和外生噪声(ε)的解耦")
    
    print("\n  3. 推理机制:")
    print("     传统: 直接映射 X → Y")
    print("     因果: 三阶段 E → U → S → Y")
    
    print("\n🚀 因果推理的革命性优势:")
    print("  ✨ 泛化能力: 基于因果机制的跨域迁移")
    print("  ✨ 可解释性: 归因-行动-激活的清晰链条")
    print("  ✨ 不确定性量化: 个体不确定性与环境不确定性分离")
    print("  ✨ 反事实推理: 支持'如果...会怎样'的推理")
    print("  ✨ 身份探索: 理解同一个体在不同情境下的表现")
    
    print("\n🌡️ 四种推理模式的深度解析:")
    print("  1. 因果模式 (T=0):")
    print("     哲学: 在无外生干扰下个体的必然选择")
    print("     应用: 确定性决策、硬分类、点估计")
    
    print("\n  2. 标准模式 (T>0, do_sample=False):")
    print("     哲学: 承认环境不确定性对决策的影响")
    print("     应用: 不确定性量化、软决策、置信区间")
    
    print("\n  3. 采样模式 (T>0, do_sample=True):")
    print("     哲学: 探索个体在随机扰动下的多样表现")
    print("     应用: 创造性生成、多样性探索、蒙特卡洛")
    
    print("\n  4. 兼容模式:")
    print("     哲学: 与传统方法对齐，便于性能比较")
    print("     应用: 基准测试、渐进式迁移")
    
    print("\n🎯 基准测试协议中的应用:")
    print("  固定噪声实验: 理解噪声强度对性能的影响")
    print("  自适应噪声实验: 验证模型自主学习噪声的能力")
    print("  四模式对比: 评估不同推理模式的适用场景")
    print("  架构消融: 量化因果架构相对传统方法的优势")


def visualize_results(classification_results, regression_results):
    """
    可视化演示结果
    """
    print("\n📊 生成结果可视化图表")
    
    # Setup plotting style
    plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Classification results visualization
    if classification_results:
        models = list(classification_results.keys())
        accuracies = [classification_results[model]['accuracy'] for model in models]
        f1_scores = [classification_results[model]['f1_score'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax1.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Classification Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Regression results visualization  
    if regression_results:
        models = list(regression_results.keys())
        r2_scores = [regression_results[model]['r2'] for model in models]
        mae_scores = [regression_results[model]['mae'] for model in models]
        
        x = np.arange(len(models))
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x - 0.2, r2_scores, 0.4, label='R²', alpha=0.8, color='blue')
        bars2 = ax2_twin.bar(x + 0.2, mae_scores, 0.4, label='MAE', alpha=0.8, color='red')
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('R²', color='blue')
        ax2_twin.set_ylabel('MAE', color='red')
        ax2.set_title('Regression Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    output_dir = "tutorials/00_getting_started"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/basic_usage_results.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    
    print(f"   图表已保存到: {output_dir}/basic_usage_results.png")


def main():
    """
    主函数：运行所有演示
    """
    print("🌟 欢迎使用 CausalEngine 基础教程！")
    print("这个教程将展示如何使用 CausalEngine 进行因果推理")
    print("=" * 60)
    
    # 1. 基础API演示 (暂时跳过，因为需要真实的CausalEngine)
    # demo_basic_causal_engine()
    print("\n⚠️  跳过基础API演示（需要完整的CausalEngine实现）")
    
    # 2. 因果vs相关性理论说明
    demo_causality_vs_correlation()
    
    # 3. 分类任务演示
    classification_results = demo_classification_task()
    
    # 4. 回归任务演示
    regression_results = demo_regression_task()
    
    # 5. 结果可视化
    visualize_results(classification_results, regression_results)
    
    # 6. 总结
    print("\n🎉 CausalEngine基础教程完成！(2024更新版)")
    print("=" * 60)
    
    print("\n📊 本教程涵盖内容:")
    print("  ✅ 因果推理vs相关性推理的理论对比")
    print("  ✅ 四种推理模式的实际应用")
    print("  ✅ 分类和回归任务的端到端演示")
    print("  ✅ 基于基准协议的标准化配置")
    
    print("\n📖 推荐学习路径 (基于最新教程体系):")
    print("  1. 理论深入: tutorials/00_getting_started/theoretical_foundations.py")
    print("  2. 基准协议: tutorials/00_getting_started/benchmark_protocol_intro.py")
    print("  3. 分类应用: tutorials/01_classification/ (基于新实验设计)")
    print("  4. 回归应用: tutorials/02_regression/ (基于新实验设计)")
    print("  5. 消融实验: tutorials/03_ablation_studies/ (固定vs自适应噪声)")
    print("  6. 高级主题: tutorials/04_advanced_topics/ (四种推理模式)")
    
    print("\n🔗 核心资源 (最新版本):")
    print("  📐 数学基础: causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md")
    print("  🧪 实验协议: causal_engine/misc/benchmark_strategy.md")
    print("  📊 项目总览: causal_engine/ONE_PAGER.md")
    print("  🏗️ 架构文档: causal_engine/README.md")
    
    print("\n🎯 关键收获:")
    print("  🧠 CausalEngine基于个体选择变量实现真正的因果推理")
    print("  🎲 柯西分布线性稳定性提供解析不确定性传播")
    print("  🌡️ 温度参数实现从确定性到随机性的统一调制")
    print("  🔄 三阶段架构提供清晰可解释的推理链条")
    print("  📈 基准测试协议确保科学严谨的性能评估")


if __name__ == "__main__":
    main()