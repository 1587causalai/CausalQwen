#!/usr/bin/env python
"""
门控策略对比实验

测试不同的门控系数 alpha 对模型性能的影响：
- alpha = 1.0: 无门控（默认）
- alpha = 0.5: 混合门控
- alpha = 0.1: 强门控
- alpha = 0.0: 完全门控
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.data.evaluation_data import get_all_evaluation_datasets
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator


def run_experiment(alpha, base_config, tokenizer, device, args):
    """运行单个门控策略实验"""
    print(f"\n{'='*60}")
    print(f"实验：门控系数 alpha = {alpha}")
    print(f"{'='*60}")
    
    # 创建配置（深拷贝以避免修改原始配置）
    from copy import deepcopy
    config = deepcopy(base_config)
    config.reg_loss_gating_alpha = alpha
    
    # 描述门控策略
    if alpha == 1.0:
        strategy = "无门控（硬掩码）"
        desc = "回归损失仅在 <NUM> 位置激活，无概率加权"
    elif alpha == 0.0:
        strategy = "完全门控（软注意力）"
        desc = "回归损失由 P(<NUM>) 概率完全控制"
    else:
        strategy = f"混合门控（{alpha:.0%} 基础 + {1-alpha:.0%} 概率）"
        desc = f"回归损失 = {alpha} + {1-alpha} * P(<NUM>)"
    
    print(f"策略: {strategy}")
    print(f"描述: {desc}")
    
    # 创建模型
    model = CausalLanguageModel(config).to(device)
    model.init_weights()
    
    # 训练
    print(f"\n开始训练...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        config=config  # 确保传递配置对象
    )
    
    training_metrics = trainer.train(
        num_epochs=args.epochs, 
        num_samples=args.num_samples
    )
    
    # 评估
    print(f"\n开始评估...")
    evaluator = Evaluator(model, tokenizer, device, config)
    datasets = get_all_evaluation_datasets(tokenizer)
    
    results = {}
    for name, dataset in datasets.items():
        print(f"  评估 {name} 数据集...")
        eval_results = evaluator.evaluate(dataset, batch_size=args.batch_size)
        results[name] = eval_results
        
        print(f"    - 分类 F1: {eval_results.get('cls_f1', 0):.4f}")
        print(f"    - 回归 MAE: {eval_results.get('reg_mae', 0):.4f}")
    
    return {
        'alpha': alpha,
        'strategy': strategy,
        'training_metrics': training_metrics,
        'eval_results': results
    }


def main(args):
    """主函数"""
    print_section("CausalQwen 知识迁移验证")
    
    # 设置
    device = torch.device('cpu')
    qwen_model_path = os.path.expanduser(args.qwen_model_path)
    
    # 初始化分词器
    print("\n初始化分词器...")
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    
    # 获取词汇表信息
    vocab_info = tokenizer.vocab_size_info()
    
    # 基础配置
    base_config = CausalLMConfig(
        vocab_size=vocab_info['causalqwen_vocab'],  # 使用 151936
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        causal_dim=896,
        use_real_qwen=True,
        use_mock_feature_network=False,  # 明确设置
        qwen_model_path=qwen_model_path,
        reg_loss_gating_alpha=1.0,  # 默认无门控
        reg_loss_weight=1.0
    )
    
    # 测试不同的门控策略
    alphas = [1.0, 0.5, 0.1, 0.0]
    all_results = []
    
    for alpha in alphas:
        result = run_experiment(alpha, base_config, tokenizer, device, args)
        all_results.append(result)
    
    # 总结结果
    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    
    print("\n门控策略性能对比:")
    print(f"{'Alpha':<8} {'策略':<20} {'Basic F1':<10} {'Basic MAE':<10}")
    print("-" * 60)
    
    for result in all_results:
        alpha = result['alpha']
        strategy = result['strategy'].split('（')[0]  # 简短描述
        basic_f1 = result['eval_results'].get('basic', {}).get('cls_f1', 0)
        basic_mae = result['eval_results'].get('basic', {}).get('reg_mae', 0)
        
        print(f"{alpha:<8.1f} {strategy:<20} {basic_f1:<10.4f} {basic_mae:<10.4f}")
    
    # 分析最佳策略
    print("\n🏆 分析:")
    
    # 找到最佳 F1
    best_f1_result = max(all_results, 
                         key=lambda r: r['eval_results'].get('basic', {}).get('cls_f1', 0))
    print(f"  最佳分类 (F1): alpha = {best_f1_result['alpha']}")
    
    # 找到最佳 MAE
    best_mae_result = min(all_results, 
                          key=lambda r: r['eval_results'].get('basic', {}).get('reg_mae', float('inf')))
    print(f"  最佳回归 (MAE): alpha = {best_mae_result['alpha']}")
    
    # 建议
    print("\n💡 建议:")
    if best_f1_result['alpha'] == 1.0 and best_mae_result['alpha'] == 1.0:
        print("  ✅ 无门控策略（alpha=1.0）在两项任务上都表现最佳")
        print("  这验证了我们的默认选择是正确的")
    elif best_f1_result['alpha'] == best_mae_result['alpha']:
        print(f"  ✅ alpha={best_f1_result['alpha']} 在两项任务上都表现最佳")
        print(f"  考虑将默认值改为 {best_f1_result['alpha']}")
    else:
        print(f"  ⚠️  分类和回归任务的最佳 alpha 不同")
        print(f"  可能需要针对具体任务调整 alpha")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gating_experiment_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 详细结果已保存至: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="门控策略对比实验")
    parser.add_argument('--qwen_model_path', type=str, 
                       default='~/models/Qwen2.5-0.5B',
                       help='Qwen 模型路径')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='训练样本数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    
    args = parser.parse_args()
    args.qwen_model_path = os.path.expanduser(args.qwen_model_path)
    
    main(args)
