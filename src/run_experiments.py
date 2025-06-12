#!/usr/bin/env python
"""
Unified Experiment Runner for the Causal Language Model.

This script orchestrates the execution of various experiments as defined
in the design documents, including:
- basic: A simple run to validate the baseline model's functionality.
- comprehensive: Evaluates the baseline model across multiple datasets.
- comparison: Compares model performance across different hyperparameter settings.
- ablation: Conducts an ablation study to validate core architectural choices.
- initialization: Compares different initialization strategies.
"""
import os
import sys
import torch
import json
import argparse
from datetime import datetime
from copy import deepcopy
import numpy as np
from dataclasses import asdict
import wandb

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.data.evaluation_data import get_all_evaluation_datasets
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def get_model_configs(base_config, experiment_type='ablation'):
    """
    Factory function to get model configurations for a given experiment type.

    Args:
        base_config (CausalLMConfig): The base configuration.
        experiment_type (str): The type of experiment ('ablation', 'comparison', etc.).

    Returns:
        dict: A dictionary mapping configuration names to CausalLMConfig objects.
    """
    configs = {}
    
    if experiment_type == 'ablation':
        # Full model (baseline for ablation)
        configs['full_model'] = deepcopy(base_config)
        
        # Ablation: No OVR (use Softmax)
        config_no_ovr = deepcopy(base_config)
        config_no_ovr.use_ovr_classifier = False
        configs['no_ovr'] = config_no_ovr
        
        # Ablation: No Cauchy (use Normal distribution)
        config_no_cauchy = deepcopy(base_config)
        config_no_cauchy.use_cauchy_distribution = False
        configs['no_cauchy'] = config_no_cauchy
        
        # Ablation: No OVR and No Cauchy (traditional baseline)
        config_no_ovr_no_cauchy = deepcopy(base_config)
        config_no_ovr_no_cauchy.use_ovr_classifier = False
        config_no_ovr_no_cauchy.use_cauchy_distribution = False
        configs['no_ovr_no_cauchy'] = config_no_ovr_no_cauchy

    elif experiment_type == 'comparison':
        # Base model (baseline for comparison)
        configs['base'] = deepcopy(base_config)
        
        # Comparison: Different OvR thresholds
        config_low_threshold = deepcopy(base_config)
        config_low_threshold.ovr_threshold = 1.0
        configs['low_threshold'] = config_low_threshold
        
        config_high_threshold = deepcopy(base_config)
        config_high_threshold.ovr_threshold = 100.0
        configs['high_threshold'] = config_high_threshold
        
        # Comparison: Different causal dimensions (only if not using identity mapping)
        if base_config.causal_dim != base_config.hidden_size:
            config_small_causal = deepcopy(base_config)
            config_small_causal.causal_dim = 64
            configs['small_causal'] = config_small_causal
            
            config_large_causal = deepcopy(base_config)
            config_large_causal.causal_dim = 256
            configs['large_causal'] = config_large_causal
        else:
            print("📌 跳过因果维度比较（当前使用恒等映射，causal_dim == hidden_size）")
        
        # Comparison: Different regression loss weights
        config_high_reg_weight = deepcopy(base_config)
        config_high_reg_weight.reg_loss_weight = 2.0
        configs['high_reg_weight'] = config_high_reg_weight
        
        config_low_reg_weight = deepcopy(base_config)
        config_low_reg_weight.reg_loss_weight = 0.5
        configs['low_reg_weight'] = config_low_reg_weight
        
    elif experiment_type == 'initialization':
        # Initialization strategy comparison experiment
        
        # Default initialization (our standard approach)
        configs['default'] = deepcopy(base_config)
        
        # Different initial uncertainty levels
        config_low_uncertainty = deepcopy(base_config)
        config_low_uncertainty.initial_scale_bias = 1.0  # exp(1.0) ≈ 2.7
        configs['low_uncertainty'] = config_low_uncertainty
        
        config_high_uncertainty = deepcopy(base_config)
        config_high_uncertainty.initial_scale_bias = 3.0  # exp(3.0) ≈ 20
        configs['high_uncertainty'] = config_high_uncertainty
        
        # No identity mapping (test Xavier initialization)
        if base_config.hidden_size != 64:  # Avoid creating duplicate if already 64
            config_no_identity = deepcopy(base_config)
            config_no_identity.causal_dim = 64  # Force different dimension
            configs['no_identity_mapping'] = config_no_identity
        
    elif experiment_type in ['basic', 'comprehensive']:
        configs['base'] = deepcopy(base_config)
        
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
        
    return configs

def print_experiment_header(experiment_type, timestamp, device):
    """Print a formatted experiment header with key information."""
    print("\n" + "="*80)
    print(f"🚀 CausalQwen 实验运行器")
    print("="*80)
    print(f"📅 时间戳: {timestamp}")
    print(f"🧪 实验类型: {experiment_type}")
    print(f"💻 设备: {device}")
    print(f"🏗️  架构版本: 推断-行动范式 v3")
    print("="*80 + "\n")

def print_config_info(config_name, config, base_config=None):
    """Print configuration information in a formatted way."""
    print(f"\n{'='*60}")
    print(f"⚙️  配置: {config_name}")
    print(f"{'='*60}")
    
    # Print key configuration parameters
    print(f"📊 模型参数:")
    print(f"   - 词汇表大小: {config.vocab_size:,} (Qwen 完整配置容量)")
    print(f"   - 隐藏维度: {config.hidden_size}")
    print(f"   - 因果维度: {config.causal_dim}")
    print(f"   - 恒等映射: {'是' if config.causal_dim == config.hidden_size else '否'}")
    
    print(f"\n🎯 训练设置:")
    print(f"   - OvR 分类: {'是' if config.use_ovr_classifier else '否 (Softmax)'}")
    print(f"   - 柯西分布: {'是' if config.use_cauchy_distribution else '否 (正态分布)'}")
    print(f"   - OvR 阈值: {config.ovr_threshold}")
    print(f"   - 回归损失权重: {config.reg_loss_weight}")
    print(f"   - 初始不确定性: exp({config.initial_scale_bias}) ≈ {np.exp(config.initial_scale_bias):.1f}")
    
    # Print differences from base config if provided
    if base_config and config_name not in ['base', 'full_model', 'default']:
        print(f"\n🔄 与基准配置的差异:")
        differences = []
        for attr in ['causal_dim', 'use_ovr_classifier', 'use_cauchy_distribution', 
                     'ovr_threshold', 'reg_loss_weight', 'initial_scale_bias']:
            if hasattr(config, attr) and hasattr(base_config, attr):
                base_val = getattr(base_config, attr)
                config_val = getattr(config, attr)
                if base_val != config_val:
                    differences.append(f"   - {attr}: {base_val} → {config_val}")
        
        if differences:
            for diff in differences:
                print(diff)
        else:
            print("   - 无差异")

def main(args):
    """Main function to orchestrate the experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_base_dir, f"{args.experiment}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Print experiment header
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_experiment_header(args.experiment, timestamp, device)
    
    # --- 1. Setup ---
    print("📚 初始化分词器...")
    tokenizer = QwenTokenizerWrapper(model_path=args.qwen_model_path, use_real_tokenizer=True)
    print(f"✅ 分词器加载完成")
    
    # 获取详细的词汇表信息
    vocab_info = tokenizer.vocab_size_info()
    print(f"   - Qwen 配置容量: {vocab_info['config_capacity']:,}")
    print(f"   - Qwen 实际使用: {vocab_info['qwen_used']:,}")
    print(f"   - CausalQwen 词汇表: {vocab_info['causalqwen_vocab']:,}")
    print(f"   - <NUM> token ID: {vocab_info['num_token_id']}")
    print(f"   - 预留槽位: {vocab_info['reserved_slots']} (已用 {vocab_info['reserved_used']}, 剩余 {vocab_info['reserved_remaining']})")
    
    # Create base configuration - 使用完整的词汇表大小
    base_config = CausalLMConfig(
        vocab_size=vocab_info['causalqwen_vocab'],  # 使用 151936 而非 151666
        num_token_id=tokenizer.num_token_id,
        hidden_size=args.hidden_size,
        # Force causal_dim = hidden_size for identity mapping by default
        causal_dim=args.hidden_size,
        use_real_qwen=True,
        use_mock_feature_network=False,  # 明确设置
        qwen_model_path=args.qwen_model_path,
        ovr_threshold=args.ovr_threshold,
        reg_loss_weight=args.reg_loss_weight,
        reg_loss_gating_alpha=getattr(args, 'reg_loss_gating_alpha', 1.0),  # 默认无门控
        initial_scale_bias=getattr(args, 'initial_scale_bias', 2.3)  # Default: exp(2.3) ≈ 10
    )
    
    print(f"\n📋 基础配置创建完成")
    print(f"   - 使用预训练 Qwen: {base_config.use_real_qwen}")
    print(f"   - 模型路径: {base_config.qwen_model_path}")
    
    # --- 2. Get configurations and datasets ---
    print(f"\n🔧 生成实验配置...")
    model_configs = get_model_configs(base_config, args.experiment)
    print(f"✅ 生成 {len(model_configs)} 个配置变体")
    
    print(f"\n📊 加载评估数据集...")
    evaluation_datasets = get_all_evaluation_datasets(tokenizer)
    
    if args.experiment == 'basic':
        # Basic experiment only runs on the basic dataset
        evaluation_datasets = {'basic': evaluation_datasets['basic']}
        print("✅ 基础实验模式：仅使用 basic 数据集")
    else:
        print(f"✅ 加载 {len(evaluation_datasets)} 个评估数据集")
        for name in evaluation_datasets.keys():
            print(f"   - {name}")

    # Save experiment metadata
    experiment_info = {
        'timestamp': timestamp,
        'experiment_type': args.experiment,
        'device': str(device),
        'base_config': asdict(base_config),
        'num_configs': len(model_configs),
        'num_datasets': len(evaluation_datasets),
        'training_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_training_samples': args.num_samples
    }
    
    with open(os.path.join(results_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=4)
    
    print(f"\n💾 实验元数据已保存")
    print(f"   - 结果目录: {results_dir}")

    # --- 3. Run Experiment Loop ---
    all_results = {}
    
    for config_idx, (config_name, config) in enumerate(model_configs.items()):
        print_config_info(config_name, config, base_config)
        
        # --- WandB Initialization ---
        wandb_run = None
        if args.use_wandb:
            try:
                wandb_run = wandb.init(
                    project="CausalQwen",
                    name=f"{args.experiment}_{config_name}_{timestamp}",
                    config=asdict(config),
                    tags=[args.experiment, "v3", "abduction-action"],
                    reinit=True
                )
                print("\n📊 Weights & Biases 初始化成功")
            except Exception as e:
                print(f"\n⚠️  无法初始化 Weights & Biases: {e}")
                wandb_run = None

        # Instantiate model
        print(f"\n🏗️  创建模型...")
        model = CausalLanguageModel(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ 模型创建完成")
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        
        # Initialize model weights
        print(f"\n🔧 初始化模型权重...")
        model.init_weights()
        print("✅ 权重初始化完成")
        print("   - 归因推断网络: 恒等映射" if config.causal_dim == config.hidden_size else "   - 归因推断网络: Xavier 初始化")
        print("   - 分类头: 从 Qwen lm_head 完整迁移 (151,936 个权重)")
        print("   - 回归头: 小随机初始化 (gain=0.01)")
        
        # Train model if not skipped
        if not args.no_train:
            print(f"\n🎯 开始训练...")
            print(f"   - 训练轮数: {args.epochs}")
            print(f"   - 批次大小: {args.batch_size}")
            print(f"   - 学习率: {args.lr}")
            print(f"   - 训练样本数: {args.num_samples}")
            
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                config=config,
                wandb_run=wandb_run
            )
            
            # Train and monitor key metrics
            training_metrics = trainer.train(num_epochs=args.epochs, num_samples=args.num_samples)
            
            # Save the trained model
            model_path = os.path.join(results_dir, f"model_{config_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"\n💾 模型已保存: {model_path}")
            
            # Log training summary
            if training_metrics:
                print(f"\n📈 训练完成摘要:")
                print(f"   - 最终总损失: {training_metrics.get('final_loss', 'N/A'):.4f}")
                print(f"   - 最终分类损失: {training_metrics.get('final_cls_loss', 'N/A'):.4f}")
                print(f"   - 最终回归损失: {training_metrics.get('final_reg_loss', 'N/A'):.4f}")
                
                if wandb_run:
                    wandb_run.summary.update({
                        'final_loss': training_metrics.get('final_loss', 0),
                        'final_cls_loss': training_metrics.get('final_cls_loss', 0),
                        'final_reg_loss': training_metrics.get('final_reg_loss', 0)
                    })
        else:
            print(f"\n⏭️  跳过训练（--no_train 模式）")
        
        # Evaluate model
        print(f"\n📊 开始模型评估...")
        evaluator = Evaluator(model, tokenizer, device, config)
        
        config_results = {}
        for dataset_idx, (name, dataset) in enumerate(evaluation_datasets.items()):
            print(f"\n📋 评估数据集 [{dataset_idx+1}/{len(evaluation_datasets)}]: {name}")
            
            # Define path for saving raw evaluation outputs
            eval_output_path = os.path.join(results_dir, f"eval_{config_name}_{name}.pt")
            
            # Evaluate
            results = evaluator.evaluate(dataset, batch_size=args.batch_size, save_path=eval_output_path)
            config_results[name] = results
            
            # Report results
            print(f"   ✅ 评估完成")
            print(f"      - 分类 F1: {results.get('cls_f1', 0):.4f}")
            print(f"      - 分类精确率: {results.get('cls_precision', 0):.4f}")
            print(f"      - 分类召回率: {results.get('cls_recall', 0):.4f}")
            print(f"      - 回归 MAE: {results.get('reg_mae', 0):.4f}")
            print(f"      - 回归 RMSE: {results.get('reg_rmse', 0):.4f}")
            print(f"      - 回归 PICP: {results.get('reg_picp', 0):.4f}")
            
            # Log to wandb if available
            if wandb_run:
                wandb_run.log({
                    f"{name}/cls_f1": results.get('cls_f1', 0),
                    f"{name}/cls_precision": results.get('cls_precision', 0),
                    f"{name}/cls_recall": results.get('cls_recall', 0),
                    f"{name}/reg_mae": results.get('reg_mae', 0),
                    f"{name}/reg_rmse": results.get('reg_rmse', 0),
                    f"{name}/reg_picp": results.get('reg_picp', 0)
                })
            
        all_results[config_name] = config_results
        
        # Finish WandB run
        if wandb_run:
            wandb_run.finish()
            print("\n📊 Weights & Biases 运行已结束")
        
        print(f"\n✅ 配置 [{config_idx+1}/{len(model_configs)}] 完成: {config_name}")

    # --- 4. Save all results and generate summary ---
    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    
    # Convert numpy types to standard Python types for JSON serialization
    all_results_serializable = convert_numpy_types(all_results)
    
    # Save detailed results
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results_serializable, f, indent=4)
    print(f"\n💾 详细结果已保存: {results_path}")
    
    # Generate experiment summary
    summary_path = os.path.join(results_dir, "summary.md")
    with open(summary_path, 'w') as f:
        f.write(f"# CausalQwen 实验总结\n\n")
        f.write(f"**实验类型:** {args.experiment}\n")
        f.write(f"**时间戳:** {timestamp}\n")
        f.write(f"**设备:** {device}\n")
        f.write(f"**架构:** 推断-行动范式 v3\n\n")
        
        f.write("## 配置总览\n\n")
        f.write(f"- **基础模型:** Qwen-{args.hidden_size/1000:.1f}B\n")
        f.write(f"- **Qwen 已用词汇:** 151,665\n")
        f.write(f"- **CausalQwen 词汇表:** {base_config.vocab_size} (151,665 + 1)\n")
        f.write(f"- **Qwen 配置容量:** 151,936 (含 271 个预留)\n")
        f.write(f"- **隐藏维度:** {base_config.hidden_size}\n")
        f.write(f"- **因果维度:** {base_config.causal_dim}\n")
        f.write(f"- **恒等映射:** {'启用' if base_config.causal_dim == base_config.hidden_size else '禁用'}\n\n")
        
        f.write("## 实验结果\n\n")
        
        # Create results table
        if len(evaluation_datasets) > 0:
            # Table header
            f.write("| 配置 | ")
            for dataset_name in evaluation_datasets.keys():
                f.write(f"{dataset_name} F1 | {dataset_name} MAE | ")
            f.write("\n")
            
            # Table separator
            f.write("|------|")
            for _ in evaluation_datasets.keys():
                f.write("-------|--------|")
            f.write("\n")
            
            # Table rows
            for config_name, config_results in all_results_serializable.items():
                f.write(f"| {config_name} | ")
                for dataset_name in evaluation_datasets.keys():
                    metrics = config_results.get(dataset_name, {})
                    f1 = metrics.get('cls_f1', 0)
                    mae = metrics.get('reg_mae', 0)
                    f.write(f"{f1:.4f} | {mae:.4f} | ")
                f.write("\n")
        
        # Best performing configurations
        if len(all_results) > 1 and len(evaluation_datasets) > 0:
            f.write("\n## 最佳配置\n\n")
            for dataset_name in evaluation_datasets.keys():
                # Best F1
                best_f1_config = max(all_results.keys(), 
                                   key=lambda k: all_results[k].get(dataset_name, {}).get('cls_f1', 0))
                best_f1_score = all_results[best_f1_config][dataset_name]['cls_f1']
                
                # Best MAE (lower is better)
                best_mae_config = min(all_results.keys(), 
                                    key=lambda k: all_results[k].get(dataset_name, {}).get('reg_mae', float('inf')))
                best_mae_score = all_results[best_mae_config][dataset_name]['reg_mae']
                
                f.write(f"### {dataset_name} 数据集\n")
                f.write(f"- **最佳分类 (F1):** {best_f1_config} ({best_f1_score:.4f})\n")
                f.write(f"- **最佳回归 (MAE):** {best_mae_config} ({best_mae_score:.4f})\n\n")
    
    print(f"💾 实验总结已保存: {summary_path}")
    
    # Print summary to console
    print(f"\n🏆 性能总结:")
    if len(all_results) > 1 and len(evaluation_datasets) > 0:
        for dataset_name in evaluation_datasets.keys():
            best_f1_config = max(all_results.keys(), 
                               key=lambda k: all_results[k].get(dataset_name, {}).get('cls_f1', 0))
            best_f1_score = all_results[best_f1_config][dataset_name]['cls_f1']
            print(f"   {dataset_name}:")
            print(f"      - 最佳 F1: {best_f1_config} ({best_f1_score:.4f})")
    
    print(f"\n🎉 实验完成！所有结果已保存到: {results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Experiment Runner for Causal Language Model.")
    parser.add_argument(
        'experiment', 
        type=str, 
        choices=['basic', 'comprehensive', 'comparison', 'ablation', 'initialization'],
        help='The type of experiment to run.'
    )
    parser.add_argument(
        '--qwen_model_path', 
        type=str, 
        default='~/models/Qwen2.5-0.5B',
        help='Path to the pre-trained Qwen model.'
    )
    parser.add_argument(
        '--results_base_dir', 
        type=str, 
        default='results',
        help='Base directory to save experiment results.'
    )
    
    # Model architecture args
    parser.add_argument('--hidden_size', type=int, default=896, help='Hidden size of the model (for Qwen-0.5B).')
    parser.add_argument('--ovr_threshold', type=float, default=10.0, help='OvR decision threshold.')
    parser.add_argument('--reg_loss_weight', type=float, default=1.0, help='Weight for regression loss in total loss.')
    parser.add_argument('--initial_scale_bias', type=float, default=2.3, help='Initial bias for scale parameter (log scale).')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('--no_train', action='store_true', help="Skip training and only run evaluation.")
    parser.add_argument('--use_wandb', action='store_true', help="Use Weights & Biases for logging.")

    args = parser.parse_args()
    
    # Expand user path
    args.qwen_model_path = os.path.expanduser(args.qwen_model_path)
    
    main(args)

