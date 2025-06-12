#!/usr/bin/env python
"""
Unified Experiment Runner for the Causal Language Model.

This script orchestrates the execution of various experiments as defined
in the design documents, including:
- basic: A simple run to validate the baseline model's functionality.
- comprehensive: Evaluates the baseline model across multiple datasets.
- comparison: Compares model performance across different hyperparameter settings.
- ablation: Conducts an ablation study to validate core architectural choices.
- initialization: Compares first-principles vs heuristic initialization strategies.
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
    
    Args:
        obj: Any object that might contain numpy types.
        
    Returns:
        Object with numpy types converted to standard Python types.
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
        # Full model (baseline for ablation) - uses first-principles initialization
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
        # Base model (baseline for comparison) - uses first-principles initialization
        configs['base'] = deepcopy(base_config)
        
        # Comparison: Different OvR thresholds
        config_low_threshold = deepcopy(base_config)
        config_low_threshold.ovr_threshold = 100.0  # 从 1.0 改为 100.0
        configs['low_threshold'] = config_low_threshold
        
        config_high_threshold = deepcopy(base_config)
        config_high_threshold.ovr_threshold = 100000.0  # 从 50.0 改为 100000.0
        configs['high_threshold'] = config_high_threshold
        
        # Comparison: Different causal dimensions (only if not using identity mapping)
        if base_config.causal_dim != base_config.hidden_size:
            config_small_causal = deepcopy(base_config)
            config_small_causal.causal_dim = 64
            configs['small_causal'] = config_small_causal
            
            config_large_causal = deepcopy(base_config)
            config_large_causal.causal_dim = 256
            configs['large_causal'] = config_large_causal
        
        # Comparison: Different regression loss weights
        config_high_reg_weight = deepcopy(base_config)
        config_high_reg_weight.reg_loss_weight = 2.0
        configs['high_reg_weight'] = config_high_reg_weight
        
        config_low_reg_weight = deepcopy(base_config)
        config_low_reg_weight.reg_loss_weight = 0.5
        configs['low_reg_weight'] = config_low_reg_weight
        
    elif experiment_type == 'initialization':
        # NEW: Initialization strategy comparison experiment
        
        # First-principles initialization (our new approach)
        configs['first_principles'] = deepcopy(base_config)
        # This will use the default first-principles initialization in ActionNetwork
        
        # Note: We can't easily test the old heuristic initialization without 
        # modifying the ActionNetwork code, but we can document the differences
        # and compare against models trained with different initialization strategies
        
        # Different initial uncertainty levels
        config_low_uncertainty = deepcopy(base_config)
        config_low_uncertainty.initial_scale_bias = 1.0  # exp(1.0) ≈ 2.7
        configs['low_uncertainty'] = config_low_uncertainty
        
        config_high_uncertainty = deepcopy(base_config)
        config_high_uncertainty.initial_scale_bias = 3.0  # exp(3.0) ≈ 20
        configs['high_uncertainty'] = config_high_uncertainty
        
    elif experiment_type in ['basic', 'comprehensive']:
        configs['base'] = deepcopy(base_config)
        
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
        
    return configs

def main(args):
    """Main function to orchestrate the experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_base_dir, f"{args.experiment}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"=== 运行实验: {args.experiment} ===")
    print(f"结果将保存到: {results_dir}")
    print(f"🧮 使用更新的初始化策略")
    print(f"   - 分类头：完全复用 Qwen 的 lm_head（包括权重和偏置）")
    print(f"   - 回归头：零初始化")
    print(f"   - 保留词汇：自动处理，无需特殊配置")
    
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    tokenizer = QwenTokenizerWrapper(model_path=args.qwen_model_path, use_real_tokenizer=True)
    print(f"加载分词器: vocab_size={tokenizer.vocab_size}, <NUM>_id={tokenizer.num_token_id}")
    
    base_config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=args.hidden_size,
        # CRITICAL: Force causal_dim = hidden_size for identity mapping initialization
        # This ensures AbductionNetwork uses identity mapping (C=H constraint)
        causal_dim=args.hidden_size,
        use_real_qwen=True,
        qwen_model_path=args.qwen_model_path,
        ovr_threshold=args.ovr_threshold,
        reg_loss_weight=args.reg_loss_weight,
        # Add support for different initial uncertainty levels
        initial_scale_bias=getattr(args, 'initial_scale_bias', 2.3)  # Default: exp(2.3) ≈ 10
    )
    
    print(f"基础配置: hidden_size={base_config.hidden_size}, causal_dim={base_config.causal_dim}")
    print(f"         ovr_threshold={base_config.ovr_threshold}, reg_loss_weight={base_config.reg_loss_weight}")
    
    # --- 2. Get configurations and datasets ---
    model_configs = get_model_configs(base_config, args.experiment)
    evaluation_datasets = get_all_evaluation_datasets(tokenizer)
    
    if args.experiment == 'basic':
        # Basic experiment only runs on the basic dataset
        evaluation_datasets = {'basic': evaluation_datasets['basic']}
        print("运行基础实验，仅在 basic 数据集上测试")
    else:
        print(f"运行 {args.experiment} 实验，共 {len(evaluation_datasets)} 个数据集")

    # --- 3. Run Experiment Loop ---
    all_results = {}
    for config_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"🚀 运行配置: {config_name}")
        print(f"{'='*60}")
        
        # Print config differences from base
        if config_name != 'base' and config_name != 'full_model':
            print("配置与基础配置的差异:")
            for attr in dir(config):
                if not attr.startswith('_') and hasattr(base_config, attr):
                    base_val = getattr(base_config, attr)
                    config_val = getattr(config, attr)
                    if base_val != config_val:
                        print(f"  {attr}: {base_val} → {config_val}")
        
        # --- WandB Initialization ---
        wandb_run = None
        if args.use_wandb:
            try:
                wandb_run = wandb.init(
                    project="CausalQwen2",  # 移除 FirstPrinciples 后缀
                    name=f"{args.experiment}_{config_name}_{timestamp}",
                    config=asdict(config),
                    tags=[args.experiment, "updated_init"],  # 更新标签
                    reinit=True
                )
                print("✅ Weights & Biases 初始化成功")
            except Exception as e:
                print(f"⚠️  无法初始化 Weights & Biases。错误: {e}")
                wandb_run = None

        # Instantiate model
        model = CausalLanguageModel(config).to(device)
        print(f"📊 模型创建完成，共 {sum(p.numel() for p in model.parameters()):,} 个参数")
        
        # Initialize model weights with knowledge transfer
        print("🔧 初始化模型权重（知识传输）...")
        model.init_weights()  # 不再需要传递数值统计参数
        print("✅ 模型初始化完成")
        
        # Train model if not skipped
        if not args.no_train:
            print("🎯 开始训练模型...")
            
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
            print(f"💾 已保存训练后的模型到 {model_path}")
            
            # Log training summary
            if training_metrics:
                print(f"📈 训练完成:")
                print(f"   最终损失 (Final loss): {training_metrics.get('final_loss', 'N/A'):.4f}")
                print(f"   最终分类损失 (Final cls_loss): {training_metrics.get('final_cls_loss', 'N/A'):.4f}")
                print(f"   最终回归损失 (Final reg_loss): {training_metrics.get('final_reg_loss', 'N/A'):.4f}")
        
        # Evaluate model
        print("📊 评估模型...")
        evaluator = Evaluator(model, tokenizer, device, config)
        
        config_results = {}
        for name, dataset in evaluation_datasets.items():
            print(f"  📋 在数据集 {name} 上评估")
            # Define path for saving raw evaluation outputs
            eval_output_path = os.path.join(results_dir, f"evaluation_outputs_{config_name}_{name}.pt")
            results = evaluator.evaluate(dataset, batch_size=args.batch_size, save_path=eval_output_path)
            config_results[name] = results
            
            # Enhanced result reporting
            cls_f1 = results.get('cls_f1', 0)
            reg_mae = results.get('reg_mae', 0)
            reg_picp = results.get('reg_picp', 0)
            print(f"    📊 结果: 分类 F1: {cls_f1:.4f}, 回归 MAE: {reg_mae:.4f}, 回归 PICP: {reg_picp:.4f}")
            
            # Log to wandb if available
            if wandb_run:
                wandb_run.log({
                    f"{name}_cls_f1": cls_f1,
                    f"{name}_reg_mae": reg_mae,
                    f"{name}_reg_picp": reg_picp
                })
            
        all_results[config_name] = config_results

        # --- Finish WandB Run ---
        if wandb_run:
            wandb_run.finish()
            print("✅ Weights & Biases 运行结束")

    # --- 4. Save all results and generate summary ---
    # Convert numpy types to standard Python types for JSON serialization
    all_results_serializable = convert_numpy_types(all_results)
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results_serializable, f, indent=4)
    
    # Generate experiment summary
    summary_path = os.path.join(results_dir, "experiment_summary.md")
    with open(summary_path, 'w') as f:
        f.write(f"# 实验总结: {args.experiment}\n\n")
        f.write(f"**时间戳:** {timestamp}\n")
        f.write(f"**初始化策略:** 完全复用 Qwen lm_head + 零初始化回归头\n")
        f.write(f"**设备:** {device}\n")
        f.write(f"**基础配置:** hidden_size={base_config.hidden_size}, causal_dim={base_config.causal_dim}\n\n")
        
        f.write("## 结果总结\n\n")
        for config_name, config_results in all_results_serializable.items():
            f.write(f"### 配置: {config_name}\n\n")
            for dataset_name, metrics in config_results.items():
                f.write(f"**{dataset_name}:**\n")
                f.write(f"- 分类 F1 分数: {metrics.get('cls_f1', 0):.4f}\n")
                f.write(f"- 回归 MAE: {metrics.get('reg_mae', 0):.4f}\n")
                f.write(f"- 回归 PICP: {metrics.get('reg_picp', 0):.4f}\n\n")
    
    print(f"\n🎉 实验完成！")
    print(f"📁 所有结果已保存到: {results_path}")
    print(f"📄 实验总结已保存到: {summary_path}")
    
    # Print best performing configuration
    if len(all_results) > 1:
        print(f"\n🏆 性能总结:")
        for dataset_name in evaluation_datasets.keys():
            best_f1_config = max(all_results.keys(), 
                               key=lambda k: all_results[k].get(dataset_name, {}).get('cls_f1', 0))
            best_f1_score = all_results[best_f1_config][dataset_name]['cls_f1']
            print(f"   {dataset_name} - 最佳分类 F1: {best_f1_config} ({best_f1_score:.4f})")

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
    parser.add_argument('--ovr_threshold', type=float, default=10000.0, help='OvR decision threshold.')  # 从 100.0 改为 10000.0
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

