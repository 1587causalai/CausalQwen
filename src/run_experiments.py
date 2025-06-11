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
        config_low_threshold.ovr_threshold = 1.0
        configs['low_threshold'] = config_low_threshold
        
        config_high_threshold = deepcopy(base_config)
        config_high_threshold.ovr_threshold = 50.0
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
    
    print(f"=== Running Experiment: {args.experiment} ===")
    print(f"Results will be saved to: {results_dir}")
    print(f"🧮 Using FIRST PRINCIPLES initialization strategy")
    print(f"   - All ActionNetwork biases set to 0.0 (no magic numbers)")
    print(f"   - Uncertainty expressed purely through AbductionNetwork scale_U")
    print(f"   - Mathematical consistency with Cauchy framework maintained")
    
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = QwenTokenizerWrapper(model_path=args.qwen_model_path, use_real_tokenizer=True)
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}, <NUM>_id={tokenizer.num_token_id}")
    
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
    
    print(f"Base config: hidden_size={base_config.hidden_size}, causal_dim={base_config.causal_dim}")
    print(f"             ovr_threshold={base_config.ovr_threshold}, reg_loss_weight={base_config.reg_loss_weight}")
    
    # --- 2. Get configurations and datasets ---
    model_configs = get_model_configs(base_config, args.experiment)
    evaluation_datasets = get_all_evaluation_datasets(tokenizer)
    
    if args.experiment == 'basic':
        # Basic experiment only runs on the basic dataset
        evaluation_datasets = {'basic': evaluation_datasets['basic']}
        print("Running basic experiment on basic dataset only")
    else:
        print(f"Running {args.experiment} experiment on {len(evaluation_datasets)} datasets")

    # --- 3. Run Experiment Loop ---
    all_results = {}
    for config_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"🚀 Running configuration: {config_name}")
        print(f"{'='*60}")
        
        # Print config differences from base
        if config_name != 'base' and config_name != 'full_model':
            print("Configuration differences from base:")
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
                    project="CausalQwen2-FirstPrinciples",  # Updated project name
                    name=f"{args.experiment}_{config_name}_{timestamp}",
                    config=asdict(config),
                    tags=[args.experiment, "first_principles_init"],  # Add tags
                    reinit=True # Allows multiple runs in one script
                )
                print("✅ Weights & Biases initialized successfully.")
            except Exception as e:
                print(f"⚠️  Could not initialize Weights & Biases. Error: {e}")
                wandb_run = None

        # Instantiate model
        model = CausalLanguageModel(config).to(device)
        print(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model if not skipped
        if not args.no_train:
            print("🎯 Training model with first-principles initialization...")
            
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
            print(f"💾 Saved trained model to {model_path}")
            
            # Log training summary
            if training_metrics:
                print(f"📈 Training completed:")
                print(f"   Final loss: {training_metrics.get('final_loss', 'N/A'):.4f}")
                print(f"   Final cls_loss: {training_metrics.get('final_cls_loss', 'N/A'):.4f}")
                print(f"   Final reg_loss: {training_metrics.get('final_reg_loss', 'N/A'):.4f}")
        
        # Evaluate model
        print("📊 Evaluating model...")
        evaluator = Evaluator(model, tokenizer, device, config)
        
        config_results = {}
        for name, dataset in evaluation_datasets.items():
            print(f"  📋 Evaluating on dataset: {name}")
            # Define path for saving raw evaluation outputs
            eval_output_path = os.path.join(results_dir, f"evaluation_outputs_{config_name}_{name}.pt")
            results = evaluator.evaluate(dataset, batch_size=args.batch_size, save_path=eval_output_path)
            config_results[name] = results
            
            # Enhanced result reporting
            cls_f1 = results.get('cls_f1', 0)
            reg_mae = results.get('reg_mae', 0)
            reg_picp = results.get('reg_picp', 0)
            print(f"    📊 Results: Cls F1: {cls_f1:.4f}, Reg MAE: {reg_mae:.4f}, Reg PICP: {reg_picp:.4f}")
            
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
            print("✅ Weights & Biases run finished.")

    # --- 4. Save all results and generate summary ---
    # Convert numpy types to standard Python types for JSON serialization
    all_results_serializable = convert_numpy_types(all_results)
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results_serializable, f, indent=4)
    
    # Generate experiment summary
    summary_path = os.path.join(results_dir, "experiment_summary.md")
    with open(summary_path, 'w') as f:
        f.write(f"# Experiment Summary: {args.experiment}\n\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Initialization Strategy:** First Principles (no magic number biases)\n")
        f.write(f"**Device:** {device}\n")
        f.write(f"**Base Config:** hidden_size={base_config.hidden_size}, causal_dim={base_config.causal_dim}\n\n")
        
        f.write("## Results Summary\n\n")
        for config_name, config_results in all_results_serializable.items():
            f.write(f"### Configuration: {config_name}\n\n")
            for dataset_name, metrics in config_results.items():
                f.write(f"**{dataset_name}:**\n")
                f.write(f"- Classification F1: {metrics.get('cls_f1', 0):.4f}\n")
                f.write(f"- Regression MAE: {metrics.get('reg_mae', 0):.4f}\n")
                f.write(f"- Regression PICP: {metrics.get('reg_picp', 0):.4f}\n\n")
    
    print(f"\n🎉 Experiment complete!")
    print(f"📁 All results saved to: {results_path}")
    print(f"📄 Summary saved to: {summary_path}")
    
    # Print best performing configuration
    if len(all_results) > 1:
        print(f"\n🏆 Performance Summary:")
        for dataset_name in evaluation_datasets.keys():
            best_f1_config = max(all_results.keys(), 
                               key=lambda k: all_results[k].get(dataset_name, {}).get('cls_f1', 0))
            best_f1_score = all_results[best_f1_config][dataset_name]['cls_f1']
            print(f"   {dataset_name} - Best Cls F1: {best_f1_config} ({best_f1_score:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Experiment Runner for Causal Language Model with First-Principles Initialization.")
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
    parser.add_argument('--ovr_threshold', type=float, default=100.0, help='OvR decision threshold.')
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

