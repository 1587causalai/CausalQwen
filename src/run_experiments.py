#!/usr/bin/env python
"""
Unified Experiment Runner for the Causal Language Model.

This script orchestrates the execution of various experiments as defined
in the design documents, including:
- basic: A simple run to validate the baseline model's functionality.
- comprehensive: Evaluates the baseline model across multiple datasets.
- comparison: Compares model performance across different hyperparameter settings.
- ablation: Conducts an ablation study to validate core architectural choices.
"""
import os
import sys
import torch
import json
import argparse
from datetime import datetime
from copy import deepcopy
import numpy as np

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
        
        # Comparison: Smaller causal dimension
        config_small_causal = deepcopy(base_config)
        config_small_causal.causal_dim = 16
        configs['small_causal'] = config_small_causal
        
        # Comparison: Larger causal dimension
        config_large_causal = deepcopy(base_config)
        config_large_causal.causal_dim = 128
        configs['large_causal'] = config_large_causal
        
        # Comparison: Higher regression loss weight
        config_high_reg_weight = deepcopy(base_config)
        config_high_reg_weight.reg_loss_weight = 2.0
        configs['high_reg_weight'] = config_high_reg_weight
        
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
    
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = QwenTokenizerWrapper(model_path=args.qwen_model_path, use_real_tokenizer=True)
    
    base_config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=args.hidden_size,
        causal_dim=args.causal_dim,
        use_real_qwen=True,
        qwen_model_path=args.qwen_model_path
    )
    
    # --- 2. Get configurations and datasets ---
    model_configs = get_model_configs(base_config, args.experiment)
    evaluation_datasets = get_all_evaluation_datasets(tokenizer)
    
    if args.experiment == 'basic':
        # Basic experiment only runs on the basic dataset
        evaluation_datasets = {'basic': evaluation_datasets['basic']}

    # --- 3. Run Experiment Loop ---
    all_results = {}
    for config_name, config in model_configs.items():
        print(f"\n--- Running configuration: {config_name} ---")
        
        # Instantiate model
        model = CausalLanguageModel(config).to(device)
        
        # Train model if not skipped
        if not args.no_train:
            print("Training model...")
            # Apply custom weight initialization for new components
            # This is crucial to prevent gradient explosion
            if hasattr(model, 'abduction_network'):
                 model.abduction_network.apply(Trainer.weights_init)
            if hasattr(model, 'action_network'):
                 model.action_network.apply(Trainer.weights_init)

            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                config=config
            )
            trainer.train(num_epochs=args.epochs, num_samples=args.num_samples)
            
            # Save the trained model
            model_path = os.path.join(results_dir, f"model_{config_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved trained model to {model_path}")
        
        # Evaluate model
        print("Evaluating model...")
        evaluator = Evaluator(model, tokenizer, device, config)
        
        config_results = {}
        for name, dataset in evaluation_datasets.items():
            print(f"  on dataset: {name}")
            results = evaluator.evaluate(dataset, batch_size=args.batch_size)
            config_results[name] = results
            print(f"    -> Cls F1: {results.get('cls_f1', 0):.4f}, Reg MAE: {results.get('reg_mae', 0):.4f}, Reg PICP: {results.get('reg_picp', 0):.4f}")
            
        all_results[config_name] = config_results

    # --- 4. Save all results ---
    # Convert numpy types to standard Python types for JSON serialization
    all_results_serializable = convert_numpy_types(all_results)
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results_serializable, f, indent=4)
    print(f"\nExperiment complete. All results saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Experiment Runner for Causal Language Model.")
    parser.add_argument(
        'experiment', 
        type=str, 
        choices=['basic', 'comprehensive', 'comparison', 'ablation'],
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
        default='docs/results',
        help='Base directory to save experiment results.'
    )
    # Model architecture args
    parser.add_argument('--hidden_size', type=int, default=896, help='Hidden size of the model (for Qwen-0.5B).')
    parser.add_argument('--causal_dim', type=int, default=64, help='Dimension of the causal state.')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('--no_train', action='store_true', help="Skip training and only run evaluation.")

    args = parser.parse_args()
    
    # Expand user path
    args.qwen_model_path = os.path.expanduser(args.qwen_model_path)
    
    main(args)

