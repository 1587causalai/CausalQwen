#!/usr/bin/env python
"""
Experiment runner script.

This script runs experiments to evaluate the causal language model
on various datasets and configurations.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
from src.data.synthetic import TextWithNumbersGenerator
from src.data.tokenizer import MockTokenizer, QwenTokenizerWrapper
from src.evaluate import (
    evaluate_on_synthetic_data,
    evaluate_on_qa_data,
    evaluate_on_extreme_data,
    run_comprehensive_evaluation,
    compare_models
)
from src.utils.visualization import (
    plot_metrics,
    visualize_causal_state,
    visualize_decision_boundary,
    visualize_uncertainty,
    visualize_regression_performance,
    visualize_training_history,
    visualize_model_comparison
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run experiments for causal language model')
    
    parser.add_argument('--experiment', type=str, default='basic',
                        choices=['basic', 'comprehensive', 'comparison', 'ablation'],
                        help='Type of experiment to run')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run experiments on')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    parser.add_argument('--causal_dim', type=int, default=64,
                        help='Dimension of causal state')
    
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden size for feature network')
    
    parser.add_argument('--vocab_size', type=int, default=1000,
                        help='Vocabulary size')
    
    parser.add_argument('--use_real_qwen', action='store_true', default=False,
                        help='Use real Qwen2.5-0.5B model instead of mock implementation')
    
    parser.add_argument('--qwen_model_path', type=str, default='~/models/Qwen2.5-0.5B',
                        help='Path to Qwen model directory')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_tokenizer_and_config(args):
    """Create tokenizer and model configuration based on arguments."""
    if args.use_real_qwen:
        print("Using real Qwen model and tokenizer...")
        # Create Qwen tokenizer wrapper
        tokenizer = QwenTokenizerWrapper(
            model_path=args.qwen_model_path,
            use_real_tokenizer=True
        )
        
        # Create model configuration with real Qwen
        config = CausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_id=tokenizer.num_token_id,
            hidden_size=args.hidden_size,
            causal_dim=args.causal_dim,
            use_mock_feature_network=False,
            use_real_qwen=True,
            qwen_model_path=args.qwen_model_path
        )
    else:
        print("Using mock model and tokenizer...")
        # Create mock tokenizer
        tokenizer = MockTokenizer(vocab_size=args.vocab_size)
        
        # Create model configuration with mock feature network
        config = CausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_id=tokenizer.num_token_id,
            hidden_size=args.hidden_size,
            causal_dim=args.causal_dim,
            use_mock_feature_network=True,
            use_real_qwen=False
        )
    
    return tokenizer, config


def create_model(config):
    """Create a causal language model with the given configuration."""
    model = CausalLanguageModel(config)
    return model


def create_config_variants(base_tokenizer, base_config, args, variant_type='comparison'):
    """Create configuration variants for comparison and ablation experiments."""
    configs = {}
    
    if variant_type == 'comparison':
        # Comparison experiment variants (hyperparameter sensitivity)
        configs = {
            'base': base_config,
            'small_causal': CausalLMConfig(
                vocab_size=base_tokenizer.vocab_size,
                num_token_id=base_tokenizer.num_token_id,
                hidden_size=args.hidden_size,
                causal_dim=16,  # Smaller causal dimension
                use_mock_feature_network=not args.use_real_qwen,
                use_real_qwen=args.use_real_qwen,
                qwen_model_path=args.qwen_model_path if args.use_real_qwen else None
            ),
            'large_causal': CausalLMConfig(
                vocab_size=base_tokenizer.vocab_size,
                num_token_id=base_tokenizer.num_token_id,
                hidden_size=args.hidden_size,
                causal_dim=128,  # Larger causal dimension
                use_mock_feature_network=not args.use_real_qwen,
                use_real_qwen=args.use_real_qwen,
                qwen_model_path=args.qwen_model_path if args.use_real_qwen else None
            ),
            'high_reg_weight': CausalLMConfig(
                vocab_size=base_tokenizer.vocab_size,
                num_token_id=base_tokenizer.num_token_id,
                hidden_size=args.hidden_size,
                causal_dim=args.causal_dim,
                reg_loss_weight=2.0,  # Higher regression loss weight
                use_mock_feature_network=not args.use_real_qwen,
                use_real_qwen=args.use_real_qwen,
                qwen_model_path=args.qwen_model_path if args.use_real_qwen else None
            )
        }
    elif variant_type == 'ablation':
        # Ablation experiment variants (component contribution)
        configs = {
            'full_model': CausalLMConfig(
                vocab_size=base_tokenizer.vocab_size,
                num_token_id=base_tokenizer.num_token_id,
                hidden_size=args.hidden_size,
                causal_dim=args.causal_dim,
                use_mock_feature_network=not args.use_real_qwen,
                use_real_qwen=args.use_real_qwen,
                qwen_model_path=args.qwen_model_path if args.use_real_qwen else None,
                use_ovr_classifier=True,
                use_cauchy_distribution=True
            ),
            'no_ovr': CausalLMConfig(
                vocab_size=base_tokenizer.vocab_size,
                num_token_id=base_tokenizer.num_token_id,
                hidden_size=args.hidden_size,
                causal_dim=args.causal_dim,
                use_mock_feature_network=not args.use_real_qwen,
                use_real_qwen=args.use_real_qwen,
                qwen_model_path=args.qwen_model_path if args.use_real_qwen else None,
                use_ovr_classifier=False,  # Use softmax instead of OvR
                use_cauchy_distribution=True
            ),
            'no_cauchy': CausalLMConfig(
                vocab_size=base_tokenizer.vocab_size,
                num_token_id=base_tokenizer.num_token_id,
                hidden_size=args.hidden_size,
                causal_dim=args.causal_dim,
                use_mock_feature_network=not args.use_real_qwen,
                use_real_qwen=args.use_real_qwen,
                qwen_model_path=args.qwen_model_path if args.use_real_qwen else None,
                use_ovr_classifier=True,
                use_cauchy_distribution=False  # Use normal distribution instead of Cauchy
            ),
            'no_ovr_no_cauchy': CausalLMConfig(
                vocab_size=base_tokenizer.vocab_size,
                num_token_id=base_tokenizer.num_token_id,
                hidden_size=args.hidden_size,
                causal_dim=args.causal_dim,
                use_mock_feature_network=not args.use_real_qwen,
                use_real_qwen=args.use_real_qwen,
                qwen_model_path=args.qwen_model_path if args.use_real_qwen else None,
                use_ovr_classifier=False,
                use_cauchy_distribution=False
            )
        }
    
    return configs


def run_basic_experiment(args):
    """Run a basic experiment to validate the model."""
    print("Running basic experiment...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tokenizer and model configuration
    tokenizer, config = create_tokenizer_and_config(args)
    
    # Create model
    model = create_model(config)
    model.to(args.device)
    
    # Evaluate on synthetic data
    metrics = evaluate_on_synthetic_data(
        model,
        tokenizer,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed
    )
    
    # Plot metrics
    plot_metrics(metrics, output_dir=args.output_dir, prefix='basic')
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'basic_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Basic experiment completed.")
    print(f"Results saved to {args.output_dir}")


def run_comprehensive_experiment(args):
    """Run a comprehensive experiment to evaluate the model on various datasets."""
    print("Running comprehensive experiment...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tokenizer and model configuration
    tokenizer, config = create_tokenizer_and_config(args)
    
    # Create model
    model = create_model(config)
    model.to(args.device)
    
    # Run comprehensive evaluation
    all_metrics = run_comprehensive_evaluation(
        model,
        tokenizer,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    print("Comprehensive experiment completed.")
    print(f"Results saved to {args.output_dir}")


def run_comparison_experiment(args):
    """Run an experiment to compare different model configurations."""
    print("Running comparison experiment...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tokenizer and base configuration
    tokenizer, base_config = create_tokenizer_and_config(args)
    
    # Create configuration variants
    configs = create_config_variants(tokenizer, base_config, args, variant_type='comparison')
    
    # Create models
    models = {name: create_model(config).to(args.device) for name, config in configs.items()}
    
    # Compare models
    comparison_results = compare_models(
        models,
        tokenizer,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    print("Comparison experiment completed.")
    print(f"Results saved to {args.output_dir}")


def run_ablation_experiment(args):
    """Run an ablation study to evaluate the contribution of different components."""
    print("Running ablation experiment...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tokenizer and base configuration
    tokenizer, base_config = create_tokenizer_and_config(args)
    
    # Create configuration variants for ablation study
    configs = create_config_variants(tokenizer, base_config, args, variant_type='ablation')
    
    # Create models
    models = {name: create_model(config).to(args.device) for name, config in configs.items()}
    
    # Compare models
    comparison_results = compare_models(
        models,
        tokenizer,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    print("Ablation experiment completed.")
    print(f"Results saved to {args.output_dir}")


def main():
    """Main function to run experiments."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"{args.experiment}_{timestamp}")
    
    # Run the specified experiment
    if args.experiment == 'basic':
        run_basic_experiment(args)
    elif args.experiment == 'comprehensive':
        run_comprehensive_experiment(args)
    elif args.experiment == 'comparison':
        run_comparison_experiment(args)
    elif args.experiment == 'ablation':
        run_ablation_experiment(args)
    else:
        print(f"Unknown experiment type: {args.experiment}")
        sys.exit(1)


if __name__ == "__main__":
    main()

