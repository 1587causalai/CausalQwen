#!/usr/bin/env python
"""
Qwen experiment runner script.

This script runs experiments using real Qwen2.5-0.5B model to evaluate 
the causal language model architecture with a real language model backbone.
"""

import os
import sys
import argparse
from datetime import datetime

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.run_experiments import (
    run_basic_experiment, 
    run_comprehensive_experiment, 
    run_comparison_experiment, 
    run_ablation_experiment,
    set_seed
)


def parse_qwen_args():
    """Parse command line arguments for Qwen experiments."""
    parser = argparse.ArgumentParser(description='Run experiments with real Qwen2.5-0.5B model')
    
    parser.add_argument('--experiment', type=str, default='basic',
                        choices=['basic', 'comprehensive', 'comparison', 'ablation'],
                        help='Type of experiment to run')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run experiments on (auto/cpu/cuda)')
    
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples for evaluation (reduced for real model)')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation (reduced for real model)')
    
    parser.add_argument('--causal_dim', type=int, default=64,
                        help='Dimension of causal state')
    
    parser.add_argument('--hidden_size', type=int, default=896,
                        help='Hidden size for feature network (Qwen2.5-0.5B has 896)')
    
    parser.add_argument('--qwen_model_path', type=str, default='~/models/Qwen2.5-0.5B',
                        help='Path to Qwen model directory')
    
    return parser.parse_args()


def main():
    """Main function to run Qwen experiments."""
    args = parse_qwen_args()
    
    # Force using real Qwen model
    args.use_real_qwen = True
    args.vocab_size = 1000  # This will be overridden by the tokenizer
    
    # Auto-detect device if not specified
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running {args.experiment} experiment with real Qwen2.5-0.5B model")
    print(f"Model path: {args.qwen_model_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of samples: {args.num_samples}")
    print("=" * 60)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"qwen_{args.experiment}_{timestamp}")
    
    # Run the specified experiment
    try:
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
    except Exception as e:
        print(f"Error running experiment: {e}")
        print("This might be due to:")
        print("1. Qwen model not found at the specified path")
        print("2. Insufficient GPU memory")
        print("3. Missing dependencies")
        print("\nTry reducing batch_size or num_samples, or check the model path.")
        sys.exit(1)


if __name__ == "__main__":
    main() 