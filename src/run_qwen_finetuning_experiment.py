#!/usr/bin/env python
"""
Run Qwen Fine-tuning Before-and-After Experiment.

This script automates the process of:
1. Evaluating the performance of the pre-trained Qwen model (baseline).
2. Fine-tuning the Qwen model using our causal LM framework.
3. Evaluating the performance of the fine-tuned model.
4. Saving the results for comparison and reporting.
"""

import os
import sys
import torch
import json
import argparse
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.tokenizer import QwenTokenizerWrapper
from src.data.evaluation_data import get_all_evaluation_datasets
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator


def main(args):
    """Main function to run the experiment."""
    
    # Setup base directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"qwen_finetuning_experiment_{timestamp}"
    results_dir = os.path.join(args.results_base_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"=== Running Qwen Fine-tuning Experiment: {experiment_name} ===")
    print(f"Results will be saved to: {results_dir}")
    
    # --- 1. Setup: Model and Tokenizer ---
    print("\n--- Step 1: Initializing Model and Tokenizer ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = QwenTokenizerWrapper(
        model_path=args.qwen_model_path,
        use_real_tokenizer=True
    )
    
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=896, # For Qwen2.5-0.5B
        causal_dim=64,
        use_real_qwen=True,
        qwen_model_path=args.qwen_model_path
    )
    
    # --- 2. Pre-trained Model Evaluation (Baseline) ---
    print("\n--- Step 2: Evaluating Pre-trained Model (Baseline) ---")
    
    # Load fresh pre-trained model
    baseline_model = CausalLanguageModel(config).to(device)
    
    evaluator = Evaluator(baseline_model, tokenizer, device)
    evaluation_datasets = get_all_evaluation_datasets(tokenizer)
    
    baseline_results = {}
    for name, dataset in evaluation_datasets.items():
        print(f"Evaluating on: {name}")
        results = evaluator.evaluate(dataset, batch_size=args.batch_size)
        baseline_results[name] = results
        print(f"  Accuracy: {results['accuracy']:.4f}, Mean Rank: {results['mean_rank']:.2f}")

    baseline_results_path = os.path.join(results_dir, "baseline_results.json")
    with open(baseline_results_path, 'w') as f:
        json.dump(baseline_results, f, indent=4)
    print(f"Baseline results saved to {baseline_results_path}")
    
    # --- 3. Model Fine-tuning ---
    print("\n--- Step 3: Fine-tuning the Model ---")
    
    # We need a new model instance for training
    finetune_model = CausalLanguageModel(config)
    
    # Apply custom weight initialization for new components
    # This is crucial to prevent gradient explosion
    finetune_model.abduction_network.apply(Trainer.weights_init)
    finetune_model.action_network.apply(Trainer.weights_init)
    finetune_model.to(device)

    trainer = Trainer(
        model=finetune_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    trainer.train(num_epochs=args.epochs, num_samples=args.num_samples)
    
    # Save the fine-tuned model
    finetuned_model_path = os.path.join(results_dir, "finetuned_model.pth")
    torch.save(finetune_model.state_dict(), finetuned_model_path)
    print(f"Fine-tuned model saved to {finetuned_model_path}")
    
    # --- 4. Fine-tuned Model Evaluation ---
    print("\n--- Step 4: Evaluating Fine-tuned Model ---")
    
    # Evaluator will now use the fine-tuned model
    evaluator_finetuned = Evaluator(finetune_model, tokenizer, device)
    
    finetuned_results = {}
    for name, dataset in evaluation_datasets.items():
        print(f"Evaluating on: {name}")
        results = evaluator_finetuned.evaluate(dataset, batch_size=args.batch_size)
        finetuned_results[name] = results
        print(f"  Accuracy: {results['accuracy']:.4f}, Mean Rank: {results['mean_rank']:.2f}")

    finetuned_results_path = os.path.join(results_dir, "finetuned_results.json")
    with open(finetuned_results_path, 'w') as f:
        json.dump(finetuned_results, f, indent=4)
    print(f"Finetuned results saved to {finetuned_results_path}")
    
    print("\n=== Experiment Complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Qwen Fine-tuning Before-and-After Experiment.")
    parser.add_argument(
        '--qwen_model_path', 
        type=str, 
        default='~/models/Qwen2.5-0.5B',
        help='Path to the pre-trained Qwen model.'
    )
    parser.add_argument(
        '--results_base_dir', 
        type=str, 
        default='docs-site/results',
        help='Base directory to save experiment results.'
    )
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
    
    args = parser.parse_args()
    main(args) 