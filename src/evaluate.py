"""
Model evaluation script.

This script provides functions for evaluating the causal language model
on various tasks and datasets, including text generation, numerical prediction,
and mixed tasks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.utils.data import DataLoader, TensorDataset

from src.models.causal_lm import CausalLanguageModel
from src.utils.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_calibration_metrics
)
from src.data.synthetic import TextWithNumbersGenerator
from src.data.tokenizer import MockTokenizer
from src.utils.visualization import plot_metrics


def evaluate_model(
    model: CausalLanguageModel,
    test_loader: DataLoader,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Evaluate the model on a test dataset.
    
    Args:
        model: The causal language model to evaluate
        test_loader: DataLoader for the test dataset
        device: Device to run evaluation on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    model.to(device)
    
    # Initialize metric accumulators
    cls_preds = []
    cls_targets = []
    reg_preds = []
    reg_targets = []
    cls_probs = []
    reg_scales = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            numerical_values = batch[2].to(device)
            targets = batch[3].to(device)
            target_values = batch[4].to(device)
            
            # Get model predictions
            outputs = model(input_ids, numerical_values, attention_mask=attention_mask)
            predictions = model.predict(input_ids, numerical_values, attention_mask=attention_mask)
            
            # Collect predictions and targets
            cls_preds.append(predictions['cls_pred'].cpu())
            cls_targets.append(targets.cpu())
            cls_probs.append(outputs['cls_probs'].cpu())
            
            # Collect regression predictions and targets if available
            if target_values is not None:
                reg_preds.append(predictions['reg_pred'].cpu())
                reg_targets.append(target_values.cpu())
                reg_scales.append(outputs['reg_scale'].cpu())
    
    # Concatenate results
    cls_preds = torch.cat(cls_preds, dim=0)
    cls_targets = torch.cat(cls_targets, dim=0)
    cls_probs = torch.cat(cls_probs, dim=0)
    
    # Calculate classification metrics
    cls_metrics = calculate_classification_metrics(cls_preds, cls_targets, cls_probs)
    
    # Calculate regression metrics if available
    reg_metrics = {}
    if reg_preds:
        reg_preds = torch.cat(reg_preds, dim=0)
        reg_targets = torch.cat(reg_targets, dim=0)
        reg_scales = torch.cat(reg_scales, dim=0)
        reg_metrics = calculate_regression_metrics(reg_preds, reg_targets, reg_scales)
    
    # Calculate calibration metrics
    calib_metrics = calculate_calibration_metrics(cls_probs, cls_targets)
    
    # Combine all metrics
    metrics = {**cls_metrics, **reg_metrics, **calib_metrics}
    
    return metrics


def evaluate_on_synthetic_data(
    model: CausalLanguageModel,
    tokenizer: MockTokenizer,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu'),
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate the model on synthetic data.
    
    Args:
        model: The causal language model to evaluate
        tokenizer: The tokenizer to use for processing text
        num_samples: Number of synthetic samples to generate
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        seed: Random seed for reproducibility
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Generate synthetic data
    generator = TextWithNumbersGenerator(seed=seed)
    texts, true_values = generator.generate_text(num_samples=num_samples)
    
    # Tokenize texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    numerical_values = inputs['numerical_values']
    
    # Prepare targets (predicting the numerical value)
    # The target token is always <NUM>
    targets = torch.full((num_samples,), tokenizer.num_token_id, dtype=torch.long)
    target_values = torch.tensor(true_values, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        input_ids, attention_mask, numerical_values, targets, target_values
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, device)
    
    return metrics


def evaluate_on_qa_data(
    model: CausalLanguageModel,
    tokenizer: MockTokenizer,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu'),
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate the model on question-answering data.
    
    Args:
        model: The causal language model to evaluate
        tokenizer: The tokenizer to use for processing text
        num_samples: Number of synthetic samples to generate
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        seed: Random seed for reproducibility
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Generate QA data
    generator = TextWithNumbersGenerator(seed=seed)
    contexts, questions, answers = generator.generate_qa_pairs(num_samples=num_samples)
    
    # Combine context and question
    combined_texts = [f"{context} {question}" for context, question in zip(contexts, questions)]
    
    # Tokenize texts
    inputs = tokenizer(
        combined_texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # Numerical values from context are part of the input, handled by tokenizer
    numerical_values = inputs['numerical_values']
    
    # Prepare targets (answer prediction)
    targets = torch.full((num_samples,), tokenizer.num_token_id, dtype=torch.long)
    target_values = torch.tensor(answers, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        input_ids, attention_mask, numerical_values, targets, target_values
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, device)
    
    return metrics


def evaluate_on_extreme_data(
    model: CausalLanguageModel,
    tokenizer: MockTokenizer,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu'),
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate the model on data with extreme values.
    
    Args:
        model: The causal language model to evaluate
        tokenizer: The tokenizer to use for processing text
        num_samples: Number of synthetic samples to generate
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        seed: Random seed for reproducibility
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Generate extreme value data
    generator = TextWithNumbersGenerator(seed=seed)
    texts, values = generator.generate_extreme_text(num_samples=num_samples)
    
    # Tokenize texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    numerical_values = inputs['numerical_values']
    
    # Prepare targets (predicting the numerical value)
    targets = torch.full((num_samples,), tokenizer.num_token_id, dtype=torch.long)
    target_values = torch.tensor(values, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        input_ids, attention_mask, numerical_values, targets, target_values
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, device)
    
    return metrics


def visualize_results(
    metrics: Dict[str, float],
    output_dir: str = 'results',
    prefix: str = 'eval'
) -> None:
    """
    Visualize evaluation results by calling the central plotting function.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save visualizations
        prefix: Prefix for output files
    """
    plot_metrics(metrics, output_dir=output_dir, prefix=prefix)


def run_comprehensive_evaluation(
    model: CausalLanguageModel,
    tokenizer: MockTokenizer,
    output_dir: str = 'results',
    device: torch.device = torch.device('cpu'),
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Run a comprehensive evaluation of the model on various datasets.
    
    Args:
        model: The causal language model to evaluate
        tokenizer: The tokenizer to use
        output_dir: Directory to save results and visualizations
        device: Device to run evaluation on
        seed: Random seed for reproducibility
        
    Returns:
        all_metrics: Dictionary of metrics for each dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on basic text-number data
    print("Evaluating on basic text-number data...")
    basic_metrics = evaluate_on_synthetic_data(
        model, tokenizer, num_samples=1000, batch_size=32, device=device, seed=seed
    )
    visualize_results(basic_metrics, output_dir=output_dir, prefix='basic')
    
    # Evaluate on QA data
    print("Evaluating on question-answering data...")
    qa_metrics = evaluate_on_qa_data(
        model, tokenizer, num_samples=1000, batch_size=32, device=device, seed=seed
    )
    visualize_results(qa_metrics, output_dir=output_dir, prefix='qa')
    
    # Evaluate on extreme value data
    print("Evaluating on extreme value data...")
    extreme_metrics = evaluate_on_extreme_data(
        model, tokenizer, num_samples=1000, batch_size=32, device=device, seed=seed
    )
    visualize_results(extreme_metrics, output_dir=output_dir, prefix='extreme')
    
    # Combine all metrics
    all_metrics = {
        'basic': basic_metrics,
        'qa': qa_metrics,
        'extreme': extreme_metrics
    }
    
    # Save metrics to file
    import json
    with open(os.path.join(output_dir, 'all_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics


def compare_models(
    models: Dict[str, CausalLanguageModel],
    tokenizer: MockTokenizer,
    output_dir: str = 'results',
    device: torch.device = torch.device('cpu'),
    seed: int = 42
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare multiple models on various datasets.
    
    Args:
        models: Dictionary of models to compare
        tokenizer: The tokenizer to use
        output_dir: Directory to save results and visualizations
        device: Device to run evaluation on
        seed: Random seed for reproducibility
        
    Returns:
        comparison_results: Dictionary of metrics for each model and dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    comparison_results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Run comprehensive evaluation
        model_metrics = run_comprehensive_evaluation(
            model, tokenizer, output_dir=model_dir, device=device, seed=seed
        )
        
        comparison_results[model_name] = model_metrics
    
    # Generate comparison visualizations
    generate_comparison_plots(comparison_results, output_dir=output_dir)
    
    return comparison_results


def generate_comparison_plots(
    comparison_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str = 'results'
) -> None:
    """
    Generate plots comparing different models.
    
    Args:
        comparison_results: Dictionary of metrics for each model and dataset
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all model names
    model_names = list(comparison_results.keys())
    
    # Get all dataset names
    dataset_names = list(next(iter(comparison_results.values())).keys())
    
    # Select key metrics to compare
    key_metrics = [
        'cls_accuracy', 'cls_f1', 'reg_mse', 'reg_mae', 'calib_error'
    ]
    
    # Generate comparison plots for each dataset and metric
    for dataset in dataset_names:
        for metric in key_metrics:
            # Check if metric exists for this dataset
            if not all(metric in comparison_results[model][dataset] for model in model_names):
                continue
            
            # Extract metric values for each model
            values = [comparison_results[model][dataset][metric] for model in model_names]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.bar(model_names, values)
            plt.title(f'{metric} Comparison ({dataset} dataset)')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            
            # Add value labels
            for i, v in enumerate(values):
                plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'compare_{dataset}_{metric}.png'))
            plt.close()
    
    # Generate overall comparison plot
    plt.figure(figsize=(15, 10))
    
    # Number of metrics and models
    n_metrics = len(key_metrics)
    n_models = len(model_names)
    
    # Set width of bars
    bar_width = 0.8 / n_models
    
    # Set positions of bars on x-axis
    r = np.arange(n_metrics)
    
    # Create bars for each model
    for i, model in enumerate(model_names):
        # Calculate average metrics across datasets
        avg_values = []
        for j, metric in enumerate(key_metrics):
            values = []
            for dataset in dataset_names:
                if metric in comparison_results[model][dataset]:
                    # Normalize regression metrics (lower is better)
                    if metric.startswith('reg_') or metric == 'calib_error':
                        values.append(1.0 / (1.0 + comparison_results[model][dataset][metric]))
                    else:
                        values.append(comparison_results[model][dataset][metric])
            
            if values:
                avg_values.append(np.mean(values))
            else:
                avg_values.append(0)
        
        # Create bars
        plt.bar(r + i * bar_width, avg_values, width=bar_width, label=model)
    
    # Add labels and legend
    plt.xlabel('Metrics')
    plt.ylabel('Performance')
    plt.title('Overall Model Comparison')
    plt.xticks(r + bar_width * (n_models - 1) / 2, key_metrics, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'))
    plt.close()


if __name__ == "__main__":
    # Example usage
    from src.models.causal_lm import CausalLanguageModel, CausalLMConfig
    from src.data.tokenizer import MockTokenizer

    # Create a mock tokenizer
    tokenizer = MockTokenizer(vocab_size=1000)

    # Create a mock model for testing
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=768,
        causal_dim=64,
        use_mock_feature_network=True
    )
    model = CausalLanguageModel(config)
    
    # Run evaluation
    metrics = evaluate_on_synthetic_data(model, tokenizer, num_samples=100)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Visualize results
    visualize_results(metrics, output_dir='results', prefix='test')

