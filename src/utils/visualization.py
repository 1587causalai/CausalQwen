"""
Visualization module.

This module provides functions for visualizing model performance,
causal state distributions, decision boundaries, and other aspects
of the causal language model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models.causal_lm import CausalLanguageModel
from src.utils.distributions import cauchy_sample


def plot_metrics(
    metrics: Dict[str, float],
    output_dir: str = 'results',
    prefix: str = 'eval'
) -> str:
    """
    Plot key evaluation metrics, filtering for readability.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define key metrics to display for better readability and to avoid scale issues.
    KEY_CLS_METRICS = ['cls_accuracy', 'cls_f1']
    KEY_REG_METRICS = ['reg_mse', 'reg_mae']
    KEY_CALIB_METRICS = ['calib_ece', 'calib_mce']
    
    # Group metrics by type, only including the key metrics.
    cls_metrics = {k: v for k, v in metrics.items() if k in KEY_CLS_METRICS}
    reg_metrics = {k: v for k, v in metrics.items() if k in KEY_REG_METRICS}
    calib_metrics = {k: v for k, v in metrics.items() if k in KEY_CALIB_METRICS}
    
    # Determine number of subplots needed
    metric_groups = [cls_metrics, reg_metrics, calib_metrics]
    plot_groups = [group for group in metric_groups if group]
    num_plots = len(plot_groups)
    
    if num_plots == 0:
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), squeeze=False)
    
    plot_idx = 0
    group_titles = ['Key Classification Metrics', 'Key Regression Metrics', 'Key Calibration Metrics']

    for i, group in enumerate(metric_groups):
        if not group:
            continue
        
        ax = axes[plot_idx, 0]
        names = list(group.keys())
        values = list(group.values())
        
        bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_title(group_titles[i])
        ax.set_ylabel('Value')
        # Set tick positions first, then labels
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on top of bars
        ax.bar_label(bars, fmt='%.4f', padding=3)

        plot_idx += 1
    
    plt.tight_layout(pad=2.0)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_metrics.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_calibration_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    output_dir: str = 'results',
    prefix: str = 'eval'
) -> str:
    """
    Plot calibration curve.
    
    Args:
        confidences: Predicted confidences
        accuracies: Actual accuracies
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot diagonal (perfect calibration)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Plot actual calibration curve
    plt.plot(confidences, accuracies, 'o-', label='Model calibration')
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(confidences - accuracies))
    
    plt.title(f'Calibration Curve (Error: {calibration_error:.4f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_calibration.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def visualize_causal_state(
    model: CausalLanguageModel,
    inputs: Dict[str, torch.Tensor],
    output_dir: str = 'results',
    prefix: str = 'eval',
    num_samples: int = 1000
) -> str:
    """
    Visualize causal state distribution.
    
    Args:
        model: Causal language model
        inputs: Model inputs
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        num_samples: Number of samples to draw from distribution
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs.get('numerical_values', None))
    
    # Get causal state distribution parameters
    causal_loc = outputs['causal_loc'][0].cpu().numpy()
    causal_scale = outputs['causal_scale'][0].cpu().numpy()
    
    # If causal dimension is greater than 2, use PCA for visualization
    if causal_loc.shape[0] > 2:
        # Sample from distribution
        samples = []
        for _ in range(num_samples):
            sample = cauchy_sample(
                torch.tensor(causal_loc), 
                torch.tensor(causal_scale)
            ).numpy()
            samples.append(sample)
        
        samples = np.array(samples)
        
        # Apply PCA
        pca = PCA(n_components=2)
        samples_2d = pca.fit_transform(samples)
        loc_2d = pca.transform([causal_loc])[0]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot samples
        plt.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.3, label='Samples')
        plt.scatter([loc_2d[0]], [loc_2d[1]], color='red', s=100, label='Location')
        
        plt.title('Causal State Distribution (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
    else:
        # Use first two dimensions directly
        # Sample from distribution
        samples_2d = np.array([
            cauchy_sample(
                torch.tensor(causal_loc[:2]), 
                torch.tensor(causal_scale[:2])
            ).numpy()
            for _ in range(num_samples)
        ])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot samples
        plt.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.3, label='Samples')
        plt.scatter([causal_loc[0]], [causal_loc[1]], color='red', s=100, label='Location')
        
        plt.title('Causal State Distribution')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.grid(True)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_causal_state.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def visualize_decision_boundary(
    model: CausalLanguageModel,
    output_dir: str = 'results',
    prefix: str = 'eval',
    feature_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100
) -> str:
    """
    Visualize OvR classification decision boundary.
    
    Args:
        model: Causal language model
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        feature_range: Range of feature values to plot
        resolution: Grid resolution
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create grid
    x = np.linspace(feature_range[0], feature_range[1], resolution)
    y = np.linspace(feature_range[0], feature_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Prepare grid points
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Calculate decision scores for each grid point
    scores = []
    
    with torch.no_grad():
        for point in grid_points:
            # Create causal state
            causal_loc = torch.tensor(
                [point[0], point[1]] + [0] * (model.config.causal_dim - 2), 
                dtype=torch.float32
            )
            causal_scale = torch.ones_like(causal_loc) * 0.1
            
            # Get decision scores
            outputs = model.action_network(causal_loc.unsqueeze(0), causal_scale.unsqueeze(0))
            cls_probs = outputs['cls_probs'][0].cpu().numpy()
            
            scores.append(cls_probs)
    
    scores = np.array(scores)
    
    # Find predicted class for each point
    pred_classes = np.argmax(scores, axis=1)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=pred_classes, cmap='viridis', alpha=0.5, s=10)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Predicted Class')
    
    plt.title('Decision Boundary')
    plt.xlabel('Causal Dimension 1')
    plt.ylabel('Causal Dimension 2')
    plt.grid(True)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_decision_boundary.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def visualize_uncertainty(
    model: CausalLanguageModel,
    test_loader: torch.utils.data.DataLoader,
    output_dir: str = 'results',
    prefix: str = 'eval'
) -> str:
    """
    Visualize model uncertainty.
    
    Args:
        model: Causal language model
        test_loader: Test data loader
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect uncertainties for correct and incorrect predictions
    correct_uncertainties = []
    incorrect_uncertainties = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to model device
            input_ids = batch['input_ids'].to(model.device)
            numerical_values = batch.get('numerical_values', None)
            if numerical_values is not None:
                numerical_values = numerical_values.to(model.device)
            
            targets = batch['targets'].to(model.device)
            
            # Get model outputs
            outputs = model(input_ids, numerical_values)
            predictions = model.predict(input_ids, numerical_values)
            
            # Get uncertainties (use causal scale mean as uncertainty measure)
            uncertainties = outputs['causal_scale'].mean(dim=1).cpu().numpy()
            
            # Separate correct and incorrect predictions
            correct_mask = (predictions['cls_pred'] == targets).cpu().numpy()
            
            correct_uncertainties.extend(uncertainties[correct_mask])
            incorrect_uncertainties.extend(uncertainties[~correct_mask])
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(correct_uncertainties, bins=30, alpha=0.5, label='Correct predictions')
    plt.hist(incorrect_uncertainties, bins=30, alpha=0.5, label='Incorrect predictions')
    
    plt.title('Uncertainty Distribution')
    plt.xlabel('Uncertainty (Causal Scale Mean)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_uncertainty.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def visualize_causal_state_tsne(
    model: CausalLanguageModel,
    test_loader: torch.utils.data.DataLoader,
    output_dir: str = 'results',
    prefix: str = 'eval',
    num_samples: int = 500
) -> str:
    """
    Visualize causal states using t-SNE.
    
    Args:
        model: Causal language model
        test_loader: Test data loader
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        num_samples: Maximum number of samples to visualize
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect causal states and targets
    causal_locs = []
    targets = []
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if count >= num_samples:
                break
            
            # Move batch to model device
            input_ids = batch['input_ids'].to(model.device)
            numerical_values = batch.get('numerical_values', None)
            if numerical_values is not None:
                numerical_values = numerical_values.to(model.device)
            
            batch_targets = batch['targets'].to(model.device)
            
            # Get model outputs
            outputs = model(input_ids, numerical_values)
            
            # Collect causal locations and targets
            batch_size = min(input_ids.size(0), num_samples - count)
            causal_locs.append(outputs['causal_loc'][:batch_size].cpu().numpy())
            targets.append(batch_targets[:batch_size].cpu().numpy())
            
            count += batch_size
    
    # Concatenate results
    causal_locs = np.concatenate(causal_locs, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    causal_locs_2d = tsne.fit_transform(causal_locs)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot points colored by target
    scatter = plt.scatter(causal_locs_2d[:, 0], causal_locs_2d[:, 1], c=targets, cmap='viridis', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Target Class')
    
    plt.title('Causal State t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_causal_tsne.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def visualize_regression_performance(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    output_dir: str = 'results',
    prefix: str = 'eval'
) -> str:
    """
    Visualize regression performance.
    
    Args:
        predictions: Predicted values
        targets: Target values
        uncertainties: Prediction uncertainties
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot scatter with optional uncertainty
    if uncertainties is not None:
        # Scale marker size based on uncertainty
        sizes = 20 + 100 * (uncertainties / np.max(uncertainties))
        plt.scatter(targets, predictions, alpha=0.7, s=sizes)
        
        # Add colorbar for uncertainty
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=np.min(uncertainties), vmax=np.max(uncertainties))))
    else:
        plt.scatter(targets, predictions, alpha=0.7)
    
    # Plot diagonal (perfect predictions)
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    plt.title(f'Regression Performance (MSE: {mse:.4f}, MAE: {mae:.4f})')
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_regression.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def visualize_training_history(
    history: Dict[str, List[float]],
    output_dir: str = 'results',
    prefix: str = 'train'
) -> List[str]:
    """
    Visualize training history.
    
    Args:
        history: Dictionary of training metrics
        output_dir: Directory to save the plots
        prefix: Prefix for the output file names
        
    Returns:
        output_paths: List of paths to the saved plots
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    
    # Group metrics by type
    loss_metrics = {k: v for k, v in history.items() if 'loss' in k.lower()}
    accuracy_metrics = {k: v for k, v in history.items() if 'acc' in k.lower() or 'accuracy' in k.lower()}
    other_metrics = {k: v for k, v in history.items() 
                    if not ('loss' in k.lower() or 'acc' in k.lower() or 'accuracy' in k.lower())}
    
    # Plot loss metrics
    if loss_metrics:
        plt.figure(figsize=(10, 6))
        
        for name, values in loss_metrics.items():
            plt.plot(values, label=name)
        
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        output_path = os.path.join(output_dir, f'{prefix}_loss.png')
        plt.savefig(output_path)
        plt.close()
        output_paths.append(output_path)
    
    # Plot accuracy metrics
    if accuracy_metrics:
        plt.figure(figsize=(10, 6))
        
        for name, values in accuracy_metrics.items():
            plt.plot(values, label=name)
        
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        output_path = os.path.join(output_dir, f'{prefix}_accuracy.png')
        plt.savefig(output_path)
        plt.close()
        output_paths.append(output_path)
    
    # Plot other metrics
    if other_metrics:
        for name, values in other_metrics.items():
            plt.figure(figsize=(10, 6))
            
            plt.plot(values)
            
            plt.title(f'Training {name}')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.grid(True)
            
            # Save figure
            output_path = os.path.join(output_dir, f'{prefix}_{name}.png')
            plt.savefig(output_path)
            plt.close()
            output_paths.append(output_path)
    
    return output_paths


def visualize_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str],
    output_dir: str = 'results',
    prefix: str = 'compare'
) -> str:
    """
    Visualize model comparison.
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        metric_names: Names of metrics to compare
        output_dir: Directory to save the plot
        prefix: Prefix for the output file name
        
    Returns:
        output_path: Path to the saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model names
    model_names = list(metrics_dict.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 5 * len(metric_names)))
    
    # Handle case with only one metric
    if len(metric_names) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        # Extract values for this metric
        values = []
        for model in model_names:
            if metric in metrics_dict[model]:
                values.append(metrics_dict[model][metric])
            else:
                values.append(0)
        
        # Create bar plot
        ax.bar(model_names, values)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(j, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{prefix}_metrics.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy metrics
    metrics = {
        'cls_accuracy': 0.85,
        'cls_precision': 0.82,
        'cls_recall': 0.79,
        'cls_f1': 0.80,
        'reg_mse': 0.25,
        'reg_mae': 0.40,
        'calib_error': 0.12
    }
    
    # Plot metrics
    plot_path = plot_metrics(metrics, output_dir='results', prefix='test')
    print(f"Metrics plot saved to: {plot_path}")
    
    # Create dummy calibration data
    confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    accuracies = np.array([0.15, 0.35, 0.45, 0.65, 0.8])
    
    # Plot calibration curve
    calib_path = plot_calibration_curve(confidences, accuracies, output_dir='results', prefix='test')
    print(f"Calibration plot saved to: {calib_path}")
    
    # Create dummy regression data
    predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = np.array([1.2, 1.8, 3.1, 3.9, 5.2])
    uncertainties = np.array([0.2, 0.1, 0.3, 0.1, 0.2])
    
    # Plot regression performance
    reg_path = visualize_regression_performance(predictions, targets, uncertainties, output_dir='results', prefix='test')
    print(f"Regression plot saved to: {reg_path}")
    
    # Create dummy training history
    history = {
        'train_loss': [2.5, 2.0, 1.5, 1.2, 1.0],
        'val_loss': [2.7, 2.2, 1.8, 1.5, 1.3],
        'train_accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.85]
    }
    
    # Plot training history
    history_paths = visualize_training_history(history, output_dir='results', prefix='test')
    print(f"Training history plots saved to: {history_paths}")
    
    # Create dummy model comparison data
    metrics_dict = {
        'Model A': {'cls_accuracy': 0.85, 'reg_mse': 0.25},
        'Model B': {'cls_accuracy': 0.82, 'reg_mse': 0.30},
        'Model C': {'cls_accuracy': 0.88, 'reg_mse': 0.22}
    }
    
    # Plot model comparison
    compare_path = visualize_model_comparison(metrics_dict, ['cls_accuracy', 'reg_mse'], output_dir='results', prefix='test')
    print(f"Model comparison plot saved to: {compare_path}")

