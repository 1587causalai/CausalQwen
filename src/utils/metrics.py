"""
Metrics module.

This module provides functions for calculating various evaluation metrics
for the causal language model, including classification metrics, regression metrics,
and calibration metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted class indices
        targets: Target class indices
        probabilities: Predicted class probabilities
        
    Returns:
        metrics: Dictionary of classification metrics
    """
    # Convert tensors to numpy arrays
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate basic metrics
    accuracy = accuracy_score(targets_np, preds_np)
    
    # Calculate precision, recall, and F1 score
    # Use weighted average to handle class imbalance
    precision = precision_score(targets_np, preds_np, average='weighted', zero_division=0)
    recall = recall_score(targets_np, preds_np, average='weighted', zero_division=0)
    f1 = f1_score(targets_np, preds_np, average='weighted', zero_division=0)
    
    # Calculate additional metrics if probabilities are provided
    entropy = 0.0
    confidence = 0.0
    
    if probabilities is not None:
        probs_np = probabilities.cpu().numpy()
        
        # Calculate entropy of predicted probabilities
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probs_np * np.log(probs_np + epsilon), axis=1).mean()
        
        # Calculate average confidence (max probability)
        confidence = np.max(probs_np, axis=1).mean()
    
    # Compile metrics
    metrics = {
        'cls_accuracy': float(accuracy),
        'cls_precision': float(precision),
        'cls_recall': float(recall),
        'cls_f1': float(f1),
        'cls_entropy': float(entropy),
        'cls_confidence': float(confidence)
    }
    
    return metrics


def calculate_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    scales: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        predictions: Predicted values
        targets: Target values
        scales: Predicted scale parameters (uncertainty)
        
    Returns:
        metrics: Dictionary of regression metrics
    """
    # Convert tensors to numpy arrays
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate mean squared error
    mse = np.mean((preds_np - targets_np) ** 2)
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(preds_np - targets_np))
    
    # Calculate mean absolute percentage error
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((targets_np - preds_np) / (np.abs(targets_np) + epsilon))) * 100
    
    # Calculate R-squared
    ss_total = np.sum((targets_np - np.mean(targets_np)) ** 2)
    ss_residual = np.sum((targets_np - preds_np) ** 2)
    r2 = 1 - (ss_residual / (ss_total + epsilon))
    
    # Calculate normalized root mean squared error
    target_range = np.max(targets_np) - np.min(targets_np)
    nrmse = np.sqrt(mse) / (target_range + epsilon)
    
    # Calculate uncertainty metrics if scales are provided
    uncertainty_error = 0.0
    picp = 0.0  # Prediction Interval Coverage Probability
    
    if scales is not None:
        scales_np = scales.cpu().numpy()
        
        # Calculate mean uncertainty
        mean_uncertainty = np.mean(scales_np)
        
        # Calculate correlation between absolute error and uncertainty
        abs_errors = np.abs(preds_np - targets_np)
        uncertainty_error_corr = np.corrcoef(abs_errors, scales_np)[0, 1]
        
        # Calculate PICP (for 95% prediction interval)
        # For Cauchy distribution, 95% interval is approximately ±12.7 times the scale
        interval_multiplier = 12.7
        lower_bounds = preds_np - interval_multiplier * scales_np
        upper_bounds = preds_np + interval_multiplier * scales_np
        in_interval = (targets_np >= lower_bounds) & (targets_np <= upper_bounds)
        picp = np.mean(in_interval)
        
        # Update metrics
        metrics_uncertainty = {
            'reg_mean_uncertainty': float(mean_uncertainty),
            'reg_uncertainty_error_corr': float(uncertainty_error_corr),
            'reg_picp': float(picp)
        }
    else:
        metrics_uncertainty = {}
    
    # Compile basic metrics
    metrics_basic = {
        'reg_mse': float(mse),
        'reg_mae': float(mae),
        'reg_mape': float(mape),
        'reg_r2': float(r2),
        'reg_nrmse': float(nrmse)
    }
    
    # Combine all metrics
    metrics = {**metrics_basic, **metrics_uncertainty}
    
    return metrics


def calculate_calibration_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate calibration metrics.
    
    Args:
        probabilities: Predicted class probabilities
        targets: Target class indices
        num_bins: Number of bins for calibration curve
        
    Returns:
        metrics: Dictionary of calibration metrics
    """
    # Convert tensors to numpy arrays
    probs_np = probabilities.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Get predicted classes and their probabilities
    pred_classes = np.argmax(probs_np, axis=1)
    pred_probs = np.max(probs_np, axis=1)
    
    # Check if predictions are correct
    correct = (pred_classes == targets_np)
    
    # Create bins for confidence
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(pred_probs, bin_edges) - 1
    
    # Calculate calibration metrics
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        bin_mask = (bin_indices == i)
        if np.any(bin_mask):
            bin_accuracies[i] = np.mean(correct[bin_mask])
            bin_confidences[i] = np.mean(pred_probs[bin_mask])
            bin_counts[i] = np.sum(bin_mask)
    
    # Calculate Expected Calibration Error (ECE)
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(bin_counts)
    
    # Calculate Maximum Calibration Error (MCE)
    mce = np.max(np.abs(bin_accuracies - bin_confidences))
    
    # Calculate Brier Score
    # For multi-class, use one-hot encoding of targets
    num_classes = probs_np.shape[1]
    targets_one_hot = np.zeros((len(targets_np), num_classes))
    targets_one_hot[np.arange(len(targets_np)), targets_np] = 1
    brier_score = np.mean(np.sum((probs_np - targets_one_hot) ** 2, axis=1))
    
    # Compile metrics
    metrics = {
        'calib_ece': float(ece),
        'calib_mce': float(mce),
        'calib_brier': float(brier_score),
        'calib_error': float(ece)  # Alias for main calibration error
    }
    
    return metrics


def calculate_combined_metrics(
    cls_metrics: Dict[str, float],
    reg_metrics: Dict[str, float],
    alpha: float = 0.5
) -> Dict[str, float]:
    """
    Calculate combined metrics for mixed tasks.
    
    Args:
        cls_metrics: Classification metrics
        reg_metrics: Regression metrics
        alpha: Weight for classification metrics (1-alpha for regression)
        
    Returns:
        metrics: Dictionary of combined metrics
    """
    # Normalize classification metrics (higher is better)
    cls_score = cls_metrics.get('cls_accuracy', 0.0)
    
    # Normalize regression metrics (lower is better)
    reg_mse = reg_metrics.get('reg_mse', float('inf'))
    reg_score = 1.0 / (1.0 + reg_mse)  # Transform to [0, 1] range
    
    # Calculate combined score
    combined_score = alpha * cls_score + (1 - alpha) * reg_score
    
    # Compile metrics
    metrics = {
        'combined_score': float(combined_score),
        'cls_weight': float(alpha),
        'reg_weight': float(1 - alpha)
    }
    
    return metrics


def calculate_uncertainty_metrics(
    causal_loc: torch.Tensor,
    causal_scale: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    Calculate metrics related to uncertainty representation.
    
    Args:
        causal_loc: Location parameters of causal state distribution
        causal_scale: Scale parameters of causal state distribution
        targets: Target class indices
        num_samples: Number of samples to draw from distribution
        
    Returns:
        metrics: Dictionary of uncertainty metrics
    """
    # Convert tensors to numpy arrays
    loc_np = causal_loc.cpu().numpy()
    scale_np = causal_scale.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate mean scale (uncertainty)
    mean_scale = np.mean(scale_np)
    
    # Calculate scale variance
    scale_variance = np.var(scale_np)
    
    # Calculate scale range
    scale_range = np.max(scale_np) - np.min(scale_np)
    
    # Compile metrics
    metrics = {
        'uncertainty_mean': float(mean_scale),
        'uncertainty_variance': float(scale_variance),
        'uncertainty_range': float(scale_range)
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create dummy data for classification
    predictions = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 1, 0, 0])
    probabilities = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
        [0.3, 0.6, 0.1]
    ])
    
    # Calculate classification metrics
    cls_metrics = calculate_classification_metrics(predictions, targets, probabilities)
    print("Classification metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Create dummy data for regression
    reg_predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    reg_targets = torch.tensor([1.2, 1.8, 3.1, 3.9, 5.2])
    reg_scales = torch.tensor([0.2, 0.1, 0.3, 0.1, 0.2])
    
    # Calculate regression metrics
    reg_metrics = calculate_regression_metrics(reg_predictions, reg_targets, reg_scales)
    print("\nRegression metrics:")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Calculate calibration metrics
    calib_metrics = calculate_calibration_metrics(probabilities, targets)
    print("\nCalibration metrics:")
    for k, v in calib_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Calculate combined metrics
    combined_metrics = calculate_combined_metrics(cls_metrics, reg_metrics)
    print("\nCombined metrics:")
    for k, v in combined_metrics.items():
        print(f"  {k}: {v:.4f}")

