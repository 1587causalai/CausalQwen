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
    probabilities: Optional[torch.Tensor] = None,
    num_token_id: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted class indices [batch_size, seq_len] or [batch_size]
        targets: Target class indices [batch_size, seq_len] or [batch_size]
        probabilities: Predicted class probabilities
        num_token_id: Token ID for <NUM> to calculate specific metrics
        
    Returns:
        metrics: Dictionary of classification metrics
    """
    # Convert tensors to numpy arrays
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Flatten if needed
    if preds_np.ndim > 1:
        preds_np = preds_np.flatten()
    if targets_np.ndim > 1:
        targets_np = targets_np.flatten()
    
    # Remove invalid indices (e.g., padding)
    valid_mask = targets_np >= 0
    preds_np = preds_np[valid_mask]
    targets_np = targets_np[valid_mask]
    
    if len(preds_np) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(targets_np, preds_np),
        'precision': precision_score(targets_np, preds_np, average='weighted', zero_division=0),
        'recall': recall_score(targets_np, preds_np, average='weighted', zero_division=0),
        'f1': f1_score(targets_np, preds_np, average='weighted', zero_division=0)
    }
    
    # <NUM> token specific metrics if provided
    if num_token_id is not None:
        num_mask = targets_np == num_token_id
        if num_mask.sum() > 0:
            metrics['num_precision'] = precision_score(
                num_mask, preds_np == num_token_id, zero_division=0
            )
            metrics['num_recall'] = recall_score(
                num_mask, preds_np == num_token_id, zero_division=0
            )
            metrics['num_f1'] = f1_score(
                num_mask, preds_np == num_token_id, zero_division=0
            )
    
    return metrics


def calculate_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        predictions: Predicted values
        targets: Target values
        mask: Boolean mask for valid positions
        
    Returns:
        metrics: Dictionary of regression metrics
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Apply mask if provided
    if mask is not None:
        mask_np = mask.cpu().numpy().astype(bool)
        preds_np = preds_np[mask_np]
        targets_np = targets_np[mask_np]
    
    if len(preds_np) == 0:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'mape': 0.0
        }
    
    # Calculate metrics
    mae = np.mean(np.abs(preds_np - targets_np))
    rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    non_zero_mask = targets_np != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((targets_np[non_zero_mask] - preds_np[non_zero_mask]) / targets_np[non_zero_mask])) * 100
    else:
        mape = 0.0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }


def calculate_calibration_metrics(
    predicted_probs: torch.Tensor,
    predicted_scales: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    confidence_levels: List[float] = [0.5, 0.9, 0.95]
) -> Dict[str, float]:
    """
    Calculate calibration metrics for uncertainty quantification.
    
    Args:
        predicted_probs: Predicted probabilities (for classification)
        predicted_scales: Predicted scales (uncertainty)
        targets: True targets
        mask: Valid positions mask
        confidence_levels: Confidence levels to evaluate
        
    Returns:
        metrics: Dictionary of calibration metrics
    """
    # This is a placeholder - implement proper calibration metrics
    # based on the Cauchy distribution properties
    
    metrics = {}
    
    # For now, return dummy values
    for level in confidence_levels:
        metrics[f'coverage_{int(level*100)}'] = 0.0
        metrics[f'width_{int(level*100)}'] = 0.0
    
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

