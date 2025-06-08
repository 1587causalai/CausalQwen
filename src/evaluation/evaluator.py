"""
Evaluator for the Causal Language Model.
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import math

# Constant for 95% confidence interval for Cauchy distribution
# tan(pi * (0.975 - 0.5)) = tan(pi * 0.475) ~= 12.706
CAUCHY_95_CONF_FACTOR = 12.706 
# Constant for 95% confidence interval for Normal distribution
NORMAL_95_CONF_FACTOR = 1.96

class Evaluator:
    """
    Handles comprehensive evaluation of the Causal Language Model.
    It calculates a wide range of metrics for classification, regression, and calibration.
    """
    
    def __init__(self, model, tokenizer, device, model_config):
        """
        Initializes the Evaluator.
        Args:
            model (nn.Module): The model to evaluate.
            tokenizer: The tokenizer.
            device (torch.device): The device to run evaluation on.
            model_config (CausalLMConfig): The model's configuration object.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_config = model_config

    def evaluate(self, dataset, batch_size=16, save_path=None):
        """
        Runs a full evaluation on a given dataset.
        Args:
            dataset (torch.utils.data.Dataset): The dataset to evaluate.
            batch_size (int): Batch size for the DataLoader.
            save_path (str, optional): If provided, saves raw prediction outputs to this path.
        Returns:
            dict: A dictionary of comprehensive evaluation metrics.
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Lists to store predictions and ground truth for metric calculation
        all_pred_tokens, all_true_tokens = [], []
        all_true_values, all_is_num_mask = [], []
        all_reg_locs, all_reg_scales = [], []
        all_pred_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = [t.to(self.device) for t in batch]
                input_ids, attention_mask, numerical_values, targets, target_values = batch

                # --- Main Forward Pass ---
                # Get full model output, including distribution parameters
                outputs = self.model(input_ids, numerical_values, attention_mask)
                
                # --- Predictions ---
                all_true_tokens.extend(targets.cpu().numpy())
                
                # For ECE, we need to convert OvR probabilities to normalized probabilities
                # Step 1: Get OvR probabilities (independent probabilities)
                ovr_probs = outputs['cls_probs']  # Shape: [batch_size, num_classes]
                
                # Step 2: Convert to normalized probability distribution (sum to 1)
                # We use softmax on the logits (inverse of the OvR probability transformation)
                # First, convert OvR probabilities back to decision scores
                # P(S_k > 0) = 0.5 + (1/π) * arctan(loc_S_k / scale_S_k)
                # So: arctan(loc_S_k / scale_S_k) = π * (P - 0.5)
                # And: loc_S_k / scale_S_k = tan(π * (P - 0.5))
                # We'll use the raw decision scores as logits for softmax normalization
                
                # Get decision score distributions from model
                cls_loc = outputs['cls_loc']  # Location parameters (like logits)
                cls_scale = outputs['cls_scale']  # Scale parameters (uncertainty)
                
                # Use location parameters as logits for softmax normalization
                # This converts OvR independent decisions to proper probability distribution
                normalized_probs = torch.softmax(cls_loc, dim=1)  # Sum to 1 across classes
                
                # Step 3: Use maximum probability as confidence for ECE
                max_probs, pred_classes = torch.max(normalized_probs, dim=1)
                pred_confidences_batch = max_probs
                all_pred_confidences.extend(pred_confidences_batch.cpu().numpy())
                
                # Update predicted tokens to use normalized prediction
                pred_tokens = pred_classes
                all_pred_tokens.extend(pred_tokens.cpu().numpy())

                # Collect data for regression and calibration metrics, now collecting for all samples
                is_num_mask = (targets == self.tokenizer.num_token_id)
                all_is_num_mask.extend(is_num_mask.cpu().numpy())
                all_true_values.extend(target_values.cpu().numpy())
                all_reg_locs.extend(outputs['reg_loc'].cpu().numpy())
                all_reg_scales.extend(outputs['reg_scale'].cpu().numpy())
        
        # Convert all lists to numpy arrays for consistent processing
        all_pred_tokens = np.array(all_pred_tokens)
        all_true_tokens = np.array(all_true_tokens)
        all_true_values = np.array(all_true_values)
        all_is_num_mask = np.array(all_is_num_mask, dtype=bool)
        all_reg_locs = np.array(all_reg_locs)
        all_reg_scales = np.array(all_reg_scales)
        all_pred_confidences = np.array(all_pred_confidences)

        # Filter regression-related data using the mask
        true_reg_values = all_true_values[all_is_num_mask]
        pred_reg_locs = all_reg_locs[all_is_num_mask]
        pred_reg_scales = all_reg_scales[all_is_num_mask]
        
        # Compute all metrics from the collected data
        metrics = self._compute_metrics(
            all_pred_tokens, all_true_tokens,
            true_reg_values, # Only pass actual regression values
            all_pred_confidences,
            pred_reg_locs, pred_reg_scales
        )
        
        # Save raw outputs if a path is provided
        if save_path:
            output_data = {
                'pred_tokens': all_pred_tokens,
                'true_tokens': all_true_tokens,
                'is_num_mask': all_is_num_mask,
                'true_values': all_true_values,
                'reg_locs': all_reg_locs,
                'reg_scales': all_reg_scales,
                'confidences': all_pred_confidences,
                'num_token_id': self.tokenizer.num_token_id
            }
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(output_data, save_path)
            print(f"Saved raw evaluation outputs to {save_path}")

        return metrics

    def _compute_metrics(self, pred_tokens, true_tokens, true_values, pred_confidences, reg_locs, reg_scales):
        """Computes and returns a dictionary of all metrics."""
        metrics = {}
        
        # --- Classification Metrics ---
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_tokens, pred_tokens, average='weighted', zero_division=0
        )
        metrics['cls_accuracy'] = accuracy_score(true_tokens, pred_tokens)
        metrics['cls_precision'] = precision
        metrics['cls_recall'] = recall
        metrics['cls_f1'] = f1
        
        # --- Regression Metrics ---
        if true_values.size > 0:
            # Note: The deterministic regression prediction is just the location 'loc'
            pred_values = reg_locs
            metrics['reg_mse'] = np.mean((true_values - pred_values)**2)
            metrics['reg_mae'] = np.mean(np.abs(true_values - pred_values))
        else:
            metrics['reg_mse'] = 0
            metrics['reg_mae'] = 0

        # --- Calibration Metrics ---
        # Use the new multi-class calibration metrics
        calib_metrics = self._compute_multiclass_calibration_metrics(pred_confidences, pred_tokens, true_tokens)
        metrics.update(calib_metrics)
        
        # PICP (Prediction Interval Coverage Probability) for Regression
        if reg_locs.size > 0:
            metrics['reg_picp'] = self._compute_picp(
                true_values, reg_locs, reg_scales
            )
        else:
            metrics['reg_picp'] = 0
            
        return metrics

    def _compute_ece(self, confidences, predictions, labels, n_bins=10):
        """Computes the Expected Calibration Error (ECE)."""
        confidences, predictions, labels = np.array(confidences), np.array(predictions), np.array(labels)
        accuracies = (predictions == labels)
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

    def _compute_picp(self, true_values, locs, scales):
        """Computes Prediction Interval Coverage Probability (PICP)."""
        if self.model_config.use_cauchy_distribution:
            factor = CAUCHY_95_CONF_FACTOR
        else:
            # Assumes Normal distribution if not Cauchy
            factor = NORMAL_95_CONF_FACTOR
            
        lower_bounds = locs - factor * scales
        upper_bounds = locs + factor * scales
        
        covered = ((true_values >= lower_bounds) & (true_values <= upper_bounds)).mean()
        return float(covered)

    def _compute_multiclass_calibration_metrics(self, pred_confidences, pred_tokens, true_tokens):
        """
        Compute calibration metrics for multi-class classification.
        
        This method computes ECE using properly normalized probabilities
        (sum to 1 across classes) rather than independent OvR probabilities.
        
        Args:
            pred_confidences (np.ndarray): Maximum probabilities from normalized distribution
            pred_tokens (np.ndarray): Predicted class tokens
            true_tokens (np.ndarray): True class tokens
            
        Returns:
            dict: Dictionary containing calibration metrics
        """
        # Compute the corrected ECE using normalized probabilities
        ece = self._compute_ece(pred_confidences, pred_tokens, true_tokens)
        
        return {
            'calib_ece': ece
        } 