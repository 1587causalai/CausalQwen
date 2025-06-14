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
        all_true_values, all_pred_values = [], []
        all_pred_confidences, all_reg_scales = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = [t.to(self.device) for t in batch]
                input_ids, attention_mask, numerical_values, targets, target_values = batch

                # --- Main Forward Pass ---
                # Get full model output, including distribution parameters
                outputs = self.model(input_ids, numerical_values, attention_mask)
                
                # --- Handle sequence-to-sequence predictions ---
                cls_loc = outputs['cls_loc']  # [B, S, C]
                cls_scale = outputs['cls_scale']  # [B, S, C]
                reg_loc = outputs['reg_loc']  # [B, S]
                reg_scale = outputs['reg_scale']  # [B, S]
                
                # Flatten to handle as individual predictions
                batch_size, seq_len, vocab_size = cls_loc.shape
                
                # Create mask for valid (non-ignored) positions
                valid_mask = (targets != -100)  # [B, S]
                
                # Flatten and filter valid positions
                valid_cls_loc = cls_loc[valid_mask]  # [N_valid, C]
                valid_cls_scale = cls_scale[valid_mask]  # [N_valid, C]
                valid_targets = targets[valid_mask]  # [N_valid]
                valid_reg_loc = reg_loc[valid_mask]  # [N_valid]
                valid_reg_scale = reg_scale[valid_mask]  # [N_valid]
                valid_target_values = target_values[valid_mask]  # [N_valid]
                
                # --- CONFIDENCE CALCULATION FOR ECE ---
                # Apply softmax to convert logits to a valid probability distribution
                softmax_probs = torch.softmax(valid_cls_loc, dim=1)  # [N_valid, C]
                
                # The confidence for ECE is the maximum probability from the distribution
                max_probs, pred_classes = torch.max(softmax_probs, dim=1)  # [N_valid]
                pred_confidences_batch = max_probs
                all_pred_confidences.extend(pred_confidences_batch.cpu().numpy())
                
                # Update predicted tokens
                pred_tokens = pred_classes
                all_pred_tokens.extend(pred_tokens.cpu().numpy())
                all_true_tokens.extend(valid_targets.cpu().numpy())
                all_true_values.extend(valid_target_values.cpu().numpy())
                all_pred_values.extend(valid_reg_loc.cpu().numpy())
                all_reg_scales.extend(valid_reg_scale.cpu().numpy())
        
        # Convert all lists to numpy arrays for consistent processing
        all_pred_tokens = np.array(all_pred_tokens)
        all_true_tokens = np.array(all_true_tokens)
        all_true_values = np.array(all_true_values)
        all_pred_values = np.array(all_pred_values)
        all_reg_scales = np.array(all_reg_scales)
        all_pred_confidences = np.array(all_pred_confidences)

        # Compute all metrics from the collected data
        metrics = self._compute_metrics(
            all_pred_tokens, all_true_tokens,
            all_pred_values, all_true_values,
            all_pred_confidences, all_reg_scales
        )
        
        # Save raw outputs if a path is provided
        if save_path:
            output_data = {
                'pred_tokens': all_pred_tokens,
                'true_tokens': all_true_tokens,
                'true_values': all_true_values,
                'pred_values': all_pred_values,
                'confidences': all_pred_confidences,
                'num_token_id': self.tokenizer.num_token_id
            }
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(output_data, save_path)
            print(f"Saved raw evaluation outputs to {save_path}")

        return metrics

    def _compute_metrics(self, pred_tokens, true_tokens, pred_values, true_values, pred_confidences, reg_scales):
        """Computes and returns a dictionary of all metrics."""
        metrics = {}
        num_token_id = self.tokenizer.num_token_id

        # --- Overall Classification Metrics (on all tokens) ---
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_tokens, pred_tokens, average='weighted', zero_division=0
        )
        metrics['cls_accuracy_all'] = accuracy_score(true_tokens, pred_tokens)
        metrics['cls_precision_all'] = precision
        metrics['cls_recall_all'] = recall
        metrics['cls_f1_all'] = f1
        
        # --- <NUM> Token Detection Metrics ---
        is_true_num = (true_tokens == num_token_id)
        is_pred_num = (pred_tokens == num_token_id)

        tp = np.sum(is_true_num & is_pred_num)
        fp = np.sum(~is_true_num & is_pred_num)
        fn = np.sum(is_true_num & ~is_pred_num)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics['num_precision'] = precision
        metrics['num_recall'] = recall
        metrics['num_f1'] = f1

        # --- Regression Metrics (only on True Positive <NUM> predictions) ---
        tp_mask = is_true_num & is_pred_num
        tp_true_values = true_values[tp_mask]
        tp_pred_values = pred_values[tp_mask]
        
        # Filter out potential NaNs from targets
        valid_reg_mask = ~np.isnan(tp_true_values)
        tp_true_values = tp_true_values[valid_reg_mask]
        tp_pred_values = tp_pred_values[valid_reg_mask]

        if tp_true_values.size > 0:
            metrics['reg_mae'] = np.mean(np.abs(tp_true_values - tp_pred_values))
            metrics['reg_mdae'] = np.median(np.abs(tp_true_values - tp_pred_values))
            metrics['reg_mse'] = np.mean((tp_true_values - tp_pred_values)**2)
        else:
            metrics['reg_mae'] = 0.0
            metrics['reg_mdae'] = 0.0
            metrics['reg_mse'] = 0.0

        # --- Calibration Metrics ---
        # ECE for overall classification
        calib_metrics = self._compute_multiclass_calibration_metrics(pred_confidences, pred_tokens, true_tokens)
        metrics.update(calib_metrics)
        
        # PICP for regression (only on True Positives)
        tp_reg_scales = reg_scales[tp_mask][valid_reg_mask]
        if tp_true_values.size > 0:
            metrics['reg_picp'] = self._compute_picp(
                tp_true_values, tp_pred_values, tp_reg_scales
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