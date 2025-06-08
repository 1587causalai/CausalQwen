"""
Evaluator for the Causal Language Model.
"""

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

    def evaluate(self, dataset, batch_size=16):
        """
        Runs a full evaluation on a given dataset.
        Args:
            dataset (torch.utils.data.Dataset): The dataset to evaluate.
            batch_size (int): Batch size for the DataLoader.
        Returns:
            dict: A dictionary of comprehensive evaluation metrics.
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Lists to store predictions and ground truth for metric calculation
        all_pred_tokens, all_true_tokens = [], []
        all_pred_values, all_true_values = [], []
        all_pred_probs, all_pred_confidences = [], []
        all_reg_locs, all_reg_scales = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = [t.to(self.device) for t in batch]
                input_ids, attention_mask, numerical_values, targets, target_values = batch

                # --- Main Forward Pass ---
                # Get full model output, including distribution parameters
                outputs = self.model(input_ids, numerical_values, attention_mask)
                
                # --- Predictions ---
                # Get deterministic predictions using the .predict() method for consistency
                predictions = self.model.predict(input_ids, numerical_values, attention_mask)
                pred_tokens = predictions['cls_pred']
                pred_values = predictions['reg_pred']

                all_pred_tokens.extend(pred_tokens.cpu().numpy())
                all_true_tokens.extend(targets.cpu().numpy())
                all_pred_confidences.extend(torch.max(outputs['cls_probs'], dim=-1).values.cpu().numpy())
                all_pred_probs.extend(outputs['cls_probs'].cpu().numpy())

                # Collect data for regression and calibration metrics
                is_num_mask = (targets == self.tokenizer.num_token_id)
                if is_num_mask.sum() > 0:
                    all_true_values.extend(target_values[is_num_mask].cpu().numpy())
                    all_pred_values.extend(pred_values[is_num_mask].cpu().numpy())
                    all_reg_locs.extend(outputs['reg_loc'][is_num_mask].cpu().numpy())
                    all_reg_scales.extend(outputs['reg_scale'][is_num_mask].cpu().numpy())
        
        # Compute all metrics from the collected data
        metrics = self._compute_metrics(
            all_pred_tokens, all_true_tokens,
            all_pred_values, all_true_values,
            all_pred_probs, all_pred_confidences,
            all_reg_locs, all_reg_scales
        )
        return metrics

    def _compute_metrics(self, pred_tokens, true_tokens, pred_values, true_values, pred_probs, pred_confidences, reg_locs, reg_scales):
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
        if true_values:
            true_values = np.array(true_values)
            pred_values = np.array(pred_values)
            metrics['reg_mse'] = np.mean((true_values - pred_values)**2)
            metrics['reg_mae'] = np.mean(np.abs(true_values - pred_values))
        else:
            metrics['reg_mse'] = 0
            metrics['reg_mae'] = 0

        # --- Calibration Metrics ---
        # ECE (Expected Calibration Error)
        metrics['calib_ece'] = self._compute_ece(pred_confidences, pred_tokens, true_tokens)
        
        # PICP (Prediction Interval Coverage Probability) for Regression
        if reg_locs:
            metrics['reg_picp'] = self._compute_picp(
                np.array(true_values), np.array(reg_locs), np.array(reg_scales)
            )
        else:
            metrics['reg_picp'] = 0
            
        return metrics

    def _compute_ece(self, confidences, predictions, labels, n_bins=15):
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