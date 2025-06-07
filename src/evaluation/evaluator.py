"""
Evaluator module for the Causal Language Model.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

class Evaluator:
    """Handles the evaluation of the model on various datasets."""
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize the Evaluator.
        
        Args:
            model (nn.Module): The model to be evaluated.
            tokenizer: The tokenizer to use.
            device (torch.device): The device to run evaluation on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, dataset, batch_size=16):
        """
        Run evaluation on a given dataset.
        
        Args:
            dataset (TensorDataset): The dataset to evaluate on.
            batch_size (int): The batch size for evaluation.
            
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        total_correct = 0
        total_samples = 0
        all_num_ranks = []
        all_num_probs = []
        all_reg_errors = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = [t.to(self.device) for t in batch]
                input_ids, attention_mask, numerical_values, targets, target_values = batch
                
                # Forward pass to get probabilities and full output
                outputs = self.model(input_ids, numerical_values, attention_mask)
                cls_probs = outputs['cls_probs']
                
                # Get predictions for accuracy calculation
                predictions = self.model.predict(input_ids, numerical_values, attention_mask)
                pred_tokens = predictions['cls_pred']
                
                # --- Metric Calculation ---
                # 1. Accuracy
                total_correct += (pred_tokens == targets).sum().item()
                total_samples += targets.size(0)
                
                # 2. <NUM> token rank and probability
                num_token_id = self.tokenizer.num_token_id
                ranks = (cls_probs.argsort(dim=1, descending=True) == num_token_id).nonzero(as_tuple=True)[1] + 1
                all_num_ranks.extend(ranks.cpu().tolist())
                
                num_probs = cls_probs[:, num_token_id]
                all_num_probs.extend(num_probs.cpu().tolist())
                
                # 3. Regression error (only for correct predictions)
                correct_mask = (pred_tokens == targets)
                if correct_mask.sum() > 0:
                    pred_values = predictions['reg_pred'][correct_mask]
                    true_vals = target_values[correct_mask]
                    errors = F.mse_loss(pred_values, true_vals, reduction='none')
                    all_reg_errors.extend(torch.sqrt(errors).cpu().tolist())

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        mean_rank = np.mean(all_num_ranks) if all_num_ranks else -1
        mean_prob = np.mean(all_num_probs) if all_num_probs else -1
        rmse = np.mean(all_reg_errors) if all_reg_errors else -1
        
        return {
            "accuracy": accuracy,
            "mean_rank": mean_rank,
            "mean_prob": mean_prob,
            "rmse": rmse,
            "num_samples": total_samples
        } 