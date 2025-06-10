"""
Loss functions module.

This module implements the loss functions for the causal language model,
including One-vs-Rest (OvR) classification loss and gated regression loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distributions import cauchy_cdf, cauchy_nll_loss


def compute_ovr_probabilities(loc, scale, threshold=0.0):
    """
    Computes OvR probabilities based on decision score distributions.

    Args:
        loc (torch.Tensor): Location parameters for each class decision score.
        scale (torch.Tensor): Scale parameters for each class decision score.
        threshold (float, optional): Decision threshold. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed probabilities for each class.
    """
    # Implements the formula P(S > C) = 1/2 + (1/π) * arctan((loc - C)/scale)
    return 0.5 + (1 / math.pi) * torch.atan((loc - threshold) / scale)


class OvRClassificationLoss(nn.Module):
    """
    One-vs-Rest (OvR) classification loss.
    
    Instead of using softmax, this loss treats each class as an independent
    binary classification problem, which aligns with the causal decision-making
    framework.
    """
    
    def __init__(self, num_classes, threshold=0.0):
        """
        Initialize the OvR classification loss.
        
        Args:
            num_classes (int): Number of classes
            threshold (float, optional): Decision threshold. Defaults to 0.0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        
    def forward(self, loc, scale, targets):
        """
        Compute the OvR classification loss.
        
        Args:
            loc (torch.Tensor): Location parameters for each class decision score
                                Shape: [batch_size, num_classes]
            scale (torch.Tensor): Scale parameters for each class decision score
                                 Shape: [batch_size, num_classes]
            targets (torch.Tensor): Target class indices
                                   Shape: [batch_size]
            
        Returns:
            torch.Tensor: OvR classification loss
        """
        
        # Use the utility function to compute probabilities
        probs = compute_ovr_probabilities(loc, scale, self.threshold)
        
        # Create one-hot target tensor
        target_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # Binary cross-entropy loss for each class
        # For the true class (y_k = 1), we want P(S_k > threshold) to be high
        # For other classes (y_k = 0), we want P(S_k > threshold) to be low
        bce_loss = -(target_one_hot * torch.log(probs + 1e-10) + 
                     (1 - target_one_hot) * torch.log(1 - probs + 1e-10))
        
        # Sum over classes, mean over batch
        return bce_loss.sum(dim=1).mean()


class GatedRegressionLoss(nn.Module):
    """
    Hard-Gated regression loss for numerical prediction.
    
    This loss is only activated when the target is a numerical value (indicated by <NUM> token).
    It uses a "hard" gate (the target must be <NUM>) instead of being scaled by
    the model's prediction probability, which proved to be unstable.
    """
    
    def __init__(self, num_token_id):
        """
        Initialize the gated regression loss.
        
        Args:
            num_token_id (int): Token ID for the <NUM> token
        """
        super().__init__()
        self.num_token_id = num_token_id
        
    def forward(self, reg_loc, reg_scale, targets, target_values):
        """
        Compute the hard-gated regression loss.
        
        Args:
            reg_loc (torch.Tensor): Predicted location parameter for regression
            reg_scale (torch.Tensor): Predicted scale parameter for regression
            targets (torch.Tensor): Target token IDs
            target_values (torch.Tensor): Target numerical values
            
        Returns:
            torch.Tensor: Hard-gated regression loss
        """
        # Create mask for samples where target is <NUM>
        is_num_mask = (targets == self.num_token_id).float()
        num_count = is_num_mask.sum()

        if num_count == 0:
            return torch.tensor(0.0, device=reg_loc.device)

        # Compute the standard Cauchy NLL loss
        cauchy_loss = cauchy_nll_loss(reg_loc, reg_scale, target_values)
        
        # Apply a "hard" gate: only compute loss for <NUM> tokens.
        # We do not scale by the probability, as it created instabilities.
        hard_gated_loss = is_num_mask * cauchy_loss
        
        # Return the mean loss over the <NUM> samples in the batch
        return hard_gated_loss.sum() / num_count


class CausalLMLoss(nn.Module):
    """
    Combined loss function for the causal language model.
    
    This combines the OvR classification loss and the gated regression loss,
    and it's designed to handle sequence-to-sequence targets.
    """
    
    def __init__(self, num_classes, num_token_id, regression_weight=1.0, ovr_threshold=0.0, ignore_index=-100):
        """
        Initialize the combined loss function.
        
        Args:
            num_classes (int): Number of classes (vocabulary size)
            num_token_id (int): Token ID for the <NUM> token
            regression_weight (float, optional): Weight for the regression loss. Defaults to 1.0.
            ovr_threshold (float, optional): Decision threshold for OvR loss. Defaults to 0.0.
            ignore_index (int, optional): Specifies a target value that is ignored
                                          and does not contribute to the input gradient.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_token_id = num_token_id
        self.regression_weight = regression_weight
        self.ovr_threshold = ovr_threshold
        self.ignore_index = ignore_index
        
    def forward(self, cls_loc, cls_scale, reg_loc, reg_scale, labels, target_values):
        """
        Compute the combined, sequence-to-sequence loss.
        
        Args:
            cls_loc (torch.Tensor): Location params for classification. Shape: [B, S, C]
            cls_scale (torch.Tensor): Scale params for classification. Shape: [B, S, C]
            reg_loc (torch.Tensor): Location param for regression. Shape: [B, S]
            reg_scale (torch.Tensor): Scale param for regression. Shape: [B, S]
            labels (torch.Tensor): Target token IDs. Shape: [B, S]
            target_values (torch.Tensor): Target numerical values. Shape: [B, S]
            
        Returns:
            dict: Dictionary containing total loss and individual loss components
        """
        batch_size, seq_len, _ = cls_loc.shape
        
        # --- 1. Classification Loss (Sequence-wise) ---
        # Create a mask for valid (non-ignored) tokens
        active_loss_mask = labels.view(-1) != self.ignore_index
        
        # Flatten predictions and labels to [B*S, C] and [B*S]
        active_cls_loc = cls_loc.view(-1, self.num_classes)[active_loss_mask]
        active_cls_scale = cls_scale.view(-1, self.num_classes)[active_loss_mask]
        active_labels = labels.view(-1)[active_loss_mask]
        
        # Compute OvR probabilities for active tokens
        cls_probs = compute_ovr_probabilities(active_cls_loc, active_cls_scale, self.ovr_threshold)
        
        # Create one-hot targets for active tokens
        target_one_hot = F.one_hot(active_labels, self.num_classes).float()
        
        # BCE loss for each class, on active tokens only
        bce_loss = -(target_one_hot * torch.log(cls_probs + 1e-9) + 
                     (1 - target_one_hot) * torch.log(1 - cls_probs + 1e-9))
        
        # Sum over classes, and then take the mean over all active tokens
        classification_loss = bce_loss.sum(dim=1).mean()

        # --- 2. Gated Regression Loss (Sequence-wise) ---
        # Mask for positions where the label is <NUM> AND is active
        is_num_mask = (labels == self.num_token_id) & (labels != self.ignore_index)
        num_count = is_num_mask.sum()

        if num_count == 0:
            gated_regression_loss = torch.tensor(0.0, device=reg_loc.device)
        else:
            # Select only the active <NUM> predictions and targets
            active_reg_loc = reg_loc[is_num_mask]
            active_reg_scale = reg_scale[is_num_mask]
            active_target_values = target_values[is_num_mask]

            # Compute the base Cauchy NLL loss for the active <NUM> positions
            unweighted_regression_loss_per_sample = cauchy_nll_loss(
                active_reg_loc, active_reg_scale, active_target_values, reduction='none'
            )
            
            # Get the gate probability P(<NUM>) at the active <NUM> positions
            # Note: cls_probs is already filtered for active tokens, so we need a new mask
            active_num_mask_flat = (active_labels == self.num_token_id)
            num_prob = cls_probs[active_num_mask_flat, self.num_token_id]
            
            # Apply the soft gate: P(<NUM>) * L_cauchy
            gated_loss_per_sample = num_prob * unweighted_regression_loss_per_sample
            
            # The final loss is the mean over all active <NUM> samples
            gated_regression_loss = gated_loss_per_sample.mean()

        # --- 3. Combine losses ---
        total_loss = classification_loss + self.regression_weight * gated_regression_loss
        
        return {
            'loss': total_loss,
            'cls_loss': classification_loss,
            'gated_reg_loss': gated_regression_loss,
        }

