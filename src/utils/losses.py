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
        self.register_buffer('thresholds', torch.ones(num_classes) * threshold)
        
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
        batch_size = loc.size(0)
        
        # Direct implementation of the formula from core-design.md
        # P(S_k > C_k) = 1/2 + (1/π) * arctan((loc_Sk - C_k)/scale_Sk)
        probs = 0.5 + (1 / math.pi) * torch.atan((loc - self.thresholds) / scale)
        
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
    Gated regression loss for numerical prediction.
    
    This loss is only activated when the target is a numerical value (indicated by <NUM> token),
    and is weighted by the model's confidence in predicting the <NUM> token.
    """
    
    def __init__(self, num_token_id):
        """
        Initialize the gated regression loss.
        
        Args:
            num_token_id (int): Token ID for the <NUM> token
        """
        super().__init__()
        self.num_token_id = num_token_id
        
    def forward(self, reg_loc, reg_scale, num_prob, targets, target_values):
        """
        Compute the gated regression loss.
        
        Args:
            reg_loc (torch.Tensor): Predicted location parameter for regression
                                   Shape: [batch_size]
            reg_scale (torch.Tensor): Predicted scale parameter for regression
                                     Shape: [batch_size]
            num_prob (torch.Tensor): Probability of predicting <NUM> token
                                    Shape: [batch_size]
            targets (torch.Tensor): Target token IDs
                                   Shape: [batch_size]
            target_values (torch.Tensor): Target numerical values (only valid when target is <NUM>)
                                         Shape: [batch_size]
            
        Returns:
            torch.Tensor: Gated regression loss
        """
        # Create mask for samples where target is <NUM>
        is_num_mask = (targets == self.num_token_id).float()
        
        # Compute Cauchy NLL loss for regression
        cauchy_loss = cauchy_nll_loss(reg_loc, reg_scale, target_values)
        
        # Gate the regression loss by:
        # 1. Only applying it to samples where target is <NUM>
        # 2. Weighting it by the model's confidence in predicting <NUM>
        gated_loss = is_num_mask * num_prob * cauchy_loss
        
        # Return mean loss over batch
        # If no <NUM> targets in batch, return zero loss
        num_count = is_num_mask.sum()
        if num_count > 0:
            return gated_loss.sum() / num_count
        else:
            return torch.tensor(0.0, device=gated_loss.device)


class CausalLMLoss(nn.Module):
    """
    Combined loss function for the causal language model.
    
    This combines the OvR classification loss and the gated regression loss.
    """
    
    def __init__(self, num_classes, num_token_id, regression_weight=1.0):
        """
        Initialize the combined loss function.
        
        Args:
            num_classes (int): Number of classes (vocabulary size)
            num_token_id (int): Token ID for the <NUM> token
            regression_weight (float, optional): Weight for the regression loss. Defaults to 1.0.
        """
        super().__init__()
        self.cls_loss = OvRClassificationLoss(num_classes)
        self.reg_loss = GatedRegressionLoss(num_token_id)
        self.regression_weight = regression_weight
        
    def forward(self, cls_loc, cls_scale, reg_loc, reg_scale, targets, target_values):
        """
        Compute the combined loss.
        
        Args:
            cls_loc (torch.Tensor): Location parameters for classification
                                   Shape: [batch_size, num_classes]
            cls_scale (torch.Tensor): Scale parameters for classification
                                     Shape: [batch_size, num_classes]
            reg_loc (torch.Tensor): Location parameter for regression
                                   Shape: [batch_size]
            reg_scale (torch.Tensor): Scale parameter for regression
                                     Shape: [batch_size]
            targets (torch.Tensor): Target token IDs
                                   Shape: [batch_size]
            target_values (torch.Tensor): Target numerical values
                                         Shape: [batch_size]
            
        Returns:
            dict: Dictionary containing total loss and individual loss components
        """
        # Compute classification loss
        classification_loss = self.cls_loss(cls_loc, cls_scale, targets)
        
        # Get probability of <NUM> token for gating
        # Direct implementation from core-design.md: P(S_k > C_k) = 1/2 + (1/π) * arctan((loc_Sk - C_k)/scale_Sk)
        num_prob = 0.5 + (1 / math.pi) * torch.atan(
            (cls_loc[:, self.reg_loss.num_token_id] - 0.0) / cls_scale[:, self.reg_loss.num_token_id]
        )
        
        # Compute regression loss
        regression_loss = self.reg_loss(reg_loc, reg_scale, num_prob, targets, target_values)
        
        # Combine losses
        total_loss = classification_loss + self.regression_weight * regression_loss
        
        return {
            'loss': total_loss,
            'cls_loss': classification_loss,
            'reg_loss': regression_loss
        }

