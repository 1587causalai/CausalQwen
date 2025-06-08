"""
Action Network module.

This module implements the action network for the causal language model,
which transforms the latent causal state into classification and regression outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.distributions import CauchyLinear


class ClassificationHead(nn.Module):
    """
    Classification head for token prediction.
    
    This implements the One-vs-Rest (OvR) classification approach,
    where each class has an independent decision score.
    """
    
    def __init__(self, causal_dim, num_classes, threshold=0.0):
        """
        Initialize the classification head.
        
        Args:
            causal_dim (int): Dimensionality of the latent causal state
            num_classes (int): Number of classes (vocabulary size)
            threshold (float, optional): Decision threshold. Defaults to 0.0.
        """
        super().__init__()
        self.causal_dim = causal_dim
        self.num_classes = num_classes
        self.threshold = threshold
        
        # Linear layer to map causal state to class decision scores
        self.causal_linear = CauchyLinear(causal_dim, num_classes)
        
        # Register threshold as a buffer (not a parameter)
        self.register_buffer('thresholds', torch.ones(num_classes) * threshold)
        
    def forward(self, causal_loc, causal_scale):
        """
        Transform causal state distribution to class decision score distributions.
        
        Args:
            causal_loc (torch.Tensor): Location parameter of causal state
                                      Shape: [batch_size, causal_dim]
            causal_scale (torch.Tensor): Scale parameter of causal state
                                        Shape: [batch_size, causal_dim]
        
        Returns:
            tuple: (score_loc, score_scale) - Parameters of the decision score distributions
                  Each has shape: [batch_size, num_classes]
        """
        # Transform causal state distribution to decision score distributions
        score_loc, score_scale = self.causal_linear(causal_loc, causal_scale)
        
        return score_loc, score_scale
    
    def compute_probabilities(self, score_loc, score_scale):
        """
        Compute class probabilities using the Cauchy CDF.
        
        From core-design.md: P(S_k > C_k) = 1/2 + (1/π) * arctan((loc_Sk - C_k)/scale_Sk)
        
        Args:
            score_loc (torch.Tensor): Location parameters of decision scores
                                     Shape: [batch_size, num_classes]
            score_scale (torch.Tensor): Scale parameters of decision scores
                                       Shape: [batch_size, num_classes]
        
        Returns:
            torch.Tensor: Class probabilities
                         Shape: [batch_size, num_classes]
        """
        # Direct implementation of the formula from core-design.md
        # P(S_k > C_k) = 1/2 + (1/π) * arctan((loc_Sk - C_k)/scale_Sk)
        probs = 0.5 + (1 / torch.pi) * torch.atan((score_loc - self.thresholds) / score_scale)
        
        return probs
    
    def predict(self, score_loc, score_scale):
        """
        Make class predictions based on decision scores.
        
        Args:
            score_loc (torch.Tensor): Location parameters of decision scores
                                     Shape: [batch_size, num_classes]
            score_scale (torch.Tensor): Scale parameters of decision scores
                                       Shape: [batch_size, num_classes]
        
        Returns:
            torch.Tensor: Predicted class indices
                         Shape: [batch_size]
        """
        # Compute probabilities
        probs = self.compute_probabilities(score_loc, score_scale)
        
        # Return class with highest probability
        return torch.argmax(probs, dim=-1)


class RegressionHead(nn.Module):
    """
    Regression head for numerical prediction.
    """
    
    def __init__(self, causal_dim):
        """
        Initialize the regression head.
        
        Args:
            causal_dim (int): Dimensionality of the latent causal state
        """
        super().__init__()
        self.causal_dim = causal_dim
        
        # Linear layer to map causal state to regression value
        self.causal_linear = CauchyLinear(causal_dim, 1)
        
    def forward(self, causal_loc, causal_scale):
        """
        Transform causal state distribution to regression value distribution.
        
        Args:
            causal_loc (torch.Tensor): Location parameter of causal state
                                      Shape: [batch_size, causal_dim]
            causal_scale (torch.Tensor): Scale parameter of causal state
                                        Shape: [batch_size, causal_dim]
        
        Returns:
            tuple: (value_loc, value_scale) - Parameters of the regression value distribution
                  Each has shape: [batch_size, 1]
        """
        # Transform causal state distribution to regression value distribution
        value_loc, value_scale = self.causal_linear(causal_loc, causal_scale)
        
        # Squeeze the last dimension to get [batch_size] tensors
        return value_loc.squeeze(-1), value_scale.squeeze(-1)
    
    def predict(self, value_loc, value_scale):
        """
        Make regression predictions.
        
        For Cauchy distribution, the median (location parameter) is the best
        point estimate.
        
        Args:
            value_loc (torch.Tensor): Location parameter of regression value
                                     Shape: [batch_size]
            value_scale (torch.Tensor): Scale parameter of regression value
                                       Shape: [batch_size]
        
        Returns:
            torch.Tensor: Predicted regression values
                         Shape: [batch_size]
        """
        # For Cauchy distribution, the median (location parameter) is the best point estimate
        return value_loc


class ActionNetwork(nn.Module):
    """
    Action Network that combines classification and regression heads.
    
    This network transforms the latent causal state into both classification
    and regression outputs.
    """
    
    def __init__(self, causal_dim, num_classes, num_token_id):
        """
        Initialize the action network.
        
        Args:
            causal_dim (int): Dimensionality of the latent causal state
            num_classes (int): Number of classes (vocabulary size)
            num_token_id (int): Token ID for the <NUM> token
        """
        super().__init__()
        self.causal_dim = causal_dim
        self.num_classes = num_classes
        self.num_token_id = num_token_id
        
        # Classification head for token prediction
        self.classification_head = ClassificationHead(causal_dim, num_classes)
        
        # Regression head for numerical prediction
        self.regression_head = RegressionHead(causal_dim)
        
    def init_weights(self, qwen_lm_head, num_target_mean, num_target_std, num_token_id):
        """
        Initialize weights based on pretrained Qwen model and data statistics.
        
        Args:
            qwen_lm_head (nn.Linear): The pretrained language model head from Qwen.
            num_target_mean (float): The mean of the numerical target values.
            num_target_std (float): The standard deviation of the numerical target values.
            num_token_id (int): The token ID for the <NUM> token.
        """
        # 1. Initialize Classification Head
        cls_head = self.classification_head.causal_linear
        
        # Handle vocabulary size mismatch between our tokenizer and Qwen model
        qwen_vocab_size = qwen_lm_head.weight.shape[0]
        our_vocab_size = self.num_classes  # This includes our added <NUM> token
        
        # The overlapping vocabulary size (excluding our <NUM> token)
        overlapping_vocab_size = min(qwen_vocab_size, our_vocab_size - 1)
        
        print(f"  - Qwen vocab size: {qwen_vocab_size}, Our vocab size: {our_vocab_size}")
        print(f"  - Copying weights for {overlapping_vocab_size} overlapping tokens")

        # Copy weights and biases for overlapping tokens
        cls_head.loc_layer.weight.data[:overlapping_vocab_size, :].copy_(
            qwen_lm_head.weight.data[:overlapping_vocab_size, :]
        )
        if qwen_lm_head.bias is not None:
            cls_head.loc_layer.bias.data[:overlapping_vocab_size].copy_(
                qwen_lm_head.bias.data[:overlapping_vocab_size]
            )
        
        # Initialize any remaining tokens (between overlapping_vocab_size and our_vocab_size-1) to zero
        if overlapping_vocab_size < our_vocab_size - 1:
            cls_head.loc_layer.weight.data[overlapping_vocab_size:our_vocab_size-1, :].fill_(0)
            cls_head.loc_layer.bias.data[overlapping_vocab_size:our_vocab_size-1].fill_(0)
        
        # Initialize the new <NUM> token row for loc to zero
        cls_head.loc_layer.weight.data[num_token_id, :].fill_(0)
        # Penalize the bias for <NUM> token to suppress initial predictions
        cls_head.loc_layer.bias.data[num_token_id].fill_(-10.0)

        # Initialize all scale parameters to a high uncertainty state
        # Weight = zero, Bias = large positive value
        cls_head.scale_layer.weight.data.fill_(0)
        cls_head.scale_layer.bias.data.fill_(2.3) # exp(2.3) approx 10

        # 2. Initialize Regression Head
        reg_head = self.regression_head.causal_linear
        
        # Initialize loc to predict the mean of the data
        reg_head.loc_layer.weight.data.fill_(0)
        reg_head.loc_layer.bias.data.fill_(num_target_mean)

        # Initialize scale to reflect the standard deviation of the data
        reg_head.scale_layer.weight.data.fill_(0)
        # Use log of std dev, ensure std > 0
        reg_head.scale_layer.bias.data.fill_(torch.log(torch.tensor(num_target_std) + 1e-6))

    def forward(self, causal_loc, causal_scale):
        """
        Transform causal state distribution to classification and regression outputs.
        
        Args:
            causal_loc (torch.Tensor): Location parameter of causal state
                                      Shape: [batch_size, causal_dim]
            causal_scale (torch.Tensor): Scale parameter of causal state
                                        Shape: [batch_size, causal_dim]
        
        Returns:
            dict: Dictionary containing all output distribution parameters
        """
        # Get classification outputs
        cls_loc, cls_scale = self.classification_head(causal_loc, causal_scale)
        
        # Get regression outputs
        reg_loc, reg_scale = self.regression_head(causal_loc, causal_scale)
        
        # Compute class probabilities
        cls_probs = self.classification_head.compute_probabilities(cls_loc, cls_scale)
        
        return {
            'cls_loc': cls_loc,
            'cls_scale': cls_scale,
            'reg_loc': reg_loc,
            'reg_scale': reg_scale,
            'cls_probs': cls_probs
        }
    
    def predict(self, causal_loc, causal_scale):
        """
        Make predictions based on causal state.
        
        Args:
            causal_loc (torch.Tensor): Location parameter of causal state
                                      Shape: [batch_size, causal_dim]
            causal_scale (torch.Tensor): Scale parameter of causal state
                                        Shape: [batch_size, causal_dim]
        
        Returns:
            dict: Dictionary containing predictions
        """
        # Get output distributions
        outputs = self.forward(causal_loc, causal_scale)
        
        # Make classification prediction
        cls_pred = self.classification_head.predict(outputs['cls_loc'], outputs['cls_scale'])
        
        # Make regression prediction
        reg_pred = self.regression_head.predict(outputs['reg_loc'], outputs['reg_scale'])
        
        # Get probability of <NUM> token
        num_prob = outputs['cls_probs'][:, self.num_token_id]
        
        return {
            'cls_pred': cls_pred,
            'reg_pred': reg_pred,
            'num_prob': num_prob,
            **outputs
        }

