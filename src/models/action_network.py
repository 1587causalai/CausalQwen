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
    
    def __init__(self, causal_dim, num_classes, threshold=10.0):
        """
        Initialize the classification head.
        
        Args:
            causal_dim (int): Dimensionality of the latent causal state
            num_classes (int): Number of classes (vocabulary size)
            threshold (float, optional): Decision threshold. Defaults to 10.0.
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
    Action Network for the Causal Language Model.
    
    This network takes the parameters of the individual causal representation distribution
    and transforms them into the parameters for the output distributions
    (classification scores and regression values).
    """
    
    def __init__(self, causal_dim, vocab_size, num_token_id):
        """
        Initialize the Action Network.
        
        Args:
            causal_dim (int): Dimensionality of the individual causal representation
            vocab_size (int): Size of the vocabulary
            num_token_id (int): The token ID for the <NUM> token.
        """
        super().__init__()
        self.causal_dim = causal_dim
        self.vocab_size = vocab_size
        self.num_token_id = num_token_id
        
        # Classification head for token prediction
        self.classification_head = ClassificationHead(causal_dim, vocab_size)
        
        # Regression head for numerical prediction
        self.regression_head = RegressionHead(causal_dim)

    def init_weights(self, qwen_lm_head, num_target_median, num_target_scale, num_token_id):
        """
        Initialize weights based on pretrained Qwen model and data statistics.
        
        Args:
            qwen_lm_head (nn.Linear): The pretrained language model head from Qwen.
            num_target_median (float): The median of the numerical target values (Cauchy location parameter).
            num_target_scale (float): The scale parameter for numerical targets (Cauchy scale parameter).
            num_token_id (int): The token ID for the <NUM> token.
        """
        # 1. Initialize Classification Head
        cls_head = self.classification_head.causal_linear
        
        # Handle vocabulary size mismatch between our tokenizer and Qwen model
        qwen_vocab_size = qwen_lm_head.weight.shape[0]
        our_vocab_size = self.vocab_size  # This includes our added <NUM> token
        
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
        
        # Initialize loc to predict the median of the data (Cauchy location parameter)
        reg_head.loc_layer.weight.data.fill_(0)
        reg_head.loc_layer.bias.data.fill_(num_target_median)

        # Initialize scale to reflect the scale parameter of the data (Cauchy scale parameter)
        reg_head.scale_layer.weight.data.fill_(0)
        # Use log of scale parameter, ensure scale > 0
        reg_head.scale_layer.bias.data.fill_(torch.log(torch.tensor(num_target_scale) + 1e-6))

    def forward(self, causal_loc, causal_scale):
        """
        Transform the individual causal representation distribution to output distributions.
        
        Args:
            causal_loc (torch.Tensor): Location parameters of the individual causal representation
            causal_scale (torch.Tensor): Scale parameters of the individual causal representation
        
        Returns:
            dict: A dictionary containing the parameters for classification and regression.
        """
        # 1. Classification
        cls_loc, cls_scale = self.classification_head(causal_loc, causal_scale)
        
        # 2. Regression
        reg_loc, reg_scale = self.regression_head(causal_loc, causal_scale)
        
        return {
            'cls_loc': cls_loc,
            'cls_scale': cls_scale,
            'reg_loc': reg_loc,
            'reg_scale': reg_scale
        }
    
    def predict(self, causal_loc, causal_scale=None):
        """
        Make a deterministic prediction using the median of the individual causal representation.
        
        Args:
            causal_loc (torch.Tensor): Location parameters of the individual causal representation
            causal_scale (torch.Tensor, optional): Scale parameters (not used for median prediction).
        
        Returns:
            dict: A dictionary containing the predicted token and regression value.
        """
        # For prediction, we use the location parameters (median) as point estimates
        # 1. Classification: Use the classification head's loc layer directly for deterministic scores  
        cls_scores = self.classification_head.causal_linear.loc_layer(causal_loc)
        
        # 2. Regression: Use the regression head's loc layer directly for deterministic value
        reg_value = self.regression_head.causal_linear.loc_layer(causal_loc).squeeze(-1)
        
        # Get probability of <NUM> token (computed using the deterministic scores)
        # Apply Cauchy CDF formula: P(S > threshold) = 0.5 + (1/π) * arctan((loc - threshold)/scale)
        # For prediction, we can use a default scale of 1.0 or compute it if needed
        threshold = self.classification_head.thresholds[self.num_token_id]
        num_prob = 0.5 + (1 / torch.pi) * torch.atan((cls_scores[:, self.num_token_id] - threshold) / 1.0)
        
        return {
            'cls_pred': torch.argmax(cls_scores, dim=-1),
            'reg_pred': reg_value,
            'num_prob': num_prob
        }

