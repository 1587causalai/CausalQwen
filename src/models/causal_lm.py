"""
Causal Language Model.

This module implements the complete causal language model by integrating
all components: feature network, abduction network, and action network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from .feature_network import MockFeatureNetwork, NumAwareFeatureNetwork, QwenFeatureNetwork
from .abduction_network import AbductionNetwork
from .action_network import ActionNetwork
from ..utils.distributions import cauchy_sample_reparameterized


@dataclass
class CausalLMConfig:
    """
    Configuration class for Causal Language Model.
    
    This class holds all the configuration parameters needed to initialize
    a CausalLanguageModel instance.
    """
    vocab_size: int = 1000
    hidden_size: int = 1024
    causal_dim: int = 64
    num_token_id: Optional[int] = None
    use_mock_feature_network: bool = True
    use_real_qwen: bool = False  # New flag for using real Qwen model
    qwen_model_path: str = "~/models/Qwen2.5-0.5B"  # Path to Qwen model
    use_num_aware_features: bool = True
    use_ovr_classifier: bool = True
    use_cauchy_distribution: bool = True
    reg_loss_weight: float = 1.0
    ovr_threshold: float = 0.0
    
    def __post_init__(self):
        if self.num_token_id is None:
            # Set default NUM token ID to vocab_size (assuming it's added to vocab)
            self.num_token_id = self.vocab_size


class CausalLanguageModel(nn.Module):
    """
    Complete Causal Language Model.
    
    This model integrates all components of the causal language model architecture:
    1. Feature Network: Extracts features from input tokens
    2. Abduction Network: Infers the distribution of the latent individual causal representation
    3. Action Network: Transforms the individual causal representation into classification and regression outputs
    """
    
    def __init__(
        self,
        config=None,
        vocab_size=None,
        num_token_id=None,
        hidden_size=1024,
        causal_dim=64,
        use_mock_feature_network=True,
        use_num_aware_features=True
    ):
        """
        Initialize the causal language model.
        
        Args:
            config (CausalLMConfig, optional): Configuration object. If provided, other args are ignored.
            vocab_size (int, optional): Size of the vocabulary
            num_token_id (int, optional): Token ID for the <NUM> token
            hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
            causal_dim (int, optional): Dimensionality of the latent individual causal representation. Defaults to 64.
            use_mock_feature_network (bool, optional): Whether to use a mock feature network. 
                                                      Defaults to True.
            use_num_aware_features (bool, optional): Whether to use numerical-aware features. 
                                                    Defaults to True.
        """
        super().__init__()
        
        # If config is provided, use it; otherwise use direct parameters
        if config is not None:
            self.vocab_size = config.vocab_size
            self.num_token_id = config.num_token_id
            self.hidden_size = config.hidden_size
            self.causal_dim = config.causal_dim
            use_mock_feature_network = config.use_mock_feature_network
            use_num_aware_features = config.use_num_aware_features
            use_real_qwen = config.use_real_qwen
            qwen_model_path = config.qwen_model_path
        else:
            self.vocab_size = vocab_size or 1000
            self.num_token_id = num_token_id or self.vocab_size
            self.hidden_size = hidden_size
            self.causal_dim = causal_dim
            use_real_qwen = False
            qwen_model_path = "~/models/Qwen2.5-0.5B"
        
        # Initialize feature network based on configuration
        if use_real_qwen:
            # Use real Qwen model as feature network
            print("Initializing with real Qwen model...")
            base_feature_network = QwenFeatureNetwork(
                model_path=qwen_model_path,
                hidden_size=self.hidden_size,
                use_real_model=True
            )
        else:
            # Use mock feature network
            base_feature_network = MockFeatureNetwork(hidden_size=self.hidden_size)
        
        # Wrap with numerical-aware feature network if requested
        if use_num_aware_features:
            self.feature_network = NumAwareFeatureNetwork(
                base_feature_network, self.num_token_id, self.hidden_size
            )
        else:
            self.feature_network = base_feature_network
        
        # Initialize abduction network
        self.abduction_network = AbductionNetwork(self.hidden_size, self.causal_dim)
        
        # Initialize action network
        self.action_network = ActionNetwork(
            self.causal_dim, 
            self.vocab_size, 
            self.num_token_id,
            ovr_threshold=config.ovr_threshold if config is not None else 10.0
        )
        
    def init_weights(self, num_target_median, num_target_scale):
        """
        Initialize the weights of abduction and action networks using the
        knowledge transfer strategy.
        
        Args:
            num_target_median (float): The median of the numerical target values (Cauchy location parameter).
            num_target_scale (float): The scale parameter for numerical targets (Cauchy scale parameter).
        """
        print("Applying knowledge transfer initialization...")
        
        # 1. Initialize Abduction Network
        # This assumes hidden_size and causal_dim are the same
        if self.hidden_size == self.causal_dim:
            self.abduction_network.init_weights()
            print("  - Abduction network initialized for identity mapping.")
        else:
            print("  - WARNING: Abduction network not initialized (hidden_size != causal_dim).")

        # 2. Initialize Action Network
        qwen_lm_head = None
        # We need to get the lm_head from the underlying QwenFeatureNetwork
        if isinstance(self.feature_network, NumAwareFeatureNetwork) and \
           isinstance(self.feature_network.base_network, QwenFeatureNetwork):
            qwen_lm_head = self.feature_network.base_network.get_lm_head()
        elif isinstance(self.feature_network, QwenFeatureNetwork):
            qwen_lm_head = self.feature_network.get_lm_head()

        if qwen_lm_head is not None:
            self.action_network.init_weights(
                qwen_lm_head=qwen_lm_head,
                num_target_median=num_target_median,
                num_target_scale=num_target_scale,
                num_token_id=self.num_token_id
            )
            print("  - Action network initialized from Qwen's lm_head and data stats.")
        else:
            print("  - WARNING: Action network not initialized (Qwen lm_head not available).")

    def forward(self, input_ids, numerical_values=None, attention_mask=None):
        """
        Forward pass of the causal language model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens
                                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            dict: Dictionary containing all output distribution parameters and intermediate states
        """
        # Extract features
        features = self.feature_network(input_ids, numerical_values, attention_mask)
        
        # Infer individual causal representation distribution
        causal_loc, causal_scale = self.abduction_network(features)
        
        # Transform individual causal representation to outputs
        outputs = self.action_network(causal_loc, causal_scale)
        
        # Add intermediate states to outputs
        outputs.update({
            'features': features,
            'causal_loc': causal_loc,
            'causal_scale': causal_scale
        })
        
        return outputs
    
    def sample_and_predict(self, input_ids, numerical_values=None, attention_mask=None):
        """
        Sample from the individual causal representation distribution and make predictions.
        
        This method is used for exploration or when simulating real-world randomness.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens
                                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            dict: Dictionary containing predictions and sampled states
        """
        # Extract features
        features = self.feature_network(input_ids, numerical_values, attention_mask)
        
        # Infer individual causal representation distribution
        causal_loc, causal_scale = self.abduction_network(features)
        
        # Sample from the individual causal representation distribution
        causal_sample = cauchy_sample_reparameterized(causal_loc, causal_scale)
        
        # Make predictions using the sampled individual causal representation
        # For sampled prediction, we use the same location parameter for all samples
        # but with zero scale (deterministic)
        predictions = self.action_network.predict(causal_sample, torch.zeros_like(causal_scale))
        
        # Add sampled state to predictions
        predictions['causal_sample'] = causal_sample
        
        return predictions
    
    def predict(self, input_ids, numerical_values=None, attention_mask=None):
        """
        Make deterministic predictions without sampling.
        
        This method uses the median (location parameter) of the individual causal representation distribution
        for prediction, which is more stable and efficient.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens
                                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            dict: Dictionary containing predictions
        """
        # Extract features
        features = self.feature_network(input_ids, numerical_values, attention_mask)
        
        # Infer individual causal representation distribution
        causal_loc, causal_scale = self.abduction_network(features)
        
        # Make predictions using the median of the individual causal representation distribution
        predictions = self.action_network.predict(causal_loc, torch.zeros_like(causal_scale))
        
        return predictions


class CausalQwen(nn.Module):
    """
    CausalQwen model that extends the base causal language model.
    
    This is a placeholder for the full implementation that would integrate
    with the Qwen-0.5B model.
    """
    
    def __init__(
        self,
        vocab_size,
        num_token_id,
        hidden_size=1024,
        causal_dim=64
    ):
        """
        Initialize the CausalQwen model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            num_token_id (int): Token ID for the <NUM> token
            hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
            causal_dim (int, optional): Dimensionality of the latent individual causal representation. Defaults to 64.
        """
        super().__init__()
        
        # This is a placeholder. In a real implementation, we would initialize
        # the model with the actual Qwen-0.5B backbone.
        self.model = CausalLanguageModel(
            vocab_size=vocab_size,
            num_token_id=num_token_id,
            hidden_size=hidden_size,
            causal_dim=causal_dim,
            use_mock_feature_network=True
        )
        
    def forward(self, input_ids, numerical_values=None, attention_mask=None):
        """
        Forward pass of the CausalQwen model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens
                                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            dict: Dictionary containing all output distribution parameters and intermediate states
        """
        return self.model(input_ids, numerical_values, attention_mask)
    
    def predict(self, input_ids, numerical_values=None, attention_mask=None):
        """
        Make deterministic predictions.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens
                                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            dict: Dictionary containing predictions
        """
        return self.model.predict(input_ids, numerical_values, attention_mask)
    
    def sample_and_predict(self, input_ids, numerical_values=None, attention_mask=None):
        """
        Sample from the individual causal representation distribution and make predictions.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens
                                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            dict: Dictionary containing predictions and sampled states
        """
        return self.model.sample_and_predict(input_ids, numerical_values, attention_mask)

