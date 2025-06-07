"""
Feature Network module.

This module implements the feature extraction network for the causal language model.
In the initial implementation, we use a mock feature network that generates random features.
Later, this can be replaced with a real Qwen-0.5B backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNetworkBase(nn.Module):
    """
    Base class for feature extraction networks.
    """
    
    def __init__(self, hidden_size):
        """
        Initialize the base feature network.
        
        Args:
            hidden_size (int): Size of the hidden representation
        """
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, input_ids, attention_mask=None):
        """
        Extract features from input tokens.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Feature representation
                         Shape: [batch_size, hidden_size]
        """
        raise NotImplementedError("Subclasses must implement forward method")


class MockFeatureNetwork(FeatureNetworkBase):
    """
    Mock feature network that generates random features.
    
    This is used for testing the causal inference components without
    requiring a full language model backbone.
    """
    
    def __init__(self, hidden_size=1024, seed=42):
        """
        Initialize the mock feature network.
        
        Args:
            hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__(hidden_size)
        self.seed = seed
        torch.manual_seed(seed)
        
        # Create a simple embedding layer to ensure some meaningful structure in the features
        self.embedding = nn.Embedding(10000, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Generate mock features from input tokens.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Mock feature representation
                         Shape: [batch_size, hidden_size]
        """
        # Get the last token ID from each sequence
        last_token_ids = input_ids[:, -1]
        
        # Generate embeddings for the last token
        embeddings = self.embedding(last_token_ids)
        
        # Apply a linear transformation
        features = self.linear(embeddings)
        
        # Add some noise for variability
        noise = torch.randn_like(features) * 0.1
        features = features + noise
        
        return features


class QwenFeatureNetwork(FeatureNetworkBase):
    """
    Feature network based on Qwen-0.5B.
    
    This implementation uses the actual Qwen model as a feature extractor.
    """
    
    def __init__(self, model_path="~/models/Qwen2.5-0.5B", hidden_size=1024, use_real_model=True):
        """
        Initialize the Qwen feature network.
        
        Args:
            model_path (str): Path to the Qwen model directory
            hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
            use_real_model (bool): Whether to use the real Qwen model or mock implementation
        """
        super().__init__(hidden_size)
        self.model_path = model_path
        self.use_real_model = use_real_model
        
        if use_real_model:
            try:
                from transformers import AutoModel, AutoTokenizer
                import os
                
                # Expand the tilde in the path
                expanded_path = os.path.expanduser(model_path)
                
                print(f"Loading Qwen model from {expanded_path}")
                self.model = AutoModel.from_pretrained(expanded_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(expanded_path, trust_remote_code=True)
                
                # Get the actual hidden size from the model
                self.model_hidden_size = self.model.config.hidden_size
                
                # Add projection layer if hidden sizes don't match
                if self.model_hidden_size != hidden_size:
                    self.projection = nn.Linear(self.model_hidden_size, hidden_size)
                else:
                    self.projection = None
                    
                print(f"Successfully loaded Qwen model with hidden size {self.model_hidden_size}")
                
                # Set model to evaluation mode
                self.model.eval()
                
            except Exception as e:
                print(f"Failed to load Qwen model: {e}")
                print("Falling back to mock implementation")
                self.use_real_model = False
                self.mock_network = MockFeatureNetwork(hidden_size)
        else:
            # Use mock implementation
            self.mock_network = MockFeatureNetwork(hidden_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Extract features using the Qwen model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Feature representation
                         Shape: [batch_size, hidden_size]
        """
        if not self.use_real_model:
            # Use mock implementation
            return self.mock_network(input_ids, attention_mask)
        
        try:
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get the last hidden state [batch_size, seq_len, hidden_size]
                last_hidden_state = outputs.last_hidden_state
                
                # Pool over the sequence dimension (use mean pooling)
                if attention_mask is not None:
                    # Mask out padding tokens before pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    features = sum_embeddings / sum_mask
                else:
                    # Simple mean pooling
                    features = last_hidden_state.mean(dim=1)
                
                # Apply projection if needed
                if self.projection is not None:
                    features = self.projection(features)
                
                return features
                
        except Exception as e:
            print(f"Error during Qwen model forward pass: {e}")
            # Fallback to mock implementation
            return self.mock_network(input_ids, attention_mask)


class NumAwareFeatureNetwork(nn.Module):
    """
    Feature network that is aware of numerical values.
    
    This network processes the <NUM> token specially by modulating its
    embedding with the actual numerical value.
    """
    
    def __init__(self, base_network, num_token_id, hidden_size=1024):
        """
        Initialize the numerical-aware feature network.
        
        Args:
            base_network (FeatureNetworkBase): Base feature extraction network
            num_token_id (int): Token ID for the <NUM> token
            hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
        """
        super().__init__()
        self.base_network = base_network
        self.num_token_id = num_token_id
        self.hidden_size = hidden_size
        
        # Projection for numerical values
        self.num_projection = nn.Linear(1, hidden_size)
        
    def forward(self, input_ids, numerical_values=None, attention_mask=None):
        """
        Extract features with special handling for numerical values.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens
                                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Feature representation
                         Shape: [batch_size, hidden_size]
        """
        # Get base features
        features = self.base_network(input_ids, attention_mask)
        
        # If no numerical values provided, return base features
        if numerical_values is None:
            return features
        
        # Create mask for <NUM> tokens
        num_mask = (input_ids == self.num_token_id)
        
        # If no <NUM> tokens in the input, return base features
        if not num_mask.any():
            return features
        
        # Get the numerical values for <NUM> tokens
        # We assume numerical_values has the same shape as input_ids
        # and contains the actual values for <NUM> tokens
        batch_size, seq_len = input_ids.shape
        
        # Process each sequence in the batch
        for i in range(batch_size):
            # Find positions of <NUM> tokens in this sequence
            num_positions = num_mask[i].nonzero(as_tuple=True)[0]
            
            # If no <NUM> tokens in this sequence, continue
            if len(num_positions) == 0:
                continue
            
            # Process each <NUM> token
            for pos in num_positions:
                # Get the numerical value
                value = numerical_values[i, pos].unsqueeze(0).unsqueeze(0)  # [1, 1]
                
                # Project the numerical value to the hidden space
                value_embedding = self.num_projection(value)  # [1, hidden_size]
                
                # Modulate the feature with the numerical value
                # We use a simple multiplication here, but more complex fusion
                # mechanisms could be used
                if pos == seq_len - 1:  # If <NUM> is the last token
                    features[i] = features[i] * torch.sigmoid(value_embedding.squeeze(0))
        
        return features

