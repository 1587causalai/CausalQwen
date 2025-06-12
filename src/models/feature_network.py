"""
Feature Network module.

This module implements the feature extraction network for the causal language model.
In the initial implementation, we use a mock feature network that generates random features.
Later, this can be replaced with a real Qwen-0.5B backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    
    def __init__(self, hidden_size=1024, vocab_size=151936, seed=42):
        """
        Initialize the mock feature network.
        
        Args:
            hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
            vocab_size (int, optional): Size of the vocabulary. Defaults to 151936 (Qwen's full space).
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__(hidden_size)
        self.vocab_size = vocab_size
        self.seed = seed
        torch.manual_seed(seed)
        
        # Create a simple embedding layer to ensure some meaningful structure in the features
        # Use Qwen's full vocabulary size to handle all token IDs
        self.embedding = nn.Embedding(vocab_size, hidden_size)
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
                         Shape: [batch_size, seq_len, hidden_size]
        """
        # For sequence-to-sequence, we need to return features for each position
        batch_size, seq_len = input_ids.shape
        
        # Generate embeddings for all tokens in the sequence
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Apply a linear transformation
        features = self.linear(embeddings)  # [batch_size, seq_len, hidden_size]
        
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
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import os
                
                # Expand the tilde in the path
                expanded_path = os.path.expanduser(model_path)
                
                print(f"Loading Qwen model from {expanded_path}")
                # Load the full causal language model (including lm_head)
                self.model = AutoModelForCausalLM.from_pretrained(expanded_path, trust_remote_code=True)
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
                self.mock_network = MockFeatureNetwork(hidden_size, vocab_size=151936)
        
        # Ensure mock_network is always available as fallback
        if not hasattr(self, 'mock_network') or self.mock_network is None:
            self.mock_network = MockFeatureNetwork(hidden_size, vocab_size=151936)
        
    def get_lm_head(self):
        """
        Returns the language model head of the underlying Qwen model.
        
        This is needed for weight initialization in the ActionNetwork.
        
        Returns:
            nn.Linear: The language model head module.
        """
        if not self.use_real_model:
            return None
            
        # For AutoModelForCausalLM, the lm_head should be directly accessible
        if hasattr(self.model, 'lm_head'):
            print(f"Found language model head: lm_head")
            return self.model.lm_head
        
        # Try different possible attribute names for the language model head
        possible_lm_head_names = ['output', 'classifier', 'language_model_head', 'head']
        
        for attr_name in possible_lm_head_names:
            if hasattr(self.model, attr_name):
                lm_head = getattr(self.model, attr_name)
                if isinstance(lm_head, nn.Linear):
                    print(f"Found language model head: {attr_name}")
                    return lm_head
        
        # If no direct lm_head found, look for it in nested modules
        for name, module in self.model.named_modules():
            if 'lm_head' in name.lower() and isinstance(module, nn.Linear):
                print(f"Found language model head in nested module: {name}")
                return module
        
        print("WARNING: Could not find language model head.")
        return None

    def forward(self, input_ids, attention_mask=None):
        """
        Extract features using the Qwen model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                                     Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask
                                                   Shape: [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Feature representation for each position
                         Shape: [batch_size, seq_len, hidden_size]
        """
        if not self.use_real_model:
            # For mock implementation, we now return sequence features directly
            return self.mock_network(input_ids, attention_mask)
        
        try:
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Extract hidden states from the outputs object
                last_hidden_state = None
                
                # For AutoModelForCausalLM, use hidden_states instead of last_hidden_state
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden_state = outputs.hidden_states[-1]  # Last layer hidden states
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    last_hidden_state = outputs.last_hidden_state
                else:
                    # If neither works, try accessing from the output object directly
                    # Sometimes the output object itself might have the hidden states
                    print(f"Debug: outputs type: {type(outputs)}")
                    print(f"Debug: outputs attributes: {dir(outputs)}")
                    print("Warning: Cannot find hidden states in model output, falling back to mock network")
                    return self.mock_network(input_ids, attention_mask)
                
                # Ensure we have valid hidden states
                if last_hidden_state is None:
                    print("Warning: Hidden states are None, falling back to mock network")
                    return self.mock_network(input_ids, attention_mask)
                
                # Ensure the hidden states are actually a tensor
                if not isinstance(last_hidden_state, torch.Tensor):
                    print(f"Warning: Hidden states are not a tensor (type: {type(last_hidden_state)}), falling back to mock network")
                    return self.mock_network(input_ids, attention_mask)
                
                # NO MORE POOLING! Return the full sequence
                # Shape: [batch_size, seq_len, model_hidden_size]
                
                # Apply projection if needed
                if self.projection is not None:
                    # Reshape for linear layer: [batch_size * seq_len, model_hidden_size]
                    batch_size, seq_len, _ = last_hidden_state.shape
                    features_flat = last_hidden_state.view(-1, self.model_hidden_size)
                    features_proj = self.projection(features_flat)
                    # Reshape back: [batch_size, seq_len, hidden_size]
                    features = features_proj.view(batch_size, seq_len, self.hidden_size)
                else:
                    features = last_hidden_state
                
                # Final safety check
                if not isinstance(features, torch.Tensor):
                    print(f"Warning: Final features are not a tensor (type: {type(features)}), falling back to mock network")
                    return self.mock_network(input_ids, attention_mask)
                
                return features
                
        except Exception as e:
            print(f"Error during Qwen model forward pass: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback to mock implementation - now returns correct sequence features
            print("Falling back to mock network due to exception")
            return self.mock_network(input_ids, attention_mask)


class NumAwareFeatureNetwork(FeatureNetworkBase):
    """支持数值感知的特征网络 - 统一的直接对数编码"""
    
    def __init__(self, base_network: FeatureNetworkBase, hidden_size: int, 
                 num_token_id: int, scale_factor: float = 1.0):
        super().__init__(hidden_size)
        self.base_network = base_network
        self.num_token_id = num_token_id
        self.scale_factor = scale_factor  # 可选的缩放因子
        
        # 方向向量：固定为均匀分布
        self.register_buffer('direction_vector', 
                           torch.ones(hidden_size) / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float)))
        
    def encode_numerical_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        将数值编码为向量：sign(v) * ln(1 + |v|)
        
        注意：当 v=0 时，编码结果为 0，这提供了完美的统一性
        
        Args:
            value: 数值张量 [batch_size, seq_len]
            
        Returns:
            编码后的标量 [batch_size, seq_len]
        """
        return torch.sign(value) * torch.log1p(torch.abs(value)) * self.scale_factor
    
    def _create_numerical_embeddings(self, numerical_values, num_mask, base_features):
        """
        创建数值感知的嵌入。
        
        Args:
            numerical_values: [batch_size, seq_len] 数值张量
            num_mask: [batch_size, seq_len] 布尔掩码，指示哪些位置是 <NUM>
            base_features: [batch_size, seq_len, hidden_size] 基础特征
            
        Returns:
            output_features: [batch_size, seq_len, hidden_size] 融合后的特征
        """
        batch_size, seq_len, hidden_size = base_features.shape
        
        # 创建数值嵌入向量 [batch_size, seq_len, hidden_size]
        num_embeddings = torch.zeros_like(base_features)
        
        if num_mask.any():
            # 获取有数值的位置
            num_positions = num_mask.nonzero(as_tuple=True)
            num_values = numerical_values[num_positions]  # [num_positions]
            
            # 计算数值嵌入: sign(v) * ln(1 + |v|)
            signs = torch.sign(num_values)
            magnitudes = torch.log1p(torch.abs(num_values))
            num_scalars = signs * magnitudes  # [num_positions]
            
            # 将标量扩展到 hidden_size 维度
            # 使用归一化的方向向量，确保数值信息分布均匀
            direction_vector = torch.ones(hidden_size, device=base_features.device) / math.sqrt(hidden_size)
            
            # 广播到对应位置: [num_positions, hidden_size]
            num_vectors = num_scalars.unsqueeze(-1) * direction_vector.unsqueeze(0)
            
            # 将数值嵌入放到对应位置
            num_embeddings[num_positions[0], num_positions[1]] = num_vectors
        
        # 加性融合
        output_features = base_features + num_embeddings
        
        return output_features

    def forward(self, input_ids: torch.Tensor, 
                numerical_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播 - 统一的加性融合
        
        对于所有位置 i：feature_i = h(x_i) + sign(v_i) * ln(1 + |v_i|) * e
        当 x_i != <NUM> 时，v_i = 0，因此 sign(v_i) * ln(1 + |v_i|) = 0
        
        这种统一性使得每个位置都有一个关联的数值（默认为0），
        为未来的扩展（如位置编码、时间戳等）提供了优雅的框架。
        
        Args:
            input_ids: token IDs [batch_size, seq_len]
            numerical_values: 数值 [batch_size, seq_len]，非<NUM>位置应为0
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            特征表示 [batch_size, seq_len, hidden_size]
        """
        # 获取基础特征（token embeddings）
        base_features = self.base_network(input_ids, attention_mask)
        
        # 创建数值掩码，指示哪些位置是 <NUM>
        num_mask = (input_ids == self.num_token_id)
        
        # 仅对 <NUM> 位置创建数值嵌入
        output_features = self._create_numerical_embeddings(numerical_values, num_mask, base_features)
        
        return output_features

