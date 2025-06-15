import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Tuple, Dict, List

# --- Preprocessing Utility (Not an nn.Module) ---

class NumericalAwareTokenizer:
    """
    非 nn.Module 的工具类
    职责: 负责在模型外部进行数据预处理。
    """
    def __init__(self, tokenizer_path: str, num_token: str = "<NUM>", placeholder_token: str = "_NUM_HOLDER_"):
        # This is a conceptual representation. A real implementation would need
        # to handle tokenizer loading and potential addition of special tokens.
        print(f"Conceptual Tokenizer: Loaded from {tokenizer_path}")
        self.num_token = num_token
        self.placeholder_token = placeholder_token

    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Mocker implementation for the preprocessing step.
        In a real scenario, this would involve regex, tokenization, and padding.
        """
        print("--- Running NumericalAwareTokenizer.preprocess (Mocker) ---")
        # Dummy output for demonstration purposes
        batch_size = len(texts)
        seq_length = 20 # fixed sequence length for this mock
        return {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'numeric_values': torch.randn(batch_size, seq_length),
            'attention_mask': torch.ones(batch_size, seq_length)
        }


# --- Granular and Composite Module Definitions (V2) ---

class NumericalEmbedding(nn.Module):
    """模块 1.1 (子模块): 数值编码"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.direction_vector = nn.Parameter(torch.randn(hidden_size))

    def forward(self, numeric_values: torch.Tensor) -> torch.Tensor:
        print("--- Running NumericalEmbedding (Mocker) ---")
        # phi(v) = sign(v) * ln(1 + |v|) * w_num
        # Placeholder logic: returns a tensor of the correct shape
        return torch.zeros(numeric_values.shape[0], numeric_values.shape[1], self.direction_vector.shape[0], device=numeric_values.device)

class NumericalAwareEmbedding(nn.Module):
    """模块 1 (主模块): 数值感知嵌入"""
    def __init__(self, token_embedding_layer: nn.Embedding, hidden_size: int):
        super().__init__()
        self.token_embedding = token_embedding_layer
        self.numerical_embedding = NumericalEmbedding(hidden_size)

    def forward(self, input_ids: torch.Tensor, numeric_values: torch.Tensor) -> torch.Tensor:
        print("--- Running NumericalAwareEmbedding (Mocker) ---")
        # Placeholder logic: combines outputs from sub-modules
        base_embed = self.token_embedding(input_ids)
        num_embed = self.numerical_embedding(numeric_values)
        return base_embed + num_embed # In mock, this just adds zeros to base_embed

class FeatureExtractionNetwork(nn.Module):
    """模块二: 特征提取网络"""
    def __init__(self, qwen_transformer_model: nn.Module):
        super().__init__()
        self.qwen_transformer = qwen_transformer_model

    def forward(self, enhanced_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        print("--- Running FeatureExtractionNetwork (Mocker) ---")
        # Placeholder logic: returns a zero tensor of the correct shape
        return torch.zeros_like(enhanced_embeddings)

class AbductionNetwork(nn.Module):
    """模块三: 归因推断网络"""
    def __init__(self, hidden_size: int, causal_representation_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.causal_representation_dim = causal_representation_dim

    def forward(self, context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        print("--- Running AbductionNetwork (Mocker) ---")
        batch_size, seq_len, _ = context_features.shape
        loc_U = torch.zeros(batch_size, seq_len, self.causal_representation_dim, device=context_features.device)
        scale_U = torch.ones(batch_size, seq_len, self.causal_representation_dim, device=context_features.device)
        return loc_U, scale_U

class ActionNetwork(nn.Module):
    """模块四: 行动决策网络"""
    def __init__(self, causal_representation_dim: int, vocab_size: int):
        super().__init__()
        self.causal_representation_dim = causal_representation_dim
        self.vocab_size = vocab_size

    def forward(self, loc_U: torch.Tensor, scale_U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        print("--- Running ActionNetwork (Mocker) ---")
        batch_size, seq_len, _ = loc_U.shape
        loc_S = torch.zeros(batch_size, seq_len, self.vocab_size, device=loc_U.device)
        scale_S = torch.ones(batch_size, seq_len, self.vocab_size, device=loc_U.device)
        loc_Y = torch.zeros(batch_size, seq_len, device=loc_U.device)
        scale_Y = torch.ones(batch_size, seq_len, device=loc_U.device)
        return loc_S, scale_S, loc_Y, scale_Y

# --- Granular Loss Modules ---

class OvrClassificationLoss(nn.Module):
    """模块 5.1 (子模块): OvR 分类损失"""
    def __init__(self, ovr_threshold: float, ignore_index: int = -100):
        super().__init__()
        self.ovr_threshold = ovr_threshold
        self.ignore_index = ignore_index

    def forward(self, loc_S: torch.Tensor, scale_S: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        print("--- Running OvrClassificationLoss (Mocker) ---")
        # Placeholder logic: returns a zero tensor of the correct shape
        return torch.zeros(labels.shape[0], labels.shape[1], device=loc_S.device)

class RegressionLoss(nn.Module):
    """模块 5.2 (子模块): 回归损失 (Cauchy NLL)"""
    def __init__(self):
        super().__init__()

    def forward(self, loc_Y: torch.Tensor, scale_Y: torch.Tensor, true_numeric_values: torch.Tensor) -> torch.Tensor:
        print("--- Running RegressionLoss (Mocker) ---")
        # Placeholder logic: returns a zero tensor of the correct shape
        return torch.zeros_like(loc_Y)

# --- Composite Loss Module ---

class CausalLoss(nn.Module):
    """模块 5 (主模块): 组合损失"""
    def __init__(self, num_token_id: int, reg_loss_weight: float, alpha_gated_reg: float, ovr_threshold: float, ignore_index: int = -100):
        super().__init__()
        self.num_token_id = num_token_id
        self.reg_loss_weight = reg_loss_weight
        self.alpha_gated_reg = alpha_gated_reg
        self.ovr_cls_loss = OvrClassificationLoss(ovr_threshold, ignore_index)
        self.reg_loss = RegressionLoss()

    def forward(self, loc_S, scale_S, loc_Y, scale_Y, labels, numeric_values, attention_mask) -> Tuple[torch.Tensor, Dict[str, float]]:
        print("--- Running CausalLoss (Mocker) ---")
        # Placeholder logic for the entire composite loss
        cls_loss_unreduced = self.ovr_cls_loss(loc_S, scale_S, labels)
        reg_nll_unreduced = self.reg_loss(loc_Y, scale_Y, numeric_values)
        
        # Mock total loss calculation
        total_loss = torch.tensor(0.0, device=loc_S.device, requires_grad=True)
        loss_dict = {
            'total_loss': 0.0,
            'cls_loss_mean': 0.0,
            'reg_loss_effective': 0.0
        }
        return total_loss, loss_dict

# --- Main Model ---

class CausalQwenModel(nn.Module):
    """CausalQwen 主模型"""
    def __init__(self, qwen_model_path: str):
        super().__init__()
        qwen_config = AutoConfig.from_pretrained(qwen_model_path, trust_remote_code=True)
        qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_path, trust_remote_code=True)
        
        self.hidden_size = qwen_config.hidden_size
        self.vocab_size = qwen_config.vocab_size
        self.causal_representation_dim = self.hidden_size

        self.numerical_embedding = NumericalAwareEmbedding(
            token_embedding_layer=qwen_model.model.embed_tokens,
            hidden_size=self.hidden_size
        )
        self.feature_extraction = FeatureExtractionNetwork(
            qwen_transformer_model=qwen_model.model
        )
        self.abduction_network = AbductionNetwork(
            hidden_size=self.hidden_size,
            causal_representation_dim=self.causal_representation_dim
        )
        self.action_network = ActionNetwork(
            causal_representation_dim=self.causal_representation_dim,
            vocab_size=self.vocab_size
        )

    def forward(self, input_ids, numeric_values, attention_mask=None):
        enhanced_embeddings = self.numerical_embedding(input_ids, numeric_values)
        context_features = self.feature_extraction(enhanced_embeddings, attention_mask)
        loc_U, scale_U = self.abduction_network(context_features)
        loc_S, scale_S, loc_Y, scale_Y = self.action_network(loc_U, scale_U)
        return loc_S, scale_S, loc_Y, scale_Y

# --- Example Usage and Test ---
if __name__ == '__main__':
    print("Starting CausalQwen V2 Mocker Test...")

    # --- Configuration ---
    QWEN_MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B")
    NUM_TOKEN_ID = 151643

    # --- Preprocessing Step ---
    print("\n1. Preprocessing raw text...")
    tokenizer_util = NumericalAwareTokenizer(QWEN_MODEL_PATH)
    dummy_texts = ["The price is 99.9 dollars.", "Another text with value -10.5."]
    processed_inputs = tokenizer_util.preprocess(dummy_texts)
    input_ids = processed_inputs['input_ids']
    numeric_values = processed_inputs['numeric_values']
    attention_mask = processed_inputs['attention_mask']
    
    # Dummy labels for loss calculation
    labels = torch.randint(0, 1000, input_ids.shape)
    true_numeric_values = torch.randn_like(numeric_values)
    labels[0, 3] = NUM_TOKEN_ID # Ensure some <NUM> tokens are present
    labels[1, 5] = NUM_TOKEN_ID

    print("   Preprocessing finished.")

    # --- Model and Loss Instantiation ---
    print("\n2. Instantiating Model and Loss Function...")
    causal_qwen = CausalQwenModel(QWEN_MODEL_PATH)
    loss_fn = CausalLoss(
        num_token_id=NUM_TOKEN_ID,
        ovr_threshold=100.0,
        reg_loss_weight=1.0,
        alpha_gated_reg=0.0
    )
    print("   Model and Loss function instantiated successfully.")

    # --- Forward Pass ---
    print("\n3. Performing forward pass through the model...")
    loc_S, scale_S, loc_Y, scale_Y = causal_qwen(input_ids, numeric_values, attention_mask)
    print("   Forward pass completed.")
    print(f"   Output shapes: \n"
          f"     loc_S: {loc_S.shape}\n"
          f"     scale_S: {scale_S.shape}\n"
          f"     loc_Y: {loc_Y.shape}\n"
          f"     scale_Y: {scale_Y.shape}")

    # --- Loss Calculation ---
    print("\n4. Calculating loss...")
    total_loss, loss_dict = loss_fn(loc_S, scale_S, loc_Y, scale_Y, labels, true_numeric_values, attention_mask)
    print("   Loss calculation completed.")
    print(f"   Total Loss: {total_loss.item()}")
    print(f"   Loss Dictionary: {loss_dict}")

    print("\nCausalQwen V2 Mocker Test finished successfully!")


