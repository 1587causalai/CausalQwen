import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class CausalQwen(nn.Module):
    def __init__(self, qwen_model_path, ovr_threshold=100.0, gamma_init=10.0, alpha_gated_reg=0.0):
        super().__init__()
        self.ovr_threshold = ovr_threshold
        self.gamma_init = gamma_init
        self.alpha_gated_reg = alpha_gated_reg

        # Load Qwen model and tokenizer
        self.qwen_config = AutoConfig.from_pretrained(qwen_model_path)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

        self.vocab_size = self.qwen_config.vocab_size
        self.hidden_size = self.qwen_config.hidden_size

        # 1. 数值感知嵌入 (Numerical-aware Embedding)
        # Use Qwen's original embedding layer
        self.token_embedding = self.qwen_model.model.embed_tokens

        # Direction vector for numerical encoding, initialized as per doc
        self.numerical_direction_vector = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.numerical_direction_vector.data = F.normalize(self.numerical_direction_vector.data, p=2, dim=-1)

        # 2. 特征提取网络 (Feature Extraction Network) - Use Qwen's transformer layers
        # We will use the full qwen_model.model in the forward pass
        # self.qwen_model_norm = self.qwen_model.model.norm # This might not be needed if we use qwen_model.model directly

        # 3. 归因推断网络 (Abduction Network)
        # C = H, so input/output dimensions are hidden_size
        self.abduction_loc_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.abduction_scale_linear = nn.Linear(self.hidden_size, self.hidden_size)

        # 4. 行动决策网络 (Action Network)
        # Classification Action Network - Use Qwen's lm_head for initialization
        self.action_cls_loc_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.action_cls_scale_linear = nn.Linear(self.hidden_size, self.vocab_size)

        # Regression Action Network
        self.action_reg_loc_linear = nn.Linear(self.hidden_size, 1) # Output 1 for scalar regression
        self.action_reg_scale_linear = nn.Linear(self.hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Step 1: Numerical-aware Embedding - numerical_direction_vector initialized in __init__
        # token_embedding is Qwen's original, no special init needed.

        # Step 2: Feature Extraction Network - Qwen's transformer layers are already initialized.

        # Step 3: Abduction Network -> Identity mapping
        # loc_U = z_i => W_loc = I, b_loc = 0
        nn.init.eye_(self.abduction_loc_linear.weight)
        nn.init.zeros_(self.abduction_loc_linear.bias)

        # scale_U = gamma_i (large value) => W_scale = 0, b_scale = log(gamma)
        nn.init.zeros_(self.abduction_scale_linear.weight)
        nn.init.constant_(self.abduction_scale_linear.bias, math.log(self.gamma_init))

        # Step 4: Action Network (Classification) -> Copy Qwen weights
        # Copy Qwen's lm_head weights
        self.action_cls_loc_linear.weight.data.copy_(self.qwen_model.lm_head.weight.data)
        if self.qwen_model.lm_head.bias is not None:
            self.action_cls_loc_linear.bias.data.copy_(self.qwen_model.lm_head.bias.data)
        else:
            nn.init.zeros_(self.action_cls_loc_linear.bias)

        # action_cls_scale_linear should be initialized such that scale_S is large initially
        nn.init.zeros_(self.action_cls_scale_linear.weight)
        nn.init.constant_(self.action_cls_scale_linear.bias, math.log(self.gamma_init)) # Large scale initially

        # Step 4: Action Network (Regression) -> Conventional initialization
        nn.init.xavier_uniform_(self.action_reg_loc_linear.weight)
        nn.init.zeros_(self.action_reg_loc_linear.bias)

        nn.init.xavier_uniform_(self.action_reg_scale_linear.weight)
        nn.init.zeros_(self.action_reg_scale_linear.bias)

    def numerical_encoding_function(self, numeric_values):
        # phi(v) = sign(v) * ln(1 + |v|) * e_vec
        sign_v = torch.sign(numeric_values)
        log_term = torch.log(1 + torch.abs(numeric_values))
        # Expand numerical_direction_vector to match batch and sequence dimensions
        e_vec_expanded = self.numerical_direction_vector.unsqueeze(0).unsqueeze(0).expand(numeric_values.shape[0], numeric_values.shape[1], -1)
        return sign_v.unsqueeze(-1) * log_term.unsqueeze(-1) * e_vec_expanded

    def forward(self, input_ids, numeric_values, attention_mask=None):
        # input_ids: [B, S]
        # numeric_values: [B, S]

        # 1. 数值感知嵌入 (Numerical-aware Embedding)
        base_embeddings = self.token_embedding(input_ids) # [B, S, H]
        phi_v = self.numerical_encoding_function(numeric_values) # [B, S, H]
        enhanced_embeddings = base_embeddings + phi_v # [B, S, H]

        # 2. 特征提取网络 (Feature Extraction Network)
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Qwen2 attention_mask expects a boolean mask or a float mask with -inf for masked positions
        # If attention_mask is provided as 0s and 1s, convert it to boolean
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        # Pass enhanced_embeddings directly to the Qwen model's core
        # The Qwen2Model's forward method will handle its internal layers, attention, and positional embeddings
        qwen_output = self.qwen_model.model(
            inputs_embeds=enhanced_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True # We need the last hidden state
        )
        
        # The last hidden state is typically in qwen_output.last_hidden_state
        context_features = qwen_output.last_hidden_state # [B, S, H]

        # 3. 归因推断网络 (Abduction Network)
        loc_U = self.abduction_loc_linear(context_features) # [B, S, C]
        # Ensure scale_U is positive using softplus or exp
        scale_U = F.softplus(self.abduction_scale_linear(context_features)) # [B, S, C]

        # 4. 行动决策网络 (Action Network)
        # Classification
        loc_S = self.action_cls_loc_linear(loc_U) # [B, S, V_full]
        scale_S = F.softplus(self.action_cls_scale_linear(loc_U)) # [B, S, V_full]

        # Regression
        loc_Y = self.action_reg_loc_linear(loc_U).squeeze(-1) # [B, S] (squeeze last dim if it's 1)
        scale_Y = F.softplus(self.action_reg_scale_linear(loc_U)).squeeze(-1) # [B, S]

        return loc_U, scale_U, loc_S, scale_S, loc_Y, scale_Y

    def calculate_loss(self, loc_S, scale_S, loc_Y, scale_Y, true_input_ids, true_numeric_values, num_token_id, lambda_reg=1.0):
        # OvR Classification Loss
        # P_k,i = 0.5 + (1/pi) * arctan((loc_S_k,i - C_k) / scale_S_k,i)
        # For simplicity, C_k is a constant ovr_threshold
        ovr_threshold_tensor = torch.full_like(loc_S, self.ovr_threshold)
        
        # Clamp scale_S to avoid division by zero or very small values
        scale_S_clamped = torch.clamp(scale_S, min=1e-6)
        
        prob_S = 0.5 + (1/math.pi) * torch.atan((loc_S - ovr_threshold_tensor) / scale_S_clamped)
        
        # Create one-hot encoded true_input_ids for OvR
        true_labels_one_hot = F.one_hot(true_input_ids, num_classes=self.vocab_size).float()
        
        # Binary Cross Entropy Loss
        # Clamp probabilities to avoid log(0)
        prob_S_clamped = torch.clamp(prob_S, min=1e-7, max=1-1e-7)
        cls_loss = - (true_labels_one_hot * torch.log(prob_S_clamped) + (1 - true_labels_one_hot) * torch.log(1 - prob_S_clamped)).sum(dim=-1).mean()

        # Gated Regression Loss
        # m_i is mask where true_input_ids == num_token_id
        is_num_mask = (true_input_ids == num_token_id).float()

        # P_<NUM>,i is the probability of the <NUM> token from classification output
        # Assuming num_token_id is a valid index in vocab_size
        num_prob_from_cls = prob_S[:, :, num_token_id]

        # Cauchy NLL Loss: log(pi * scale_Y) + log(1 + ((y_true - loc_Y) / scale_Y)^2)
        # Clamp scale_Y to avoid division by zero or very small values
        scale_Y_clamped = torch.clamp(scale_Y, min=1e-6)
        
        cauchy_nll = torch.log(math.pi * scale_Y_clamped) + torch.log(1 + ((true_numeric_values - loc_Y) / scale_Y_clamped)**2)

        # L_reg_gated,i = m_i * (alpha + (1-alpha) * P_<NUM>,i) * L_cauchy_nll,i
        reg_gated_loss = is_num_mask * (self.alpha_gated_reg + (1 - self.alpha_gated_reg) * num_prob_from_cls) * cauchy_nll
        reg_gated_loss = reg_gated_loss.mean()

        total_loss = cls_loss + lambda_reg * reg_gated_loss
        return total_loss, cls_loss, reg_gated_loss

    def deterministic_inference(self, loc_S, scale_S, loc_Y, scale_Y, num_token_id):
        # Classification prediction: argmax_k P(S_k,i > C_k)
        ovr_threshold_tensor = torch.full_like(loc_S, self.ovr_threshold)
        scale_S_clamped = torch.clamp(scale_S, min=1e-6)
        prob_S = 0.5 + (1/math.pi) * torch.atan((loc_S - ovr_threshold_tensor) / scale_S_clamped)
        predicted_token_ids = torch.argmax(prob_S, dim=-1) # [B, S]

        # Regression prediction: loc_Y
        predicted_numeric_values = loc_Y # [B, S]

        # Apply regression prediction only where <NUM> token is predicted
        # This is a simplified logic for demonstration. In a real scenario, you'd check if the predicted_token_id is <NUM_ID>
        # and then use the predicted_numeric_value. For this minimalist script, we'll just return both.
        
        return predicted_token_ids, predicted_numeric_values

    def causal_sampling(self, loc_U, scale_U, num_samples=1, epsilon_seed=None):
        # Sample 'cause' (U) based on Cauchy(loc_U, scale_U)
        # Reparameterization Trick: u_i = loc_U_i + scale_U_i * tan(pi * (epsilon - 0.5))
        
        if epsilon_seed is None:
            # If no seed, generate a new epsilon for each sample
            epsilon = torch.rand(loc_U.shape[0], loc_U.shape[1], loc_U.shape[2], num_samples, device=loc_U.device)
        else:
            # Use provided epsilon_seed, expand if necessary
            epsilon = epsilon_seed.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(loc_U.shape[0], loc_U.shape[1], loc_U.shape[2], num_samples)

        # Ensure epsilon is within (0, 1) for tan function
        epsilon = torch.clamp(epsilon, min=1e-7, max=1-1e-7)

        # Expand loc_U and scale_U for sampling
        loc_U_expanded = loc_U.unsqueeze(-1).expand(-1, -1, -1, num_samples)
        scale_U_expanded = scale_U.unsqueeze(-1).expand(-1, -1, -1, num_samples)

        sampled_U = loc_U_expanded + scale_U_expanded * torch.tan(math.pi * (epsilon - 0.5))
        # sampled_U shape: [B, S, C, num_samples]

        # Pass sampled_U through Action Network to get deterministic predictions
        # This requires reshaping sampled_U to [B*S*num_samples, C] for linear layers
        B, S, C, N = sampled_U.shape
        sampled_U_reshaped = sampled_U.view(B * S * N, C)

        # Classification
        sampled_loc_S = self.action_cls_loc_linear(sampled_U_reshaped) # [B*S*N, V_full]
        sampled_scale_S = F.softplus(self.action_cls_scale_linear(sampled_U_reshaped)) # [B*S*N, V_full]

        # Regression
        sampled_loc_Y = self.action_reg_loc_linear(sampled_U_reshaped).squeeze(-1) # [B*S*N]
        sampled_scale_Y = F.softplus(self.action_reg_scale_linear(sampled_U_reshaped)).squeeze(-1) # [B*S*N]

        # Reshape back to [B, S, num_samples, ...] for predictions
        sampled_loc_S = sampled_loc_S.view(B, S, N, self.vocab_size)
        sampled_scale_S = sampled_scale_S.view(B, S, N, self.vocab_size)
        sampled_loc_Y = sampled_loc_Y.view(B, S, N)
        sampled_scale_Y = sampled_scale_Y.view(B, S, N)

        # Deterministic prediction from sampled distributions
        ovr_threshold_tensor = torch.full_like(sampled_loc_S, self.ovr_threshold)
        sampled_scale_S_clamped = torch.clamp(sampled_scale_S, min=1e-6)
        sampled_prob_S = 0.5 + (1/math.pi) * torch.atan((sampled_loc_S - ovr_threshold_tensor) / sampled_scale_S_clamped)
        sampled_predicted_token_ids = torch.argmax(sampled_prob_S, dim=-1) # [B, S, num_samples]

        sampled_predicted_numeric_values = sampled_loc_Y # [B, S, num_samples]

        return sampled_predicted_token_ids, sampled_predicted_numeric_values, sampled_U

    def compatible_traditional_sampling(self, loc_S):
        # Use loc_S as logits for traditional softmax-based sampling
        # loc_S: [B, S, V_full]
        softmax_probs = F.softmax(loc_S, dim=-1)
        return softmax_probs


# Example Usage and Test
if __name__ == '__main__':
    # Hyperparameters
    QWEN_MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B")  # Path to the downloaded Qwen model

    # Qwen2.5-0.5B tokenizer adds <|endoftext|> as pad_token, and its ID is 151643
    # The document says <NUM> is the first reserved token, which is usually after vocab_size
    # For Qwen2.5-0.5B, the vocab_size is 151936. The reserved tokens start from 151643.
    # Let's assume <NUM> is 151643 for now, as it's a common practice for special tokens.
    # You might need to verify this with Qwen's tokenizer.add_special_tokens or similar.
    NUM_TOKEN_ID = 151643 # Placeholder, verify with Qwen tokenizer
    BATCH_SIZE = 2
    SEQUENCE_LENGTH = 10

    # Instantiate the model
    model = CausalQwen(QWEN_MODEL_PATH, ovr_threshold=100.0, gamma_init=10.0, alpha_gated_reg=0.0)
    print("Model initialized successfully.")
    print(f"Vocab Size: {model.vocab_size}, Hidden Size: {model.hidden_size}")

    # Create dummy input data
    # input_ids: Example sequence with some <NUM> tokens
    input_ids = torch.randint(0, model.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH))
    # Replace some random tokens with <NUM> token_id
    num_indices = torch.randint(0, SEQUENCE_LENGTH, (BATCH_SIZE, BATCH_SIZE)) # Randomly select some positions to be <NUM>
    for i in range(BATCH_SIZE):
        input_ids[i, num_indices[i]] = NUM_TOKEN_ID

    # numeric_values: Corresponding numerical values. 0.0 for non-<NUM> tokens.
    numeric_values = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.float)
    for i in range(BATCH_SIZE):
        for j in range(SEQUENCE_LENGTH):
            if input_ids[i, j] == NUM_TOKEN_ID:
                numeric_values[i, j] = torch.randn(1).item() * 100 # Assign a random numerical value

    # Dummy true labels for loss calculation
    true_input_ids = torch.randint(0, model.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH))
    true_numeric_values = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH) * 100
    # Ensure true_input_ids has some NUM_TOKEN_ID for gated regression loss to be active
    for i in range(BATCH_SIZE):
        true_input_ids[i, num_indices[i]] = NUM_TOKEN_ID

    print("Dummy input data created.")

    # Forward pass
    loc_U, scale_U, loc_S, scale_S, loc_Y, scale_Y = model(input_ids, numeric_values)
    print(f"Forward pass output shapes:")
    print(f"  loc_U: {loc_U.shape}, scale_U: {scale_U.shape}")
    print(f"  loc_S: {loc_S.shape}, scale_S: {loc_S.shape}")
    print(f"  loc_Y: {loc_Y.shape}, scale_Y: {loc_Y.shape}")

    # Loss calculation
    total_loss, cls_loss, reg_gated_loss = model.calculate_loss(loc_S, scale_S, loc_Y, scale_Y, true_input_ids, true_numeric_values, NUM_TOKEN_ID)
    print(f"Losses:")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Classification Loss: {cls_loss.item():.4f}")
    print(f"  Gated Regression Loss: {reg_gated_loss.item():.4f}")

    # Deterministic Inference
    predicted_token_ids, predicted_numeric_values = model.deterministic_inference(loc_S, scale_S, loc_Y, scale_Y, NUM_TOKEN_ID)
    print(f"Deterministic Inference Output Shapes:")
    print(f"  Predicted Token IDs: {predicted_token_ids.shape}")
    print(f"  Predicted Numeric Values: {predicted_numeric_values.shape}")

    # Causal Sampling
    # Example: Sample 3 times
    sampled_token_ids, sampled_numeric_values, sampled_U = model.causal_sampling(loc_U, scale_U, num_samples=3)
    print(f"Causal Sampling Output Shapes (3 samples):\n  Sampled Token IDs: {sampled_token_ids.shape}\n  Sampled Numeric Values: {sampled_numeric_values.shape}\n  Sampled U: {sampled_U.shape}")

    # Compatible Traditional Sampling
    softmax_probs = model.compatible_traditional_sampling(loc_S)
    print(f"Compatible Traditional Sampling (Softmax Probs) Shape: {softmax_probs.shape}")

    print("All core functionalities tested successfully.")


