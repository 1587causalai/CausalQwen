"""
Causal Language Model.

This module implements the complete causal language model by integrating
all components: feature network, abduction network, and action network.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_network import QwenFeatureNetwork, MockFeatureNetwork
from .numerical_aware_embedding import NumericalAwareEmbedding
from .abduction_network import AbductionNetwork
from .action_network import ActionNetwork


@dataclass
class CausalLMConfig:
    """Configuration for the Causal Language Model."""
    # Model architecture
    vocab_size: int = 151936  # 使用完整的Qwen配置容量
    num_token_id: int = 151665  # Token ID for <NUM>
    hidden_size: int = 896  # Hidden size (896 for Qwen-0.5B)
    causal_dim: int = 896  # Causal representation dimension
    
    # Feature network settings
    use_real_qwen: bool = True
    use_mock_feature_network: bool = False  # 补全缺失的属性
    qwen_model_path: str = "~/models/Qwen2.5-0.5B"
    use_numerical_features: bool = True  # 控制是否启用数值嵌入
    
    # OvR classification settings
    use_ovr_classifier: bool = True
    ovr_threshold: float = 0.0  # 使用 0.0 作为默认阈值
    
    # Regression loss settings
    reg_loss_weight: float = 1.0
    reg_loss_gating_alpha: float = 1.0  # 门控系数：1.0 = 无门控，0.0 = 完全门控
    
    # Distribution settings
    use_cauchy_distribution: bool = True
    initial_scale_bias: float = 10.0  # softplus(10.0) ≈ 10.0

class CausalLanguageModel(nn.Module):
    """
    因果语言模型 (Causal Language Model)
    
    通过推断-行动范式，将预训练的语言模型扩展为因果推理模型。
    """
    
    def __init__(self, config: CausalLMConfig):
        super().__init__()
        self.config = config
        
        # 保存常用配置为实例属性，方便访问
        self.vocab_size = config.vocab_size
        self.num_token_id = config.num_token_id
        self.hidden_size = config.hidden_size
        self.causal_dim = config.causal_dim
        
        # 特征网络选择逻辑（修复导入问题）
        use_mock_feature_network = getattr(config, 'use_mock_feature_network', False)
        
        if config.use_real_qwen and not use_mock_feature_network:
            print("Initializing with real Qwen model...")
            self.feature_network = QwenFeatureNetwork(
                model_path=config.qwen_model_path,
                hidden_size=config.hidden_size
            )
            
            # 检查是否需要数值感知功能
            if hasattr(config, 'use_numerical_features') and config.use_numerical_features:
                self.numerical_aware_embedding = NumericalAwareEmbedding(
                    base_embedding_layer=self.feature_network.qwen_model.model.embed_tokens,
                    num_token_id=config.num_token_id,
                    hidden_size=config.hidden_size
                )
            else:
                self.numerical_aware_embedding = None
        else:
            print("Initializing with mock feature network...")
            # 需要导入MockFeatureNetwork
            self.feature_network = MockFeatureNetwork(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size
            )
            self.numerical_aware_embedding = None # Mock network handles this internally for now
        
        # Abduction network - 推断个体因果表征分布
        self.abduction_network = AbductionNetwork(self.hidden_size, self.causal_dim)
        
        # Action network - 基于因果表征生成输出
        self.action_network = ActionNetwork(
            input_dim=self.causal_dim,
            hidden_size=self.hidden_size,
            num_token_id=self.num_token_id,
            vocab_size=self.vocab_size
        )

    def init_weights(self):
        """
        Initialize the weights of abduction and action networks using the
        updated knowledge transfer strategy.
        
        知识传输策略：
        - 分类头：完全复用 Qwen 的 lm_head
        - 回归头：使用小的随机值初始化
        """
        print("应用知识传输初始化...")
        
        # 1. Initialize Abduction Network
        # This assumes hidden_size and causal_dim are the same
        if self.hidden_size == self.causal_dim:
            self.abduction_network.initialize_for_identity_mapping(
                scale_bias=self.config.initial_scale_bias
            )
        else:
            print("  - WARNING: Abduction network not initialized for identity mapping (hidden_size != causal_dim).")
            # Here you might want a different initialization for the non-identity case
            # For now, we do nothing and rely on the default nn.Linear init.

        # 2. Initialize Action Network by transferring knowledge from Qwen's lm_head
        qwen_lm_head = None
        try:
            if hasattr(self.feature_network, 'get_lm_head'):
                qwen_lm_head = self.feature_network.get_lm_head()
                if qwen_lm_head is not None:
                    print("Found language model head: lm_head")
                else:
                    print("Feature network returned None for lm_head (likely MockFeatureNetwork)")
        except Exception as e:
            print(f"Error getting lm_head: {e}")

        # 将 lm_head 传递给 Action Network 以完成其权重的初始化
        self.action_network.init_weights(qwen_lm_head=qwen_lm_head)
            
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
        # 如果没有提供numerical_values，创建全零向量
        if numerical_values is None:
            numerical_values = torch.zeros_like(input_ids, dtype=torch.float)
        
        # 步骤 1: 数值感知嵌入 e
        enhanced_embeddings = self.numerical_aware_embedding(input_ids, numerical_values)
        
        # 步骤 2: 特征提取 z
        features = self.feature_network(inputs_embeds=enhanced_embeddings, attention_mask=attention_mask)
        
        # 步骤 3: 归因推断 U
        causal_loc, causal_scale = self.abduction_network(features)
        
        # 步骤 4: 行动决策 S, Y
        action_outputs = self.action_network(causal_loc, causal_scale)
        
        cls_loc = action_outputs['loc_S']
        cls_scale = action_outputs['scale_S']
        
        # 组织输出
        outputs = {
            'features': features,
            'enhanced_embeddings': enhanced_embeddings,
            'causal_loc': causal_loc,
            'causal_scale': causal_scale,
            'cls_loc': cls_loc,
            'cls_scale': cls_scale,
            'reg_loc': action_outputs['loc_Y'],
            'reg_scale': action_outputs['scale_Y']
        }
        
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
    
    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, 
                numerical_values: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                strategy: str = 'deterministic') -> Dict[str, Union[torch.Tensor, float]]:
        """
        Performs a single-step prediction for the next token.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            numerical_values (torch.Tensor, optional): Numerical values for <NUM> tokens.
            attention_mask (torch.Tensor, optional): Attention mask.
            strategy (str): The prediction strategy.
                            - 'deterministic': Use the median of the distribution (loc).
                            - 'causal_sampling': Sample from the causal representation U.

        Returns:
            A dictionary containing the predicted token ID, regression value, and <NUM> probability
            for the last token in the sequence.
        """
        # 1. Get model outputs from the forward pass
        outputs = self.forward(input_ids, numerical_values, attention_mask)
        
        # We only care about the prediction for the last token in the sequence
        causal_loc = outputs['causal_loc'][:, -1, :]
        causal_scale = outputs['causal_scale'][:, -1, :]

        # 2. Use the ActionNetwork's predict method with the chosen strategy
        if strategy == 'causal_sampling':
            # For sampling, we pass both loc and scale
            predictions = self.action_network.predict(causal_loc, causal_scale)
        else: # 'deterministic'
            # For deterministic, we only pass loc (scale is assumed zero)
            predictions = self.action_network.predict(causal_loc, None)

        return {
            "pred_token_id": predictions['cls_pred'],
            "pred_value": predictions['reg_pred'],
            "num_prob": predictions['num_prob']
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, 
                 max_new_tokens: int,
                 numerical_values: Optional[torch.Tensor] = None, 
                 attention_mask: Optional[torch.Tensor] = None,
                 strategy: str = 'deterministic',
                 stop_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generates a sequence of tokens auto-regressively.

        Args:
            input_ids (torch.Tensor): The initial sequence of token IDs.
            max_new_tokens (int): The maximum number of new tokens to generate.
            numerical_values (torch.Tensor, optional): Initial numerical values.
            attention_mask (torch.Tensor, optional): Initial attention mask.
            strategy (str): 'deterministic' or 'causal_sampling'.
            stop_token_id (int, optional): A token ID that stops generation.

        Returns:
            The generated sequence of token IDs, including the initial input.
        """
        if numerical_values is None:
            numerical_values = torch.zeros_like(input_ids, dtype=torch.float32)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        generated_ids = input_ids

        for _ in range(max_new_tokens):
            # Get the prediction for the next token
            next_token_preds = self.predict(
                input_ids=generated_ids,
                numerical_values=numerical_values,
                attention_mask=attention_mask,
                strategy=strategy
            )
            
            next_token_id = next_token_preds['pred_token_id'].unsqueeze(-1)
            
            # Check for stop token
            if stop_token_id is not None and next_token_id.item() == stop_token_id:
                break
            
            # Append the new token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # For the next iteration, we need to handle numerical values and attention mask
            # If the predicted token is <NUM>, we use its predicted value. Otherwise, 0.
            next_value = torch.zeros_like(next_token_id, dtype=torch.float32)
            if next_token_id.item() == self.num_token_id:
                next_value[0, 0] = next_token_preds['pred_value']
            
            numerical_values = torch.cat([numerical_values, next_value], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

        return generated_ids

    def _apply_knowledge_transfer_initialization(self, num_target_median: float, num_target_scale: float):
        """
        DEPRECATED: This method is kept for backward compatibility.
        """
        print("Applying knowledge transfer initialization...")
        
        # Initialize abduction network for identity mapping
        self.abduction_network.initialize_for_identity_mapping(scale_bias=2.3)
        
        # Transfer knowledge from Qwen to action network
        if self.use_real_qwen and hasattr(self.features_network, 'qwen_model'):
            qwen_model = self.features_network.qwen_model
            
            # Find the language model head
            lm_head = None
            for name in ['lm_head', 'cls', 'output_layer']:
                if hasattr(qwen_model, name):
                    lm_head = getattr(qwen_model, name)
                    print(f"Found language model head: {name}")
                    break
            
            if lm_head is not None:
                print(f"  🧮 Applying knowledge transfer initialization...")
                qwen_vocab_size = lm_head.weight.shape[0]
                our_vocab_size = self.vocab_size
                print(f"    - Qwen vocab size: {qwen_vocab_size}, Our vocab size: {our_vocab_size}")
                
                # Copy weights for tokens that exist in both vocabularies
                with torch.no_grad():
                    # Classification head
                    cls_linear = self.action_network.classification_head.causal_linear
                    
                    # Only copy weights for tokens in our vocabulary
                    copy_size = min(our_vocab_size, qwen_vocab_size)
                    cls_linear.weight[:copy_size].copy_(lm_head.weight[:copy_size])
                    print(f"    - Copied weights for {copy_size} tokens from Qwen model")
                    
                    # Handle the case where Qwen has more tokens (reserved tokens)
                    if qwen_vocab_size > our_vocab_size:
                        print(f"    - Qwen has {qwen_vocab_size - our_vocab_size} reserved tokens that won't be used")
                    
                    # Special handling for <NUM> token if it's beyond Qwen's original vocab
                    if self.num_token_id >= copy_size:
                        print(f"    - <NUM> token (ID: {self.num_token_id}) uses pre-initialized weights")
                    else:
                        print(f"    - <NUM> token inherits Qwen's token {self.num_token_id} weights")
                    
                    # Initialize biases
                    if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                        if cls_linear.bias is not None:
                            cls_linear.bias[:copy_size].copy_(lm_head.bias[:copy_size])
                    else:
                        if cls_linear.bias is not None:
                            cls_linear.bias.zero_()
                        print(f"    - Initialized all biases to 0 (Qwen has no bias)")
                    
                    # Regression head initialization
                    reg_linear = self.action_network.regression_head.causal_linear
                    torch.nn.init.xavier_uniform_(reg_linear.weight, gain=0.01)
                    if reg_linear.bias is not None:
                        reg_linear.bias.data.fill_(0.0)
                    print(f"    - Regression head: weight Xavier(gain=0.01), bias = 0.0")
                
                print(f"  ✅ Knowledge transfer initialization complete:")
                print(f"    * Classification head inherits Qwen's language modeling knowledge")
                print(f"    * Reserved tokens are preserved but not used")
                print(f"    * Regression head initialized with zero bias (no data dependency)")

    def compute_loss(self, outputs, targets, numerical_values, attention_mask=None):
        """
        Compute the loss for training the causal language model.
        
        This implements the gated loss function from core-design.md:
        L_total = Σ(L_cls_i + λ * L_reg_gated_i)
        
        现在支持混合门控策略：
        L_reg_gated = mask * (alpha + (1-alpha) * P_NUM) * L_cauchy_nll
        
        Args:
            outputs: Model outputs dictionary
            targets: Target token IDs [batch_size, seq_len]
            numerical_values: Target numerical values [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing total loss and component losses
        """
        # 导入损失函数和概率计算工具
        from ..losses.loss_functions import compute_total_loss
        from ..utils.losses import compute_ovr_probabilities
        
        # 获取模型输出
        cls_loc = outputs['cls_loc']
        cls_scale = outputs['cls_scale']
        reg_loc = outputs['reg_loc']
        reg_scale = outputs['reg_scale']
        
        # 根据loc和scale计算分类概率
        cls_probs = compute_ovr_probabilities(cls_loc, cls_scale, self.config.ovr_threshold)
        
        # 创建数值掩码
        num_mask = (targets == self.num_token_id).float()
        
        # 获取 <NUM> token 的预测概率
        num_probs = cls_probs[:, :, self.num_token_id]
        
        # 处理attention mask
        if attention_mask is not None:
            # 将padding位置的损失设置为0
            active_mask = attention_mask.float()
            num_mask = num_mask * active_mask
            # 可以考虑将padding位置的targets设置为-100（忽略索引）
        
        # 计算损失
        loss_dict = compute_total_loss(
            cls_probs=cls_probs,
            cls_targets=targets,
            reg_loc=reg_loc,
            reg_scale=reg_scale,
            reg_targets=numerical_values,
            num_probs=num_probs,
            num_mask=num_mask,
            cls_weight=1.0,
            reg_weight=self.config.reg_loss_weight,
            gating_alpha=self.config.reg_loss_gating_alpha,
            attention_mask=active_mask
        )
        
        # 返回格式化的结果 - 保持与测试一致的键名
        return {
            'total': loss_dict['total'],      # 测试期望的键名
            'cls': loss_dict['cls'],          # 测试期望的键名
            'reg': loss_dict['reg'],          # 测试期望的键名
            'loss': loss_dict['total'],       # 向后兼容
            'cls_loss': loss_dict['cls'],     # 向后兼容
            'reg_loss': loss_dict['reg'],     # 向后兼容
            'effective_reg_loss': loss_dict['effective_reg_loss'], # 新增：更有意义的回归损失
            'gate_weights_mean': loss_dict['avg_gate_weight'],
            'num_positions': loss_dict['num_positions'].item(),
            'num_prob_mean': loss_dict['avg_gate_weight']  # 平均门控权重反映了平均概率
        }

