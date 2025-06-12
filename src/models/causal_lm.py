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

from .feature_network import QwenFeatureNetwork, NumAwareFeatureNetwork
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
    use_numerical_features: bool = True  # 添加数值感知功能控制
    
    # OvR classification settings
    use_ovr_classifier: bool = True
    ovr_threshold: float = 0.0  # 使用 0.0 作为默认阈值
    
    # Regression loss settings
    reg_loss_weight: float = 1.0
    reg_loss_gating_alpha: float = 1.0  # 门控系数：1.0 = 无门控，0.0 = 完全门控
    
    # Distribution settings
    use_cauchy_distribution: bool = True
    initial_scale_bias: float = 2.3  # log(10) ≈ 2.3

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
        
        # 验证词汇表大小配置
        if config.use_real_qwen and config.vocab_size != 151936:
            print(f"⚠️  警告：使用真实 Qwen 时，建议 vocab_size=151936（完整配置容量）")
            print(f"   当前设置：{config.vocab_size}")
        
        # 特征网络选择逻辑（修复导入问题）
        use_mock_feature_network = getattr(config, 'use_mock_feature_network', False)
        
        if config.use_real_qwen and not use_mock_feature_network:
            print("Initializing with real Qwen model...")
            # 使用数值感知特征网络包装QwenFeatureNetwork
            base_qwen_network = QwenFeatureNetwork(
                model_path=config.qwen_model_path,
                hidden_size=config.hidden_size
            )
            
            # 检查是否需要数值感知功能
            if hasattr(config, 'use_numerical_features') and config.use_numerical_features:
                self.feature_network = NumAwareFeatureNetwork(
                    base_network=base_qwen_network,
                    num_token_id=config.num_token_id,
                    hidden_size=config.hidden_size
                )
            else:
                # 直接使用QwenFeatureNetwork，但需要确保接口兼容
                self.feature_network = base_qwen_network
        else:
            print("Initializing with mock feature network...")
            # 需要导入MockFeatureNetwork
            from .feature_network import MockFeatureNetwork
            self.feature_network = MockFeatureNetwork(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size
            )
        
        # Abduction network - 推断个体因果表征分布
        self.abduction_network = AbductionNetwork(self.hidden_size, self.causal_dim)
        
        # Action network - 基于因果表征生成输出
        self.action_network = ActionNetwork(
            causal_dim=self.causal_dim,
            vocab_size=self.vocab_size,
            num_token_id=self.num_token_id,
            ovr_threshold=config.ovr_threshold
        )

        # 数值编码器 - 根据配置决定是否使用
        if config.use_numerical_features:
            # 可以从 feature_network 模块导入，如果存在的话
            try:
                from .feature_network import NumericalEncoder
                self.numerical_encoder = NumericalEncoder(
                    embedding_dim=config.hidden_size
                )
            except ImportError:
                print("Warning: NumericalEncoder not found, numerical features will be handled by feature network")
                self.numerical_encoder = None
        else:
            self.numerical_encoder = None

    def init_weights(self, num_target_median=None, num_target_scale=None):
        """
        Initialize the weights of abduction and action networks using the
        updated knowledge transfer strategy.
        
        知识传输策略：
        - 分类头：完全复用 Qwen 的 lm_head（包括我们添加的 <NUM> token）
        - 回归头：使用 <NUM> token 的权重初始化（利用保留词汇）
        - 保留词汇：Qwen 已经为这些位置分配了权重，我们直接使用
        
        Args:
            num_target_median (float, optional): Deprecated, no longer used
            num_target_scale (float, optional): Deprecated, no longer used
        """
        print("应用知识传输初始化...")
        
        # 1. Initialize Abduction Network
        # This assumes hidden_size and causal_dim are the same
        if self.hidden_size == self.causal_dim:
            # Get initial_scale_bias from config if available
            initial_scale_bias = getattr(self, 'config', None)
            if initial_scale_bias is not None and hasattr(initial_scale_bias, 'initial_scale_bias'):
                initial_scale_bias = initial_scale_bias.initial_scale_bias
            else:
                initial_scale_bias = 2.3  # Default value
            
            self.abduction_network.initialize_for_identity_mapping(scale_bias=initial_scale_bias)
            print(f"  - Abduction network initialized for identity mapping (scale_bias={initial_scale_bias}).")
        else:
            print("  - WARNING: Abduction network not initialized (hidden_size != causal_dim).")

        # 2. Initialize Action Network
        qwen_lm_head = None
        
        # 获取 lm_head 的逻辑需要更加健壮
        try:
            if isinstance(self.feature_network, NumAwareFeatureNetwork):
                # 如果是数值感知特征网络，获取其基础网络的lm_head
                if hasattr(self.feature_network, 'base_network'):
                    base_network = self.feature_network.base_network
                    if hasattr(base_network, 'get_lm_head'):
                        qwen_lm_head = base_network.get_lm_head()
                        print("Found language model head: lm_head (from NumAwareFeatureNetwork.base_network)")
            elif hasattr(self.feature_network, 'get_lm_head'):
                # 直接从特征网络获取lm_head
                qwen_lm_head = self.feature_network.get_lm_head()
                if qwen_lm_head is not None:
                    print("Found language model head: lm_head")
                else:
                    print("Feature network returned None for lm_head (likely MockFeatureNetwork)")
            else:
                print("Feature network does not have get_lm_head method")
                
        except Exception as e:
            print(f"Error getting lm_head: {e}")
            qwen_lm_head = None

        if qwen_lm_head is not None:
            self.action_network.init_weights(
                qwen_lm_head=qwen_lm_head,
                num_target_median=0.0,  # No longer used, passing dummy value
                num_target_scale=1.0,   # No longer used, passing dummy value
                num_token_id=self.num_token_id
            )
            print("  - Action network initialized from Qwen's lm_head (no data dependency).")
        else:
            print("  - WARNING: Action network not initialized (Qwen lm_head not available).")
            print("  - Using random initialization for ActionNetwork")
            # 如果没有Qwen的lm_head，使用随机初始化
            self.action_network.init_weights(
                qwen_lm_head=None,
                num_target_median=0.0,
                num_target_scale=1.0,
                num_token_id=self.num_token_id
            )

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
        
        # Extract features
        features = self.feature_network(input_ids, numerical_values, attention_mask)
        
        # Infer individual causal representation distribution
        causal_loc, causal_scale = self.abduction_network(features)
        
        # Transform individual causal representation to outputs
        action_outputs = self.action_network(causal_loc, causal_scale)
        
        # 计算OvR概率
        from ..utils.distributions import cauchy_cdf
        
        cls_loc = action_outputs['cls_loc']
        cls_scale = action_outputs['cls_scale']
        
        # 计算每个类别的概率：P(S_k > C_k)
        thresholds = self.action_network.classification_head.thresholds
        cls_probs = 0.5 + (1 / torch.pi) * torch.atan((cls_loc - thresholds) / cls_scale)
        
        # 组织输出
        outputs = {
            'features': features,
            'causal_loc': causal_loc,
            'causal_scale': causal_scale,
            'cls_loc': cls_loc,
            'cls_scale': cls_scale,
            'cls_probs': cls_probs,
            'reg_loc': action_outputs['reg_loc'],
            'reg_scale': action_outputs['reg_scale']
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

    def _apply_knowledge_transfer_initialization(self, num_target_median: float, num_target_scale: float):
        """Apply knowledge transfer from Qwen model to action network."""
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
        # 导入损失函数
        from ..losses.loss_functions import compute_total_loss
        
        # 获取模型输出
        cls_probs = outputs['cls_probs']  # [batch_size, seq_len, vocab_size]
        reg_loc = outputs['reg_loc']      # [batch_size, seq_len]
        reg_scale = outputs['reg_scale']  # [batch_size, seq_len]
        
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
            gating_alpha=self.config.reg_loss_gating_alpha
        )
        
        # 返回格式化的结果 - 保持与测试一致的键名
        return {
            'total': loss_dict['total'],      # 测试期望的键名
            'cls': loss_dict['cls'],          # 测试期望的键名
            'reg': loss_dict['reg'],          # 测试期望的键名
            'loss': loss_dict['total'],       # 向后兼容
            'cls_loss': loss_dict['cls'],     # 向后兼容
            'reg_loss': loss_dict['reg'],     # 向后兼容
            'gate_weights_mean': loss_dict['avg_gate_weight'],
            'num_positions': loss_dict['num_positions'].item(),
            'num_prob_mean': loss_dict['avg_gate_weight']  # 平均门控权重反映了平均概率
        }

