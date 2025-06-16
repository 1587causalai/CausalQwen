```python
@dataclass
class CausalLMConfig:
    """因果语言模型的配置类"""
    vocab_size: int = 50257
    hidden_size: int = 768
    causal_dim: int = 512
    num_layers: int = 12
    num_heads: int = 12
    
    # 模型选择
    use_real_qwen: bool = False
    qwen_model_path: str = None
    use_mock_feature_network: bool = True
    
    # 数值感知功能控制
    use_numerical_features: bool = True  # 添加这一行
    
    # 特殊token
    num_token_id: int = 50256  # <NUM> token的ID
    
    # ...existing code...
```