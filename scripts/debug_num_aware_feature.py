#!/usr/bin/env python
"""
深入诊断 NumAwareFeatureNetwork 的数值处理
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.feature_network import NumAwareFeatureNetwork, QwenFeatureNetwork
from src.data.tokenizer import QwenTokenizerWrapper
from src.models.causal_lm import CausalLMConfig

def main():
    print("=== NumAwareFeatureNetwork 诊断 ===\n")
    
    # 初始化
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    tokenizer = QwenTokenizerWrapper(model_path=qwen_model_path, use_real_tokenizer=True)
    
    # 创建配置
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=896,
        causal_dim=896,
        use_real_qwen=True,
        qwen_model_path=qwen_model_path
    )
    
    # 创建基础网络
    base_network = QwenFeatureNetwork(
        model_path=qwen_model_path,
        hidden_size=config.hidden_size
    )
    
    # 创建 NumAwareFeatureNetwork（纯加性版本）
    num_aware_network = NumAwareFeatureNetwork(
        base_network=base_network,
        hidden_size=config.hidden_size,
        num_token_id=tokenizer.num_token_id
    )
    
    # 测试多个文本
    test_texts = [
        "The price is 99.99 dollars.",
        "I have 3 apples and 5 oranges.",
        "The temperature is -15.5 degrees.",
        "No numbers here, just text.",
        "The year 2024 is here."
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"测试文本: '{text}'")
        
        # 分词
        encoded = tokenizer(text, return_tensors='pt')
        input_ids = encoded['input_ids']
        numerical_values = encoded['numerical_values']
        attention_mask = encoded['attention_mask']
        
        # 转换为 tokens 查看
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        print(f"\nTokens: {tokens}")
        print(f"Input IDs: {input_ids[0].tolist()}")
        print(f"Numerical values: {numerical_values[0].tolist()}")
        
        # 检查是否有 <NUM> token
        num_positions = (input_ids == tokenizer.num_token_id)
        print(f"\n<NUM> token 位置: {num_positions[0].tolist()}")
        print(f"<NUM> token 数量: {num_positions.sum().item()}")
        
        # 如果有 <NUM> token，显示对应的数值
        if num_positions.any():
            num_indices = num_positions[0].nonzero(as_tuple=True)[0]
            for idx in num_indices:
                print(f"  位置 {idx}: token='{tokens[idx]}', 数值={numerical_values[0][idx].item()}")
        
        # 测试特征提取
        print("\n测试特征提取...")
        with torch.no_grad():
            # 获取基础特征
            base_features = base_network(input_ids, attention_mask)
            print(f"基础特征形状: {base_features.shape}")
            print(f"基础特征统计: mean={base_features.mean().item():.4f}, std={base_features.std().item():.4f}")
            
            # 获取数值感知特征
            num_aware_features = num_aware_network(input_ids, numerical_values, attention_mask)
            print(f"数值感知特征形状: {num_aware_features.shape}")
            print(f"数值感知特征统计: mean={num_aware_features.mean().item():.4f}, std={num_aware_features.std().item():.4f}")
            
            # 比较 <NUM> token 位置的特征差异
            if num_positions.any():
                print("\n<NUM> token 位置的特征分析:")
                for idx in num_indices:
                    base_feat = base_features[0, idx]
                    aware_feat = num_aware_features[0, idx]
                    diff = aware_feat - base_feat
                    
                    print(f"\n位置 {idx} (数值={numerical_values[0][idx].item():.2f}):")
                    print(f"  基础特征范数: {base_feat.norm().item():.4f}")
                    print(f"  数值感知特征范数: {aware_feat.norm().item():.4f}")
                    print(f"  差异向量范数: {diff.norm().item():.4f}")
                    print(f"  余弦相似度: {torch.cosine_similarity(base_feat.unsqueeze(0), aware_feat.unsqueeze(0)).item():.4f}")
    
    # 测试数值量化的分布
    print(f"\n{'='*60}")
    print("测试数值量化分布...")
    test_values = torch.tensor([0.0, 0.1, 1.0, 10.0, 99.99, 1000.0, -5.2, -100.0])
    for val in test_values:
        quantized = num_aware_network.quantize_value(val.unsqueeze(0))
        # 获取对应的嵌入
        with torch.no_grad():
            embedding = num_aware_network.numerical_embeddings(quantized)
            print(f"  数值 {val:8.2f} -> 量化桶 {quantized.item():4d} -> 嵌入范数 {embedding.norm().item():.4f}")
    
    # 测试数值编码函数
    print(f"\n{'='*60}")
    print("测试直接对数编码...")
    test_values = torch.tensor([0.0, 0.1, 1.0, 10.0, 99.99, 1000.0, -5.2, -100.0])
    for val in test_values:
        # 计算编码值
        encoded = torch.sign(val) * torch.log1p(torch.abs(val))
        print(f"  数值 {val:8.2f} -> 编码值 {encoded.item():8.4f}")
    
    # 可视化编码函数
    print(f"\n{'='*60}")
    print("编码函数的性质分析...")
    import numpy as np
    values = np.logspace(-2, 3, 100)  # 从0.01到1000
    positive_encoding = np.log1p(values)
    negative_encoding = -np.log1p(values)
    
    print(f"正数编码范围: [{positive_encoding.min():.4f}, {positive_encoding.max():.4f}]")
    print(f"编码函数在v=1处的导数: {1/(1+1):.4f}")
    print(f"编码函数在v=100处的导数: {1/(1+100):.4f}")

if __name__ == '__main__':
    main()
