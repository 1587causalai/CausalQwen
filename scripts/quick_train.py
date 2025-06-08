#!/usr/bin/env python
"""
Quick training script for the causal language model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Add the project root to Python path
# Correctly point to the project root, which is one level up from the /scripts directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.tokenizer import QwenTokenizerWrapper, MockTokenizer
from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
from src.data.synthetic import TextWithNumbersGenerator
from src.utils.losses import CausalLMLoss


def create_training_data(tokenizer, num_samples=1000):
    """Create training data."""
    print(f"生成 {num_samples} 个训练样本...")
    
    generator = TextWithNumbersGenerator(seed=42)
    texts, true_values = generator.generate_text(num_samples=num_samples)
    
    # Tokenize texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask'] 
    numerical_values = inputs['numerical_values']
    
    # Create targets (always <NUM> token)
    targets = torch.full((num_samples,), tokenizer.num_token_id, dtype=torch.long)
    target_values = torch.tensor(true_values, dtype=torch.float32)
    
    print(f"数据形状: input_ids={input_ids.shape}, targets={targets.shape}")
    print(f"目标词元ID: {tokenizer.num_token_id}")
    print(f"数值范围: {target_values.min():.2f} 到 {target_values.max():.2f}")
    
    return input_ids, attention_mask, numerical_values, targets, target_values


def train_quick_model(use_real_qwen=False, num_epochs=5, batch_size=8, num_samples=500):
    """Train the model quickly."""
    print("=== 快速训练因果语言模型 ===")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # Create tokenizer
    if use_real_qwen:
        print("使用真实 Qwen tokenizer...")
        tokenizer = QwenTokenizerWrapper(
            model_path="~/models/Qwen2.5-0.5B",
            use_real_tokenizer=True
        )
        hidden_size = 896
    else:
        print("使用模拟 tokenizer...")
        tokenizer = MockTokenizer(vocab_size=1000)
        hidden_size = 768
    
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"<NUM> token ID: {tokenizer.num_token_id}")
    
    # Create model configuration
    config = CausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        hidden_size=hidden_size,
        causal_dim=64,
        use_mock_feature_network=not use_real_qwen,
        use_real_qwen=use_real_qwen,
        qwen_model_path="~/models/Qwen2.5-0.5B" if use_real_qwen else None
    )
    
    # Create model
    print("创建模型...")
    model = CausalLanguageModel(config)

    def weights_init(m):
        """Custom weight initialization for linear layers."""
        if isinstance(m, nn.Linear):
            # Use Xavier initialization with a small gain to prevent explosion
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                # Initialize bias to a small non-zero value
                torch.nn.init.constant_(m.bias, 0.01)

    # Apply custom initialization to new components
    print("Applying custom weight initialization...")
    model.abduction_network.apply(weights_init)
    model.action_network.apply(weights_init)

    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # Create training data
    input_ids, attention_mask, numerical_values, targets, target_values = create_training_data(
        tokenizer, num_samples=num_samples
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, numerical_values, targets, target_values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create loss function and optimizer
    loss_fn = CausalLMLoss(
        num_classes=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        regression_weight=1.0
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\n开始训练 {num_epochs} 个epoch...")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (batch_input_ids, batch_attention_mask, batch_numerical_values, batch_targets, batch_target_values) in enumerate(pbar):
            # Move to device
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_numerical_values = batch_numerical_values.to(device)
            batch_targets = batch_targets.to(device)
            batch_target_values = batch_target_values.to(device)
            
            # Forward pass
            outputs = model(batch_input_ids, batch_numerical_values, batch_attention_mask)
            
            # Compute loss
            loss_dict = loss_fn(
                outputs["cls_loc"],
                outputs["cls_scale"], 
                outputs["reg_loc"],
                outputs["reg_scale"],
                batch_targets,
                batch_target_values
            )
            
            loss = loss_dict["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            predictions = model.predict(batch_input_ids, batch_numerical_values, batch_attention_mask)
            pred_tokens = predictions['cls_pred']
            correct += (pred_tokens == batch_targets).sum().item()
            total += batch_targets.size(0)
            
            # Update metrics
            total_loss += loss.item()
            total_cls_loss += loss_dict["cls_loss"].item()
            total_reg_loss += loss_dict["reg_loss"].item()
            
            # Update progress bar
            accuracy = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'cls_loss': f'{loss_dict["cls_loss"].item():.4f}',
                'reg_loss': f'{loss_dict["reg_loss"].item():.4f}'
            })
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, "
              f"Cls_Loss={avg_cls_loss:.4f}, Reg_Loss={avg_reg_loss:.4f}")
    
    print("\n训练完成！")
    
    # Save model
    save_path = f"results/trained_model_{'qwen' if use_real_qwen else 'mock'}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
    
    return model, tokenizer


def test_trained_model(model, tokenizer, device):
    """Test the trained model."""
    print("\n=== 测试训练后的模型 ===")
    
    model.eval()
    
    # Test on new data
    test_texts = [
        "The price is 42.5 dollars.",
        "Temperature: 25.0 degrees", 
        "Distance: 100.0 meters",
        "Weight is 75.5 kg"
    ]
    
    for text in test_texts:
        print(f"\n测试文本: {text}")
        
        # Tokenize - this already adds batch dimension for real tokenizers
        inputs = tokenizer.encode(text, return_tensors='pt')
        
        # Ensure inputs have correct dimensions
        if inputs['input_ids'].dim() == 1:
            # Add batch dimension if missing
            inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
            inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0) 
            inputs['numerical_values'] = inputs['numerical_values'].unsqueeze(0)
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['numerical_values'], inputs['attention_mask'])
            predictions = model.predict(inputs['input_ids'], inputs['numerical_values'], inputs['attention_mask'])
        
        pred_token_id = predictions['cls_pred'].item()
        pred_value = predictions['reg_pred'].item()
        
        print(f"预测词元ID: {pred_token_id}")
        print(f"目标词元ID: {tokenizer.num_token_id}")
        print(f"预测正确: {pred_token_id == tokenizer.num_token_id}")
        print(f"预测数值: {pred_value:.4f}")
        
        # Show top-3 probabilities
        cls_probs = outputs['cls_probs'].squeeze()
        top_k = torch.topk(cls_probs, k=5)
        print("前5个预测:")
        for i, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices)):
            is_num = " (*NUM*)" if token_id.item() == tokenizer.num_token_id else ""
            print(f"  {i+1}. ID={token_id.item()}, prob={prob.item():.6f}{is_num}")
        
        # Find the rank of <NUM> token
        _, sorted_indices = torch.sort(cls_probs, descending=True)
        num_rank = (sorted_indices == tokenizer.num_token_id).nonzero(as_tuple=True)[0].item() + 1
        
        print(f"<NUM>词元概率: {cls_probs[tokenizer.num_token_id].item():.6f}")
        print(f"<NUM>词元排名: {num_rank}")
        
        # Check if there are ties at the top
        max_prob = cls_probs.max().item()
        tied_tokens = (cls_probs == max_prob).sum().item()
        print(f"最高概率: {max_prob:.6f}, 并列数量: {tied_tokens}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick train causal language model')
    parser.add_argument('--use_real_qwen', action='store_true', help='Use real Qwen model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of training samples')
    
    args = parser.parse_args()
    
    # Train model
    model, tokenizer = train_quick_model(
        use_real_qwen=args.use_real_qwen,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_trained_model(model, tokenizer, device) 