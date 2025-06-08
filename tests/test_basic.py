"""
简单的模型测试脚本

验证因果语言模型的基本功能。
"""

import torch
import sys
import os

# 将项目根目录添加到sys.path, 确保模块可以被找到
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.causal_lm import CausalLanguageModel
from src.utils.cauchy import cauchy_pdf, cauchy_linear_transform
from src.data.synthetic import SimpleTokenizer, CausalDataset


def test_cauchy_functions():
    """测试柯西分布函数"""
    print("Testing Cauchy distribution functions...")
    
    # 测试PDF
    x = torch.tensor([0.0, 1.0, -1.0])
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    
    pdf_values = cauchy_pdf(x, loc, scale)
    print(f"Cauchy PDF at x={x.tolist()}: {pdf_values.tolist()}")
    
    # 测试线性变换
    input_loc = torch.randn(2, 3)
    input_scale = torch.abs(torch.randn(2, 3)) + 0.1
    weight = torch.randn(4, 3)
    
    output_loc, output_scale = cauchy_linear_transform(input_loc, input_scale, weight)
    print(f"Linear transform output shapes: {output_loc.shape}, {output_scale.shape}")
    
    print("Cauchy functions test passed!\n")


def test_model_creation():
    """测试模型创建"""
    print("Testing model creation...")
    
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 128
    causal_dim = 16
    
    model = CausalLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        causal_dim=causal_dim
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Model output keys: {list(outputs.keys())}")
    print(f"Classification probabilities shape: {outputs['cls_probs'].shape}")
    print(f"Regression predictions shape: {outputs['reg_pred'].shape}")
    
    print("Model creation test passed!\n")


def test_data_generation():
    """测试数据生成"""
    print("Testing data generation...")
    
    # 测试分词器
    tokenizer = SimpleTokenizer(vocab_size=100)
    text = "The price is 25.5 dollars."
    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"Original text: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {decoded_text}")
    
    # 测试数据集
    dataset = CausalDataset(data_type='basic', size=10, vocab_size=100)
    sample = dataset[0]
    
    print(f"Dataset sample keys: {list(sample.keys())}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Sample text: {sample['text']}")
    print(f"Sample value: {sample['value']}")
    
    print("Data generation test passed!\n")


def test_training_step():
    """测试训练步骤"""
    print("Testing training step...")
    
    # 创建小模型和数据
    model = CausalLanguageModel(vocab_size=100, embed_dim=32, hidden_dim=64, causal_dim=8)
    dataset = CausalDataset(data_type='basic', size=8, vocab_size=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    
    # 创建损失函数和优化器
    from src.utils.losses import CausalLanguageModelLoss
    tokenizer = SimpleTokenizer(vocab_size=100)
    num_token_id = tokenizer.vocab[tokenizer.num_token]
    
    loss_fn = CausalLanguageModelLoss(num_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 执行一个训练步骤
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids']
        cls_targets = batch['cls_target']
        reg_targets = batch['reg_target']
        
        # 前向传播
        predictions = model(input_ids)
        
        # 计算损失
        losses = loss_fn(predictions, cls_targets, reg_targets)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        print(f"Training step completed:")
        print(f"  Total loss: {losses['total_loss'].item():.4f}")
        print(f"  Classification loss: {losses['cls_loss'].item():.4f}")
        print(f"  Regression loss: {losses['reg_loss'].item():.4f}")
        
        break  # 只测试一个批次
    
    print("Training step test passed!\n")


def main():
    """主测试函数"""
    print("=" * 50)
    print("CausalQwen Model Tests")
    print("=" * 50)
    
    try:
        test_cauchy_functions()
        test_model_creation()
        test_data_generation()
        test_training_step()
        
        print("=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

