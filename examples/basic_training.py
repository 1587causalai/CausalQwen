"""
因果语言模型训练示例

展示如何训练和使用因果语言模型。
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到Python路径
sys.path.insert(0, project_root)

# 从src包导入
from src.models.causal_lm import CausalLanguageModel
from src.utils.losses import CausalLanguageModelLoss
from src.data.synthetic import create_dataloader, SimpleTokenizer


def train_model(model, dataloader, loss_fn, optimizer, device, epochs=10):
    """
    训练模型
    
    Args:
        model: 因果语言模型
        dataloader: 数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        device: 设备
        epochs: 训练轮数
    """
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            cls_targets = batch['cls_targets'].to(device)
            reg_targets = batch['reg_targets'].to(device)
            
            # 前向传播
            predictions = model(input_ids)
            
            # 计算损失
            losses = loss_fn(predictions, cls_targets, reg_targets)
            
            # 反向传播
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            # 累计损失
            total_loss += losses['total_loss'].item()
            total_cls_loss += losses['cls_loss'].item()
            total_reg_loss += losses['reg_loss'].item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, '
                      f'Loss: {losses["total_loss"].item():.4f}, '
                      f'Cls: {losses["cls_loss"].item():.4f}, '
                      f'Reg: {losses["reg_loss"].item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        
        print(f'Epoch {epoch} completed. '
              f'Avg Loss: {avg_loss:.4f}, '
              f'Avg Cls: {avg_cls_loss:.4f}, '
              f'Avg Reg: {avg_reg_loss:.4f}')


def evaluate_model(model, dataloader, device):
    """
    评估模型
    
    Args:
        model: 因果语言模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        评估结果
    """
    model.eval()
    
    correct_cls = 0
    total_samples = 0
    reg_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            cls_targets = batch['cls_targets'].to(device)
            reg_targets = batch['reg_targets'].to(device)
            
            # 预测
            predictions = model.predict(input_ids)
            
            # 分类准确率
            correct_cls += (predictions['cls_pred'] == cls_targets).sum().item()
            total_samples += cls_targets.size(0)
            
            # 回归误差（只对正确分类的样本计算）
            correct_mask = (predictions['cls_pred'] == cls_targets)
            if correct_mask.any():
                reg_error = torch.abs(predictions['reg_pred'][correct_mask] - reg_targets[correct_mask])
                reg_errors.extend(reg_error.cpu().tolist())
    
    cls_accuracy = correct_cls / total_samples
    reg_mae = sum(reg_errors) / len(reg_errors) if reg_errors else float('inf')
    
    return {
        'cls_accuracy': cls_accuracy,
        'reg_mae': reg_mae,
        'total_samples': total_samples,
        'correct_cls': correct_cls
    }


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512
    causal_dim = 32
    batch_size = 16
    epochs = 20
    learning_rate = 1e-3
    
    # 创建模型
    model = CausalLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        causal_dim=causal_dim
    ).to(device)
    
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # 创建数据加载器
    train_loader = create_dataloader('basic', batch_size, size=1000, vocab_size=vocab_size)
    test_loader = create_dataloader('basic', batch_size, size=200, vocab_size=vocab_size)
    
    # 创建损失函数和优化器
    tokenizer = SimpleTokenizer(vocab_size)
    num_token_id = tokenizer.vocab[tokenizer.num_token]
    
    loss_fn = CausalLanguageModelLoss(num_token_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练前评估
    print("Evaluating before training...")
    eval_results = evaluate_model(model, test_loader, device)
    print(f"Before training - Cls Accuracy: {eval_results['cls_accuracy']:.4f}, "
          f"Reg MAE: {eval_results['reg_mae']:.4f}")
    
    # 训练模型
    print("Starting training...")
    train_model(model, train_loader, loss_fn, optimizer, device, epochs)
    
    # 训练后评估
    print("Evaluating after training...")
    eval_results = evaluate_model(model, test_loader, device)
    print(f"After training - Cls Accuracy: {eval_results['cls_accuracy']:.4f}, "
          f"Reg MAE: {eval_results['reg_mae']:.4f}")
    
    # 示例预测
    print("\nExample predictions:")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 3:  # 只显示前3个批次
                break
                
            input_ids = batch['input_ids'][:3].to(device)  # 只取前3个样本
            predictions = model.predict(input_ids)
            
            for j in range(3):
                text = batch['texts'][j]
                true_value = batch['values'][j]
                pred_cls = predictions['cls_pred'][j].item()
                pred_reg = predictions['reg_pred'][j].item()
                
                print(f"Text: {text}")
                print(f"True value: {true_value:.2f}")
                print(f"Predicted cls: {pred_cls} (should be {num_token_id})")
                print(f"Predicted reg: {pred_reg:.2f}")
                print(f"Cls correct: {pred_cls == num_token_id}")
                print("-" * 50)


if __name__ == "__main__":
    main()

