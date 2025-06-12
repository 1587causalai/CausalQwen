"""
Trainer module for fine-tuning the Causal Language Model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from ..data.synthetic import TextWithNumbersGenerator
from ..utils.losses import CausalLMLoss, compute_ovr_probabilities

class Trainer:
    """Handles the training loop, optimizer, and data loading for fine-tuning."""
    
    def __init__(self, model, tokenizer, device, learning_rate=1e-4, batch_size=16, config=None, wandb_run=None):
        """
        Initialize the Trainer.
        
        Args:
            model (nn.Module): The model to be trained.
            tokenizer: The tokenizer to use.
            device (torch.device): The device to train on.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The batch size for training.
            config: The model's configuration object.
            wandb_run: An active Weights & Biases run object.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.wandb_run = wandb_run
        self.learning_rate = learning_rate
        self.config = config  # 确保保存config为实例属性
        
        # 从配置中获取门控系数
        self.gating_alpha = config.reg_loss_gating_alpha if config else 1.0
        
        # 在日志中记录门控策略
        if self.gating_alpha == 1.0:
            gating_strategy = "无门控（硬掩码）"
        elif self.gating_alpha == 0.0:
            gating_strategy = "完全门控（软注意力）"
        else:
            gating_strategy = f"混合门控（alpha={self.gating_alpha}）"
        
        print(f"   - 回归损失门控策略: {gating_strategy}")
        
        # Calculate data statistics for initialization
        print("Pre-calculating data statistics for initialization...")
        temp_dataloader = self._create_training_data(1000, shuffle=False)
        all_target_values = []
        for batch in temp_dataloader:
            # batch is now: [input_ids, attention_mask, numerical_values, labels, target_values]
            # target_values is [B, S] with NaN for non-numerical positions
            batch_target_values = batch[4]  # [B, S]
            # Only collect non-NaN values  
            valid_values = batch_target_values[~torch.isnan(batch_target_values)]
            if valid_values.size(0) > 0:
                all_target_values.append(valid_values)
        
        if all_target_values:
            all_target_values = torch.cat(all_target_values)
            # 对于柯西分布，使用中位数估计位置参数，使用 IQR/2 估计尺度参数
            self.num_target_median = torch.median(all_target_values).item()
            # 计算四分位数
            q1 = torch.quantile(all_target_values, 0.25).item()
            q3 = torch.quantile(all_target_values, 0.75).item()
            # 柯西分布的尺度参数估计：IQR / 2 (因为柯西分布的IQR ≈ 2 * scale)
            self.num_target_scale = (q3 - q1) / 2.0
            print(f"  - Calculated Median (location): {self.num_target_median:.2f}")
            print(f"  - Calculated IQR/2 (scale): {self.num_target_scale:.2f}")
        else:
            self.num_target_median = 0.0
            self.num_target_scale = 1.0
            print("  - No numerical targets found. Using default stats (Median=0, Scale=1).")
        
        # Initialize model weights using the new strategy
        if hasattr(self.model, 'init_weights'):
            self.model.init_weights(self.num_target_median, self.num_target_scale)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = CausalLMLoss(
            num_classes=self.tokenizer.vocab_size,
            num_token_id=self.tokenizer.num_token_id,
            regression_weight=self.config.reg_loss_weight,
            ovr_threshold=self.config.ovr_threshold
        )
        
    @staticmethod
    def weights_init(m):
        """Custom weight initialization for linear layers."""
        if isinstance(m, nn.Linear):
            # Use Xavier initialization with a small gain to prevent explosion
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                # Initialize bias to a small non-zero value
                torch.nn.init.constant_(m.bias, 0.01)

    def _create_training_data(self, num_samples, shuffle=True):
        """Create synthetic training data in sequence-to-sequence format."""
        print(f"Generating {num_samples} synthetic training samples...")
        generator = TextWithNumbersGenerator(seed=42)
        texts, true_values = generator.generate_text(num_samples=num_samples)
        
        # Use the enhanced tokenizer to encode texts properly
        inputs = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask'] 
        numerical_values = inputs['numerical_values']
        
        # --- Create sequence-to-sequence labels and targets ---
        # Labels: each position predicts the next token
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding positions
        labels[:, :-1] = labels[:, 1:].clone()  # Shift left: predict next token
        labels[:, -1] = -100  # Last position has no next token to predict
        
        # Target values: numerical values for positions where label is <NUM>
        target_values = torch.full_like(numerical_values, float('nan'))
        shifted_numerical_values = numerical_values.clone()
        shifted_numerical_values[:, :-1] = numerical_values[:, 1:].clone()  # Shift left 
        shifted_numerical_values[:, -1] = 0.0  # Last position default
        
        # Only set target values where the label is <NUM> 
        num_mask = (labels == self.tokenizer.num_token_id)
        target_values[num_mask] = shifted_numerical_values[num_mask]
        
        dataset = TensorDataset(input_ids, attention_mask, numerical_values, labels, target_values)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, num_epochs, num_samples):
        """
        Run the training loop.
        
        Args:
            num_epochs (int): Number of epochs to train for.
            num_samples (int): Number of synthetic samples to generate for training.
        """
        self.model.train()
        
        print("Creating final training data loader...")
        dataloader = self._create_training_data(num_samples, shuffle=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        global_step = 0
        for epoch in range(num_epochs):
            total_loss = 0
            total_cls_loss = 0
            total_reg_loss = 0
            total_correct = 0
            total_num_correct = 0
            total_num_samples = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i, batch in enumerate(pbar):
                batch = [t.to(self.device) for t in batch]
                batch_input_ids, batch_attention_mask, batch_numerical_values, batch_labels, batch_target_values = batch
                
                # Forward pass
                outputs = self.model(batch_input_ids, batch_numerical_values, batch_attention_mask)

                # Compute loss
                loss_dict = self.loss_fn(
                    outputs["cls_loc"], outputs["cls_scale"],
                    outputs["reg_loc"], outputs["reg_scale"],
                    batch_labels, batch_target_values
                )
                loss = loss_dict["loss"]
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # --- METRICS LOGGING ---
                total_loss += loss.item()
                total_cls_loss += loss_dict["cls_loss"].item()
                total_reg_loss += loss_dict["gated_reg_loss"].item()
                
                # Classification metrics - compute predictions from model outputs
                cls_loc = outputs["cls_loc"]  # [B, S, C]
                cls_scale = outputs["cls_scale"]  # [B, S, C]
                # Compute OvR probabilities
                cls_probs = compute_ovr_probabilities(cls_loc, cls_scale, self.config.ovr_threshold)
                pred_tokens = torch.argmax(cls_probs, dim=-1)  # [B, S]
                
                # Create mask for valid (non-ignored) tokens  
                valid_mask = (batch_labels != -100)
                
                # Overall accuracy (only on valid tokens)
                valid_pred = pred_tokens[valid_mask]
                valid_labels = batch_labels[valid_mask]
                total_correct += (valid_pred == valid_labels).sum().item()
                total_samples += valid_labels.size(0)
                
                # <NUM> token accuracy (only on valid <NUM> tokens)
                num_mask = (batch_labels == self.tokenizer.num_token_id)
                num_acc = None
                if num_mask.any():
                    num_pred = pred_tokens[num_mask]
                    num_labels = batch_labels[num_mask]
                    total_num_correct += (num_pred == num_labels).sum().item()
                    total_num_samples += num_labels.size(0)
                    num_acc = (num_pred == num_labels).sum().item() / num_labels.size(0)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{total_correct / total_samples:.4f}',
                    'num_acc': f'{total_num_correct / total_num_samples if total_num_samples > 0 else 0:.4f}'
                })
                
                # Log to Weights & Biases if enabled
                if self.wandb_run:
                    # Calculate regression MAE for logging
                    reg_mae = None
                    if num_mask.any():
                        valid_reg_targets = batch_target_values[num_mask]
                        valid_reg_targets = valid_reg_targets[~torch.isnan(valid_reg_targets)]
                        if valid_reg_targets.size(0) > 0:
                            valid_reg_preds = outputs['reg_loc'][num_mask]
                            valid_reg_preds = valid_reg_preds[:valid_reg_targets.size(0)]
                            reg_mae = torch.abs(valid_reg_preds - valid_reg_targets).mean().item()

                    log_data = {
                        "total_loss": loss.item(),
                        "cls_loss": loss_dict["cls_loss"].item(),
                        "gated_reg_loss": loss_dict["gated_reg_loss"].item(),
                        "reg_mae": reg_mae,
                        "units_mean_loc": outputs['causal_loc'].mean().item(),
                        "units_mean_scale": outputs['causal_scale'].mean().item(),
                        "ovr_prob_sum": cls_probs.sum(dim=-1).mean().item(),
                        "num_accuracy": num_acc,
                        "accuracy": (valid_pred == valid_labels).sum().item() / valid_labels.size(0) if valid_labels.size(0) > 0 else 0.0
                    }
                    
                    self.wandb_run.log({k: v for k, v in log_data.items() if v is not None}, step=global_step)
                
                global_step += 1
        
            avg_loss = total_loss / len(dataloader)
            avg_cls_loss = total_cls_loss / len(dataloader)
            avg_reg_loss = total_reg_loss / len(dataloader)
            accuracy = total_correct / total_samples
            num_accuracy = total_num_correct / total_num_samples if total_num_samples > 0 else 0
            print(f"Epoch {epoch+1} Complete: Avg Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, <NUM> Accuracy={num_accuracy:.4f}")

            # Log epoch-level summary to wandb
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch_avg_loss": avg_loss,
                    "epoch_accuracy": accuracy,
                    "epoch_num_accuracy": num_accuracy
                }, step=global_step)

        # Return training metrics instead of dataloader
        return {
            "final_loss": avg_loss,
            "final_cls_loss": avg_cls_loss,
            "final_reg_loss": avg_reg_loss,
            "final_accuracy": accuracy,
            "final_num_accuracy": num_accuracy,
            "total_epochs": num_epochs
        }

    def train_step(self, batch):
        """执行单个训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 解包批次数据
        batch_input_ids = batch['input_ids']
        batch_labels = batch['labels']
        batch_numerical_values = batch['numerical_values']
        batch_attention_mask = batch.get('attention_mask', None)
        
        # 前向传播
        outputs = self.model(batch_input_ids, batch_numerical_values, batch_attention_mask)
        
        # 计算损失
        loss_info = self.model.compute_loss(
            outputs, 
            batch_labels, 
            batch_numerical_values, 
            batch_attention_mask
        )
        
        # 反向传播
        loss = loss_info['loss']
        loss.backward()
        self.optimizer.step()
        
        # 记录指标
        metrics = {
            'loss': loss.item(),
            'cls_loss': loss_info.get('cls_loss', 0),
            'reg_loss': loss_info.get('reg_loss', 0),
            'num_positions': loss_info.get('num_positions', 0)
        }
        
        # 记录门控相关的统计信息
        if hasattr(loss_info, 'gate_weights_mean'):
            metrics['avg_gate_weight'] = loss_info.get('gate_weights_mean', 0.0)
            if self.wandb_run:
                self.wandb_run.log({
                    'train/avg_gate_weight': metrics['avg_gate_weight'],
                    'train/gating_alpha': self.gating_alpha
                })
        
        if self.wandb_run:
            self.wandb_run.log({
                'train/loss': metrics['loss'],
                'train/cls_loss': metrics['cls_loss'],
                'train/reg_loss': metrics['reg_loss']
            })
        
        return metrics