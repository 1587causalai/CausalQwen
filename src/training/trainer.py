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
    
    def __init__(self, model, tokenizer, device, config, learning_rate=1e-4, batch_size=16, wandb_run=None):
        """
        Initialize the Trainer.
        
        Args:
            model (nn.Module): The model to be trained.
            tokenizer: The tokenizer to use.
            device (torch.device): The device to train on.
            config (CausalLMConfig): The model's configuration object.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The batch size for training.
            wandb_run: An active Weights & Biases run object.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.batch_size = batch_size
        self.wandb_run = wandb_run
        
        # Calculate data statistics for initialization
        print("Pre-calculating data statistics for initialization...")
        temp_dataloader = self._create_training_data(1000, shuffle=False)
        all_target_values = []
        for batch in temp_dataloader:
            # batch_target_values is the 5th element in the batch tuple
            is_num_mask = (batch[3] == self.tokenizer.num_token_id)
            if is_num_mask.any():
                all_target_values.append(batch[4][is_num_mask])
        
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
        """Create synthetic training data."""
        print(f"Generating {num_samples} synthetic training samples...")
        generator = TextWithNumbersGenerator(seed=42)
        texts, true_values = generator.generate_text(num_samples=num_samples)
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        numerical_values = inputs['numerical_values']
        
        targets = torch.full((num_samples,), self.tokenizer.num_token_id, dtype=torch.long)
        target_values = torch.tensor(true_values, dtype=torch.float32)
        
        dataset = TensorDataset(input_ids, attention_mask, numerical_values, targets, target_values)
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
            total_correct = 0
            total_num_correct = 0
            total_num_samples = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i, batch in enumerate(pbar):
                batch = [t.to(self.device) for t in batch]
                batch_input_ids, batch_attention_mask, batch_numerical_values, batch_targets, batch_target_values = batch
                
                # Forward pass
                outputs = self.model(batch_input_ids, batch_numerical_values, batch_attention_mask)

                # Compute loss
                loss_dict = self.loss_fn(
                    outputs["cls_loc"], outputs["cls_scale"],
                    outputs["reg_loc"], outputs["reg_scale"],
                    batch_targets, batch_target_values
                )
                loss = loss_dict["loss"]
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # --- METRICS LOGGING ---
                total_loss += loss.item()
                
                # Classification metrics
                cls_probs = loss_dict["cls_probs"]
                pred_tokens = torch.argmax(cls_probs, dim=-1)
                
                # Overall accuracy
                total_correct += (pred_tokens == batch_targets).sum().item()
                total_samples += batch_targets.size(0)
                
                # <NUM> token accuracy
                num_mask = (batch_targets == self.tokenizer.num_token_id)
                num_acc = None
                if num_mask.any():
                    total_num_correct += (pred_tokens[num_mask] == batch_targets[num_mask]).sum().item()
                    total_num_samples += num_mask.sum().item()
                    num_acc = (pred_tokens[num_mask] == batch_targets[num_mask]).sum().item() / num_mask.sum().item()

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
                        reg_mae = torch.abs(outputs['reg_loc'][num_mask] - batch_target_values[num_mask]).mean().item()

                    log_data = {
                        "total_loss": loss.item(),
                        "cls_loss": loss_dict["cls_loss"].item(),
                        "reg_mae": reg_mae,
                        "units_mean_loc": outputs['causal_loc'].mean().item(),
                        "units_mean_scale": outputs['causal_scale'].mean().item(),
                        "ovr_prob_sum": cls_probs.sum(dim=1).mean().item(),
                        "num_accuracy": num_acc,
                        "accuracy": (pred_tokens == batch_targets).sum().item() / batch_targets.size(0)
                    }
                    
                    self.wandb_run.log({k: v for k, v in log_data.items() if v is not None}, step=global_step)
                
                global_step += 1
        
            avg_loss = total_loss / len(dataloader)
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

        return dataloader 