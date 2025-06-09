"""
Trainer module for fine-tuning the Causal Language Model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..data.synthetic import TextWithNumbersGenerator
from ..utils.losses import CausalLMLoss, compute_ovr_probabilities

class Trainer:
    """Handles the training loop, optimizer, and data loading for fine-tuning."""
    
    def __init__(self, model, tokenizer, device, config, learning_rate=1e-4, batch_size=16):
        """
        Initialize the Trainer.
        
        Args:
            model (nn.Module): The model to be trained.
            tokenizer: The tokenizer to use.
            device (torch.device): The device to train on.
            config (CausalLMConfig): The model's configuration object.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The batch size for training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.batch_size = batch_size
        
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
            self.num_target_mean = all_target_values.mean().item()
            self.num_target_std = all_target_values.std().item()
            print(f"  - Calculated Mean: {self.num_target_mean:.2f}")
            print(f"  - Calculated Std Dev: {self.num_target_std:.2f}")
        else:
            self.num_target_mean = 0.0
            self.num_target_std = 1.0
            print("  - No numerical targets found. Using default stats (Mean=0, Std=1).")
        
        # Initialize model weights using the new strategy
        if hasattr(self.model, 'init_weights'):
            self.model.init_weights(self.num_target_mean, self.num_target_std)
        
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
        # Data is now created once for stats, so we reuse it or recreate it for training
        print("Creating final training data loader...")
        dataloader = self._create_training_data(num_samples, shuffle=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i, batch in enumerate(pbar):
                batch = [t.to(self.device) for t in batch]
                batch_input_ids, batch_attention_mask, batch_numerical_values, batch_targets, batch_target_values = batch
                
                # Forward pass
                outputs = self.model(batch_input_ids, batch_numerical_values, batch_attention_mask)
                
                # --- BEGIN DIAGNOSTIC LOGGING ---
                if epoch == 0 and i < 5:
                    print(f"\n--- Batch {i+1} Diagnostics ---")
                    
                    # 1. Analyze target values
                    is_num_mask = (batch_targets == self.tokenizer.num_token_id)
                    if is_num_mask.any():
                        reg_targets = batch_target_values[is_num_mask]
                        print(f"  Target Values (for regression):")
                        print(f"    - Count: {len(reg_targets)}")
                        print(f"    - Mean: {reg_targets.mean().item():.2f}")
                        print(f"    - Std: {reg_targets.std().item():.2f}")
                        print(f"    - Range: [{reg_targets.min().item():.2f}, {reg_targets.max().item():.2f}]")
                    else:
                        print("  Target Values: No regression targets in this batch.")

                    # 2. Analyze model's regression predictions
                    if torch.any(is_num_mask):
                        pred_locs = outputs['reg_loc'][is_num_mask]
                        pred_scales = torch.exp(outputs['reg_scale'][is_num_mask])
                        print("  Model Predictions (for regression):")
                        print(f"    - Predicted Loc (μ): Mean={pred_locs.mean().item():.4f}, Std={pred_locs.std().item():.4f}, Range=[{pred_locs.min().item():.4f}, {pred_locs.max().item():.4f}]")
                        print(f"    - Predicted Scale (γ): Mean={pred_scales.mean().item():.4f}, Std={pred_scales.std().item():.4f}, Range=[{pred_scales.min().item():.4f}, {pred_scales.max().item():.4f}]")

                    # 3. OvR Probability Sum Diagnostics
                    cls_loc = outputs['cls_loc']
                    cls_scale = torch.exp(outputs['cls_scale'])
                    threshold = self.loss_fn.cls_loss_fn.threshold
                    ovr_probs = 0.5 + (1 / torch.pi) * torch.atan((cls_loc - threshold) / cls_scale)
                    sum_ovr_probs = ovr_probs.sum(dim=1)
                    
                    print("  OvR Probability Sum (Σ Pk) Diagnostics:")
                    print(f"    - Mean: {sum_ovr_probs.mean().item():.4f}")
                    print(f"    - Std: {sum_ovr_probs.std().item():.4f}")
                    print(f"    - Range: [{sum_ovr_probs.min().item():.4f}, {sum_ovr_probs.max().item():.4f}]")
                # --- END DIAGNOSTIC LOGGING ---

                # Compute loss
                loss_dict = self.loss_fn(
                    outputs["cls_loc"], outputs["cls_scale"],
                    outputs["reg_loc"], outputs["reg_scale"],
                    batch_targets, batch_target_values
                )
                loss = loss_dict["loss"]
                
                # --- BEGIN DIAGNOSTIC LOGGING (Loss) ---
                if epoch == 0 and i < 5:
                    if "unweighted_reg_loss" in loss_dict and "gated_reg_loss" in loss_dict:
                        unweighted_loss = loss_dict['unweighted_reg_loss'].mean().item()
                        gated_loss = loss_dict['gated_reg_loss'].mean().item()
                        cls_loss = loss_dict['cls_loss'].mean().item()
                        num_prob = loss_dict.get('num_prob', torch.tensor(0.0)).mean().item()
                        
                        print(f"  Loss Components:")
                        print(f"    - Classification Loss: {cls_loss:.4f}")
                        print(f"    - Unweighted Regression NLL Loss: {unweighted_loss:.4f}")
                        print(f"    - <NUM> Probability (Gate): {num_prob:.4f}")
                        print(f"    - Gated Regression Loss (final component): {gated_loss:.4f}")
                        print(f"  ---------------------------\n")
                # --- END DIAGNOSTIC LOGGING (Loss) ---

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Calculate accuracy using the pre-computed probabilities from the loss function
                cls_probs = loss_dict["cls_probs"]
                pred_tokens = torch.argmax(cls_probs, dim=-1)
                total_correct += (pred_tokens == batch_targets).sum().item()
                
                total_samples += batch_targets.size(0)
                total_loss += loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{total_correct / total_samples:.4f}'
                })
        
            avg_loss = total_loss / len(dataloader)
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch+1} Complete: Avg Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}") 