"""
Trainer module for fine-tuning the Causal Language Model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..data.synthetic import TextWithNumbersGenerator
from ..utils.losses import CausalLMLoss

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
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = CausalLMLoss(
            num_classes=self.tokenizer.vocab_size,
            num_token_id=self.tokenizer.num_token_id,
            regression_weight=self.config.reg_loss_weight
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

    def _create_training_data(self, num_samples):
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
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, num_epochs, num_samples):
        """
        Run the training loop.
        
        Args:
            num_epochs (int): Number of epochs to train for.
            num_samples (int): Number of synthetic samples to generate for training.
        """
        self.model.train()
        dataloader = self._create_training_data(num_samples)
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
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
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = self.model.predict(batch_input_ids, batch_numerical_values, batch_attention_mask)
                    pred_tokens = predictions['cls_pred']
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