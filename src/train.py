"""
Training script for the causal language model.

This script implements the training loop and evaluation functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

from ..models.causal_lm import CausalLanguageModel
from ..utils.losses import CausalLMLoss
from ..utils.metrics import compute_combined_metrics
from ..data.dataset import SyntheticDataset, create_synthetic_dataloader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_synthetic_features: bool = True
):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): DataLoader for training data
        loss_fn (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to use
        use_synthetic_features (bool, optional): Whether to use synthetic features directly. 
                                                Defaults to True.
    
    Returns:
        dict: Dictionary containing training metrics
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        # Move batch to device
        if use_synthetic_features:
            # For synthetic data with pre-computed features
            features = batch["feature"].to(device)
            target_tokens = batch["target_token"].to(device)
            target_values = batch["target_value"].to(device)
            
            # Forward pass (directly using features)
            causal_loc, causal_scale = model.abduction_network(features)
            outputs = model.action_network(causal_loc, causal_scale)
            
        else:
            # For real data with input tokens
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_values = batch["numerical_values"].to(device)
            target_tokens = batch["target_ids"].to(device)
            target_values = batch["target_numerical_values"].to(device)
            
            # Forward pass
            outputs = model(input_ids, numerical_values, attention_mask)
        
        # Compute loss
        loss_dict = loss_fn(
            outputs["cls_loc"],
            outputs["cls_scale"],
            outputs["reg_loc"],
            outputs["reg_scale"],
            target_tokens,
            target_values
        )
        
        loss = loss_dict["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_cls_loss += loss_dict["cls_loss"].item()
        total_reg_loss += loss_dict["reg_loss"].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": loss.item(),
            "cls_loss": loss_dict["cls_loss"].item(),
            "reg_loss": loss_dict["reg_loss"].item()
        })
    
    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    return {
        "loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "reg_loss": avg_reg_loss
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_token_id: int,
    use_synthetic_features: bool = True
):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation data
        loss_fn (nn.Module): Loss function
        device (torch.device): Device to use
        num_token_id (int): Token ID for the <NUM> token
        use_synthetic_features (bool, optional): Whether to use synthetic features directly. 
                                                Defaults to True.
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    all_cls_preds = []
    all_reg_preds = []
    all_targets = []
    all_target_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            if use_synthetic_features:
                # For synthetic data with pre-computed features
                features = batch["feature"].to(device)
                target_tokens = batch["target_token"].to(device)
                target_values = batch["target_value"].to(device)
                
                # Forward pass (directly using features)
                causal_loc, causal_scale = model.abduction_network(features)
                outputs = model.action_network(causal_loc, causal_scale)
                predictions = model.action_network.predict(causal_loc, causal_scale)
                
            else:
                # For real data with input tokens
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                numerical_values = batch["numerical_values"].to(device)
                target_tokens = batch["target_ids"].to(device)
                target_values = batch["target_numerical_values"].to(device)
                
                # Forward pass
                outputs = model(input_ids, numerical_values, attention_mask)
                predictions = model.predict(input_ids, numerical_values, attention_mask)
            
            # Compute loss
            loss_dict = loss_fn(
                outputs["cls_loc"],
                outputs["cls_scale"],
                outputs["reg_loc"],
                outputs["reg_scale"],
                target_tokens,
                target_values
            )
            
            # Update metrics
            total_loss += loss_dict["loss"].item()
            total_cls_loss += loss_dict["cls_loss"].item()
            total_reg_loss += loss_dict["reg_loss"].item()
            num_batches += 1
            
            # Collect predictions and targets for metric computation
            all_cls_preds.append(predictions["cls_pred"].cpu())
            all_reg_preds.append(predictions["reg_pred"].cpu())
            all_targets.append(target_tokens.cpu())
            all_target_values.append(target_values.cpu())
    
    # Concatenate predictions and targets
    all_cls_preds = torch.cat(all_cls_preds)
    all_reg_preds = torch.cat(all_reg_preds)
    all_targets = torch.cat(all_targets)
    all_target_values = torch.cat(all_target_values)
    
    # Compute average loss metrics
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    # Compute performance metrics
    metrics = compute_combined_metrics(
        all_cls_preds, all_reg_preds, all_targets, all_target_values, num_token_id
    )
    
    # Add loss metrics
    metrics.update({
        "loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "reg_loss": avg_reg_loss
    })
    
    return metrics


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_token_id: int,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    use_synthetic_features: bool = True,
    save_path: Optional[str] = None,
    log_interval: int = 1
):
    """
    Train the model.
    
    Args:
        model (nn.Module): Model to train
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        num_token_id (int): Token ID for the <NUM> token
        num_epochs (int, optional): Number of epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        device (Optional[torch.device], optional): Device to use. Defaults to None.
        use_synthetic_features (bool, optional): Whether to use synthetic features directly. 
                                                Defaults to True.
        save_path (Optional[str], optional): Path to save the model. Defaults to None.
        log_interval (int, optional): Interval for logging. Defaults to 1.
    
    Returns:
        dict: Dictionary containing training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize loss function
    loss_fn = CausalLMLoss(model.vocab_size, num_token_id)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    
    # Initialize training history
    history = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_reg_loss": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_reg_loss": [],
        "val_accuracy": [],
        "val_reg_mae": []
    }
    
    # Initialize best validation loss
    best_val_loss = float("inf")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, loss_fn, optimizer, device, use_synthetic_features
        )
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_cls_loss"].append(train_metrics["cls_loss"])
        history["train_reg_loss"].append(train_metrics["reg_loss"])
        
        # Evaluate
        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate(
                model, val_dataloader, loss_fn, device, num_token_id, use_synthetic_features
            )
            
            # Update history
            history["val_loss"].append(val_metrics["loss"])
            history["val_cls_loss"].append(val_metrics["cls_loss"])
            history["val_reg_loss"].append(val_metrics["reg_loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_reg_mae"].append(val_metrics["reg_mae"])
            
            # Print metrics
            print(f"Validation metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Classification Loss: {val_metrics['cls_loss']:.4f}")
            print(f"  Regression Loss: {val_metrics['reg_loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  MAE: {val_metrics['reg_mae']:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(val_metrics["loss"])
            
            # Save best model
            if val_metrics["loss"] < best_val_loss and save_path is not None:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

            # Batch diagnostics
            if (epoch + 1) % log_interval == 0:
                print(f"--- Epoch {epoch+1} Diagnostics ---")
                is_num_mask = (all_targets == num_token_id)
                
                # Target Values
                if torch.any(is_num_mask):
                    target_reg_values = all_target_values[is_num_mask]
                    print("  Target Values (for regression):")
                    print(f"    - Count: {len(target_reg_values)}")
                    print(f"    - Mean: {target_reg_values.mean():.2f}")
                    print(f"    - Std: {target_reg_values.std():.2f}")
                    print(f"    - Range: [{target_reg_values.min():.2f}, {target_reg_values.max():.2f}]")

                # Model Predictions
                if torch.any(is_num_mask):
                    pred_locs = all_reg_preds[is_num_mask]
                    pred_scales = torch.exp(all_reg_preds[is_num_mask])
                    print("  Model Predictions (for regression):")
                    print(f"    - Predicted Loc (μ): Mean={pred_locs.mean().item():.4f}, Std={pred_locs.std().item():.4f}, Range=[{pred_locs.min().item():.4f}, {pred_locs.max().item():.4f}]")
                    print(f"    - Predicted Scale (γ): Mean={pred_scales.mean().item():.4f}, Std={pred_scales.std().item():.4f}, Range=[{pred_scales.min().item():.4f}, {pred_scales.max().item():.4f}]")
                
                # OvR Probability Sum Diagnostics
                cls_loc = all_cls_preds
                cls_scale = torch.exp(all_cls_preds)
                ovr_probs = 0.5 + (1 / torch.pi) * torch.atan(cls_loc / cls_scale)
                sum_ovr_probs = ovr_probs.sum(dim=1)
                
                print("  OvR Probability Sum (Σ Pk) Diagnostics:")
                print(f"    - Mean: {sum_ovr_probs.mean().item():.4f}")
                print(f"    - Std: {sum_ovr_probs.std().item():.4f}")
                print(f"    - Range: [{sum_ovr_probs.min().item():.4f}, {sum_ovr_probs.max().item():.4f}]")

                # Loss Components
                print("  Loss Components:")
                print(f"    - Classification Loss: {val_metrics['cls_loss']:.4f}")
                print(f"    - Regression Loss: {val_metrics['reg_loss']:.4f}")
    
    return history


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history (Dict[str, List[float]]): Training history
        save_path (Optional[str], optional): Path to save the plot. Defaults to None.
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot loss
    axs[0, 0].plot(history["train_loss"], label="Train")
    axs[0, 0].plot(history["val_loss"], label="Validation")
    axs[0, 0].set_title("Total Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    
    # Plot classification loss
    axs[0, 1].plot(history["train_cls_loss"], label="Train")
    axs[0, 1].plot(history["val_cls_loss"], label="Validation")
    axs[0, 1].set_title("Classification Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    
    # Plot regression loss
    axs[1, 0].plot(history["train_reg_loss"], label="Train")
    axs[1, 0].plot(history["val_reg_loss"], label="Validation")
    axs[1, 0].set_title("Regression Loss")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].legend()
    
    # Plot metrics
    axs[1, 1].plot(history["val_accuracy"], label="Accuracy")
    axs[1, 1].plot(history["val_reg_mae"], label="MAE")
    axs[1, 1].set_title("Validation Metrics")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    """
    Main function for training the causal language model.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set hyperparameters
    vocab_size = 1000
    num_token_id = 2
    hidden_size = 1024
    causal_dim = 64
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    # Create synthetic dataloaders
    train_dataloader = create_synthetic_dataloader(
        num_samples=10000,
        vocab_size=vocab_size,
        batch_size=batch_size,
        num_token_id=num_token_id,
        num_probability=0.2,
        hidden_size=hidden_size,
        seed=42
    )
    
    val_dataloader = create_synthetic_dataloader(
        num_samples=1000,
        vocab_size=vocab_size,
        batch_size=batch_size,
        num_token_id=num_token_id,
        num_probability=0.2,
        hidden_size=hidden_size,
        seed=43
    )
    
    # Create model
    model = CausalLanguageModel(
        vocab_size=vocab_size,
        num_token_id=num_token_id,
        hidden_size=hidden_size,
        causal_dim=causal_dim,
        use_mock_feature_network=True
    )
    
    # Train model
    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_token_id=num_token_id,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        use_synthetic_features=True,
        save_path="best_model.pt",
        log_interval=1
    )
    
    # Plot training history
    plot_training_history(history, save_path="training_history.png")


if __name__ == "__main__":
    main()

