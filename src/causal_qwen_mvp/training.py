"""
CausalQwen MVP: 训练模块实现
实现基础训练循环、损失计算和验证逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import numpy as np
from tqdm import tqdm
import logging

from .models import CausalQwenMVPForCausalLM, CausalQwen2Config


class CausalTrainer:
    """CausalQwen训练器"""
    
    def __init__(
        self,
        model: CausalQwenMVPForCausalLM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        gradient_clip_val: float = 1.0,
        log_interval: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.log_interval = log_interval
        
        # 移动模型到设备
        self.model.to(device)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练 - 占位实现"""
        # TODO: 实现更sophisticated的训练逻辑
        self.model.train()
        
        # 提取数据
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
        
        loss = outputs.loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        # 记录训练指标
        metrics = {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': self._get_grad_norm(),
        }
        
        return metrics
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步验证 - 占位实现"""
        # TODO: 添加更多验证指标
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            
            loss = outputs.loss
            
            # 计算额外指标
            metrics = {
                'val_loss': loss.item(),
                'val_perplexity': torch.exp(loss).item(),
            }
            
            # TODO: 添加因果推理相关的验证指标
            
        return metrics
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        epoch_metrics = []
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 训练步骤
            step_metrics = self.training_step(batch)
            epoch_metrics.append(step_metrics)
            
            self.global_step += 1
            
            # 更新进度条
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{step_metrics['loss']:.4f}",
                    'lr': f"{step_metrics['lr']:.2e}",
                    'step': self.global_step
                })
                
                # 日志记录
                self.logger.info(
                    f"Step {self.global_step}: loss={step_metrics['loss']:.4f}, "
                    f"lr={step_metrics['lr']:.2e}"
                )
        
        # 聚合epoch指标
        avg_metrics = self._aggregate_metrics(epoch_metrics)
        return avg_metrics
    
    def validate_epoch(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        val_metrics = []
        
        pbar = tqdm(val_dataloader, desc="Validation")
        for batch in pbar:
            step_metrics = self.validation_step(batch)
            val_metrics.append(step_metrics)
            
            pbar.set_postfix({
                'val_loss': f"{step_metrics['val_loss']:.4f}",
                'val_ppl': f"{step_metrics['val_perplexity']:.2f}"
            })
        
        # 聚合验证指标
        avg_metrics = self._aggregate_metrics(val_metrics)
        return avg_metrics
    
    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        save_path: Optional[str] = None
    ):
        """完整训练流程"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练阶段
            train_metrics = self.train_epoch(train_dataloader)
            self.logger.info(f"Epoch {epoch} train: {train_metrics}")
            
            # 验证阶段
            if val_dataloader:
                val_metrics = self.validate_epoch(val_dataloader)
                self.logger.info(f"Epoch {epoch} val: {val_metrics}")
                
                # 保存最佳模型
                current_loss = val_metrics['val_loss']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    if save_path:
                        self.save_checkpoint(save_path, is_best=True)
                        self.logger.info(f"New best model saved: val_loss={current_loss:.4f}")
            
            # 定期保存检查点
            if save_path and (epoch + 1) % 5 == 0:
                checkpoint_path = f"{save_path}_epoch_{epoch+1}"
                self.save_checkpoint(checkpoint_path)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存模型检查点 - 占位实现"""
        # TODO: 实现完整的检查点保存逻辑
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.model.config.to_dict(),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = f"{path}_best.pt" if is_best else f"{path}.pt"
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, path: str):
        """加载模型检查点 - 占位实现"""
        # TODO: 实现完整的检查点加载逻辑
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: step={self.global_step}, epoch={self.epoch}")
    
    def _get_grad_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        param_count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合指标"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = np.mean(values)
        
        return aggregated


class LossComputer:
    """损失计算器"""
    
    def __init__(self, config: CausalQwen2Config):
        self.config = config
    
    def compute_causal_loss(
        self,
        loc_S: torch.Tensor,
        scale_S: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算因果损失 - 占位实现"""
        # TODO: 实现更sophisticated的因果损失
        
        # 简化的OvR损失
        batch_size, seq_len, vocab_size = loc_S.shape
        
        # 计算OvR概率
        # P(y_k=1) = P(Cauchy(loc_S_k, scale_S_k) > 0)  # threshold=0
        probs = 0.5 + (1/torch.pi) * torch.atan(loc_S / scale_S)
        
        # 构造目标
        targets = F.one_hot(labels, num_classes=vocab_size).float()
        
        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # 应用mask（如果提供）
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss  # V2简化：移除历史遗留的权重配置
    
    def compute_regularization_loss(self, model: CausalQwenMVPForCausalLM) -> torch.Tensor:
        """计算正则化损失 - 占位实现"""
        # TODO: 添加因果模型特定的正则化
        reg_loss = 0.0
        
        # L2正则化
        for param in model.abduction_network.parameters():
            reg_loss += torch.norm(param, p=2)
        for param in model.action_network.parameters():
            reg_loss += torch.norm(param, p=2)
        
        return reg_loss * 1e-5  # 正则化权重


class TrainingValidator:
    """训练验证工具"""
    
    def __init__(self, model: CausalQwenMVPForCausalLM):
        self.model = model
    
    def validate_gradient_flow(self) -> Dict[str, float]:
        """验证梯度流动 - 占位实现"""
        # TODO: 实现梯度流动分析
        grad_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats[name] = grad_norm
        
        return grad_stats
    
    def validate_weight_updates(self) -> Dict[str, float]:
        """验证权重更新 - 占位实现"""
        # TODO: 实现权重更新分析
        weight_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_norm = param.data.norm(2).item()
                weight_stats[name] = weight_norm
        
        return weight_stats
    
    def check_training_stability(self, loss_history: List[float], window=100) -> Dict[str, Any]:
        """检查训练稳定性 - 占位实现"""
        # TODO: 实现更comprehensive的稳定性检查
        if len(loss_history) < window:
            return {'stable': True, 'reason': 'insufficient_data'}
        
        recent_losses = loss_history[-window:]
        
        # 检查损失是否发散
        if any(loss > 1000 or np.isnan(loss) for loss in recent_losses):
            return {'stable': False, 'reason': 'loss_diverged'}
        
        # 检查损失是否停滞
        loss_std = np.std(recent_losses)
        if loss_std < 1e-6:
            return {'stable': False, 'reason': 'loss_stagnated'}
        
        return {'stable': True, 'loss_std': loss_std} 