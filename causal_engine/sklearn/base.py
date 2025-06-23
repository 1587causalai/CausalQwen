"""
Base classes for sklearn-style CausalEngine estimators
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from ..engine import CausalEngine
from ..networks import AbductionNetwork, ActionNetwork
from ..heads import ActivationHead, ActivationMode


class CausalEstimatorMixin(BaseEstimator, ABC):
    """
    CausalEngine估计器的基础混入类
    
    提供sklearn风格估计器的通用功能：
    - 数据验证和预处理
    - 训练循环管理
    - 模型状态管理
    - predict()接口的统一实现
    """
    
    def __init__(self, 
                 hidden_layer_sizes=(64, 32),
                 max_iter=1000,
                 learning_rate=0.001,
                 default_mode='compatible',
                 early_stopping=True,
                 validation_fraction=0.1,
                 random_state=None,
                 verbose=False):
        """
        初始化CausalEngine估计器
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(64, 32)
            MLP隐藏层结构，与sklearn MLPRegressor/Classifier兼容
        max_iter : int, default=1000
            最大训练迭代次数
        learning_rate : float, default=0.001
            学习率
        default_mode : str, default='compatible'
            默认预测模式 ('compatible', 'standard', 'causal', 'sampling')
        early_stopping : bool, default=True
            是否使用早停
        validation_fraction : float, default=0.1
            验证集比例（用于早停）
        random_state : int, default=None
            随机种子
        verbose : bool, default=False
            是否输出训练日志
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.default_mode = default_mode
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        
        # 训练状态
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.loss_curve_ = []
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_mlp_layers(self, input_size: int, output_size: int) -> nn.Module:
        """构建MLP隐藏层（与sklearn兼容的结构）"""
        layers = []
        prev_size = input_size
        
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        # 最后一层输出到指定维度
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _build_causal_engine(self, input_size: int, causal_size: int = None, 
                           activation_mode: ActivationMode = None) -> CausalEngine:
        """构建CausalEngine核心"""
        if causal_size is None:
            causal_size = max(32, input_size // 2)  # 智能默认
            
        output_size = self._get_output_size()
        activation_mode = activation_mode or self._get_activation_mode()
        
        # 根据激活模式决定激活模式字符串
        if activation_mode == ActivationMode.REGRESSION:
            activation_modes = "regression"
        elif activation_mode == ActivationMode.CLASSIFICATION:
            activation_modes = "classification"  
        else:
            activation_modes = "classification"  # 默认
            
        return CausalEngine(
            hidden_size=input_size,
            vocab_size=output_size,
            causal_size=causal_size,
            activation_modes=activation_modes
        )
    
    def _validate_data(self, X, y=None, reset=True):
        """数据验证（sklearn风格）"""
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float32)
            
        if reset:
            self.n_features_in_ = X.shape[1]
            
        return X, y
    
    def _setup_training(self, X, y):
        """设置训练环境"""
        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
        # 数据分割（早停需要验证集）
        if self.early_stopping and len(X) > 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state
            )
            return X_train, X_val, y_train, y_val
        else:
            return X, None, y, None
            
    def _to_tensor(self, data):
        """转换为torch tensor"""
        if isinstance(data, np.ndarray):
            return torch.FloatTensor(data).to(self.device)
        return data
        
    def _from_tensor(self, tensor):
        """从torch tensor转换回numpy"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    @abstractmethod
    def _get_output_size(self) -> int:
        """获取输出维度（子类实现）"""
        pass
        
    @abstractmethod 
    def _get_activation_mode(self) -> ActivationMode:
        """获取激活模式（子类实现）"""
        pass
    
    @abstractmethod
    def _compute_loss(self, predictions, targets):
        """计算损失（子类实现）"""
        pass
    
    def _forward(self, X_batch):
        """前向传播"""
        # 通过MLP隐藏层
        hidden_features = self.model['hidden_layers'](X_batch)
        
        # CausalEngine期望3维输入: [batch_size, seq_len, features]
        # 对于sklearn风格的2维输入，我们添加seq_len=1维度
        if hidden_features.dim() == 2:
            hidden_features = hidden_features.unsqueeze(1)  # [batch_size, 1, features]
        
        # 通过CausalEngine
        predictions = self.model['causal_engine'](hidden_features)
        
        # 移除seq_len维度以匹配sklearn期望的2维输出
        if isinstance(predictions, dict):
            for key, value in predictions.items():
                if torch.is_tensor(value) and value.dim() == 3 and value.size(1) == 1:
                    predictions[key] = value.squeeze(1)  # [batch_size, features]
        elif torch.is_tensor(predictions) and predictions.dim() == 3 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
            
        return predictions
        
    def _train_step(self, X_batch, y_batch, optimizer):
        """单步训练"""
        self.model.train()
        optimizer.zero_grad()
        
        # 前向传播
        predictions = self._forward(X_batch)
        loss = self._compute_loss(predictions, y_batch)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _validate_step(self, X_val, y_val):
        """验证步骤"""
        self.model.eval()
        with torch.no_grad():
            predictions = self._forward(X_val)
            val_loss = self._compute_loss(predictions, y_val)
        return val_loss.item()
    
    def _should_stop_early(self, val_losses, patience=20, min_delta=1e-4):
        """早停判断"""
        if len(val_losses) < patience:
            return False
            
        best_loss = min(val_losses[:-patience])
        recent_loss = val_losses[-1]
        
        return recent_loss > best_loss - min_delta
    
    def predict(self, X, mode=None):
        """
        统一预测接口
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本
        mode : str, optional
            预测模式，如果为None则使用default_mode
            - 'compatible': sklearn兼容模式
            - 'standard': 标准CausalEngine输出
            - 'causal': 纯因果推理
            - 'sampling': 探索性预测
            
        Returns:
        --------
        predictions : array-like or dict
            预测结果，格式取决于mode
        """
        check_is_fitted(self)
        X, _ = self._validate_data(X, reset=False)
        X_tensor = self._to_tensor(X)
        
        if mode is None:
            mode = self.default_mode
            
        self.model.eval()
        with torch.no_grad():
            if mode == 'compatible':
                # sklearn兼容模式：返回简单数值
                predictions = self._predict_compatible(X_tensor)
            else:
                # 其他模式：返回丰富信息
                predictions = self._predict_advanced(X_tensor, mode)
                
        return self._from_tensor(predictions) if torch.is_tensor(predictions) else predictions
    
    @abstractmethod
    def _predict_compatible(self, X_tensor):
        """兼容模式预测（子类实现）"""
        pass
        
    @abstractmethod
    def _predict_advanced(self, X_tensor, mode):
        """高级模式预测（子类实现）"""
        pass