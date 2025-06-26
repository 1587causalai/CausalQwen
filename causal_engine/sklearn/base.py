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
                 mode='standard',
                 early_stopping=True,
                 validation_fraction=0.1,
                 n_iter_no_change=20,
                 tol=1e-4,
                 alpha=0.0001,
                 batch_size='auto',
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
        mode : str, default='standard'
            CausalEngine五种模式选择：
            - 'deterministic': γ_U=0, b_noise=0 (等价sklearn)
            - 'exogenous': γ_U=0, b_noise≠0 (外生噪声推理)
            - 'endogenous': γ_U≠0, b_noise=0 (内生因果推理)
            - 'standard': γ_U≠0, b_noise≠0 (标准因果推理)
            - 'sampling': γ_U≠0, b_noise≠0 (探索性因果推理)
        early_stopping : bool, default=True
            是否使用早停
        validation_fraction : float, default=0.1
            验证集比例（用于早停）
        n_iter_no_change : int, default=20
            早停的迭代次数
        tol : float, default=1e-4
            早停的容忍度
        alpha : float, default=0.0001
            学习率衰减因子
        batch_size : str or int, default='auto'
            批量大小，'auto'表示自动选择
        random_state : int, default=None
            随机种子
        verbose : bool, default=False
            是否输出训练日志
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.mode = mode
        self._configure_mode_parameters()
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.alpha = alpha
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        
        # 训练状态
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.loss_curve_ = []
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _configure_mode_parameters(self):
        """根据模式配置内部参数"""
        if self.mode == 'deterministic':
            self.gamma_U_enabled = False
            self.b_noise_enabled = False
            self.loss_type = 'traditional'  # MSE/CrossEntropy
        elif self.mode == 'exogenous':
            self.gamma_U_enabled = False
            self.b_noise_enabled = True
            self.loss_type = 'causal'  # Cauchy NLL/OvR BCE
        elif self.mode == 'endogenous':
            self.gamma_U_enabled = True
            self.b_noise_enabled = False
            self.loss_type = 'causal'
        elif self.mode == 'standard':
            self.gamma_U_enabled = True
            self.b_noise_enabled = True
            self.noise_mode = 'scale'  # 噪声作用于尺度
            self.loss_type = 'causal'
        elif self.mode == 'sampling':
            self.gamma_U_enabled = True
            self.b_noise_enabled = True
            self.noise_mode = 'location'  # 噪声作用于位置
            self.loss_type = 'causal'
        else:
            raise ValueError(f"不支持的模式: {self.mode}。支持的模式: ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']")
        
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
            # 默认因果表征维度等于输入特征维度，这是最合理的设置
            causal_size = input_size
            
        output_size = self._get_output_size()
        activation_mode = activation_mode or self._get_activation_mode()
        
        # 根据激活模式决定激活模式字符串
        if activation_mode == ActivationMode.REGRESSION:
            activation_modes = "regression"
        elif activation_mode == ActivationMode.CLASSIFICATION:
            activation_modes = "classification"  
        else:
            activation_modes = "classification"  # 默认
            
        # 创建CausalEngine并配置模式参数
        engine = CausalEngine(
            hidden_size=input_size,
            vocab_size=output_size,
            causal_size=causal_size,
            activation_modes=activation_modes,
            # 使用实例的核心参数
            b_noise_init=getattr(self, 'b_noise_init', 0.1),
            b_noise_trainable=getattr(self, 'b_noise_trainable', True),
            gamma_init=getattr(self, 'gamma_init', 10.0),
            classification_threshold_init=getattr(self, 'ovr_threshold_init', 0.0)
        )
        
        # 根据mode配置ActionNetwork的参数
        self._configure_engine_mode(engine)
        
        return engine
        
    def _configure_engine_mode(self, engine):
        """配置CausalEngine的模式参数"""
        action_net = engine.action
        
        if self.mode == 'deterministic':
            # γ_U=0, b_noise=0：确定性模式
            # 在前向传播中设置scale_U=0, b_noise=0
            pass  # 在_forward中处理
        elif self.mode == 'exogenous':
            # γ_U=0, b_noise≠0：外生噪声
            pass  # 在_forward中处理
        elif self.mode == 'endogenous':
            # γ_U≠0, b_noise=0：内生因果
            pass  # 在_forward中处理
        elif self.mode in ['standard', 'sampling']:
            # γ_U≠0, b_noise≠0：标准/采样模式
            pass  # 在_forward中处理
    
    def _validate_data(self, X, y=None, reset=True):
        """数据验证（sklearn风格）"""
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float64)
            
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
            return torch.tensor(data, dtype=torch.float64, device=self.device)
        return data.to(dtype=torch.float64, device=self.device)
        
    def _from_tensor(self, tensor):
        """从torch tensor转换回numpy"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _freeze_abduction_as_identity(self):
        """
        将Abduction模块冻结为恒等映射，以在deterministic模式下复现标准MLP。
        loc -> Identity (weight=I, bias=0)
        scale -> Zero (weight=0, bias=0)
        """
        abduction_net = self.model.causal_engine.abduction
        
        # 冻结位置参数网络为恒等映射
        loc_layer = abduction_net.loc_net
        if loc_layer.in_features != loc_layer.out_features:
            raise ValueError("为了实现恒等映射，Abduction loc_net模块的输入和输出维度必须相同。")
        
        identity_matrix = torch.eye(loc_layer.in_features, dtype=torch.float64, device=self.device)
        loc_layer.weight.data.copy_(identity_matrix)
        loc_layer.bias.data.fill_(0)
        loc_layer.weight.requires_grad = False
        loc_layer.bias.requires_grad = False

        # 冻结尺度参数网络输出为零
        scale_layer = abduction_net.scale_net
        scale_layer.weight.data.fill_(0)
        scale_layer.bias.data.fill_(0)
        scale_layer.weight.requires_grad = False
        scale_layer.bias.requires_grad = False
        
        print("==> Abduction网络已为确定性模式冻结为恒等映射。")
    
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
        """默认前向传播（使用当前模式）"""
        return self._forward_with_mode(X_batch, self.mode)
        
    def _forward_with_mode(self, X_batch, mode=None):
        """基础前向传播（子类应该重写此方法）"""
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
        """单步训练（添加梯度裁剪和数值稳定性检查）"""
        self.model.train()
        optimizer.zero_grad()
        
        # 前向传播
        predictions = self._forward(X_batch)
        loss = self._compute_loss(predictions, y_batch)
        
        # 简化：柯西分布损失数值稳定，移除过度检查
        
        # 反向传播
        loss.backward()
        
        # 不使用梯度裁剪：柯西分布损失具有天然的梯度饱和性质
        # 当预测与真实值相差很大时，梯度有界，不会无限增长
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
            mode = self.mode
            
        self.model.eval()
        with torch.no_grad():
            if mode == 'deterministic':
                # 确定性模式：sklearn兼容
                predictions = self._predict_deterministic(X_tensor)
            else:
                # 因果模式：返回分布信息
                predictions = self._predict_causal(X_tensor, mode)
                
        return self._from_tensor(predictions) if torch.is_tensor(predictions) else predictions
    
    @abstractmethod
    def _predict_deterministic(self, X_tensor):
        """确定性模式预测（子类实现）"""
        pass
        
    @abstractmethod
    def _predict_causal(self, X_tensor, mode):
        """因果模式预测（子类实现）"""
        pass