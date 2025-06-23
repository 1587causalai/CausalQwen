"""
MLPCausalRegressor - sklearn风格的因果回归器
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Dict, Any
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils.validation import check_is_fitted

from .base import CausalEstimatorMixin
from ..engine import CausalEngine, CauchyMath
from ..heads import ActivationMode


class MLPCausalRegressor(CausalEstimatorMixin, RegressorMixin):
    """
    MLP因果回归器 - sklearn风格接口
    
    这是一个drop-in替代sklearn MLPRegressor的因果回归器。
    它保持了sklearn的简单易用性，同时引入了CausalEngine的强大功能：
    
    核心优势：
    1. 噪声鲁棒性：对标签噪声具有天然免疫力
    2. 分布建模：输出完整的预测分布而非单点估计
    3. 统一接口：支持多种预测模式，从简单到高级
    4. sklearn兼容：无缝集成到现有sklearn工作流
    
    数学原理：
    - 传统回归：ŷ = f(X)，确定性输出
    - 因果回归：Y ~ Cauchy(μ, γ)，概率分布输出
    - 兼容模式：提取μ作为点估计，保持sklearn兼容性
    
    Examples:
    ---------
    >>> from causal_engine.sklearn import MLPCausalRegressor
    >>> reg = MLPCausalRegressor()
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)  # sklearn兼容
    >>> distributions = reg.predict(X_test, mode='standard')  # 分布输出
    """
    
    def __init__(self, 
                 hidden_layer_sizes=(64, 32),
                 max_iter=1000,
                 learning_rate=0.001,
                 default_mode='compatible',
                 causal_size=None,
                 early_stopping=True,
                 validation_fraction=0.1,
                 n_iter_no_change=20,
                 tol=1e-4,
                 random_state=None,
                 verbose=False):
        """
        初始化MLPCausalRegressor
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(64, 32)
            MLP隐藏层结构，格式与sklearn MLPRegressor相同
        max_iter : int, default=1000
            最大训练迭代次数
        learning_rate : float, default=0.001
            学习率
        default_mode : str, default='compatible'
            默认预测模式
            - 'compatible': 返回位置参数μ，sklearn兼容
            - 'standard': 返回完整Cauchy分布
            - 'causal': 纯因果推理（无外生噪声）
            - 'sampling': 探索性预测
        causal_size : int, optional
            因果表征维度，默认为input_size//2
        early_stopping : bool, default=True
            是否启用早停
        validation_fraction : float, default=0.1
            验证集比例
        n_iter_no_change : int, default=20
            早停耐心值
        tol : float, default=1e-4
            早停容忍度
        random_state : int, optional
            随机种子
        verbose : bool, default=False
            是否输出训练日志
        """
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate=learning_rate,
            default_mode=default_mode,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            verbose=verbose
        )
        
        self.causal_size = causal_size
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        
        # 回归特定属性
        self.n_outputs_ = None
        
    def _get_output_size(self) -> int:
        """回归任务输出维度为1"""
        return 1
        
    def _get_activation_mode(self) -> ActivationMode:
        """回归任务使用回归激活"""
        return ActivationMode.REGRESSION
        
    def _build_model(self, input_size: int):
        """构建完整模型"""
        causal_size = self.causal_size or max(32, input_size // 2)
        
        # 构建标准MLP隐藏层
        self.hidden_layers = self._build_mlp_layers(input_size, causal_size)
        
        # 构建CausalEngine（输入是MLP输出）
        self.causal_engine = self._build_causal_engine(
            input_size=causal_size,
            causal_size=causal_size
        )
        
        # 模型是MLP隐藏层和CausalEngine的组合
        self.model = nn.ModuleDict({
            'hidden_layers': self.hidden_layers,
            'causal_engine': self.causal_engine
        }).to(self.device)
        
    def _compute_loss(self, predictions, targets):
        """计算Cauchy负对数似然损失"""
        if isinstance(predictions, dict):
            # CausalEngine输出格式检查
            if 'output' in predictions:
                # 激活后的输出（回归）
                loc = predictions['output'].squeeze()
                # 对于回归，我们还可以从原始分布中获取scale
                if 'scale_S' in predictions:
                    scale = predictions['scale_S'].squeeze()
                else:
                    scale = torch.ones_like(loc) * 0.1
            elif 'loc_S' in predictions:
                # 原始决策分布
                loc = predictions['loc_S'].squeeze()
                scale = predictions['scale_S'].squeeze()
            else:
                # 兜底：使用第一个可用tensor作为loc
                for key, value in predictions.items():
                    if torch.is_tensor(value):
                        loc = value.squeeze()
                        scale = torch.ones_like(loc) * 0.1
                        break
        else:
            # 简化输出格式
            loc = predictions.squeeze()
            scale = torch.ones_like(loc) * 0.1  # 默认尺度
            
        targets = targets.squeeze()
        
        # 确保scale为正数
        scale = torch.clamp(scale, min=1e-6)
        
        # Cauchy负对数似然损失
        # -log p(y|μ,γ) = log(π) + log(γ) + log(1 + ((y-μ)/γ)²)
        cauchy_nll = (
            torch.log(torch.tensor(torch.pi, device=loc.device)) + 
            torch.log(scale) + 
            torch.log(1 + ((targets - loc) / scale) ** 2)
        )
        
        return cauchy_nll.mean()
        
    def fit(self, X, y, sample_weight=None):
        """
        训练模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练输入
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            训练目标
        sample_weight : array-like, shape (n_samples,), optional
            样本权重（暂不支持）
            
        Returns:
        --------
        self : object
            返回self以支持方法链
        """
        # 数据验证
        X, y = self._validate_data(X, y, reset=True)
        
        # 设置训练环境
        X_train, X_val, y_train, y_val = self._setup_training(X, y)
        
        # 构建模型
        self._build_model(X.shape[1])
        
        # 设置优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 转换数据
        X_train_tensor = self._to_tensor(X_train)
        y_train_tensor = self._to_tensor(y_train)
        
        if X_val is not None:
            X_val_tensor = self._to_tensor(X_val)
            y_val_tensor = self._to_tensor(y_val)
        
        # 训练循环
        self.loss_curve_ = []
        val_losses = []
        no_improvement_count = 0
        
        for epoch in range(self.max_iter):
            # 训练步骤
            train_loss = self._train_step(X_train_tensor, y_train_tensor, optimizer)
            self.loss_curve_.append(train_loss)
            
            # 验证步骤
            if X_val is not None:
                val_loss = self._validate_step(X_val_tensor, y_val_tensor)
                val_losses.append(val_loss)
                
                # 早停检查
                if self.early_stopping:
                    if len(val_losses) > self.n_iter_no_change:
                        recent_improvement = (
                            min(val_losses[-self.n_iter_no_change:]) < 
                            val_losses[-self.n_iter_no_change-1] - self.tol
                        )
                        if not recent_improvement:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch}")
                            break
            
            # 日志输出
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {train_loss:.6f}")
                
        self.n_outputs_ = 1
        self.is_fitted_ = True
        return self
        
    def _predict_compatible(self, X_tensor):
        """兼容模式：返回位置参数作为点估计"""
        predictions = self._forward(X_tensor)
        
        # CausalEngine的输出格式
        if isinstance(predictions, dict):
            # 优先使用激活后的输出
            if 'output' in predictions:
                return predictions['output'].squeeze()
            # 否则使用原始决策分布位置
            elif 'loc_S' in predictions:
                return predictions['loc_S'].squeeze()
            else:
                # 兜底：返回第一个可用的数值
                for key, value in predictions.items():
                    if torch.is_tensor(value):
                        return value.squeeze()
        else:
            return predictions.squeeze()
            
    def _predict_advanced(self, X_tensor, mode):
        """高级模式：返回完整分布信息"""
        predictions = self._forward(X_tensor)
        
        # 返回结构化输出
        if isinstance(predictions, dict):
            # 提取主要预测值
            main_pred = None
            if 'output' in predictions:
                main_pred = predictions['output'].squeeze()
            elif 'loc_S' in predictions:
                main_pred = predictions['loc_S'].squeeze()
                
            return {
                'predictions': main_pred,
                'distributions': predictions,
                'mode': mode
            }
        else:
            return {
                'predictions': predictions.squeeze(),
                'mode': mode
            }
    
    def score(self, X, y, sample_weight=None):
        """
        返回R²分数
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试输入
        y : array-like, shape (n_samples,)
            真实目标值
        sample_weight : array-like, shape (n_samples,), optional
            样本权重（暂不支持）
            
        Returns:
        --------
        score : float
            R²分数
        """
        predictions = self.predict(X, mode='compatible')
        return r2_score(y, predictions, sample_weight=sample_weight)
        
    def predict_with_uncertainty(self, X):
        """
        预测并返回不确定性信息
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本
            
        Returns:
        --------
        predictions : dict
            包含均值、不确定性和完整分布的字典
        """
        return self.predict(X, mode='standard')
        
    @property
    def feature_importances_(self):
        """特征重要性（基于梯度的简单实现）"""
        check_is_fitted(self)
        
        # 简单实现：基于第一层权重的L2范数
        if hasattr(self.hidden_layers, '0'):
            first_layer = self.hidden_layers[0]
            if isinstance(first_layer, nn.Linear):
                weights = first_layer.weight.data.detach().cpu().numpy()
                return np.linalg.norm(weights, axis=0)
        
        return np.ones(self.n_features_in_)  # 默认返回