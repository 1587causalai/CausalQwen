"""
MLPCausalClassifier - sklearn风格的因果分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Dict, Any
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels

from .base import CausalEstimatorMixin
from ..engine import CausalEngine, CauchyMath
from ..heads import ActivationMode


class MLPCausalClassifier(CausalEstimatorMixin, ClassifierMixin):
    """
    MLP因果分类器 - sklearn风格接口
    
    这是一个drop-in替代sklearn MLPClassifier的因果分类器。
    采用CausalEngine的词元激活机制和OvR（One-vs-Rest）策略，
    实现对标签噪声的天然鲁棒性。
    
    核心优势：
    1. 标签噪声鲁棒性：OvR独立激活，噪声不会传播
    2. 数学创新：Cauchy分布 + arctan激活函数
    3. 双概率策略：Softmax兼容 + OvR原生概率
    4. sklearn完全兼容：无缝替换MLPClassifier
    
    数学原理：
    - 传统分类：P_k = exp(z_k) / Σ exp(z_j)  (Softmax)
    - 因果分类：P_k = 1/2 + (1/π)arctan(μ_k/γ_k)  (OvR独立激活)
    - 决策策略：ŷ = argmax(P_k)
    
    Examples:
    ---------
    >>> from causal_engine.sklearn import MLPCausalClassifier
    >>> clf = MLPCausalClassifier()
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)  # 类别标签
    >>> probabilities = clf.predict_proba(X_test)  # 概率分布
    >>> distributions = clf.predict(X_test, mode='standard')  # 激活分布
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
        初始化MLPCausalClassifier
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(64, 32)
            MLP隐藏层结构，格式与sklearn MLPClassifier相同
        max_iter : int, default=1000
            最大训练迭代次数
        learning_rate : float, default=0.001
            学习率
        default_mode : str, default='compatible'
            默认预测模式
            - 'compatible': 返回类别标签，sklearn兼容
            - 'standard': 返回激活概率和分布
            - 'causal': 纯因果推理
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
        
        # 分类特定属性
        self.classes_ = None
        self.n_classes_ = None
        
    def _get_output_size(self) -> int:
        """分类任务输出维度等于类别数"""
        return self.n_classes_ if hasattr(self, 'n_classes_') else 2
        
    def _get_activation_mode(self) -> ActivationMode:
        """分类任务使用分类激活"""
        return ActivationMode.CLASSIFICATION
        
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
        
    def _compute_ovr_probabilities(self, distributions):
        """计算OvR激活概率"""
        if isinstance(distributions, dict):
            # 优先使用激活输出
            if 'output' in distributions:
                # 分类激活已经是概率
                return distributions['output']
            elif 'loc_S' in distributions:
                loc = distributions['loc_S']  # [batch_size, n_classes]
                scale = distributions['scale_S']  # [batch_size, n_classes]
            else:
                # 兜底：找第一个可用的tensor pair
                loc = None
                scale = None
                for key, value in distributions.items():
                    if torch.is_tensor(value) and 'loc' in key.lower():
                        loc = value
                    elif torch.is_tensor(value) and 'scale' in key.lower():
                        scale = value
                if loc is None or scale is None:
                    raise ValueError("Cannot find loc/scale in distributions")
        else:
            # 简化情况：假设分布输出
            loc = distributions
            scale = torch.ones_like(loc) * 0.1
            
        # OvR激活概率：P_k = 1/2 + (1/π) * arctan(μ_k / γ_k)
        probabilities = 0.5 + (1 / torch.pi) * torch.atan(loc / (scale + 1e-8))
        return probabilities
        
    def _compute_loss(self, predictions, targets):
        """计算OvR二元交叉熵损失"""
        # 获取激活概率
        probabilities = self._compute_ovr_probabilities(predictions)
        
        # 转换目标为one-hot编码
        targets_long = targets.long()
        targets_onehot = F.one_hot(targets_long, num_classes=self.n_classes_).float()
        
        # OvR二元交叉熵损失
        # L = -Σ_k [y_k * log(P_k) + (1-y_k) * log(1-P_k)]
        eps = 1e-8
        probabilities = torch.clamp(probabilities, eps, 1-eps)
        
        bce_loss = -(
            targets_onehot * torch.log(probabilities) + 
            (1 - targets_onehot) * torch.log(1 - probabilities)
        )
        
        return bce_loss.sum(dim=1).mean()
        
    def fit(self, X, y, sample_weight=None):
        """
        训练模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练输入
        y : array-like, shape (n_samples,)
            训练标签
        sample_weight : array-like, shape (n_samples,), optional
            样本权重（暂不支持）
            
        Returns:
        --------
        self : object
            返回self以支持方法链
        """
        # 数据验证
        X, y = self._validate_data(X, y, reset=True)
        
        # 设置类别信息
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # 标签映射
        label_map = {label: idx for idx, label in enumerate(self.classes_)}
        y_mapped = np.array([label_map[label] for label in y])
        
        # 设置训练环境
        X_train, X_val, y_train, y_val = self._setup_training(X, y_mapped)
        
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
        
        for epoch in range(self.max_iter):
            # 训练步骤
            train_loss = self._train_step(X_train_tensor, y_train_tensor, optimizer)
            self.loss_curve_.append(train_loss)
            
            # 验证步骤
            if X_val is not None:
                val_loss = self._validate_step(X_val_tensor, y_val_tensor)
                val_losses.append(val_loss)
                
                # 早停检查
                if self.early_stopping and len(val_losses) > self.n_iter_no_change:
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
                
        self.is_fitted_ = True
        return self
        
    def _predict_compatible(self, X_tensor):
        """兼容模式：返回类别标签"""
        predictions = self._forward(X_tensor)
        probabilities = self._compute_ovr_probabilities(predictions)
        
        # argmax决策
        class_indices = torch.argmax(probabilities, dim=1)
        return self.classes_[class_indices.cpu().numpy()]
        
    def _predict_advanced(self, X_tensor, mode):
        """高级模式：返回详细信息"""            
        predictions = self._forward(X_tensor)
        probabilities = self._compute_ovr_probabilities(predictions)
        
        # 类别预测
        class_indices = torch.argmax(probabilities, dim=1)
        predicted_classes = self.classes_[class_indices.cpu().numpy()]
        
        return {
            'predictions': predicted_classes,
            'probabilities': probabilities,
            'distributions': predictions,
            'mode': mode
        }
    
    def predict_proba(self, X, mode='compatible'):
        """
        预测类别概率
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本
        mode : str, default='compatible'
            概率计算模式
            - 'compatible': Softmax归一化，sklearn兼容
            - 'standard': OvR原生概率 + 简单归一化
            - 'causal': OvR纯因果概率
            - 'sampling': OvR探索性概率
            
        Returns:
        --------
        probabilities : array-like, shape (n_samples, n_classes)
            类别概率分布
        """
        check_is_fitted(self)
        X, _ = self._validate_data(X, reset=False)
        X_tensor = self._to_tensor(X)
        
        self.model.eval()
        with torch.no_grad():
            if mode == 'compatible':
                # Softmax兼容模式
                predictions = self._forward(X_tensor)
                probabilities = self._compute_ovr_probabilities(predictions)
                
                # 应用Softmax归一化以确保严格概率分布
                probabilities = F.softmax(probabilities, dim=1)
            else:
                # OvR原生概率模式    
                predictions = self._forward(X_tensor)
                probabilities = self._compute_ovr_probabilities(predictions)
                
                # 简单归一化（保持OvR特性）
                probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
                
        return self._from_tensor(probabilities)
    
    def decision_function(self, X):
        """
        决策函数（激活得分）
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本
            
        Returns:
        --------
        scores : array-like, shape (n_samples, n_classes)
            决策得分
        """
        check_is_fitted(self)
        X, _ = self._validate_data(X, reset=False)
        X_tensor = self._to_tensor(X)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self._forward(X_tensor)
            scores = self._compute_ovr_probabilities(predictions)
            
        return self._from_tensor(scores)
    
    def score(self, X, y, sample_weight=None):
        """
        返回准确率
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试输入
        y : array-like, shape (n_samples,)
            真实标签
        sample_weight : array-like, shape (n_samples,), optional
            样本权重（暂不支持）
            
        Returns:
        --------
        accuracy : float
            准确率
        """
        predictions = self.predict(X, mode='compatible')
        return accuracy_score(y, predictions, sample_weight=sample_weight)
        
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
            包含类别、概率和不确定性的字典
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