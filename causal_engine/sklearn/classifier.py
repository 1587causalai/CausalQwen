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
                 mode='standard',
                 causal_size=None,
                 activation_temperature=0.3,
                 # CausalEngine核心参数
                 b_noise_init=0.1,
                 b_noise_trainable=True,
                 gamma_init=10.0,
                 ovr_threshold_init=0.0,
                 # sklearn兼容参数
                 early_stopping=True,
                 validation_fraction=0.1,
                 n_iter_no_change=20,
                 tol=1e-4,
                 alpha=0.0001,
                 batch_size='auto',
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
        mode : str, default='standard'
            CausalEngine五种模式选择：
            - 'deterministic': γ_U=0, b_noise=0 (等价sklearn)
            - 'exogenous': γ_U=0, b_noise≠0 (外生噪声推理)
            - 'endogenous': γ_U≠0, b_noise=0 (内生因果推理) 
            - 'standard': γ_U≠0, b_noise→scale (标准因果推理)
            - 'sampling': γ_U≠0, b_noise→location (探索性因果推理)
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
        alpha : float, default=0.0001
            L2正则化系数，用于与sklearn对齐
        batch_size : int or 'auto', default='auto'
            小批量大小。'auto'表示min(200, n_samples)
        random_state : int, optional
            随机种子
        verbose : bool, default=False
            是否输出训练日志
        """
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate=learning_rate,
            mode=mode,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            verbose=verbose
        )
        
        self.causal_size = causal_size
        self.activation_temperature = activation_temperature
        # CausalEngine核心参数
        self.b_noise_init = b_noise_init
        self.b_noise_trainable = b_noise_trainable
        self.gamma_init = gamma_init
        self.ovr_threshold_init = ovr_threshold_init
        # sklearn兼容参数
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.alpha = alpha  # L2正则化系数
        self.batch_size = batch_size
        
        # 分类特定属性
        self.classes_ = None
        self.n_classes_ = None
        
    def _get_output_size(self) -> int:
        """分类任务输出维度等于类别数"""
        return self.n_classes_ if hasattr(self, 'n_classes_') else 2
        
    def _get_activation_mode(self) -> ActivationMode:
        """分类任务使用分类激活"""
        return ActivationMode.CLASSIFICATION
        
    def _init_weights_glorot(self, model):
        """使用Glorot/Xavier均匀初始化（与sklearn兼容）"""
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_model(self, input_size: int):
        """构建完整模型（统一架构）"""
        # 统一使用CausalEngine架构
        # 首先确定H的维度（MLP隐藏层的最后一层输出维度）
        if not self.hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes不能为空。")
        
        h_dim = self.hidden_layer_sizes[-1]  # H的维度
        
        # causal_size默认等于H的维度，这是最合理的设置
        causal_size = self.causal_size or h_dim
        
        # 构建标准MLP隐藏层（输出维度为h_dim）
        self.hidden_layers = self._build_mlp_layers(input_size, h_dim)
        
        # 构建CausalEngine（输入是MLP输出）
        self.causal_engine = self._build_causal_engine(
            input_size=causal_size,
            causal_size=causal_size
        )
        
        # 模型是MLP隐藏层和CausalEngine的组合
        self.model = nn.ModuleDict({
            'hidden_layers': self.hidden_layers,
            'causal_engine': self.causal_engine
        }).to(self.device).double()
        
        # 在deterministic模式下，我们通过前向传播跳过Abduction，无需冻结参数
        # 这样设计更优雅，减少内存占用和计算复杂度
        
    def _forward_with_mode(self, X_batch, mode=None):
        """统一前向传播（所有模式走相同路径）"""
        if mode is None:
            mode = self.mode

        # 1. 通过MLP隐藏层
        hidden_features = self.model['hidden_layers'](X_batch)
        
        # 2. CausalEngine期望3维输入: [batch_size, seq_len, features]
        if hidden_features.dim() == 2:
            hidden_features = hidden_features.unsqueeze(1)
            
        # 3. 通过CausalEngine统一管道：Abduction → Action → Activation
        # 简化接口：只使用mode参数
        causal_output = self.model['causal_engine'](
            hidden_features, 
            mode=mode,
            apply_activation=True,
            return_dict=True
        )
        
        # 4. 移除seq_len维度以匹配sklearn期望的2维输出
        output = causal_output['output']
        if output.dim() == 3 and output.size(1) == 1:
            output = output.squeeze(1)
            
        # 5. 构建返回结果（保持接口兼容性）
        result = {
            'output': output,  # 分类概率
            'mode': mode
        }
        
        # 添加分布参数（如果需要）
        for key in ['loc_S', 'scale_S', 'loc_U', 'scale_U']:
            if key in causal_output:
                tensor = causal_output[key]
                if tensor.dim() == 3 and tensor.size(1) == 1:
                    tensor = tensor.squeeze(1)
                result[key] = tensor
                
        return result
        
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
            
        # CausalEngine正确实现：带OvR阈值的Cauchy CDF激活概率
        # P_k = 1/2 + (1/π) * arctan((μ_k - threshold_k) / γ_k)
        scale_stable = torch.clamp(scale, min=1e-6)
        
        # 应用OvR阈值
        ovr_threshold = getattr(self, 'ovr_threshold_init', 0.0)
        if isinstance(ovr_threshold, (int, float)):
            threshold_tensor = torch.full_like(loc, ovr_threshold)
        else:
            threshold_tensor = torch.zeros_like(loc)
            
        # 应用OvR阈值
        adjusted_loc = loc - threshold_tensor
        arctan_ratio = torch.atan(adjusted_loc / scale_stable)
        
        # 直接使用Cauchy CDF计算OvR概率（您的创新设计）
        # P_{k,i} = 1/2 + (1/π) * arctan((loc_{S_{k,i}} - C_ovr) / scale_{S_{k,i}})
        probabilities = 0.5 + (1.0 / torch.pi) * arctan_ratio
        
        return probabilities
        
    def _compute_loss(self, predictions, targets):
        """根据模式计算相应损失函数（优化版：利用现有方法）"""
        if self.mode == 'deterministic':
            # Deterministic模式：直接使用CrossEntropy损失
            logits = predictions['loc_S'] if isinstance(predictions, dict) else predictions
            return F.cross_entropy(logits, targets.long())
        else:
            # 因果模式：使用OvR BCE损失
            # 简化：直接从预测结果获取概率
            probabilities = predictions['output'] if isinstance(predictions, dict) else predictions
            
            # 转换目标为one-hot编码进行OvR处理
            targets_long = targets.long()
            targets_onehot = F.one_hot(targets_long, num_classes=self.n_classes_).float()
            
            # OvR BCE损失：L = -Σ[y_k*log(P_k) + (1-y_k)*log(1-P_k)]
            eps = 1e-7
            probs_clipped = torch.clamp(probabilities, min=eps, max=1-eps)
            
            # OvR BCE损失的两个组成部分
            # 正类贡献：当真实标签为1时，希望预测概率p接近1，log(p)项
            pos_class_log_likelihood = targets_onehot * torch.log(probs_clipped)
            # 负类贡献：当真实标签为0时，希望预测概率p接近0，log(1-p)项  
            neg_class_log_likelihood = (1 - targets_onehot) * torch.log(1 - probs_clipped)
            
            # BCE损失 = -[正类对数似然 + 负类对数似然]的均值
            main_loss = -(pos_class_log_likelihood + neg_class_log_likelihood).mean()
            
            # 移除scale_S正则化：让模型自由学习最优的尺度参数
            # 人工强制target_scale=0.5缺乏理论依据，可能干扰训练
            return main_loss
        
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
        
        # 设置优化器 - 统一使用Adam优化器确保公平对比
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 转换数据
        X_train_tensor = self._to_tensor(X_train)
        y_train_tensor = self._to_tensor(y_train)
        
        if X_val is not None:
            X_val_tensor = self._to_tensor(X_val)
            y_val_tensor = self._to_tensor(y_val)
        
        # 解析batch_size
        if self.batch_size == 'auto':
            batch_size = min(200, X_train.shape[0])
        else:
            batch_size = self.batch_size

        # 训练循环
        self.loss_curve_ = []
        self.best_loss_ = np.inf
        self._no_improvement_count = 0
        self._best_model_state = None  # 保存最佳模型状态
        
        if X_val is not None:
            self.validation_scores_ = []

        for epoch in range(self.max_iter):
            # 每个epoch都打乱数据
            permutation = torch.randperm(X_train_tensor.size(0))
            epoch_losses = []

            self.model.train()
            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                X_batch, y_batch = X_train_tensor[indices], y_train_tensor[indices]
                
                # 训练步骤
                loss = self._train_step(X_batch, y_batch, optimizer)
                epoch_losses.append(loss)
            
            train_loss = np.mean(epoch_losses)
            self.loss_curve_.append(train_loss)
            
            # 验证步骤
            if X_val is not None:
                val_loss = self._validate_step(X_val_tensor, y_val_tensor)
                self.validation_scores_.append(val_loss)
                
                # 早停检查 - sklearn风格
                if self.early_stopping:
                    if val_loss < self.best_loss_ - self.tol:
                        self.best_loss_ = val_loss
                        self._no_improvement_count = 0
                        # 保存当前最佳模型状态
                        import copy
                        self._best_model_state = copy.deepcopy(self.model.state_dict())
                        if self.verbose:
                            print(f"New best validation loss: {val_loss:.6f} at epoch {epoch+1}")
                    else:
                        self._no_improvement_count += 1
                    
                    if self._no_improvement_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # 日志输出
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {train_loss:.6f}")
        
        self.n_iter_ = epoch + 1
        
        # 恢复最佳模型状态
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            if self.verbose:
                print(f"Restored best model from validation loss: {self.best_loss_:.6f}")
        
        self.is_fitted_ = True
        return self
        
    def _predict_deterministic(self, X_tensor):
        """确定性模式：返回类别标签（sklearn兼容）"""
        predictions = self._forward_with_mode(X_tensor, 'deterministic')
        
        # Deterministic模式下，CausalEngine直接输出logits作为'output'
        if isinstance(predictions, dict):
            logits = predictions.get('output')
        else:
            logits = predictions
            
        class_indices = torch.argmax(logits, dim=1)
        return self.classes_[class_indices.cpu().numpy()]
        
    def _predict_causal(self, X_tensor, mode):
        """因果模式：返回详细信息"""            
        predictions = self._forward_with_mode(X_tensor, mode)
        probabilities = self._compute_ovr_probabilities(predictions)
        
        # 类别预测
        class_indices = torch.argmax(probabilities, dim=1)
        predicted_classes = self.classes_[class_indices.cpu().numpy()]
        
        result = {
            'predictions': predicted_classes,
            'mode': mode
        }
        
        # 添加激活概率信息
        if isinstance(predictions, dict):
            result['probabilities'] = probabilities
            result['distribution_params'] = {
                'loc': predictions.get('loc_S'),
                'scale': predictions.get('scale_S')
            }
        
        return result
    
    def predict_proba(self, X, mode=None):
        """
        预测类别概率
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本
        mode : str, optional
            概率计算模式，默认为当前模式
            - 'deterministic': Softmax归一化（sklearn兼容）
            - 其他模式: OvR激活概率
            
        Returns:
        --------
        probabilities : array-like, shape (n_samples, n_classes)
            类别概率分布
        """
        if mode is None:
            mode = self.mode
            
        return self.predict_dist(X, mode=mode)
    
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
            if self.mode == 'deterministic':
                # Deterministic模式使用原始logits
                predictions = self._forward_with_mode(X_tensor, 'deterministic')
                if isinstance(predictions, dict) and 'loc_S' in predictions:
                    scores = predictions['loc_S']
                else:
                    scores = predictions
            else:
                # 因果模式使用激活概率
                predictions = self._forward_with_mode(X_tensor, self.mode)
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
        predictions = self.predict(X, mode='deterministic')
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
        
    def predict_dist(self, X, mode=None):
        """
        预测并返回激活概率分布
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本
        mode : str, optional
            预测模式，默认为当前模式
            
        Returns:
        --------
        probabilities : array-like, shape (n_samples, n_classes)
            激活概率分布
        """
        if mode is None:
            mode = self.mode
            
        check_is_fitted(self)
        X, _ = self._validate_data(X, reset=False)
        X_tensor = self._to_tensor(X)
        
        self.model.eval()
        with torch.no_grad():
            if mode == 'deterministic':
                # Deterministic模式下使用softmax转换为概率（与sklearn一致）
                predictions = self._forward_with_mode(X_tensor, mode)
                if isinstance(predictions, dict):
                    logits = predictions.get('logits', predictions.get('output'))
                else:
                    logits = predictions
                probabilities = F.softmax(logits, dim=1)
            else:
                # 因果模式下返回激活概率
                predictions = self._forward_with_mode(X_tensor, mode)
                probabilities = self._compute_ovr_probabilities(predictions)
                
        return self._from_tensor(probabilities)
        
    # 为了向后兼容，保留旧的API名称
    def predict_with_uncertainty(self, X):
        """向后兼容的API，推荐使用predict_dist"""
        return self.predict_dist(X, mode='standard')