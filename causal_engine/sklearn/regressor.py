"""
MLPCausalRegressor - sklearn风格的因果回归器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 mode='standard',
                 causal_size=None,
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
        初始化MLPCausalRegressor
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(64, 32)
            MLP隐藏层结构，格式与sklearn MLPRegressor相同
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
            因果表征维度，默认为h_dim
        b_noise_init : float, default=0.1
            外生噪声参数初始值，控制噪声强度
        b_noise_trainable : bool, default=True
            外生噪声参数是否可训练，False时b_noise固定为初始值
        gamma_init : float, default=10.0
            AbductionNetwork尺度参数初始化，影响个体不确定性
        ovr_threshold_init : float, default=0.0
            OvR分类阈值初始化（回归任务中不使用）
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
        
        # 回归特定属性
        self.n_outputs_ = None
        
    def _get_output_size(self) -> int:
        """回归任务输出维度为1"""
        return 1
        
    def _get_activation_mode(self) -> ActivationMode:
        """回归任务使用回归激活"""
        return ActivationMode.REGRESSION
        
    def _init_weights_glorot(self, model):
        """使用Glorot/Xavier均匀初始化（与sklearn兼容）"""
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_model(self, input_size: int):
        """构建完整模型（统一架构）"""
        print(f"\n为模式构建模型: {self.mode}")
        
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
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"==> 模型已构建。总可训练参数: {total_params}")

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
            'output': output,  # 回归预测值
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
        
    def _compute_loss(self, predictions, targets):
        """根据模式计算相应损失函数（优化版：利用现有方法）"""
        targets = targets.squeeze()
        
        if self.mode == 'deterministic':
            # Deterministic模式：使用传统MSE损失
            loc_S = (predictions['loc_S'] if isinstance(predictions, dict) else predictions).squeeze()
            return F.mse_loss(loc_S, targets)
        else:
            # 因果模式：使用Cauchy NLL损失
            # 简化：直接提取分布参数
            if isinstance(predictions, dict):
                loc_S = predictions['loc_S'].squeeze()
                scale_S = predictions['scale_S'].squeeze()
            else:
                loc_S = predictions.squeeze()
                scale_S = torch.ones_like(loc_S) * 0.1
            
            # 使用CauchyMath工具计算NLL（更简洁和一致）
            from ..engine import CauchyMath
            
            # 数值稳定性处理
            scale_min = 1e-4
            scale_S_stable = torch.clamp(scale_S, min=scale_min)
            
            # 使用CauchyMath计算log_pdf，然后取负数得到NLL
            log_pdf = CauchyMath.cauchy_log_pdf(targets, loc_S, scale_S_stable)
            nll = -log_pdf
            
            # 返回批次平均损失
            return nll.mean()
        
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
        
        # 设置优化器（支持L2正则化）
        if self.alpha > 0:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        else:
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
        
        self.n_outputs_ = 1
        self.is_fitted_ = True
        return self
        
    def _predict_deterministic(self, X_tensor):
        """确定性模式：返回位置参数作为点估计（sklearn兼容）"""
        predictions = self._forward_with_mode(X_tensor, 'deterministic')
        
        # Deterministic模式下，直接返回位置参数
        if isinstance(predictions, dict):
            if 'loc_S' in predictions:
                return predictions['loc_S'].squeeze()
            elif 'output' in predictions:
                return predictions['output'].squeeze()
            else:
                raise ValueError(f"无法从predictsions中提取位置参数: {predictions.keys()}")
        else:
            return predictions.squeeze()
            
    def _predict_causal(self, X_tensor, mode):
        """因果模式：返回完整分布信息"""
        predictions = self._forward_with_mode(X_tensor, mode)
        
        # 返回结构化输出，包含预测值和分布参数
        if isinstance(predictions, dict):
            loc_S = predictions.get('loc_S', predictions.get('output'))
            scale_S = predictions.get('scale_S')
            
            result = {
                'predictions': loc_S.squeeze() if loc_S is not None else None,
                'mode': mode
            }
            
            # 添加分布参数信息
            if loc_S is not None and scale_S is not None:
                result['distribution_params'] = {
                    'loc': loc_S.squeeze(),
                    'scale': scale_S.squeeze()
                }
                
            return result
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
        # 使用deterministic模式进行评分（sklearn兼容）
        predictions = self.predict(X, mode='deterministic')
        if isinstance(predictions, dict):
            predictions = predictions['predictions']
        return r2_score(y, predictions, sample_weight=sample_weight)
        
    def predict_dist(self, X, mode=None):
        """
        预测并返回完整分布参数
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本
        mode : str, optional
            预测模式，默认为当前模式
            
        Returns:
        --------
        dist_params : array-like, shape (n_samples, output_dim, 2)
            分布参数，[:, :, 0]为位置参数，[:, :, 1]为尺度参数
        """
        if mode is None:
            mode = self.mode
            
        if mode == 'deterministic':
            # Deterministic模式下无分布信息
            predictions = self.predict(X, mode=mode)
            # 返回估计的分布参数（scale=0表示确定性）
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            n_samples, n_outputs = predictions.shape
            dist_params = np.zeros((n_samples, n_outputs, 2))
            dist_params[:, :, 0] = predictions  # loc
            dist_params[:, :, 1] = 0.0  # scale = 0 (确定性)
            return dist_params
        else:
            # 因果模式下返回完整分布参数
            result = self.predict(X, mode=mode)
            if isinstance(result, dict) and 'distribution_params' in result:
                loc = self._from_tensor(result['distribution_params']['loc'])
                scale = self._from_tensor(result['distribution_params']['scale'])
                
                if loc.ndim == 1:
                    loc = loc.reshape(-1, 1)
                if scale.ndim == 1:
                    scale = scale.reshape(-1, 1)
                    
                n_samples, n_outputs = loc.shape
                dist_params = np.zeros((n_samples, n_outputs, 2))
                dist_params[:, :, 0] = loc
                dist_params[:, :, 1] = scale
                return dist_params
            else:
                # 如果没有分布参数，返回估计的参数
                predictions = result['predictions'] if isinstance(result, dict) else result
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                n_samples, n_outputs = predictions.shape
                dist_params = np.zeros((n_samples, n_outputs, 2))
                dist_params[:, :, 0] = predictions
                dist_params[:, :, 1] = 1.0  # 默认scale
                return dist_params
        
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
        
    # 为了向后兼容，保留旧的API名称
    def predict_with_uncertainty(self, X):
        """向后兼容的API，推荐使用predict_dist"""
        return self.predict_dist(X, mode='standard')