"""
MLPCausalRegressor and Robust Regressors: Scikit-learn compatible neural network regressors.

╔════════════════════════════════════════════════════════════════════════════════╗
║                        回归器模块架构图 - Regressor Suite                         ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║                          sklearn兼容的神经网络回归器集合                          ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                            输入层 (Input)                                 │  ║
║  │          X: [n_samples, n_features] + sample_weight (可选)                │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         五种回归器架构                                       │  ║
║  │                                                                           │  ║
║  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │  ║
║  │  │ MLPCausalRegressor│  │MLPPytorchRegressor│ │  Robust Suite   │          │  ║
║  │  │  因果推理回归器     │  │   标准PyTorch    │  │   鲁棒回归器      │          │  ║
║  │  │                 │  │      MLP        │  │                 │          │  ║
║  │  │  🧠 CausalEngine │  │   🔧 基准对比    │  │  💪 异常值鲁棒    │          │  ║
║  │  │  四阶段架构       │  │   标准MSE损失    │  │   多种损失函数     │          │  ║
║  │  │  五种推理模式     │  │   ReLU/Tanh     │  │                 │          │  ║
║  │  │  分布预测能力     │  │   L2正则化      │  │  ┌─────────────┐ │          │  ║
║  │  │                 │  │                 │  │  │ MLPHuber    │ │          │  ║
║  │  └─────────────────┘  └─────────────────┘  │  │ 🛡️ Huber损失 │ │          │  ║
║  │                                            │  │ 二次→线性    │ │          │  ║
║  │                                            │  └─────────────┘ │          │  ║
║  │                                            │  ┌─────────────┐ │          │  ║
║  │                                            │  │ MLPPinball  │ │          │  ║
║  │                                            │  │ 📊 分位数回归 │ │          │  ║
║  │                                            │  │ Quantile损失 │ │          │  ║
║  │                                            │  └─────────────┘ │          │  ║
║  │                                            │  ┌─────────────┐ │          │  ║
║  │                                            │  │ MLPCauchy   │ │          │  ║
║  │                                            │  │ 🎯 厚尾分布  │ │          │  ║
║  │                                            │  │ log(1+e²)   │ │          │  ║
║  │                                            │  └─────────────┘ │          │  ║
║  │                                            └─────────────────┘          │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                           核心特性                                         │  ║
║  │                                                                           │  ║
║  │  🔧 sklearn兼容: 标准fit/predict/score接口                                 │  ║
║  │  ⚖️  样本权重: 完整支持sample_weight参数                                    │  ║
║  │  📈 早停机制: validation-based early stopping                             │  ║
║  │  🎛️  批处理: 自动/手动batch size配置                                        │  ║
║  │  🎲 随机种子: 可重现的random_state控制                                      │  ║
║  │  📊 进度追踪: 可选的verbose训练日志                                         │  ║
║  │  🎯 多GPU: 自动CUDA设备检测和使用                                          │  ║
║  │  🔄 数据分割: 自动训练/验证集划分                                           │  ║
║  │  🧠 表征提取: get_representation() 统一暴露中间表征 Z                       │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         输出层 (Output)                                   │  ║
║  │     y_pred: [n_samples] + 可选的分布参数 (MLPCausalRegressor)              │  ║
║  │     R² score: 统一的性能评估指标                                           │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                           使用场景指南                                      │  ║
║  │                                                                           │  ║
║  │  🧠 MLPCausalRegressor → 需要因果推理和不确定性量化                         │  ║
║  │  🔧 MLPPytorchRegressor → 标准回归基线，性能对比                           │  ║
║  │  🛡️  MLPHuberRegressor → 数据有异常值，需要鲁棒性                           │  ║
║  │  📊 MLPPinballRegressor → 分位数回归，风险评估                             │  ║
║  │  🎯 MLPCauchyRegressor → 厚尾分布数据，极端鲁棒性                          │  ║
║  │                                                                           │  ║
║  │  📋 表征分析: 所有回归器都支持 get_representation(X) 获取中间表征 Z          │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════════════╝

This module provides:
- MLPCausalRegressor: Causal reasoning-based regressor
- MLPPytorchRegressor: Standard PyTorch MLP regressor for baseline comparison
- MLPHuberRegressor: Robust regressor with Huber loss
- MLPPinballRegressor: Quantile regressor with Pinball loss
- MLPCauchyRegressor: Robust regressor with Cauchy loss
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import inspect

from ._causal_engine.engine import CausalEngine

class MLPCausalRegressor(BaseEstimator, RegressorMixin):
    """
    Causal Multi-layer Perceptron Regressor.
    
    A scikit-learn compatible neural network regressor that uses causal reasoning
    to understand relationships in data rather than just fitting patterns.
    
    Note: This estimator does not perform automatic data preprocessing. For best
    results, consider standardizing your features using sklearn.preprocessing.StandardScaler
    or similar preprocessing techniques before training.
    
    Parameters
    ----------
    repre_size : int, optional
        The dimension of the internal representation space (Z). If None, defaults
        are handled by the CausalEngine.
        
    causal_size : int, optional
        The dimension of the causal representation space (U). If None, defaults
        are handled by the CausalEngine.

    perception_hidden_layers : tuple, default=(100,)
        The hidden layer structure for the Perception network (X -> Z).

    abduction_hidden_layers : tuple, default=()
        The hidden layer structure for the Abduction network (Z -> U).
        
    mode : str, default='standard'
        Prediction mode. Options: 'deterministic', 'standard', 'sampling'.
        
    gamma_init : float, default=10.0
        Initial scale parameter for the AbductionNetwork.
        
    b_noise_init : float, default=0.1
        Initial noise level for the ActionNetwork.
        
    b_noise_trainable : bool, default=True
        Whether the noise parameter is trainable.
        
    max_iter : int, default=1000
        Maximum number of training iterations.
        
    learning_rate : float, default=0.001
        Learning rate for optimization.
        
    early_stopping : bool, default=True
        Whether to use early stopping during training.
        
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation when early_stopping=True.
        
    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.
        
    tol : float, default=1e-4
        Tolerance for optimization convergence.
        
    random_state : int, default=None
        Random state for reproducibility.
        
    verbose : bool, default=False
        Whether to print progress messages.
    """
    
    def __init__(
        self,
        repre_size: Optional[int] = None,
        causal_size: Optional[int] = None,
        perception_hidden_layers: tuple = (100,),
        abduction_hidden_layers: tuple = (),
        mode='standard',
        gamma_init=10.0,
        b_noise_init=0.1,
        b_noise_trainable=True,
        max_iter=1000,
        learning_rate=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=None,
        verbose=False,
        alpha=0.0,
        batch_size='auto'
    ):
        self.repre_size = repre_size
        self.causal_size = causal_size
        self.perception_hidden_layers = perception_hidden_layers
        self.abduction_hidden_layers = abduction_hidden_layers
        self.mode = mode
        self.gamma_init = gamma_init
        self.b_noise_init = b_noise_init
        self.b_noise_trainable = b_noise_trainable
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Will be set during fit
        self.engine_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
    def _build_model(self, input_size):
        """Build CausalEngine model"""
        return CausalEngine(
            input_size=input_size,
            output_size=1,
            repre_size=self.repre_size,
            causal_size=self.causal_size,
            task_type='regression',
            perception_hidden_layers=self.perception_hidden_layers,
            abduction_hidden_layers=self.abduction_hidden_layers,
            gamma_init=self.gamma_init,
            b_noise_init=self.b_noise_init,
            b_noise_trainable=self.b_noise_trainable,
            alpha=self.alpha
        )
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the causal regressor to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Should be preprocessed (e.g., standardized) if needed.
        y : array-like of shape (n_samples,)
            Target values. Should be preprocessed if needed.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, the loss function will be weighted by these values.
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y, sample_weight,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            sw_train, sw_val = sample_weight, None
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # CausalEngine expects y to be 2D for regression
        if len(y_train_tensor.shape) == 1:
            y_train_tensor = y_train_tensor.unsqueeze(1)
        
        # Convert sample weights to tensor if provided
        if sw_train is not None:
            sw_train_tensor = torch.FloatTensor(sw_train)
        else:
            sw_train_tensor = None
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            # Also reshape validation y
            if len(y_val_tensor.shape) == 1:
                y_val_tensor = y_val_tensor.unsqueeze(1)
            
            # Convert validation sample weights to tensor if provided
            if sw_val is not None:
                sw_val_tensor = torch.FloatTensor(sw_val)
            else:
                sw_val_tensor = None
        
        # Create CausalEngine
        self.engine_ = self._build_model(self.n_features_in_)
        
        # Setup optimizer
        optimizer = optim.Adam(self.engine_.parameters(), lr=self.learning_rate)
        
        # Determine batch size
        n_samples = X_train.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        elif self.batch_size is None:
            batch_size = n_samples  # Full-batch training
        else:
            batch_size = min(self.batch_size, n_samples)
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # Training step with mini-batches
            self.engine_.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data for each epoch
            indices = torch.randperm(n_samples)
            X_train_shuffled = X_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]
            if sw_train_tensor is not None:
                sw_train_shuffled = sw_train_tensor[indices]
            else:
                sw_train_shuffled = None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                # Get sample weights for this batch
                if sw_train_shuffled is not None:
                    sw_batch = sw_train_shuffled[i:end_idx]
                else:
                    sw_batch = None
                
                optimizer.zero_grad()
                
                # Compute loss with sample weights
                if sw_batch is not None:
                    # Get decision scores for weighted loss computation
                    decision_scores = self.engine_._get_decision_scores(X_batch, mode=self.mode)
                    mu_pred, gamma_pred = decision_scores
                    
                    # Compute individual losses based on mode
                    if self.mode == 'deterministic':
                        # MSE loss for each sample
                        individual_losses = (y_batch.squeeze() - mu_pred.squeeze()) ** 2
                    else:
                        # Cauchy NLL loss for each sample
                        from ._causal_engine.math_utils import CauchyMath
                        individual_losses = CauchyMath.nll_loss(y_batch, mu_pred, gamma_pred, reduction='none')
                        if individual_losses.dim() > 1:
                            individual_losses = individual_losses.mean(dim=1)  # Average over output dimensions
                    
                    # Apply sample weights
                    loss = (individual_losses * sw_batch).mean()
                else:
                    # Standard loss computation
                    loss = self.engine_.compute_loss(X_batch, y_batch, mode=self.mode)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.engine_.eval()
                with torch.no_grad():
                    # Compute validation loss with sample weights
                    if sw_val_tensor is not None:
                        # Get decision scores for weighted validation loss computation
                        decision_scores = self.engine_._get_decision_scores(X_val_tensor, mode=self.mode)
                        mu_pred, gamma_pred = decision_scores
                        
                        # Compute individual validation losses based on mode
                        if self.mode == 'deterministic':
                            # MSE loss for each sample
                            individual_val_losses = (y_val_tensor.squeeze() - mu_pred.squeeze()) ** 2
                        else:
                            # Cauchy NLL loss for each sample
                            from ._causal_engine.math_utils import CauchyMath
                            individual_val_losses = CauchyMath.nll_loss(y_val_tensor, mu_pred, gamma_pred, reduction='none')
                            if individual_val_losses.dim() > 1:
                                individual_val_losses = individual_val_losses.mean(dim=1)  # Average over output dimensions
                        
                        # Apply sample weights to validation loss
                        val_loss = (individual_val_losses * sw_val_tensor).mean()
                    else:
                        # Standard validation loss computation
                        val_loss = self.engine_.compute_loss(X_val_tensor, y_val_tensor, mode=self.mode)
                    
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        # Save the best model state
                        best_state_dict = self.engine_.state_dict().copy()
                    else:
                        no_improve_count += 1
                        
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        # Restore the best model
                        if best_state_dict is not None:
                            self.engine_.load_state_dict(best_state_dict)
                        break
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_epoch_loss:.6f}")
        
        self.n_iter_ = epoch + 1
        
        # Set sklearn compatibility attributes
        self.n_layers_ = (len(self.perception_hidden_layers) + 
                         len(self.abduction_hidden_layers) + 2)  # +2 for input and output layers
        self.n_outputs_ = 1  # Regression has single output
        self.out_activation_ = 'identity'  # Linear output for regression
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"MLPCausalRegressor fitted with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Training completed in {self.n_iter_} iterations")
            
        return self
        
    def predict(self, X):
        """
        Predict using the causal regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Should be preprocessed consistently with training data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPCausalRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPCausalRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'engine_'):
            raise ValueError("This MLPCausalRegressor instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict using CausalEngine
        self.engine_.eval()
        with torch.no_grad():
            y_pred = self.engine_.predict(X_tensor, mode=self.mode)
            if isinstance(y_pred, tuple):
                # If returns (location, scale), take location for point prediction
                y_pred = y_pred[0]
            
            # Convert back to numpy
            y_pred_np = y_pred.cpu().numpy()
            if y_pred_np.ndim > 1:
                y_pred_np = y_pred_np.ravel()
            
        return y_pred_np
        
    def predict_dist(self, X):
        """
        Predict distribution parameters using the causal regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Should be preprocessed consistently with training data.
            
        Returns
        -------
        dist_params : ndarray of shape (n_samples, n_params)
            Distribution parameters (location, scale) for each sample.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.mode == 'deterministic':
            raise ValueError("Distribution prediction not available in deterministic mode.")
            
        # Check if model is fitted
        if not hasattr(self, 'engine_'):
            raise ValueError("This MLPCausalRegressor instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict distribution using CausalEngine
        self.engine_.eval()
        with torch.no_grad():
            dist_params = self.engine_.predict_distribution(X_tensor, mode=self.mode)
            
            if not isinstance(dist_params, tuple) or len(dist_params) != 2:
                raise RuntimeError("Expected distributional output (location, scale) but got different format")
            
            location, scale = dist_params
            
            # Convert to numpy
            location_np = location.cpu().numpy()
            scale_np = scale.cpu().numpy()
            
            if location_np.ndim > 1:
                location_np = location_np.ravel()
            if scale_np.ndim > 1:
                scale_np = scale_np.ravel()
            
        return np.column_stack([location_np, scale_np])
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. Should be preprocessed consistently with training data.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R² of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)
    
    def get_representation(self, X):
        """
        Extract the perception representation Z from input X.
        
        This method exposes the intermediate representation from the Perception stage
        of the CausalEngine, allowing analysis of the X → Z transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Should be preprocessed consistently with training data.
            
        Returns
        -------
        Z : ndarray of shape (n_samples, repre_size)
            The perception representation extracted from the CausalEngine.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPCausalRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPCausalRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'engine_'):
            raise ValueError("This MLPCausalRegressor instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Extract representation using perception network
        self.engine_.eval()
        with torch.no_grad():
            Z = self.engine_.perception_net(X_tensor)
            return Z.cpu().numpy()
        
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'CausalEngine parameters have complex defaults'
            }
        }

class MLPPytorchRegressor(BaseEstimator, RegressorMixin):
    """
    PyTorch Multi-layer Perceptron Regressor.
    
    A scikit-learn compatible PyTorch neural network regressor for baseline comparison.
    This provides a standard MLP implementation using PyTorch with the same interface
    as MLPCausalRegressor.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The hidden layer structure for the MLP.
        
    max_iter : int, default=1000
        Maximum number of training iterations.
        
    learning_rate : float, default=0.001
        Learning rate for optimization.
        
    early_stopping : bool, default=True
        Whether to use early stopping during training.
        
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation when early_stopping=True.
        
    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.
        
    tol : float, default=1e-4
        Tolerance for optimization convergence.
        
    random_state : int, default=None
        Random state for reproducibility.
        
    verbose : bool, default=False
        Whether to print progress messages.
        
    activation : str, default='relu'
        Activation function for hidden layers.
        
    alpha : float, default=0.0001
        L2 regularization parameter.
    """
    
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        max_iter=1000,
        learning_rate=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=None,
        verbose=False,
        activation='relu',
        alpha=0.0001,
        batch_size='auto'
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Will be set during fit
        self.model_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
    def _build_perception_network(self, input_size):
        """Build perception network: X → Z"""
        if not self.hidden_layer_sizes:
            # If no hidden layers, perception network is just identity (return input)
            return nn.Identity()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers for perception
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _build_model(self, input_size, output_size):
        """Build perception network + output layer architecture"""
        # Build perception network: X → Z
        self.perception_net = self._build_perception_network(input_size)
        
        # Determine representation size
        if self.hidden_layer_sizes:
            representation_size = self.hidden_layer_sizes[-1]  # Last hidden layer size
        else:
            representation_size = input_size  # No hidden layers, use input size
        
        # Build output layer: Z → Y
        self.output_layer = nn.Linear(representation_size, output_size)
        
        # Create a module list to hold both components
        return nn.ModuleList([self.perception_net, self.output_layer])
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the PyTorch regressor to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, the loss function will be weighted by these values.
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y, sample_weight,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            sw_train, sw_val = sample_weight, None
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Convert sample weights to tensor if provided
        if sw_train is not None:
            sw_train_tensor = torch.FloatTensor(sw_train)
        else:
            sw_train_tensor = None
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # Convert validation sample weights to tensor if provided
            if sw_val is not None:
                sw_val_tensor = torch.FloatTensor(sw_val)
            else:
                sw_val_tensor = None
        
        # Build model
        self.model_ = self._build_model(self.n_features_in_, 1)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(list(self.perception_net.parameters()) + list(self.output_layer.parameters()), 
                              lr=self.learning_rate, weight_decay=self.alpha)
        criterion = nn.MSELoss()
        
        # Determine batch size
        n_samples = X_train.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        elif self.batch_size is None:
            batch_size = n_samples  # Full-batch training
        else:
            batch_size = min(self.batch_size, n_samples)
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # Training step with mini-batches
            self.perception_net.train()
            self.output_layer.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data for each epoch
            indices = torch.randperm(n_samples)
            X_train_shuffled = X_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]
            if sw_train_tensor is not None:
                sw_train_shuffled = sw_train_tensor[indices]
            else:
                sw_train_shuffled = None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                # Get sample weights for this batch
                if sw_train_shuffled is not None:
                    sw_batch = sw_train_shuffled[i:end_idx]
                else:
                    sw_batch = None
                
                optimizer.zero_grad()
                
                # Forward pass: X → Z → Y
                Z = self.perception_net(X_batch)
                outputs = self.output_layer(Z)
                
                # Compute loss with sample weights
                if sw_batch is not None:
                    # Compute individual MSE losses for each sample
                    individual_losses = (outputs.squeeze() - y_batch) ** 2
                    loss = (individual_losses * sw_batch).mean()
                else:
                    loss = criterion(outputs.squeeze(), y_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.perception_net.eval()
                self.output_layer.eval()
                with torch.no_grad():
                    # Forward pass: X → Z → Y
                    Z_val = self.perception_net(X_val_tensor)
                    val_outputs = self.output_layer(Z_val)
                    
                    # Compute validation loss with sample weights
                    if sw_val_tensor is not None:
                        # Compute individual MSE losses for each validation sample
                        individual_val_losses = (val_outputs.squeeze() - y_val_tensor) ** 2
                        val_loss = (individual_val_losses * sw_val_tensor).mean().item()
                    else:
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                    
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = {
                            'perception_net': self.perception_net.state_dict().copy(),
                            'output_layer': self.output_layer.state_dict().copy()
                        }
                    else:
                        no_improve_count += 1
                        
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        # Restore the best model
                        if best_state_dict is not None:
                            self.perception_net.load_state_dict(best_state_dict['perception_net'])
                            self.output_layer.load_state_dict(best_state_dict['output_layer'])
                        break
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_epoch_loss:.6f}")
        
        self.n_iter_ = epoch + 1
        
        # Set sklearn compatibility attributes
        self.n_layers_ = len(self.hidden_layer_sizes) + 1  # +1 for output layer
        self.n_outputs_ = 1  # Regression has single output
        self.out_activation_ = 'identity'  # Linear output for regression
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"MLPPytorchRegressor fitted with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Training completed in {self.n_iter_} iterations")
            
        return self
        
    def predict(self, X):
        """
        Predict using the PyTorch regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPPytorchRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPytorchRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPPytorchRegressor instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict using perception network + output layer
        self.perception_net.eval()
        self.output_layer.eval()
        with torch.no_grad():
            # Forward pass: X → Z → Y
            Z = self.perception_net(X_tensor)
            y_pred = self.output_layer(Z)
            y_pred_np = y_pred.squeeze().cpu().numpy()
            
        return y_pred_np
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R² of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)
    
    def get_representation(self, X):
        """
        Extract the perception representation Z from input X.
        
        This method exposes the intermediate representation from the perception network,
        allowing analysis of the X → Z transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        Z : ndarray of shape (n_samples, representation_size)
            The perception representation extracted from the perception network.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPPytorchRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPytorchRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPPytorchRegressor instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Extract representation using perception network: X → Z
        self.perception_net.eval()
        with torch.no_grad():
            Z = self.perception_net(X_tensor)
            return Z.cpu().numpy()
        
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'PyTorch parameters have complex defaults'
            }
        }


# =============================================================================
# Robust Neural Network Regressors
# =============================================================================

class MLPHuberRegressor(BaseEstimator, RegressorMixin):
    """
    Multi-layer Perceptron regressor with Huber loss.
    
    Uses Huber loss function which is less sensitive to outliers than squared error.
    Provides sklearn-compatible interface with fit/predict methods.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
        
    max_iter : int, default=1000
        Maximum number of training iterations.
        
    learning_rate : float, default=0.001
        Learning rate for optimization.
        
    early_stopping : bool, default=True
        Whether to use early stopping during training.
        
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation when early_stopping=True.
        
    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.
        
    tol : float, default=1e-4
        Tolerance for optimization convergence.
        
    delta : float, default=1.0
        The delta parameter for Huber loss. Controls the transition from 
        quadratic to linear loss.
        
    random_state : int, default=None
        Random state for reproducibility.
        
    verbose : bool, default=False
        Whether to print progress messages.
        
    alpha : float, default=0.0
        L2 regularization parameter.
        
    batch_size : int or 'auto', default='auto'
        Size of minibatches for stochastic optimizers.
    """
    
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        max_iter=1000,
        learning_rate=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        delta=1.0,
        random_state=None,
        verbose=False,
        alpha=0.0,
        batch_size='auto'
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.delta = delta
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Will be set during fit
        self.model_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
    def _build_perception_network(self, input_size):
        """Build perception network: X → Z"""
        if not self.hidden_layer_sizes:
            # If no hidden layers, perception network is just identity (return input)
            return nn.Identity()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers for perception
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _build_model(self, input_size):
        """Build perception network + output layer architecture"""
        # Build perception network: X → Z
        self.perception_net = self._build_perception_network(input_size)
        
        # Determine representation size
        if self.hidden_layer_sizes:
            representation_size = self.hidden_layer_sizes[-1]  # Last hidden layer size
        else:
            representation_size = input_size  # No hidden layers, use input size
        
        # Build output layer: Z → Y
        self.output_layer = nn.Linear(representation_size, 1)
        
        # Create a module list to hold both components
        return nn.ModuleList([self.perception_net, self.output_layer])
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Huber regressor to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, the loss function will be weighted by these values.
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        
        # Data is assumed to be pre-scaled. No internal scaling.
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y, sample_weight,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            sw_train, sw_val = sample_weight, None
        
        # Convert to torch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        
        # Convert sample weights to tensor if provided
        if sw_train is not None:
            sw_train_tensor = torch.FloatTensor(sw_train).to(device)
        else:
            sw_train_tensor = None
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            
            # Convert validation sample weights to tensor if provided
            if sw_val is not None:
                sw_val_tensor = torch.FloatTensor(sw_val).to(device)
            else:
                sw_val_tensor = None
        
        # Build model
        self.model_ = self._build_model(self.n_features_in_).to(device)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        criterion = nn.HuberLoss(delta=self.delta)
        
        # Determine batch size
        n_samples = X_train.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        elif self.batch_size is None:
            batch_size = n_samples  # Full-batch training
        else:
            batch_size = min(self.batch_size, n_samples)
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # Training step with mini-batches
            self.perception_net.train()
            self.output_layer.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data for each epoch
            indices = torch.randperm(n_samples)
            X_train_shuffled = X_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]
            if sw_train_tensor is not None:
                sw_train_shuffled = sw_train_tensor[indices]
            else:
                sw_train_shuffled = None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                # Get sample weights for this batch
                if sw_train_shuffled is not None:
                    sw_batch = sw_train_shuffled[i:end_idx]
                else:
                    sw_batch = None
                
                optimizer.zero_grad()
                # Forward pass: X → Z → Y
                z = self.perception_net(X_batch)
                outputs = self.output_layer(z).squeeze()
                
                # Compute loss with sample weights
                if sw_batch is not None:
                    # Compute individual Huber losses for each sample
                    individual_losses = nn.functional.huber_loss(outputs, y_batch, reduction='none', delta=self.delta)
                    loss = (individual_losses * sw_batch).mean()
                else:
                    loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.perception_net.eval()
                self.output_layer.eval()
                with torch.no_grad():
                    # Forward pass: X → Z → Y
                    z_val = self.perception_net(X_val_tensor)
                    val_outputs = self.output_layer(z_val).squeeze()
                    
                    # Compute validation loss with sample weights
                    if sw_val_tensor is not None:
                        # Compute individual Huber losses for each validation sample
                        individual_val_losses = nn.functional.huber_loss(val_outputs, y_val_tensor, reduction='none', delta=self.delta)
                        val_loss = (individual_val_losses * sw_val_tensor).mean().item()
                    else:
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = self.model_.state_dict().copy()
                    else:
                        no_improve_count += 1
                        
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        # Restore the best model
                        if best_state_dict is not None:
                            self.model_.load_state_dict(best_state_dict)
                        break
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_epoch_loss:.6f}")
        
        self.n_iter_ = epoch + 1
        
        # Set sklearn compatibility attributes
        self.n_layers_ = len(self.hidden_layer_sizes) + 1  # +1 for output layer
        self.n_outputs_ = 1  # Regression has single output
        self.out_activation_ = 'identity'  # Linear output for regression
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"MLPHuberRegressor fitted with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Training completed in {self.n_iter_} iterations")
            
        return self
    
    def predict(self, X):
        """
        Predict using the Huber regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPHuberRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPHuberRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPHuberRegressor instance is not fitted yet.")
        
        # Input data is assumed to be pre-scaled
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Predict using perception_net + output_layer
        self.perception_net.eval()
        self.output_layer.eval()
        with torch.no_grad():
            # Forward pass: X → Z → Y
            z = self.perception_net(X_tensor)
            y_pred_scaled = self.output_layer(z).squeeze().cpu().numpy()
        
        # Output is on the scaled scale, benchmark runner should inverse transform if needed
        return y_pred_scaled
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R² of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)
    
    def get_representation(self, X):
        """
        Extract the perception representation Z from input X.
        
        This method exposes the intermediate representation from the perception network,
        allowing analysis of the X → Z transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        Z : ndarray of shape (n_samples, representation_size)
            The perception representation extracted from the perception network.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPHuberRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPHuberRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPHuberRegressor instance is not fitted yet.")
        
        # Convert to tensor and move to same device as model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Extract representation using perception network: X → Z
        self.perception_net.eval()
        with torch.no_grad():
            Z = self.perception_net(X_tensor)
            return Z.cpu().numpy()
    
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'PyTorch parameters have complex defaults'
            }
        }


class MLPPinballRegressor(BaseEstimator, RegressorMixin):
    """
    Multi-layer Perceptron regressor with Pinball (Quantile) loss.
    
    Uses Pinball loss function for quantile regression, robust to outliers.
    Provides sklearn-compatible interface with fit/predict methods.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
        
    max_iter : int, default=1000
        Maximum number of training iterations.
        
    learning_rate : float, default=0.001
        Learning rate for optimization.
        
    early_stopping : bool, default=True
        Whether to use early stopping during training.
        
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation when early_stopping=True.
        
    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.
        
    tol : float, default=1e-4
        Tolerance for optimization convergence.
        
    quantile : float, default=0.5
        The quantile to estimate. 0.5 corresponds to median regression.
        
    random_state : int, default=None
        Random state for reproducibility.
        
    verbose : bool, default=False
        Whether to print progress messages.
        
    alpha : float, default=0.0
        L2 regularization parameter.
        
    batch_size : int or 'auto', default='auto'
        Size of minibatches for stochastic optimizers.
    """
    
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        max_iter=1000,
        learning_rate=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        quantile=0.5,
        random_state=None,
        verbose=False,
        alpha=0.0,
        batch_size='auto'
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.quantile = quantile
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Will be set during fit
        self.model_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
    
    def _pinball_loss(self, y_pred, y_true, reduction='mean'):
        """Pinball loss (quantile loss)"""
        error = y_true - y_pred
        loss = torch.where(error >= 0, 
                          self.quantile * error, 
                          (self.quantile - 1) * error)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def _build_perception_network(self, input_size):
        """Build perception network: X → Z"""
        if not self.hidden_layer_sizes:
            # If no hidden layers, perception network is just identity (return input)
            return nn.Identity()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers for perception
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _build_model(self, input_size):
        """Build perception network + output layer architecture"""
        # Build perception network: X → Z
        self.perception_net = self._build_perception_network(input_size)
        
        # Determine representation size
        if self.hidden_layer_sizes:
            representation_size = self.hidden_layer_sizes[-1]  # Last hidden layer size
        else:
            representation_size = input_size  # No hidden layers, use input size
        
        # Build output layer: Z → Y
        self.output_layer = nn.Linear(representation_size, 1)
        
        # Create a module list to hold both components
        return nn.ModuleList([self.perception_net, self.output_layer])
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Pinball regressor to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, the loss function will be weighted by these values.
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        
        # Data is assumed to be pre-scaled. No internal scaling.
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y, sample_weight,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            sw_train, sw_val = sample_weight, None
        
        # Convert to torch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        
        # Convert sample weights to tensor if provided
        if sw_train is not None:
            sw_train_tensor = torch.FloatTensor(sw_train).to(device)
        else:
            sw_train_tensor = None
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            
            # Convert validation sample weights to tensor if provided
            if sw_val is not None:
                sw_val_tensor = torch.FloatTensor(sw_val).to(device)
            else:
                sw_val_tensor = None
        
        # Build model
        self.model_ = self._build_model(self.n_features_in_).to(device)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        
        # Determine batch size
        n_samples = X_train.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        elif self.batch_size is None:
            batch_size = n_samples  # Full-batch training
        else:
            batch_size = min(self.batch_size, n_samples)
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # Training step with mini-batches
            self.model_.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data for each epoch
            indices = torch.randperm(n_samples)
            X_train_shuffled = X_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]
            if sw_train_tensor is not None:
                sw_train_shuffled = sw_train_tensor[indices]
            else:
                sw_train_shuffled = None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                # Get sample weights for this batch
                if sw_train_shuffled is not None:
                    sw_batch = sw_train_shuffled[i:end_idx]
                else:
                    sw_batch = None
                
                optimizer.zero_grad()
                # Forward pass: X → Z → Y
                z = self.perception_net(X_batch)
                outputs = self.output_layer(z).squeeze()
                
                # Compute loss with sample weights
                if sw_batch is not None:
                    # Compute individual Pinball losses for each sample
                    individual_losses = self._pinball_loss(outputs, y_batch, reduction='none')
                    loss = (individual_losses * sw_batch).mean()
                else:
                    loss = self._pinball_loss(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.perception_net.eval()
                self.output_layer.eval()
                with torch.no_grad():
                    # Forward pass: X → Z → Y
                    z_val = self.perception_net(X_val_tensor)
                    val_outputs = self.output_layer(z_val).squeeze()
                    
                    # Compute validation loss with sample weights
                    if sw_val_tensor is not None:
                        # Compute individual Pinball losses for each validation sample
                        individual_val_losses = self._pinball_loss(val_outputs, y_val_tensor, reduction='none')
                        val_loss = (individual_val_losses * sw_val_tensor).mean().item()
                    else:
                        val_loss = self._pinball_loss(val_outputs, y_val_tensor).item()
                    
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = self.model_.state_dict().copy()
                    else:
                        no_improve_count += 1
                        
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        # Restore the best model
                        if best_state_dict is not None:
                            self.model_.load_state_dict(best_state_dict)
                        break
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_epoch_loss:.6f}")
        
        self.n_iter_ = epoch + 1
        
        # Set sklearn compatibility attributes
        self.n_layers_ = len(self.hidden_layer_sizes) + 1  # +1 for output layer
        self.n_outputs_ = 1  # Regression has single output
        self.out_activation_ = 'identity'  # Linear output for regression
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"MLPPinballRegressor fitted with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Training completed in {self.n_iter_} iterations")
            
        return self
    
    def predict(self, X):
        """
        Predict using the Pinball regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPPinballRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPinballRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPPinballRegressor instance is not fitted yet.")
        
        # Input data is assumed to be pre-scaled
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Predict using perception_net + output_layer
        self.perception_net.eval()
        self.output_layer.eval()
        with torch.no_grad():
            # Forward pass: X → Z → Y
            z = self.perception_net(X_tensor)
            y_pred_scaled = self.output_layer(z).squeeze().cpu().numpy()
        
        # Output is on the scaled scale, benchmark runner should inverse transform if needed
        return y_pred_scaled
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R² of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)
    
    def get_representation(self, X):
        """
        Extract the perception representation Z from input X.
        
        This method exposes the intermediate representation from the perception network,
        allowing analysis of the X → Z transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        Z : ndarray of shape (n_samples, representation_size)
            The perception representation extracted from the perception network.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPPinballRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPinballRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPPinballRegressor instance is not fitted yet.")
        
        # Convert to tensor and move to same device as model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Extract representation using perception network: X → Z
        self.perception_net.eval()
        with torch.no_grad():
            Z = self.perception_net(X_tensor)
            return Z.cpu().numpy()
    
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'PyTorch parameters have complex defaults'
            }
        }


class MLPCauchyRegressor(BaseEstimator, RegressorMixin):
    """
    Multi-layer Perceptron regressor with Cauchy loss.
    
    Uses Cauchy loss function for robust regression against heavy-tailed distributions.
    Provides sklearn-compatible interface with fit/predict methods.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
        
    max_iter : int, default=1000
        Maximum number of training iterations.
        
    learning_rate : float, default=0.001
        Learning rate for optimization.
        
    early_stopping : bool, default=True
        Whether to use early stopping during training.
        
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation when early_stopping=True.
        
    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.
        
    tol : float, default=1e-4
        Tolerance for optimization convergence.
        
    random_state : int, default=None
        Random state for reproducibility.
        
    verbose : bool, default=False
        Whether to print progress messages.
        
    alpha : float, default=0.0
        L2 regularization parameter.
        
    batch_size : int or 'auto', default='auto'
        Size of minibatches for stochastic optimizers.
    """
    
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        max_iter=1000,
        learning_rate=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=None,
        verbose=False,
        alpha=0.0,
        batch_size='auto'
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Will be set during fit
        self.model_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
    
    def _cauchy_loss(self, y_pred, y_true, reduction='mean'):
        """Cauchy loss function: log(1 + (y - yhat)^2)"""
        error = y_true - y_pred
        loss = torch.log(1 + error**2)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
    def _build_perception_network(self, input_size):
        """Build perception network: X → Z"""
        if not self.hidden_layer_sizes:
            # If no hidden layers, perception network is just identity (return input)
            return nn.Identity()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers for perception
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _build_model(self, input_size):
        """Build perception network + output layer architecture"""
        # Build perception network: X → Z
        self.perception_net = self._build_perception_network(input_size)
        
        # Determine representation size
        if self.hidden_layer_sizes:
            representation_size = self.hidden_layer_sizes[-1]  # Last hidden layer size
        else:
            representation_size = input_size  # No hidden layers, use input size
        
        # Build output layer: Z → Y
        self.output_layer = nn.Linear(representation_size, 1)
        
        # Create a module list to hold both components
        return nn.ModuleList([self.perception_net, self.output_layer])
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Cauchy regressor to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, the loss function will be weighted by these values.
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        
        # Data is assumed to be pre-scaled. No internal scaling.
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y, sample_weight,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            sw_train, sw_val = sample_weight, None
        
        # Convert to torch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        
        # Convert sample weights to tensor if provided
        if sw_train is not None:
            sw_train_tensor = torch.FloatTensor(sw_train).to(device)
        else:
            sw_train_tensor = None
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            
            # Convert validation sample weights to tensor if provided
            if sw_val is not None:
                sw_val_tensor = torch.FloatTensor(sw_val).to(device)
            else:
                sw_val_tensor = None
        
        # Build model
        self.model_ = self._build_model(self.n_features_in_).to(device)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        
        # Determine batch size
        n_samples = X_train.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        elif self.batch_size is None:
            batch_size = n_samples  # Full-batch training
        else:
            batch_size = min(self.batch_size, n_samples)
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # Training step with mini-batches
            self.model_.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data for each epoch
            indices = torch.randperm(n_samples)
            X_train_shuffled = X_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]
            if sw_train_tensor is not None:
                sw_train_shuffled = sw_train_tensor[indices]
            else:
                sw_train_shuffled = None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                # Get sample weights for this batch
                if sw_train_shuffled is not None:
                    sw_batch = sw_train_shuffled[i:end_idx]
                else:
                    sw_batch = None
                
                optimizer.zero_grad()
                # Forward pass: X → Z → Y
                z = self.perception_net(X_batch)
                outputs = self.output_layer(z).squeeze()
                
                # Compute loss with sample weights
                if sw_batch is not None:
                    # Compute individual Cauchy losses for each sample
                    individual_losses = self._cauchy_loss(outputs, y_batch, reduction='none')
                    loss = (individual_losses * sw_batch).mean()
                else:
                    loss = self._cauchy_loss(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.perception_net.eval()
                self.output_layer.eval()
                with torch.no_grad():
                    # Forward pass: X → Z → Y
                    z_val = self.perception_net(X_val_tensor)
                    val_outputs = self.output_layer(z_val).squeeze()
                    
                    # Compute validation loss with sample weights
                    if sw_val_tensor is not None:
                        # Compute individual Cauchy losses for each validation sample
                        individual_val_losses = self._cauchy_loss(val_outputs, y_val_tensor, reduction='none')
                        val_loss = (individual_val_losses * sw_val_tensor).mean().item()
                    else:
                        val_loss = self._cauchy_loss(val_outputs, y_val_tensor).item()
                    
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_state_dict = self.model_.state_dict().copy()
                    else:
                        no_improve_count += 1
                        
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        # Restore the best model
                        if best_state_dict is not None:
                            self.model_.load_state_dict(best_state_dict)
                        break
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_epoch_loss:.6f}")
        
        self.n_iter_ = epoch + 1
        
        # Set sklearn compatibility attributes
        self.n_layers_ = len(self.hidden_layer_sizes) + 1  # +1 for output layer
        self.n_outputs_ = 1  # Regression has single output
        self.out_activation_ = 'identity'  # Linear output for regression
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"MLPCauchyRegressor fitted with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Training completed in {self.n_iter_} iterations")
            
        return self
    
    def predict(self, X):
        """
        Predict using the Cauchy regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPCauchyRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPCauchyRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPCauchyRegressor instance is not fitted yet.")
        
        # Input data is assumed to be pre-scaled
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Predict using perception_net + output_layer
        self.perception_net.eval()
        self.output_layer.eval()
        with torch.no_grad():
            # Forward pass: X → Z → Y
            z = self.perception_net(X_tensor)
            y_pred_scaled = self.output_layer(z).squeeze().cpu().numpy()
        
        # Output is on the scaled scale, benchmark runner should inverse transform if needed
        return y_pred_scaled
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R² of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)
    
    def get_representation(self, X):
        """
        Extract the perception representation Z from input X.
        
        This method exposes the intermediate representation from the perception network,
        allowing analysis of the X → Z transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        Z : ndarray of shape (n_samples, representation_size)
            The perception representation extracted from the perception network.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.n_features_in_ is None:
            raise ValueError("This MLPCauchyRegressor instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPCauchyRegressor "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'perception_net') or self.perception_net is None:
            raise ValueError("This MLPCauchyRegressor instance is not fitted yet.")
        
        # Convert to tensor and move to same device as model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Extract representation using perception network: X → Z
        self.perception_net.eval()
        with torch.no_grad():
            Z = self.perception_net(X_tensor)
            return Z.cpu().numpy()
    
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'PyTorch parameters have complex defaults'
            }
        }