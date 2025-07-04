"""
MLPCausalClassifier: Scikit-learn compatible causal neural network classifier.

╔════════════════════════════════════════════════════════════════════════════════╗
║                        分类器模块架构图 - Classifier Suite                        ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║                         sklearn兼容的神经网络分类器集合                          ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                            输入层 (Input)                                 │  ║
║  │          X: [n_samples, n_features] + sample_weight (可选)                │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         两种分类器架构                                       │  ║
║  │                                                                           │  ║
║  │  ┌─────────────────────────────┐  ┌─────────────────────────────────┐    │  ║
║  │  │    MLPCausalClassifier      │  │    MLPPytorchClassifier         │    │  ║
║  │  │     因果推理分类器            │  │      标准PyTorch分类器           │    │  ║
║  │  │                             │  │                                 │    │  ║
║  │  │  🧠 CausalEngine 四阶段架构  │  │  🔧 标准MLP架构                  │    │  ║
║  │  │  📊 五种推理模式             │  │  ⚡ CrossEntropy损失             │    │  ║
║  │  │  🎯 OvR (One-vs-Rest)       │  │  🎛️ ReLU/Tanh/Sigmoid激活        │    │  ║
║  │  │  📈 分布预测能力             │  │  📊 Softmax概率输出              │    │  ║
║  │  │  🔄 自动数据标准化           │  │  🎲 基准对比用途                 │    │  ║
║  │  │  🎪 Cauchy/Softmax输出       │  │  💪 简单高效                    │    │  ║
║  │  └─────────────────────────────┘  └─────────────────────────────────┘    │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                           核心特性                                         │  ║
║  │                                                                           │  ║
║  │  🔧 sklearn兼容: fit/predict/predict_proba/score接口                       │  ║
║  │  ⚖️  样本权重: 完整支持sample_weight参数                                    │  ║
║  │  📈 早停机制: validation-based early stopping                             │  ║
║  │  🎛️  批处理: 自动/手动batch size配置                                        │  ║
║  │  🎲 随机种子: 可重现的random_state控制                                      │  ║
║  │  📊 多类支持: 自动处理多类分类任务                                          │  ║
║  │  🔄 标签映射: 自动处理非数值标签                                           │  ║
║  │  📈 分层采样: 验证集保持类别分布                                           │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         输出层 (Output)                                   │  ║
║  │     • predict(): 类别标签 [n_samples]                                      │  ║
║  │     • predict_proba(): 类别概率 [n_samples, n_classes]                     │  ║
║  │     • predict_log_proba(): 对数概率 [n_samples, n_classes]                 │  ║
║  │     • predict_dist(): 分布参数 [n_samples, n_classes, 2] (仅CausalEngine)  │  ║
║  │     • score(): 准确率评估                                                  │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                           使用场景指南                                      │  ║
║  │                                                                           │  ║
║  │  🧠 MLPCausalClassifier → 需要因果推理、不确定性量化、分布式分类              │  ║
║  │  🔧 MLPPytorchClassifier → 标准分类基线、性能对比、简单分类任务             │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import inspect

from ._causal_engine.engine import CausalEngine

class MLPCausalClassifier(BaseEstimator, ClassifierMixin):
    """
    Causal Multi-layer Perceptron Classifier.
    
    A scikit-learn compatible neural network classifier that uses causal reasoning
    to understand relationships in data rather than just fitting patterns.
    
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
        
    ovr_threshold : float, default=0.0
        Threshold for One-vs-Rest classification.
        
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
        ovr_threshold=0.0,
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
        self.ovr_threshold = ovr_threshold
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
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
    def _build_model(self):
        """Build CausalEngine model"""
        return CausalEngine(
            input_size=self.n_features_in_,
            output_size=len(self.classes_),
            repre_size=self.repre_size,
            causal_size=self.causal_size,
            task_type='classification',
            perception_hidden_layers=self.perception_hidden_layers,
            abduction_hidden_layers=self.abduction_hidden_layers,
            gamma_init=self.gamma_init,
            b_noise_init=self.b_noise_init,
            b_noise_trainable=self.b_noise_trainable,
            ovr_threshold=self.ovr_threshold,
            alpha=self.alpha
        )
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the causal classifier to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, the loss function will be weighted by these values.
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
            # Convert to tensor
            sample_weight_tensor = torch.FloatTensor(sample_weight)
        else:
            sample_weight_tensor = None
        
        # Store classes and input info
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Data preprocessing
        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        
        # Convert labels to indices
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_indexed = np.array([self.label_to_idx_[label] for label in y])
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight_tensor is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X_scaled, y_indexed, sample_weight_tensor, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y_indexed
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_indexed, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y_indexed
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train = X_scaled, y_indexed
            X_val, y_val = None, None
            sw_train, sw_val = sample_weight_tensor, None
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
        
        # Create CausalEngine using _build_model
        self.engine_ = self._build_model()
        
        # Setup optimizer
        optimizer = optim.Adam(self.engine_.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        
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
            sw_train_shuffled = sw_train[indices] if sw_train is not None else None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                sw_batch = sw_train_shuffled[i:end_idx] if sw_train_shuffled is not None else None
                
                optimizer.zero_grad()
                
                # Compute loss with sample weights
                if sw_batch is not None:
                    # Get individual losses for each sample
                    decision_scores = self.engine_._get_decision_scores(X_batch, self.mode)
                    individual_losses = self.engine_.decision_head.compute_loss(
                        y_true=y_batch,
                        decision_scores=decision_scores,
                        mode=self.mode,
                        reduction='none'
                    )
                    # Apply sample weights
                    weighted_losses = individual_losses * sw_batch
                    loss = torch.mean(weighted_losses)
                else:
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
                    # Compute validation loss with sample weights if available
                    if sw_val is not None:
                        decision_scores = self.engine_._get_decision_scores(X_val_tensor, self.mode)
                        individual_losses = self.engine_.decision_head.compute_loss(
                            y_true=y_val_tensor,
                            decision_scores=decision_scores,
                            mode=self.mode,
                            reduction='none'
                        )
                        weighted_losses = individual_losses * sw_val
                        val_loss = torch.mean(weighted_losses)
                    else:
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
        self.n_outputs_ = len(self.classes_)  # Number of classes
        self.out_activation_ = 'softmax'  # Multi-class classification output
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"MLPCausalClassifier fitted with {X.shape[0]} samples, "
                  f"{X.shape[1]} features, {len(self.classes_)} classes")
            print(f"Training completed in {self.n_iter_} iterations")
            
        return self
        
    def predict(self, X):
        """
        Predict class labels using the causal classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPCausalClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPCausalClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'engine_'):
            raise ValueError("This MLPCausalClassifier instance is not fitted yet.")
        
        # Preprocess input
        X_scaled = self.scaler_X_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predict using CausalEngine
        self.engine_.eval()
        with torch.no_grad():
            class_indices = self.engine_.predict(X_tensor, mode=self.mode)
            # Convert back to original class labels
            y_pred = self.classes_[class_indices.cpu().numpy()]
            
        return y_pred
        
    def predict_proba(self, X):
        """
        Predict class probabilities using the causal classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPCausalClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPCausalClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'engine_'):
            raise ValueError("This MLPCausalClassifier instance is not fitted yet.")
        
        # Preprocess input
        X_scaled = self.scaler_X_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predict probabilities using CausalEngine
        self.engine_.eval()
        with torch.no_grad():
            proba = self.engine_.predict_proba(X_tensor, mode=self.mode)
            # Convert to numpy
            proba_np = proba.cpu().numpy()
            
        return proba_np
        
    def predict_dist(self, X):
        """
        Predict distribution parameters for One-vs-Rest activations.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        dist_params : ndarray of shape (n_samples, n_classes, n_params)
            Distribution parameters (location, scale) for OvR activations.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.mode == 'deterministic':
            raise ValueError("Distribution prediction not available in deterministic mode.")
            
        if self.classes_ is None:
            raise ValueError("This MLPCausalClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPCausalClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'engine_'):
            raise ValueError("This MLPCausalClassifier instance is not fitted yet.")
        
        # Preprocess input
        X_scaled = self.scaler_X_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
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
            
            # Stack location and scale as the last dimension
            # Shape: (n_samples, n_classes, 2) where last dim is [location, scale]
            dist_params_np = np.stack([location_np, scale_np], axis=-1)
            
        return dist_params_np
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. Should be preprocessed consistently with training data.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def predict_log_proba(self, X):
        """
        Return the log of probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Should be preprocessed consistently with training data.
            
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            The log of the probability of the sample for each class.
        """
        proba = self.predict_proba(X)
        # Avoid log(0) by adding small epsilon
        proba = np.clip(proba, 1e-15, 1.0)
        return np.log(proba)
        
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'CausalEngine parameters have complex defaults'
            }
        }


class MLPPytorchClassifier(BaseEstimator, ClassifierMixin):
    """
    PyTorch Multi-layer Perceptron Classifier.
    
    A scikit-learn compatible PyTorch neural network classifier for baseline comparison.
    This provides a standard MLP implementation using PyTorch with the same interface
    as MLPCausalClassifier.
    
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
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
    def _build_model(self, input_size, output_size):
        """Build PyTorch MLP model"""
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the PyTorch classifier to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
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
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
            # Convert to tensor
            sample_weight_tensor = torch.FloatTensor(sample_weight)
        else:
            sample_weight_tensor = None
        
        # Store classes and input info
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Convert labels to indices
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_indexed = np.array([self.label_to_idx_[label] for label in y])
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight_tensor is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y_indexed, sample_weight_tensor, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y_indexed
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_indexed, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y_indexed
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train = X, y_indexed
            X_val, y_val = None, None
            sw_train, sw_val = sample_weight_tensor, None
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
        
        # Build model
        self.model_ = self._build_model(self.n_features_in_, len(self.classes_))
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        criterion = nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' for manual weighting
        
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
            sw_train_shuffled = sw_train[indices] if sw_train is not None else None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                sw_batch = sw_train_shuffled[i:end_idx] if sw_train_shuffled is not None else None
                
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                individual_losses = criterion(outputs, y_batch)
                
                # Apply sample weights if provided
                if sw_batch is not None:
                    weighted_losses = individual_losses * sw_batch
                    loss = torch.mean(weighted_losses)
                else:
                    loss = torch.mean(individual_losses)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val_tensor)
                    individual_val_losses = criterion(val_outputs, y_val_tensor)
                    
                    # Apply sample weights if provided
                    if sw_val is not None:
                        weighted_val_losses = individual_val_losses * sw_val
                        val_loss = torch.mean(weighted_val_losses).item()
                    else:
                        val_loss = torch.mean(individual_val_losses).item()
                    
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
        self.n_outputs_ = len(self.classes_)  # Number of classes
        self.out_activation_ = 'softmax'  # Multi-class classification output
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"MLPPytorchClassifier fitted with {X.shape[0]} samples, "
                  f"{X.shape[1]} features, {len(self.classes_)} classes")
            print(f"Training completed in {self.n_iter_} iterations")
            
        return self
        
    def predict(self, X):
        """
        Predict class labels using the PyTorch classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPPytorchClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPytorchClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError("This MLPPytorchClassifier instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict using PyTorch model
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            class_indices = torch.argmax(outputs, dim=1).cpu().numpy()
            # Convert back to original class labels
            y_pred = self.classes_[class_indices]
            
        return y_pred
        
    def predict_proba(self, X):
        """
        Predict class probabilities using the PyTorch classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPPytorchClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPytorchClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError("This MLPPytorchClassifier instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict probabilities using PyTorch model
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
            
        return proba
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def predict_log_proba(self, X):
        """
        Return the log of probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            The log of the probability of the sample for each class.
        """
        proba = self.predict_proba(X)
        # Avoid log(0) by adding small epsilon
        proba = np.clip(proba, 1e-15, 1.0)
        return np.log(proba)
        
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'PyTorch parameters have complex defaults'
            }
        }


class MLPSklearnOvRClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn OneVsRest MLP Classifier.
    
    A scikit-learn compatible One-vs-Rest classifier that uses MLPClassifier as base estimator.
    This implementation trains K independent binary classifiers for K classes, where each 
    classifier learns to distinguish one class from all others.
    
    数学原理：
    - 为K个类别训练K个独立的二分类MLPClassifier
    - 分类器i: f_i(x) → P(y=i | x) vs P(y≠i | x)  
    - 最终预测: ŷ = argmax_i P(y=i | x)
    - 无参数共享，每个分类器独立优化
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The hidden layer structure for each MLP classifier.
        
    max_iter : int, default=1000
        Maximum number of training iterations for each classifier.
        
    learning_rate_init : float, default=0.001
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
        
    alpha : float, default=0.0001
        L2 regularization parameter.
    """
    
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        max_iter=1000,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=None,
        verbose=False,
        alpha=0.0001,
        batch_size='auto'
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Will be set during fit
        self.ovr_classifier_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the OneVsRest classifier to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, each base classifier will be weighted by these values.
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Store classes and input info
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Create base MLPClassifier
        base_classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            random_state=self.random_state,
            verbose=False,  # We'll handle verbosity ourselves
            alpha=self.alpha,
            batch_size=self.batch_size
        )
        
        # Create OneVsRest classifier
        self.ovr_classifier_ = OneVsRestClassifier(
            base_classifier, 
            n_jobs=1  # Keep single-threaded for reproducibility
        )
        
        # Fit the OneVsRest classifier
        if self.verbose:
            print(f"🔧 训练 sklearn OvR MLP (训练{len(self.classes_)}个独立的二分类器)...")
        
        # OneVsRestClassifier doesn't support sample_weight directly, so we fit without it
        if sample_weight is not None:
            print("Warning: sample_weight is not supported with OneVsRestClassifier")
        self.ovr_classifier_.fit(X, y)
        
        # Calculate average number of iterations across all classifiers
        n_iters = []
        total_loss = 0.0
        n_classifiers = 0
        
        for estimator in self.ovr_classifier_.estimators_:
            if hasattr(estimator, 'n_iter_'):
                n_iters.append(estimator.n_iter_)
            if hasattr(estimator, 'loss_'):
                total_loss += estimator.loss_
                n_classifiers += 1
        
        self.n_iter_ = int(np.mean(n_iters)) if n_iters else self.max_iter
        
        # Set sklearn compatibility attributes
        self.n_layers_ = len(self.hidden_layer_sizes) + 1  # +1 for output layer
        self.n_outputs_ = len(self.classes_)  # Number of classes
        self.out_activation_ = 'sigmoid_ovr'  # OvR with sigmoid outputs
        self.loss_ = total_loss / n_classifiers if n_classifiers > 0 else 0.0
        
        if self.verbose:
            print(f"   OvR训练完成: 平均 {self.n_iter_} epochs, {len(self.classes_)} 个二分类器")
            print(f"   MLPSklearnOvRClassifier fitted with {X.shape[0]} samples, "
                  f"{X.shape[1]} features, {len(self.classes_)} classes")
            
        return self
        
    def predict(self, X):
        """
        Predict class labels using the OneVsRest classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPSklearnOvRClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPSklearnOvRClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'ovr_classifier_') or self.ovr_classifier_ is None:
            raise ValueError("This MLPSklearnOvRClassifier instance is not fitted yet.")
        
        return self.ovr_classifier_.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities using the OneVsRest classifier.
        
        Note: In OvR, the probabilities don't necessarily sum to 1 across classes,
        as each classifier makes independent binary decisions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities from OvR binary classifiers.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPSklearnOvRClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPSklearnOvRClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'ovr_classifier_') or self.ovr_classifier_ is None:
            raise ValueError("This MLPSklearnOvRClassifier instance is not fitted yet.")
        
        return self.ovr_classifier_.predict_proba(X)
    
    def decision_function(self, X):
        """
        Return the decision function of the samples for each class in the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        decision : ndarray of shape (n_samples, n_classes)
            Decision function values for each class.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPSklearnOvRClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPSklearnOvRClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'ovr_classifier_') or self.ovr_classifier_ is None:
            raise ValueError("This MLPSklearnOvRClassifier instance is not fitted yet.")
        
        return self.ovr_classifier_.decision_function(X)
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def predict_log_proba(self, X):
        """
        Return the log of probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            The log of the probability of the sample for each class.
        """
        proba = self.predict_proba(X)
        # Avoid log(0) by adding small epsilon
        proba = np.clip(proba, 1e-15, 1.0)
        return np.log(proba)
        
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            'multiclass_only': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'OneVsRest parameters have complex defaults'
            }
        }


class MLPPytorchSharedOvRClassifier(BaseEstimator, ClassifierMixin):
    """
    PyTorch Shared One-vs-Rest MLP Classifier.
    
    A scikit-learn compatible One-vs-Rest classifier with shared feature extraction.
    This implementation uses a shared MLP network for feature extraction, followed by 
    K independent sigmoid classification heads for K classes.
    
    数学原理：
    - 共享感知网络：h = f_shared(x; θ_shared)
    - K个独立分类头：z_i = W_i · h + b_i  
    - Sigmoid激活：p_i = σ(z_i) = 1/(1 + e^(-z_i))
    - OvR损失：L = Σ_i BCE(p_i, y_i) 其中y_i ∈ {0,1}
    - 最终预测：ŷ = argmax_i p_i
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The hidden layer structure for the shared MLP network.
        
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
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
    def _build_model(self, input_size, n_classes):
        """Build shared OvR PyTorch model"""
        
        class SharedOvRMLP(nn.Module):
            def __init__(self, input_size, hidden_sizes, n_classes, activation='relu'):
                super().__init__()
                
                # Shared feature extraction layers
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    if activation == 'relu':
                        layers.append(nn.ReLU())
                    elif activation == 'tanh':
                        layers.append(nn.Tanh())
                    elif activation == 'sigmoid':
                        layers.append(nn.Sigmoid())
                    prev_size = hidden_size
                
                self.shared_layers = nn.Sequential(*layers)
                
                # Independent classification heads for each class (OvR)
                self.classification_heads = nn.ModuleList([
                    nn.Linear(prev_size, 1) for _ in range(n_classes)
                ])
                
            def forward(self, x):
                # Shared feature extraction
                shared_features = self.shared_layers(x)
                
                # Independent binary classification for each class
                class_outputs = []
                for head in self.classification_heads:
                    class_outputs.append(head(shared_features))
                
                # Stack outputs: [n_samples, n_classes]
                return torch.cat(class_outputs, dim=1)
        
        return SharedOvRMLP(input_size, self.hidden_layer_sizes, n_classes, self.activation)
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Shared OvR classifier to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
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
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, accept_sparse=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, but X has {X.shape[0]} samples.")
            # Convert to tensor
            sample_weight_tensor = torch.FloatTensor(sample_weight)
        else:
            sample_weight_tensor = None
        
        # Store classes and input info
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Convert labels to one-hot encoding for OvR training
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_indexed = np.array([self.label_to_idx_[label] for label in y])
        
        # Create one-hot encoding for OvR
        n_classes = len(self.classes_)
        y_onehot = np.zeros((len(y), n_classes))
        y_onehot[np.arange(len(y)), y_indexed] = 1
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            if sample_weight_tensor is not None:
                X_train, X_val, y_train_onehot, y_val_onehot, sw_train, sw_val = train_test_split(
                    X, y_onehot, sample_weight_tensor, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y_indexed
                )
            else:
                X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
                    X, y_onehot, 
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y_indexed
                )
                sw_train, sw_val = None, None
        else:
            X_train, y_train_onehot = X, y_onehot
            X_val, y_val_onehot = None, None
            sw_train, sw_val = sample_weight_tensor, None
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train_onehot)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val_onehot)
        
        # Build model
        self.model_ = self._build_model(self.n_features_in_, n_classes)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        # Use BCEWithLogitsLoss for numerical stability (combines sigmoid + BCE)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        if self.verbose:
            print(f"🔧 训练 PyTorch Shared OvR (共享网络+{n_classes}个分类头)...")
        
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
            sw_train_shuffled = sw_train[indices] if sw_train is not None else None
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                sw_batch = sw_train_shuffled[i:end_idx] if sw_train_shuffled is not None else None
                
                optimizer.zero_grad()
                outputs = self.model_(X_batch)  # [batch_size, n_classes]
                
                # Compute loss for each class independently (OvR)
                individual_losses = criterion(outputs, y_batch)  # [batch_size, n_classes]
                
                # Apply sample weights if provided
                if sw_batch is not None:
                    # Expand sample weights to match [batch_size, n_classes]
                    sw_expanded = sw_batch.unsqueeze(1).expand_as(individual_losses)
                    weighted_losses = individual_losses * sw_expanded
                    loss = torch.mean(weighted_losses)
                else:
                    loss = torch.mean(individual_losses)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val_tensor)
                    individual_val_losses = criterion(val_outputs, y_val_tensor)
                    
                    # Apply sample weights if provided
                    if sw_val is not None:
                        sw_val_expanded = sw_val.unsqueeze(1).expand_as(individual_val_losses)
                        weighted_val_losses = individual_val_losses * sw_val_expanded
                        val_loss = torch.mean(weighted_val_losses).item()
                    else:
                        val_loss = torch.mean(individual_val_losses).item()
                    
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
        self.n_layers_ = len(self.hidden_layer_sizes) + 1  # +1 for classification heads
        self.n_outputs_ = len(self.classes_)  # Number of classes
        self.out_activation_ = 'sigmoid_ovr_shared'  # Shared OvR with sigmoid outputs
        self.loss_ = avg_epoch_loss  # Final training loss
        
        if self.verbose:
            print(f"   Shared OvR训练完成: {self.n_iter_} epochs")
            print(f"   MLPPytorchSharedOvRClassifier fitted with {X.shape[0]} samples, "
                  f"{X.shape[1]} features, {len(self.classes_)} classes")
            
        return self
        
    def predict(self, X):
        """
        Predict class labels using the Shared OvR classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPPytorchSharedOvRClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPytorchSharedOvRClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError("This MLPPytorchSharedOvRClassifier instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict using Shared OvR model
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)  # [n_samples, n_classes]
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs)
            # Choose class with highest probability
            class_indices = torch.argmax(probabilities, dim=1).cpu().numpy()
            # Convert back to original class labels
            y_pred = self.classes_[class_indices]
            
        return y_pred
        
    def predict_proba(self, X):
        """
        Predict class probabilities using the Shared OvR classifier.
        
        Note: In OvR, the probabilities don't necessarily sum to 1 across classes,
        as each classifier makes independent binary decisions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities from shared OvR classifier.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPPytorchSharedOvRClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPytorchSharedOvRClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError("This MLPPytorchSharedOvRClassifier instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict probabilities using Shared OvR model
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)  # [n_samples, n_classes]
            # Apply sigmoid to get independent probabilities for each class
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
        return probabilities
    
    def decision_function(self, X):
        """
        Return the decision function of the samples for each class in the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        decision : ndarray of shape (n_samples, n_classes)
            Decision function values for each class (before sigmoid).
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        if self.classes_ is None:
            raise ValueError("This MLPPytorchSharedOvRClassifier instance is not fitted yet.")
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MLPPytorchSharedOvRClassifier "
                           f"is expecting {self.n_features_in_} features.")
        
        # Check if model is fitted
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError("This MLPPytorchSharedOvRClassifier instance is not fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Get raw decision scores (logits)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor).cpu().numpy()  # [n_samples, n_classes]
            
        return outputs
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def predict_log_proba(self, X):
        """
        Return the log of probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            The log of the probability of the sample for each class.
        """
        proba = self.predict_proba(X)
        # Avoid log(0) by adding small epsilon
        proba = np.clip(proba, 1e-15, 1.0)
        return np.log(proba)
        
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            'multiclass_only': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'Shared OvR parameters have complex defaults'
            }
        }