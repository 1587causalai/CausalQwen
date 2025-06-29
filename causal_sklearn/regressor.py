"""
MLPCausalRegressor: Scikit-learn compatible causal neural network regressor.
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from ._causal_engine import create_causal_regressor

class MLPCausalRegressor(BaseEstimator, RegressorMixin):
    """
    Causal Multi-layer Perceptron Regressor.
    
    A scikit-learn compatible neural network regressor that uses causal reasoning
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
        verbose=False
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
        
        # Will be set during fit
        self._causal_engine = None
        self.n_features_in_ = None
        self.n_iter_ = None
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the causal regressor to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights (not implemented yet).
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        
        # Data preprocessing
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()
        
        X_scaled = self.scaler_X_.fit_transform(X)
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, 
                test_size=self.validation_fraction,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X_scaled, y_scaled
            X_val, y_val = None, None
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # CausalEngine expects y to be 2D for regression
        if len(y_train_tensor.shape) == 1:
            y_train_tensor = y_train_tensor.unsqueeze(1)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            # Also reshape validation y
            if len(y_val_tensor.shape) == 1:
                y_val_tensor = y_val_tensor.unsqueeze(1)
        
        # Create CausalEngine
        self.engine_ = create_causal_regressor(
            input_size=self.n_features_in_,
            output_size=1,
            repre_size=self.repre_size,
            causal_size=self.causal_size,
            perception_hidden_layers=self.perception_hidden_layers,
            abduction_hidden_layers=self.abduction_hidden_layers,
            gamma_init=self.gamma_init,
            b_noise_init=self.b_noise_init,
            b_noise_trainable=self.b_noise_trainable
        )
        
        # Setup optimizer
        optimizer = optim.Adam(self.engine_.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # Training step
            self.engine_.train()
            optimizer.zero_grad()
            
            loss = self.engine_.compute_loss(X_train_tensor, y_train_tensor, mode=self.mode)
            loss.backward()
            optimizer.step()
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.engine_.eval()
                with torch.no_grad():
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
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss.item():.6f}")
        
        self.n_iter_ = epoch + 1
        
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
            Input data.
            
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
        
        # Preprocess input
        X_scaled = self.scaler_X_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predict using CausalEngine
        self.engine_.eval()
        with torch.no_grad():
            y_pred_scaled = self.engine_.predict(X_tensor, mode=self.mode)
            if isinstance(y_pred_scaled, tuple):
                # If returns (location, scale), take location for point prediction
                y_pred_scaled = y_pred_scaled[0]
            
            # Convert back to numpy and inverse transform
            y_pred_scaled_np = y_pred_scaled.cpu().numpy()
            if y_pred_scaled_np.ndim > 1:
                y_pred_scaled_np = y_pred_scaled_np.ravel()
            
            # Inverse transform to original scale
            y_pred = self.scaler_y_.inverse_transform(y_pred_scaled_np.reshape(-1, 1)).ravel()
            
        return y_pred
        
    def predict_dist(self, X):
        """
        Predict distribution parameters using the causal regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
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
        
        # Preprocess input
        X_scaled = self.scaler_X_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predict distribution using CausalEngine
        self.engine_.eval()
        with torch.no_grad():
            dist_params = self.engine_.predict_distribution(X_tensor, mode=self.mode)
            
            if not isinstance(dist_params, tuple) or len(dist_params) != 2:
                raise RuntimeError("Expected distributional output (location, scale) but got different format")
            
            location_scaled, scale_scaled = dist_params
            
            # Convert to numpy
            location_scaled_np = location_scaled.cpu().numpy()
            scale_scaled_np = scale_scaled.cpu().numpy()
            
            if location_scaled_np.ndim > 1:
                location_scaled_np = location_scaled_np.ravel()
            if scale_scaled_np.ndim > 1:
                scale_scaled_np = scale_scaled_np.ravel()
            
            # Inverse transform location (scale doesn't need inverse transform as it's already in proper units)
            location = self.scaler_y_.inverse_transform(location_scaled_np.reshape(-1, 1)).ravel()
            
            # Scale parameter needs to be adjusted for the target scaling
            # Since we scaled y, the scale parameter also needs to be scaled back
            scale_factor = self.scaler_y_.scale_[0] if hasattr(self.scaler_y_, 'scale_') else 1.0
            scale = scale_scaled_np * scale_factor
            
        return np.column_stack([location, scale])
        
    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_parameters_default_constructible': 'CausalEngine parameters have complex defaults'
            }
        }