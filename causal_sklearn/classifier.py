"""
MLPCausalClassifier: Scikit-learn compatible causal neural network classifier.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from ._causal_engine import create_causal_classifier

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
        self.ovr_threshold = ovr_threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be set during fit
        self.engine_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.loss_ = None
        
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
            Sample weights (not implemented yet).
            
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
        
        # Data preprocessing
        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        
        # Convert labels to indices
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_indexed = np.array([self.label_to_idx_[label] for label in y])
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_indexed, 
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_indexed
            )
        else:
            X_train, y_train = X_scaled, y_indexed
            X_val, y_val = None, None
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
        
        # Create CausalEngine
        self.engine_ = create_causal_classifier(
            input_size=self.n_features_in_,
            n_classes=len(self.classes_),
            repre_size=self.repre_size,
            causal_size=self.causal_size,
            perception_hidden_layers=self.perception_hidden_layers,
            abduction_hidden_layers=self.abduction_hidden_layers,
            gamma_init=self.gamma_init,
            b_noise_init=self.b_noise_init,
            b_noise_trainable=self.b_noise_trainable,
            ovr_threshold=self.ovr_threshold
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
        
        # Set sklearn compatibility attributes
        self.n_layers_ = (len(self.perception_hidden_layers) + 
                         len(self.abduction_hidden_layers) + 2)  # +2 for input and output layers
        self.n_outputs_ = len(self.classes_)  # Number of classes
        self.out_activation_ = 'softmax'  # Multi-class classification output
        self.loss_ = loss.item()  # Final training loss
        
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
        alpha=0.0001
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
            Sample weights (not implemented yet).
            
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
        
        # Store classes and input info
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Convert labels to indices
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_indexed = np.array([self.label_to_idx_[label] for label in y])
        
        # Split for validation if early stopping is enabled
        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_indexed, 
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_indexed
            )
        else:
            X_train, y_train = X, y_indexed
            X_val, y_val = None, None
        
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
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        best_state_dict = None
        
        for epoch in range(self.max_iter):
            # Training step
            self.model_.train()
            optimizer.zero_grad()
            
            outputs = self.model_(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation step
            if self.early_stopping and X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val_tensor)
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
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss.item():.6f}")
        
        self.n_iter_ = epoch + 1
        
        # Set sklearn compatibility attributes
        self.n_layers_ = len(self.hidden_layer_sizes) + 1  # +1 for output layer
        self.n_outputs_ = len(self.classes_)  # Number of classes
        self.out_activation_ = 'softmax'  # Multi-class classification output
        self.loss_ = loss.item()  # Final training loss
        
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