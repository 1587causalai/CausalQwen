# Causal-Sklearn

A scikit-learn compatible implementation of CausalEngine for causal machine learning.

## Overview

Causal-Sklearn brings the power of causal reasoning to the familiar scikit-learn ecosystem. Built on top of the revolutionary CausalEngine™ algorithm, it provides drop-in replacements for traditional ML estimators that understand causation rather than just correlation.

## Key Features

- **Scikit-learn Compatible**: Drop-in replacements for `MLPRegressor` and `MLPClassifier`
- **Causal Reasoning**: Goes beyond pattern matching to understand causal relationships
- **Robust to Noise**: Superior performance in the presence of label noise and outliers
- **Distribution Prediction**: Provides full distributional outputs, not just point estimates
- **Multiple Modes**: Supports deterministic, standard, and sampling prediction modes

## Quick Start

```python
from causal_sklearn import MLPCausalRegressor, MLPCausalClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

# Regression example
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = MLPCausalRegressor(mode='standard', random_state=42)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# Classification example
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = MLPCausalClassifier(mode='standard', random_state=42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

## Installation

```bash
pip install causal-sklearn
```

For development installation:

```bash
git clone https://github.com/yourusername/causal-sklearn.git
cd causal-sklearn
pip install -e ".[dev]"
```

## Models

### MLPCausalRegressor

A causal neural network regressor that understands causal relationships in regression tasks.

**Key Parameters:**
- `mode`: Prediction mode ('deterministic', 'standard', 'sampling')
- `hidden_layer_sizes`: Architecture of hidden layers
- `gamma_init`: Initial scale for AbductionNetwork
- `b_noise_init`: Initial noise level for ActionNetwork

### MLPCausalClassifier

A causal neural network classifier for classification tasks.

**Key Parameters:**
- `mode`: Prediction mode ('deterministic', 'standard', 'sampling')
- `hidden_layer_sizes`: Architecture of hidden layers
- `ovr_threshold_init`: Initial threshold for One-vs-Rest classification

## Benchmarking

Compare CausalEngine with traditional methods:

```python
from causal_sklearn.benchmarks import ComparisonBenchmark

benchmark = ComparisonBenchmark()
results = benchmark.run_regression_comparison(dataset='california_housing')
results = benchmark.run_classification_comparison(dataset='wine_quality')
```

## Documentation

For detailed documentation, examples, and API reference, visit: [Documentation Link]

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Causal-Sklearn in your research, please cite:

```bibtex
@software{causal_sklearn,
  title={Causal-Sklearn: Scikit-learn Compatible Causal Machine Learning},
  author={CausalEngine Team},
  year={2024},
  url={https://github.com/yourusername/causal-sklearn}
}
```