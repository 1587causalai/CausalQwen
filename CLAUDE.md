# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CausalQwen is the first application of the breakthrough CausalEngine™ - a revolutionary algorithm that redefines intelligence from pattern matching to causal reasoning. The project implements a language model that understands causation rather than just correlation.

### Core Architecture

```
CausalQwen/
├── causal_engine/          # Core CausalEngine algorithm (independent)
│   ├── engine.py          # Main CausalEngine implementation
│   ├── networks.py        # AbductionNetwork and ActionNetwork
│   └── heads.py           # ActivationHead for different output types
├── src/causal_qwen_mvp/   # Qwen-specific application layer
│   ├── models.py          # CausalQwenMVPForCausalLM implementation
│   ├── inference.py       # Inference engine
│   └── config.py          # Configuration classes
├── tests/                 # Comprehensive test suite
└── scripts/               # Demo and validation scripts
```

**Critical Design Principle**: CausalEngine (in `causal_engine/`) is completely independent from CausalQwen. CausalEngine can work with ANY transformer model - CausalQwen is just one application.

## Essential Development Commands

### Testing
```bash
# Run all tests without requiring Qwen models
./run_tests.sh

# Specific test categories
./run_tests.sh math         # Mathematical framework tests
./run_tests.sh compatibility # Qwen compatibility tests  
./run_tests.sh generation   # Generation tests
./run_tests.sh quick        # Fast tests only
./run_tests.sh coverage     # Generate coverage report

# Direct pytest usage
pytest tests/ -v -m "not requires_qwen"  # Skip Qwen model tests
```

### Installation and Dependencies
```bash
# Basic installation
pip install torch transformers numpy

# Development dependencies  
pip install -e .[dev]  # Includes pytest, black, isort, flake8, mypy
pip install pytest-cov  # For coverage reports

# Full installation with docs
pip install -e .[dev,docs]
```

### Code Quality
The project uses standard Python tools configured in setup.py:
- **pytest**: Testing framework with custom markers (slow, requires_qwen)
- **black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Code formatting and linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Core Mathematical Framework

CausalEngine implements the fundamental equation: `Y = f(U, ε)` where:
- **U ~ Cauchy(μ, γ)**: Individual causal representation
- **ε**: Exogenous noise  
- **f**: Universal causal law (linear transformation)

### Three-Module Architecture

1. **AbductionNetwork**: Evidence → Individual representation (U)
   - Independent `loc_net` and `scale_net` pathways
   - Smart initialization based on dimension matching
   
2. **ActionNetwork**: Individual → Decisions  
   - Applies causal transformations
   
3. **ActivationHead**: Decisions → Outputs
   - Supports both classification and regression modes
   - Unified activation framework

### Four Reasoning Modes

| Mode | Temperature | do_sample | Use Case |
|------|------------|-----------|----------|
| Causal | 0 | any | Pure deterministic reasoning |
| Standard | >0 | False | Decisions with uncertainty |
| Sampling | >0 | True | Identity exploration |
| Compatible | any | N/A | Legacy comparison mode |

## Key Implementation Patterns

### CausalEngine Usage
```python
from causal_engine import CausalEngine

# Core engine - works with ANY transformer
engine = CausalEngine(
    hidden_size=768,
    vocab_size=50000,
    causal_size=768,  # Usually same as hidden_size
    activation_modes="classification"  # or ["classification", "regression", ...]
)

# Apply to any hidden states
hidden_states = any_transformer_model(input_ids)
output = engine(hidden_states, temperature=1.0, do_sample=True)
```

### Cauchy Mathematics
The engine uses `CauchyMath` class for analytically stable operations:
- No sampling required - pure analytical computation
- Linear stability: `Cauchy(μ₁,γ₁) + Cauchy(μ₂,γ₂) = Cauchy(μ₁+μ₂,γ₁+γ₂)`

### Configuration Patterns  
- CausalEngine takes atomic parameters, not config objects
- Maintains strict separation between engine and application layers
- All configurations inherit from base classes in `config.py`

## Testing Strategy

The test suite is organized with pytest markers:
- `@pytest.mark.slow`: Computationally expensive tests
- `@pytest.mark.requires_qwen`: Tests needing Qwen pretrained models

Test categories:
- **Mathematical framework**: Core Cauchy math and engine logic
- **Compatibility**: Interface compatibility with Qwen models  
- **Generation**: Text generation and inference
- **Comparison**: Performance vs original Qwen (requires models)

## Important Implementation Notes

- **Modular Design**: Never mix CausalEngine logic with application-specific code
- **Temperature Control**: Single parameter controls determinism/stochasticity boundary
- **Independence**: AbductionNetwork's loc_net and scale_net are completely independent
- **Analytical Computation**: No sampling in core math - everything is computed analytically
- **Universal Application**: CausalEngine can be applied to BERT, GPT, LLaMA, or any transformer

## Documentation Structure

Mathematical foundations are documented in:
- `causal_engine/MATHEMATICAL_FOUNDATIONS.md`: Core theory
- `docs/core_mathematical_framework.md`: Implementation details
- `causal_engine/ONE_PAGER.md`: Executive summary

The code contains extensive Chinese and English documentation reflecting the international nature of the development team.