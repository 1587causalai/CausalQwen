# Mathematical Documentation for Causal-Sklearn

This directory contains the complete mathematical foundations and theoretical framework for the CausalEngine algorithm implemented in causal-sklearn.

## Core Mathematical Documents

### 📐 Foundational Theory
- **[MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)** - Complete mathematical foundations of CausalEngine
- **[MATHEMATICAL_FOUNDATIONS_CN.md](MATHEMATICAL_FOUNDATIONS_CN.md)** - 中文版数学基础 (Chinese version)
- **[ONE_PAGER.md](ONE_PAGER.md)** - Executive summary of the mathematical framework

### 🧮 Detailed Mathematical Framework
- **[core_mathematical_framework.md](core_mathematical_framework.md)** - Core mathematical framework
- **[core_mathematical_framework_num_extended.md](core_mathematical_framework_num_extended.md)** - Extended numerical framework
- **[mathematical_equivalence_deep_dive.md](mathematical_equivalence_deep_dive.md)** - Deep dive into mathematical equivalences

### ✅ Implementation Verification
- **[MATHEMATICAL_IMPLEMENTATION_VERIFICATION.md](MATHEMATICAL_IMPLEMENTATION_VERIFICATION.md)** - Verification of mathematical implementation

## Key Mathematical Concepts

### The Four Axioms of CausalEngine

1. **Intelligence = Abduction + Action**: From observations to self-understanding to decisions
2. **Cauchy Mathematics**: The only distribution enabling analytical causal computation
3. **Temperature-Unified Control**: Single parameter controlling determinism vs stochasticity  
4. **Structural Equation Decisions**: Every choice computed by deterministic functions

### Core Mathematical Framework

The CausalEngine algorithm is built on the structural equation:

```
Y = f(U, ε)
```

Where:
- **U**: Individual causal representation (learned from context X)
- **ε**: Exogenous noise (independent random perturbation) 
- **f**: Universal causal mechanism (deterministic function)

### Two-Stage Architecture

1. **Abduction Stage**: `X → U` (Evidence to Individual Representation)
   - AbductionNetwork: Maps observations to causal representation
   - Uses Cauchy distribution for analytical uncertainty propagation

2. **Action Stage**: `U → Y` (Individual to Decision)
   - ActionNetwork: Maps representation to decision potential
   - ActivationHead: Converts potential to specific outputs (classification/regression)

### Mathematical Properties

- **Analytical Computation**: No sampling required due to Cauchy properties
- **Heavy-Tail Robustness**: Natural handling of outliers and extreme events
- **Undefined Moments**: Philosophical alignment with true uncertainty
- **Scale Invariance**: Consistent behavior across different scales

## Usage in Implementation

These mathematical documents serve as the canonical reference for:

1. **Correctness Verification**: Ensuring implementation matches theoretical framework
2. **Parameter Understanding**: Mathematical meaning of all hyperparameters
3. **Debugging**: Theoretical basis for troubleshooting implementation issues
4. **Extension**: Mathematical foundation for adding new features

## Reading Order

For implementers and developers:
1. Start with `ONE_PAGER.md` for high-level overview
2. Read `MATHEMATICAL_FOUNDATIONS.md` for complete theory
3. Consult `core_mathematical_framework.md` for detailed equations
4. Use `MATHEMATICAL_IMPLEMENTATION_VERIFICATION.md` for validation

For researchers and theorists:
1. Begin with `MATHEMATICAL_FOUNDATIONS.md`
2. Deep dive into `mathematical_equivalence_deep_dive.md`
3. Study `core_mathematical_framework_num_extended.md` for advanced theory

---

**Note**: These documents are the authoritative mathematical specification for CausalEngine. Any implementation in causal-sklearn must strictly adhere to these mathematical definitions.