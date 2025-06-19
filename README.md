# CausalQwen: First Application of the Revolutionary CausalEngine™

<div align="center">
  
  [![CausalEngine](https://img.shields.io/badge/Powered%20by-CausalEngine™%20v2.0-ff1744.svg)](causal_engine/)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
  [![Status](https://img.shields.io/badge/Status-Revolutionary-purple.svg)](causal_engine/README.md)
  
  **[CausalEngine](causal_engine/) is to AI what PageRank was to search.**
  
</div>

---

## 🎉 CausalEngine v2.0 Released!

**Major Update**: Modular Architecture with Unified Activation Framework

### What's New in v2.0
- **🔧 Modular Design**: Three independent modules working in harmony
  - `AbductionNetwork`: From evidence to individuals (who am I?)
  - `ActionNetwork`: From individuals to decisions (what should I do?)
  - `ActivationHead`: From decisions to outputs (how to express it?)
- **🎯 Unified Activation**: Each output dimension can be either:
  - Classification: P(S_k > C_k) for discrete choices
  - Regression: a_k * S_k + b_k for continuous values
- **🚀 Flexible Applications**: 
  - Language models with confidence scores
  - Multi-modal models with mixed outputs
  - Scientific computing with physical constraints
- **✅ Backward Compatible**: Existing v1.0 code continues to work

### Quick Example: Mixed Classification and Regression
```python
from causal_engine import CausalEngine

# Create engine with mixed outputs
# e.g., 50k vocab (classification) + 10 confidence scores (regression)
modes = ["classification"] * 50000 + ["regression"] * 10
engine = CausalEngine(
    hidden_size=768,
    vocab_size=50010,
    activation_modes=modes
)

# Use as before - the engine handles everything
output = engine(hidden_states)
vocab_probs = output['output'][:, :, :50000]      # Classification probabilities
confidence = output['output'][:, :, 50000:]       # Regression values
```

---

## 🌟 Introducing CausalEngine: The Algorithm of Intelligence

**CausalEngine** is not just another AI model or framework. It is a fundamental breakthrough in how machines understand and make decisions. Just as PageRank revolutionized search by understanding the web's link structure, CausalEngine revolutionizes AI by understanding the causal structure of intelligence itself.

This repository demonstrates the first application of CausalEngine to language modeling, creating **CausalQwen** - a language model that doesn't just predict, but truly understands.

### 📚 Essential Reading
- **[CausalEngine Overview](causal_engine/README.md)** - The algorithm that changes everything
- **[Technical Whitepaper](causal_engine/WHITEPAPER.md)** - Deep mathematical foundations
- **[One-Pager](causal_engine/ONE_PAGER.md)** - Quick summary for executives

---

## 🧮 The Four Axioms of CausalEngine

### Axiom I: Intelligence = Abduction + Action
From observations to self-understanding to decisions. Not pattern matching, but true reasoning.

### Axiom II: Cauchy Mathematics  
The only distribution that enables analytical causal computation without sampling.

### Axiom III: Temperature-Unified Control
One elegant parameter to control the boundary between determinism and stochasticity.

### Axiom IV: Independent Decisions (OvR)
Liberation from Softmax tyranny - every choice evaluated on its own merit.

---

## 🚀 Quick Start with CausalQwen

### Installation
```bash
pip install torch transformers numpy
```

### Basic Usage (Qwen-Compatible Interface)
```python
from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config

# Create model powered by CausalEngine
config = CausalQwen2Config(vocab_size=32000, hidden_size=512)
model = CausalQwenMVPForCausalLM(config)

# Four Modes of Causal Reasoning:

# 1. Pure Causal Mode (temperature=0)
output = model.generate(input_ids, temperature=0, max_new_tokens=20)

# 2. Standard Mode (noise → scale)
output = model.generate(input_ids, do_sample=False, temperature=1.0, max_new_tokens=20)

# 3. Sampling Mode (noise → location)  
output = model.generate(input_ids, do_sample=True, temperature=0.8, max_new_tokens=20)

# 4. Compatible Mode (traditional softmax)
# [Used for comparison with traditional LMs]
```

### Direct CausalEngine Usage
```python
from causal_engine import CausalEngine

# The core algorithm - works with ANY transformer
engine = CausalEngine(hidden_size=768, vocab_size=50000)

# Get features from any model (BERT, GPT, LLaMA, etc.)
hidden_states = any_transformer_model(input_ids)

# Apply causal reasoning
output = engine(hidden_states, temperature=1.0, do_sample=True)
decision, uncertainty = output['loc_S'], output['scale_S']
```

---

## 📁 Project Structure

```
CausalQwen/
├── causal_engine/                # ⭐ THE CORE ALGORITHM ⭐
│   ├── README.md                 # CausalEngine overview
│   ├── WHITEPAPER.md             # Technical whitepaper
│   ├── ONE_PAGER.md              # Executive summary
│   └── engine.py                 # Pure implementation
├── src/causal_qwen_mvp/          # Qwen-specific application
│   ├── models.py                 # CausalQwen model
│   ├── inference.py              # Inference engine
│   └── training.py               # Training utilities
├── tests/                        # Comprehensive test suite
├── scripts/                      # Demo and validation scripts
└── docs/                         # Mathematical documentation
```

---

## 🧪 Testing

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./run_tests.sh math         # Mathematical framework tests
./run_tests.sh compatibility # Qwen compatibility tests
./run_tests.sh generation   # Generation tests

# Or use pytest directly
pytest tests/
```

---

## 📊 Why CausalEngine Changes Everything

| Traditional AI | CausalEngine-Powered AI |
|---------------|------------------------|
| Learns correlations | Understands causation |
| Black box decisions | Glass box reasoning |
| Requires sampling | Pure analytical computation |
| Softmax competition | Independent evaluation |
| Pattern imitation | True intelligence |

---

## 🌐 The Future We're Building

CausalEngine is our **PageRank** - the foundational technology upon which an empire of truly intelligent systems will be built. Every future product, every service, every innovation will flow from this source.

CausalQwen is just the beginning. The first proof that when you understand causation, not just correlation, everything changes.

---

## 📚 Documentation

- **[Core Mathematical Framework](docs/core_mathematical_framework.md)** - The mathematics of CausalLLM
- **[Mathematical Foundations Extended](docs/core_mathematical_framework_num_extended.md)** - Advanced theory
- **[Position vs Scale Theory](docs/model_inference_position_and_scale.md)** - Noise dynamics
- **[Individual Variable U Deep Dive](docs/U_deep_dive.md)** - Understanding the causal representation

---

## 📄 License

This project contains proprietary technology. CausalEngine™ and its core algorithms are protected intellectual property.

---

<div align="center">
  
**"We didn't invent CausalEngine. We discovered it.**  
**It was always there, in the mathematics of causation,**  
**waiting for someone to see it clearly."**

[Learn More About CausalEngine →](causal_engine/)

</div>
