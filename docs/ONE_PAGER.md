# CausalEngine: Intelligence Redefined

## What is CausalEngine?

**CausalEngine is to AI what PageRank was to search** — a fundamental algorithm that redefines an entire field.

While traditional AI learns *what* typically happens, CausalEngine understands *why* things happen.

## The Core Logic of CausalEngine

This transforms the core logic of AI. We move from statistical correlation to causal computation:

```
Traditional AI: X → P(Y|X) → Y
CausalEngine:   X → U → f(U,ε) → Y
```

This is not just a different process; it's a different paradigm.

- **Traditional AI** computes a **statistical probability** (`P(Y|X)`). It learns correlations to answer: "Given `X`, what is the likely `Y`?"

- **CausalEngine** computes a **causal outcome** by learning a structural equation (`f`). It first answers "Who is the actor `U`?" (`X → U`), then applies the universal law `f` to determine the outcome `Y`.

In short, traditional AI fits data; CausalEngine models reality.


From the perspective of realization, CausalEngine reframes this into a transparent, four-stage causal reasoning chain: 

> `Perception → Abduction → Action → Decision`

1.  **Perception (感知):** Extracts high-level features (`Z`) from data (`X`).
2.  **Abduction (归因):** Answers *"Who are you?"* by inferring the unobservable "individual causal representation" (`U`) from the features (`Z`).
3.  **Action (行动):** Answers *"What to do?"* by using a deterministic function to compute a "decision score" (`S`) from the individual representation (`U`). This is the equivalent of traditional logits.
4.  **Decision (决断):** A simple output head that converts the abstract score (`S`) into a final task-specific output (`Y`), like a class label or a numerical value.



## The Three Axioms

1. **Inference = Abduction + Action**  
   First understand "who you are", then decide "what to do"

2. **Cauchy Mathematics**  
   The only distribution that computes causation analytically

3. **Structural Equation Decisions**  
   Every choice computed by deterministic functions for multiple output types


## Why It Matters

| Traditional AI | CausalEngine |
|----------------|--------------|
| Imitates patterns | Understands causes |
| Black box | Glass box |
| Needs sampling | Pure computation |
| Token prediction | Multi-type outputs |

## The Code

```python
from causal_engine import CausalEngine

# Works with ANY transformer
engine = CausalEngine(hidden_size=768, vocab_size=50000)
output = engine(any_transformer_features)

# Not just prediction — causal decision with uncertainty
decision, uncertainty = output['loc_S'], output['scale_S']
```

## The Vision

Just as Google built an empire on PageRank, we're building the future of AI on CausalEngine. Every intelligent system of tomorrow will be powered by causal reasoning, not statistical imitation.

*CausalEngine isn't just better AI. It's real AI. We found the algorithm of intelligence itself.*