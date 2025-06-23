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

## Tutorials Update Strategy

Based on the recent document references:
- `causal_engine/misc/benchmark_strategy.md`: Latest experimental design document
- `causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md`: Latest version of mathematical core understanding document

Tutorials Update Recommendations:
- Sync tutorial content with the latest benchmark strategy document
- Incorporate mathematical foundations from the CN mathematical foundations document
- Ensure consistency between Chinese and English documentation
- Update code examples to reflect the latest experimental design and mathematical understanding

## Essential Development Commands

## Memory Updates

- `tutorials/SUMMARY.md`: Need to update related content to facilitate verification and testing of updates
- `run_tutorials.py` needs to be updated

## Recent Interactions and Insights

### Mathematical Documentation Review
- Reviewed `causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md` document for potential inconsistencies and improvements
- Noted suggestions for aligning mathematical descriptions and implementation
- Recognized potential strengths in mathematical formula descriptions, particularly in early chapters