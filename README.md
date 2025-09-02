# FSU v1 — Field Sequence Unit: Research Archive
**Author: Pirassena Sabaratnam**

## Overview
This repository archives the development of **FSU (Field Sequence Unit) v1**, a series of language models that attempted to replace discrete tensor operations with **continuous neural field dynamics**. Developed between June and September 2025, this research explored whether PDE-based computation could serve as the basis for sequence modeling.

## Research Question
Can language be modeled as a continuous semantic field evolving over a manifold? Instead of discrete matrix multiplications (as in Transformers), FSU v1 used partial differential equation (PDE) solving for both forward and backward passes, with sequence positions treated as field coordinates.

## Iterations

### v1: Baseline (`v1-baseline`)
Initial implementation of continuous field evolution for sequence modeling. Established core field update and sampling mechanisms.

### v2: Streaming Data Inflow (`v2-stream-noncur`)
Replaced batch processing with a streaming data model (JSON/JSONL). Hypothesized that matching data inflow dynamics with the model's continuous learning state would stabilize training. Result: reduced gradient norms and smoother convergence.

### v3: Performance-Gated Curriculum (`v3-stream-curriculum`)
Added a curriculum dataloader that gates data complexity based on real-time performance metrics. Goal: prevent catastrophic forgetting during field maturation.

### v4: Custom Data Format (`v4-adp-protocol`)
Developed a custom data serialization format optimized for the high-frequency field updates required by the architecture. Standard tensor formats introduced unnecessary overhead.

### v5: Preprocessing Pipeline (`v5-adp-aura-protocol`)
Built a teacher/student data compilation pipeline that preprocesses raw text into a domain-aware representation using an auxiliary FSE engine. Result: significantly lower language and reasoning loss compared to all previous iterations.

### v6: Ridge-SGD Hybrid (`v6-ridgeSGD-hybrid`)
Introduced a ridge regression initialization step that "one-shots" the data representation prior to SGD fine-tuning. Result: fastest convergence to accuracy targets. However, suffered **mode collapse** during extended inference — the stability constraints (tanh clamping) needed for PDE convergence reduced field entropy to the point where representational diversity was lost.

### v7: Transformer Hybrid (`v7-fsmart-hybrid`)
Merged continuous field dynamics with a discrete Transformer encoder head, aiming to use global self-attention to resolve the context propagation issues in pure field models.

## Key Findings
1. **O(n) memory**: The architecture operates on a fixed-size state, independent of sequence length — no KV cache growth.
2. **Stability-diversity tradeoff**: The normalization required for PDE convergence (tanh clamping, gradient clipping) directly caused mode collapse. This is the fundamental tension in continuous field architectures.
3. **Incoherent inference**: Despite stable training and loss reduction, the models did not produce coherent linguistic output. The context propagation problem was never fully solved within the PDE framework.

These findings directly motivated the development of the **Gravitational Vector Network (GVN)**, which uses structured gravitational attractors to provide the stability that unconstrained field evolution lacked, while preserving the continuous dynamics.

---
*Developed June–September 2025 — Auralith Inc.*
