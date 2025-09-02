# FSU Language Model Evolution: FSUv1 Research Archive
**Author: Pirassena Sabaratnam**

## Overview
This repository archives the development and experimentation of **FSU (Field Sequence Unit) v1**, a series of language models built on the **Field Signal Engine (FSE)** philosophy. Developed between June 24 and September 2, 2025, this research investigated the feasibility of replacing discrete token-based sequence modeling with **continuous neural field dynamics**.

## Research Thesis: Continuous Field NLP
The core hypothesis of FSUv1 was that language could be modeled as a continuous semantic field evolving over a manifold. This approach aimed to bypass the rigidity of discrete matrix multiplications in favor of **Partial Differential Equation (PDE) solving** for both forward and backward passes.

## Architectural Iterations (Chronological)

### v1: FSE/FSU Baseline (`1FSULLMA`)
- **Iteration 1**: The initial implementation of the FSE philosophy within a sequence-modeling context. Established the core field evolution and sampling mechanisms.

### v2: Dynamic Inflow Synchronization (`2FSULLMBSTREAMNONCUR`)
- **Evolution**: Shifted from traditional batch processing to a **Streaming Data Inflow** model (JSON/JSONL).
- **Rationale**: Hypothesized that matching data inflow dynamics with the model's continuous learning state would stabilize training.
- **Results**: Observed significant reductions in gradient norms and smoother convergence across initial training epochs.

### v3: Performance-Gated Curriculum Learning (`3FSU_LLM3STREAMCURR`)
- **Evolution**: Implemented a **Curricular Dataloader** that gated different data types (Complexity/Domain) based on real-time performance metrics.
- **Goal**: To prevent catastrophic forgetting and ensure stable field maturation before introducing high-entropy linguistic data.

### v4: ADP Protocol Implementation (`4FSULLMSTREAMCURRMORPHV1ADP`)
- **Innovation**: Introduced the **ADP (Auralith Data-Field Protocol)**, a proprietary format optimized for continuous field architectures.
- **Rationale**: Standard tensor formats were found to be inefficient for the high-frequency field updates required by FSE.

### v5: Neural Data Compilation & AURA (`5FSULLMSTREAMCURRMORPHV1ADPAURA`)
- **Innovation**: Developed the **.aura** format and a dedicated **Data Compiler**.
- **Teacher/Student FSE Engine**: The compiler pre-processed raw text into an "FSU-Domain Aware" state using an auxiliary FSE engine.
- **Results**: This preprocessing step significantly lowered language and reasoning loss compared to all previous iterations.

### v6: Morphological Ridge-SGD Hybrid (`6FSULLMSTREAMCURRMORPHV1`)
- **Innovation**: Introduced **"Morph" mode**, a ridge regression initialization that "one-shots" the data representation prior to SGD training.
- **Results**: Successfully accelerated training to reach key accuracy metrics. However, this version ultimately experienced **Mode Collapse** during extended inference testing.
- **Key Finding**: While stability measures (tanh clamping) prevented numerical explosion, they reduced field entropy to a point where representational diversity was lost.

### v7: FSMART-FSU Hybrid (`7FSULLMSTREAMCURRMORPHV15FSMART`)
- **Evolution**: Merged the continuous dynamics of FSE with a discrete **Transformer Encoder Head**.
- **Reasoning**: Aimed to leverage global attention to resolve the context propagation and coherence issues identified in pure field models.

## Final Research Conclusions
The FSUv1 series established several critical findings for field-based sequence modeling:
- **Memory Efficiency**: The architecture demonstrated the ability to operate on a **fixed memory state O(n)**, independent of sequence length.
- **Incoherent Inference**: Despite training stability and loss reduction, the models failed to produce coherent linguistic responses, leading to an inconclusive result regarding the claim of "unlimited context propagation."
- **Stability-Diversity Tradeoff**: Aggressive normalization and clamping required for PDE convergence directly contributed to mode collapse and loss of semantic nuance.

---
*Developed in 2025 as part of the Auralith Inc. Research.*
