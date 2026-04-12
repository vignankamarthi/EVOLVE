# NN #3: Novel Homemade Framework

**Status**: STUB. Populated during Phase 5 of scaffolding. Approval required via HIP-6 before the novel architecture is implemented in `src/dl_models.py::NN3Novel`. This is the paper's central novelty claim and must survive reviewer scrutiny.

---

## Goal

Design a **novel** neural architecture for 3-class pain localization from multimodal physiological signals that offers a meaningful contribution beyond NN #2 (the current SOTA). The architecture is driven by a gap analysis of NN #2 and by physiological priors specific to the pain localization problem.

Novelty must be defensible. "It's a new combination of existing blocks" is not sufficient. The contribution should answer the question: **what does this architecture capture about pain localization from autonomic signals that prior work does not?**

---

## Research + Design Plan (to execute in Phase 5)

### Gap Analysis (scope)
- What is missing or underexploited in NN #2?
- What physiological priors are ignored by existing architectures? (vagal tone, nociceptive pathways, cardiovascular-autonomic coupling, sympathetic arousal signatures)
- Is pain localization treated as a flat 3-way classification, or as an anatomical localization task with spatial priors?
- How do existing architectures handle the small-label regime (41 train subjects)?

### Candidate Directions (to narrow during Phase 5 research)

1. **Hierarchical detection-then-localization**
   - Shared encoder, two heads: (a) binary pain detection, (b) localization conditional on pain
   - Inductive bias: "is there pain?" is easier than "where is the pain?"
   - Risk: may collapse into flat 3-way classification if the heads are not carefully regularized

2. **Cross-modal attention with physiological prior gating**
   - Attention over signal pairs (BVP <-> EDA, BVP <-> RESP, etc.)
   - Gated by a prior from physiology (autonomic coupling strength, phase coherence)
   - Novelty: learned-but-prior-constrained attention

3. **Frequency-aware branches**
   - Autonomic signals have distinct frequency bands (cardiac ~1 Hz, respiratory ~0.2 Hz, sympathetic ~0.04-0.15 Hz)
   - Per-band branches before fusion
   - Novelty: architectural decomposition mirrors the physiology

4. **Phase-aware cross-signal fusion**
   - BVP phase locked to EDA response (cardiovascular-sudomotor coupling)
   - Phase-locking value as a learnable feature
   - Novelty: first-class phase alignment in the forward pass

5. **Hybrid SSM + CNN**
   - Mamba / S5 block for long-range dependencies, CNN for local morphology
   - Novelty is in the fusion layer, not the components

### Theoretical Grounding

Each candidate must be backed by:

- Physiology citations (pain neuroscience, autonomic coupling literature)
- ML citations (attention, SSMs, cross-modal fusion)
- Any prior attempts at similar ideas (for novelty differentiation)

### Ablation Plan

The novel architecture must come with a pre-registered ablation plan: what components are tested, what's ablated, what the headline result depends on. This prevents post-hoc rationalization.

---

## Deliverable (this document, to be populated in Phase 5)

### 1. Gap Analysis
_TBD in Phase 5. What does NN #2 miss for this specific problem?_

### 2. Theoretical Grounding
_TBD in Phase 5. Physiology citations + ML citations._

### 3. Architecture Design
_TBD in Phase 5. Block diagram, layer-by-layer spec, parameter count estimate, forward-pass pseudo-code._

### 4. Ablation Plan
_TBD in Phase 5. What components are tested in ablation, what is the headline claim, what would falsify it._

### 5. Expected Contribution (for ACII 2026 paper framing)
_TBD in Phase 5. One paragraph that could become the paper's contribution statement._

### 6. Hardware Feasibility Check
_TBD in Phase 5. Parameter count, training cost on 1x H200, inference latency budget._

---

## Approval

| Stage | Status | Date |
|-------|--------|------|
| Gap analysis complete | NOT STARTED | - |
| Candidate directions narrowed | NOT STARTED | - |
| Architecture designed | NOT STARTED | - |
| Ablation plan written | NOT STARTED | - |
| Hardware feasibility verified | NOT STARTED | - |
| HIP-6 approval by Vignan | NOT GRANTED | - |
| NN3Novel implemented in dl_models.py | NOT STARTED | - |
