# NN #2: Current SOTA Research

**Status**: STUB. Populated during Phase 4 of scaffolding. Approval required via HIP-5 before the selected architecture is implemented in `src/dl_models.py::NN2SOTA`.

---

## Goal

Identify the current best neural architecture for **3-class pain localization from multimodal physiological signals** (BVP + EDA + RESP + SpO2, 65 participants, small-label regime), then implement it as NN #2 for the AI4Pain 2026 Challenge. This is the "informed baseline" against which the novel NN #3 is compared.

"Current SOTA" means:

1. Published after January 2024 in a reputable venue (NeurIPS, ICML, ICLR, EMBC, IEEE JBHI, IEEE TBME, Information Fusion, npj Digital Medicine, etc.)
2. Demonstrated on a pain or pain-adjacent physiological classification task with a comparable data regime
3. Reproducible from the paper + code (or reproducible to within a small margin)
4. Hardware-feasible on 1x H200 with the 8hr / 12 GB constraint
5. Does not depend on external data (ANTIPATTERNS rule 1)

---

## Research Plan (to execute in Phase 4)

### Scope
- Pain classification from physiological signals, 2024-2026
- Multimodal fusion for biosignals (early, mid, late fusion)
- Transformers and attention for physiological time series
- State-space models (Mamba, S4, S5) for long physiological sequences
- Self-supervised pretraining (Wave2Vec2, TS2Vec) for small-label regimes (must respect NO EXTERNAL DATA)
- Graph neural networks for multi-sensor fusion
- Pain localization specifically (hand vs arm, anatomical priors, somatotopic encoding)

### Tools
- `mcp__semantic-scholar__*` (200M papers, SPECTER recommendations, author graphs)
- `mcp__scite__search_literature` (Smart Citations for supporting / contrasting evidence)
- `mcp__claude_ai_PubMed__*` (biomedical literature)
- `mcp__parallel-research__deep_research` (async deep dives)
- WebFetch + WebSearch for venue-specific proceedings

### Deliverable (this document, to be populated in Phase 4)

1. **Problem framing** -- 3-class localization, multimodal, small-data regime (41 train subjects)
2. **Literature table** -- 15-25 candidate papers with architecture summary, reported results, applicability score (1-5 scale)
3. **Shortlist** -- 3-5 architectures with trade-off analysis
4. **Selected architecture** -- with rationale
5. **Implementation sketch** -- layers, params, expected training cost on 1x H200
6. **Novelty check** -- confirm this is current SOTA, not a paper already superseded

---

## Sections To Fill During Phase 4

### 1. Problem Framing
_TBD in Phase 4._

### 2. Literature Table
_TBD in Phase 4. Format: Paper | Year | Venue | Architecture | Data regime | Reported metric | Applicability score | Notes._

### 3. Shortlist (3-5 candidates)
_TBD in Phase 4._

### 4. Selected Architecture
_TBD in Phase 4. Requires HIP-5 approval before moving to implementation._

### 5. Implementation Sketch
_TBD in Phase 4._

### 6. Novelty Check
_TBD in Phase 4. Cross-reference Semantic Scholar for any 2025-2026 follow-up that would supersede the selection._

---

## Approval

| Stage | Status | Date |
|-------|--------|------|
| Literature review complete | NOT STARTED | - |
| Shortlist ranked | NOT STARTED | - |
| Architecture selected | NOT STARTED | - |
| HIP-5 approval by Vignan | NOT GRANTED | - |
| NN2SOTA implemented in dl_models.py | NOT STARTED | - |
