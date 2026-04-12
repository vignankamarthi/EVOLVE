# Process Flow Chart: AI4Pain 2026 Challenge

**Vignan Kamarthi** | ACII 2026 PAAIn Workshop submission | 3-class pain localization

---

## Experiment Overview

Two-track experiment on multimodal physiological signals: classical ML with signal ablation, and deep learning with three architectures (baseline, SOTA, novel). All evaluation uses balanced accuracy, macro-F1, per-class precision/recall, confusion matrix, and AUC-OVR on the fixed 41/12/12 subject split provided by the challenge organizers.

```
AI4Pain 2026 Challenge Data
(65 participants, 41 train / 12 val / 12 test)
    |
    +-- BVP (blood volume pulse)
    +-- EDA (electrodermal activity)
    +-- RESP (respiration)
    +-- SpO2 (blood oxygen saturation)
    +-- Labels: NP (No Pain) | HP (Hand Pain) | AP (Arm Pain)
    |
    +==========================================+
    |        TWO PARALLEL TRACKS               |
    +==========================================+
    |                                          |
    v                                          v
TRACK 1: Feature Engineering + ML          TRACK 2: Deep Learning
(CPU, Rust extractor)                      (GPU, H200)
    |                                          |
    v                                          v
Rust Feature Extraction              Raw multi-channel windows
40 features per signal:              (channels per ablation config)
  Catch22 (22)                         |
  Entropy (10)                         v
  Stats (8)                       3 architectures:
    |                                NN #1: ResNet-1D (baseline, ported)
    v                                NN #2: Current SOTA (Phase 4 research)
StandardScaler                       NN #3: Novel framework (Phase 5 research)
(per column, train only)               |
    |                                    v
    v                               CrossEntropyLoss + class weights
3 classifiers x 3 ablations:        Per-epoch checkpointing + early stopping
  Random Forest                        |
  XGBoost                              |
  LightGBM                             |
  (+ Logistic Regression baseline)     |
    |                                    |
    v                                    v
Optuna HPO                         Validation metrics
(maximize balanced accuracy)       on 12 validation subjects
    |                                    |
    +================+===================+
                     |
                     v
                EVALUATION
         (same metrics, both tracks)
         Balanced accuracy, macro-F1,
         per-class precision/recall, confusion matrix,
         AUC (one-vs-rest)
                     |
                     v
          TEST SET (5 submissions max)
          Default allocation:
            1. Classical ML best
            2. NN #1 ResNet-1D baseline
            3. NN #2 current SOTA
            4. NN #3 novel framework (headline)
            5. Reserve / ensemble
                     |
                     v
            ACII 2026 PAAIn Workshop
              Paper submission
```

---

## Ablation Matrix

| Config | Signals | Classical Features | DL Input Channels |
|--------|---------|--------------------|--------------------|
| `bvp_eda` | BVP + EDA | 80 | 2 |
| `bvp_eda_resp` | BVP + EDA + RESP | 120 | 3 |
| `all_four` | BVP + EDA + RESP + SpO2 | 160 | 4 |

The ablation progression tests the incremental value of respiratory and oxygenation signals over the autonomic baseline (BVP + EDA). Cross-ablation comparison is part of the paper's experimental story.

---

## Data Acquisition

```
+---------------------------------------------------------------+
|                     DATA ACQUISITION                          |
|                                                               |
|  AI4Pain 2026 Challenge release                              |
|  65 participants, fixed train/val/test split (41/12/12)      |
|  4 signals per trial:                                        |
|    BVP  -- blood volume pulse (high-rate, autonomic)          |
|    EDA  -- electrodermal activity (low-rate, sympathetic)     |
|    RESP -- respiration (mid-rate, breath cycle)               |
|    SpO2 -- blood oxygen saturation (low-rate)                 |
|  Labels: NP / HP / AP (per trial)                            |
|  Constraints:                                                 |
|    - Labeled train + val arrive April 2026                   |
|    - Unlabeled test arrives May 2026                         |
|    - 5 test label submissions maximum per team               |
|    - No external data allowed                                |
+---------------------------------------------------------------+
```

---

## Feature Extraction (Track 1)

The Rust crate `feature-extraction-rust/` is adapted from Blood-Pressure-Inference-with-BVP. For each of the four signals, the extractor computes 40 features:

```
+---------------------------------------------------------------+
|                  RUST FEATURE EXTRACTION                      |
|                                                               |
|  Per signal, per subject, per trial:                         |
|                                                               |
|  Catch22 (22 features):                                      |
|    Autocorrelation, distribution shape, binary stats,        |
|    fluctuation analysis, spectral summaries,                 |
|    motif statistics                                          |
|                                                               |
|  Entropy (10 features):                                      |
|    Permutation, statistical complexity, Fisher-Shannon,      |
|    Fisher information, Renyi PE/complexity,                  |
|    Tsallis PE/complexity, sample, approximate                |
|                                                               |
|  Statistical (8 features):                                   |
|    mean, median, std, skewness, kurtosis, RMS, min, max      |
|                                                               |
|  Output: CSV per {split, signal} with 40 feature columns     |
|          plus subject_id, trial_id, label metadata           |
|                                                               |
|  Parallelism: rayon workers on CPU                           |
|  Checkpointing: atomic writes per subject                    |
+---------------------------------------------------------------+
```

---

## Deep Learning (Track 2)

```
+---------------------------------------------------------------+
|                    DEEP LEARNING TRACK                        |
|                                                               |
|  Input: raw multi-channel windows                            |
|  Channel counts depend on ablation config (2, 3, or 4)       |
|  Per-channel StandardScaler fit on training subjects only    |
|                                                               |
|  NN #1: ResNet-1D (ported from BP inference, head 1 -> 3)   |
|    Stem -> 3 residual layers (64/128/256)                   |
|    -> AdaptiveAvgPool -> Linear(256, 128) -> Linear(128, 3) |
|                                                               |
|  NN #2: Current SOTA (Phase 4 research deliverable)          |
|    Selection requires literature review, HIP-5 approval     |
|                                                               |
|  NN #3: Novel homemade framework (Phase 5 deliverable)       |
|    Design requires gap analysis, HIP-6 approval             |
|    This is the paper's central novelty                       |
|                                                               |
|  Loss: CrossEntropyLoss + class weights                      |
|  Optimizer: Adam / AdamW                                     |
|  Training: per-epoch checkpointing, early stopping           |
|  Hardware: 1x H200, 8hr job limit, 12 GB RAM                 |
+---------------------------------------------------------------+
```

---

## Evaluation

```
+---------------------------------------------------------------+
|                       EVALUATION                              |
|                                                               |
|  Primary:    Balanced accuracy (robust to imbalance)         |
|  Secondary:  Macro-F1                                        |
|  Per-class:  Precision, recall (NP, HP, AP)                  |
|  Visual:     3x3 confusion matrix                            |
|  Ranking:    AUC (one-vs-rest)                               |
|                                                               |
|  Random baseline: 0.333 balanced accuracy                    |
|  Any model below baseline = broken (investigate immediately) |
|                                                               |
|  HP <-> AP confusion is the key localization signal          |
|  (distinguishing hand pain from arm pain is the               |
|   hard subproblem; NP vs {HP, AP} is the easy one)           |
+---------------------------------------------------------------+
```

---

## Test Submission Budget

The challenge allows 5 test-set label submissions per team. These are tracked manually in `scripts/submission_tracker.md` and each submission requires Vignan's explicit in-session approval (Runtime HIP RT-H). Submissions are numbered 1 through 5, each with a rationale, expected improvement estimate, and post-hoc result.

Default allocation:

| # | Model | Rationale |
|---|-------|-----------|
| 1 | Classical ML best | Early signal on whether the problem is tractable with handcrafted features |
| 2 | NN #1 (ResNet-1D) | Does DL help over classical at all? |
| 3 | NN #2 (current SOTA) | How much does researched SOTA close the gap? |
| 4 | NN #3 (novel) | Paper headline. Does the contribution actually win? |
| 5 | Reserve / ensemble | Last-minute improvement, top-N ensemble, or error analysis |

---

## Scaffolding-Time vs Runtime HIPs

Two tiers of Human Intervention Points govern this project.

**Scaffolding HIPs (HIP-1 through HIP-13)** live in the one-time scaffolding plan at `~/.claude/plans/staged-leaping-hippo.md`. They gate decisions during initial project construction: directory rename (HIP-1), Rust crate strategy (HIP-2), windowing strategy (HIP-4), research doc approvals (HIP-5, HIP-6), git init (HIP-10), commit review (HIP-11), PLAN.md review (HIP-12), and `/System-Sustenance` (HIP-13).

**Runtime HIPs (RT-A through RT-J)** live in `PLAN.md` and gate execution once data arrives and the project runs end-to-end. They cover data format confirmation (RT-A), feature QA (RT-B), class balance decisions (RT-C), classical ML sanity check (RT-D), windowing decisions (RT-E), NN #2 approval (RT-F), NN #3 approval (RT-G), per-submission approval (RT-H), paper narrative review (RT-I), and any git operation (RT-J).

These are distinct systems. Scaffolding HIPs are transient; runtime HIPs are permanent fixtures of the operational runbook.
