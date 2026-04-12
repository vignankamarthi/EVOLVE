# AI4Pain 2026 Challenge

**3-class pain localization from multimodal physiological signals for the PAAIn Workshop at ACII 2026, Puebla, Mexico.**

Two-track experiment: handcrafted features + classical ML, and raw-signal deep learning with three neural architectures (ResNet-1D baseline, current SOTA, and a novel homemade framework). The deep learning track is the paper's central contribution.

**Dataset**: AI4Pain 2026 Challenge release, 65 participants, fixed split 41 train / 12 validation / 12 test. Four signals: BVP (blood volume pulse), EDA (electrodermal activity), RESP (respiration), SpO2 (blood oxygen saturation). Labels: No Pain (NP), Hand Pain (HP), Arm Pain (AP).

**Hard constraints**: No external data. Five test-set label submissions maximum per team.

---

## Process Flow

```
AI4Pain 2026 (65 participants, 41/12/12 subject split)
    |
    +-- BVP (blood volume pulse)
    +-- EDA (electrodermal activity)
    +-- RESP (respiration)
    +-- SpO2 (blood oxygen saturation)
    +-- Labels: NP | HP | AP
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
40 features per signal:              (channel count per ablation)
  Catch22 (22)                         |
  Entropy (10)                         v
  Stats (8)                       3 architectures:
    |                                NN #1: ResNet-1D (baseline)
    v                                NN #2: Current SOTA (researched)
StandardScaler                       NN #3: Novel framework (researched)
(per column, train only)               |
    |                                    v
    v                               CrossEntropyLoss + class weights
3 classifiers x 3 ablations:        Per-epoch checkpointing
  Random Forest                          |
  XGBoost                                |
  LightGBM                               |
  (+ Logistic Regression baseline)       |
    |                                    |
    v                                    v
Optuna HPO                         Validation on 12 val subjects
    |                                    |
    +================+===================+
                     |
                     v
          EVALUATION (same metrics, both tracks)
          Balanced accuracy, macro-F1,
          per-class precision/recall, confusion matrix, AUC-OVR
                     |
                     v
          TEST SET (5 submissions max)
                     |
                     v
          ACII 2026 PAAIn Paper
```

**Ablation configurations** (`configs/ablation_configs.json`):

| Config | Signals | Features | DL Channels |
|--------|---------|----------|-------------|
| `bvp_eda` | BVP + EDA | 80 | 2 |
| `bvp_eda_resp` | BVP + EDA + RESP | 120 | 3 |
| `all_four` | BVP + EDA + RESP + SpO2 | 160 | 4 |

---

## Repository Structure

```
AI4Pain-2026/
├── feature-extraction-rust/    # Rust crate (Catch22 + entropy + stats)
├── src/
│   ├── data_loader.py         # AI4Pain CSV loader + StandardScaler
│   ├── models.py              # RF / XGB / LGBM classifiers
│   ├── tuning.py              # Optuna HPO wrapper
│   ├── evaluation.py          # Classification metric suite
│   ├── dl_models.py           # ResNet-1D baseline + NN #2 + NN #3
│   ├── dl_training.py         # PyTorch CE training loop
│   ├── dl_data.py             # Multi-channel raw-signal Dataset
│   └── utils.py               # Atomic writes, logging, seeding
├── scripts/
│   ├── train_models.py        # Classical ML entry point
│   ├── train_dl_models.py     # DL entry (--arch cnn1d|nn2|nn3)
│   ├── predict_test.py        # Test inference + submission CSV
│   ├── evaluate.py
│   ├── merge_features_labels.py
│   ├── submission_tracker.md  # HARD CAP of 5 submissions
│   └── *.sbatch               # SLURM jobs
├── configs/                   # Ablation + feature + model configs
├── research/
│   ├── NN2_SOTA_RESEARCH.md
│   └── NN3_NOVEL_FRAMEWORK.md
├── results/
├── reports/                   # ACII 2026 PAAIn manuscript
├── tests/
└── data/                      # Challenge data (not in git)
```

---

## Status

**Current state**: Scaffold complete, awaiting challenge data (April 2026 for train + validation, May 2026 for test).

The scaffolding produces an inert repo. Once data arrives, the linear runtime runbook (`PLAN.md`, local only) drives execution. All runtime decisions pass through Runtime HIPs (RT-A through RT-J) documented in that file.

---

## Quick Start

Prerequisites: Python 3.10+, Rust toolchain (cargo), access to the AI4Pain 2026 Challenge data release.

```bash
git clone https://github.com/vignankamarthi/AI4Pain-2026.git
cd AI4Pain-2026

# Python dependencies
pip install -r requirements.txt

# Build Rust feature extractor
cd feature-extraction-rust && cargo build --release && cd ..

# Smoke test (imports and scaffolding, no data required)
python -m pytest tests/ -v
```

Training commands (once data is in `data/raw/`):

```bash
# Classical ML HPO
python scripts/train_models.py --config bvp_eda

# Deep learning (ResNet-1D baseline)
python scripts/train_dl_models.py --arch cnn1d --config bvp_eda

# Test inference (manual submission budget applies)
python scripts/predict_test.py --model-path results/dl_nn3_novel/model.pth --output predictions_nn3.csv
```

---

## Methodology Context

This project builds on two prior AI4Pain works by the same author:

- **ICMI 2025**: Perfect binary pain classification using Catch22 + ensemble methods on multimodal physiological signals
- **Paper 2 (3-class intensity)**: 77.2% per-subject LOSO on 3-class pain intensity classification (low vs high vs none)

The AI4Pain 2026 Challenge is a **different problem**: pain **localization** (where on the body) rather than pain **intensity** (how much). The feature saliency, the likely SOTA architecture, and the physiological priors differ meaningfully. The Catch22 + entropy + stats feature extraction pipeline is reusable as a strong classical baseline, and the ResNet-1D comes from the Blood-Pressure-Inference-with-BVP project as a well-tested raw-signal starting point. NN #2 and NN #3 are researched from scratch for this localization problem.

---

## Citation

A paper is in preparation for submission to the PAAIn Workshop at ACII 2026 (Puebla, Mexico). Citation block will be added upon paper acceptance.

---

## License

Private repository until paper acceptance. Licensing decision deferred to publication time.
