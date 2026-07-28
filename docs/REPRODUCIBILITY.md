# Reproducibility Guide

This guide reproduces the results of **"The Economic Value of Textual
Information in Credit Risk Prediction: Evidence from a Machine Learning
Framework"** (Dongwoo Kim, Baekseok University).

## 1. Environment

- Python 3.9+
- `pip install -r requirements.txt`
- `sentence-transformers` is required to reproduce Stage 3 (MiniLM,
  `sentence-transformers/all-MiniLM-L6-v2`) and Stage 4 (KoSimCSE,
  `BM-K/KoSimCSE-roberta`) exactly; without it the scripts fall back to a
  TF-IDF proxy and print a warning.

## 2. Data

Place the restricted raw file (obtained from the platform under a
confidentiality agreement) at:

```
data/raw/sentiment_scoring.25.12.30.xlsx
```

The dataset contains 6,057 loan applications: 13 structured variables,
3 text fields (title, loan purpose, repayment plan) and a binary default
target (default = 1, repayment = 0; 3,352 defaults / 2,705 repayments).
Variable definitions: see `DATA_DICTIONARY.md`.

## 3. Evaluation protocol (identical across all phases)

Repeated stratified 5-fold cross-validation with 5 repeats
(`RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` → 25
test folds). Identical fold indices are shared across settings to enable
paired fold-wise comparisons (Δ). Scaling and TF–IDF vectorizers are fitted
on training folds only. Metrics: ROC–AUC, PR–AUC, H-measure (Beta(2,2)),
Recall and F1 at τ = 0.5; reported as mean [95% CI].

## 4. Reproducing the paper's tables

| Paper table | Command |
|---|---|
| Table 4 (structured-only, Phase 0) | `python code/run_repeated_cv_benchmark.py --setting structured` |
| Table 5 (text-only, Phase 1)       | `python code/run_repeated_cv_benchmark.py --setting text` (per stage; see phase1 script for Stages 2–4 features) |
| Table 6 (merged, Phase 2)          | `python code/run_repeated_cv_benchmark.py --setting merged` |
| Table 7 (tuning, Phase 3)          | `python code/phase3_hyperparameter_tuning.py` (grids in `config.TUNING_PARAM_GRID`) |
| Tables 8–10 (ensembles, Phase 4)   | `python code/phase4_ensemble_models.py` (Voting-S / Voting-W / Stacking with XGB meta-learner) |
| Table 11 / Fig 2 (SHAP)            | `python code/run_shap_analysis.py` |

Hyperparameter search grids (Phase 3, grid-search CV on training folds):

- RandomForest: n_estimators {50,100,200}; max_depth {5,10,15,None}; min_samples_split {2,5,10}
- GradientBoosting: n_estimators {50,100,200}; learning_rate {0.01,0.1,0.3}; max_depth {3,5,7}
- XGBoost: n_estimators {50,100,200}; learning_rate {0.01,0.1,0.3}; max_depth {3,5,7}

## 5. Notes

- `run_repeated_cv_benchmark.py` is the canonical implementation of the
  paper's protocol; the `phaseN_*.py` scripts additionally persist trained
  model artifacts for inspection.
- Text preprocessing: normalization, tokenization, stopword removal,
  rare-token filtering; three text fields are concatenated into a single
  narrative before vectorization (`TFIDF_MAX_FEATURES = 100`).
- The H-measure implementation (Hand, 2009; Beta(2,2)) is validated against
  boundary cases (perfect classifier → 1, random → ≈0).
