# The Economic Value of Textual Information in Credit Risk Prediction

Reproducible analysis code for the paper:

**"The Economic Value of Textual Information in Credit Risk Prediction: Evidence from a Machine Learning Framework"**
Dongwoo Kim (Baekseok University) — dongwoo.kim@bu.ac.kr — ORCID [0000-0003-2219-083X](https://orcid.org/0000-0003-2219-083X)

## Overview

The study asks whether borrower-provided narrative text adds predictive value to
credit-default models when strong structured variables are already available,
using 6,057 loan applications from an operational Korean peer-to-peer lending platform.

- **Feature settings**: structured-only (Phase 0), text-only (Phase 1), merged structured+text (Phase 2), plus hyperparameter tuning (Phase 3) and ensembles (Phase 4).
- **Structured variables (13)**: loan amount, interest rate, loan purpose, loan timing, number of investors, age, credit score, employment months, insurance status, outstanding bank loan, total applications, successful applications, success rate.
- **Text representations (Stages 1–4)**: TF–IDF (100 features), sub-word features, MiniLM sentence embeddings, KoSimCSE sentence embeddings.
- **Classifiers (6)**: LR, DT, NB, RF, GB, XGB.
- **Evaluation protocol**: repeated stratified 5-fold cross-validation with 5 repeats (25 test folds; `RepeatedStratifiedKFold`, random_state=42), reporting mean values with 95% CIs for ROC–AUC, PR–AUC, H-measure, Recall and F1 (τ = 0.5). All preprocessing is fitted inside training folds only (no leakage).
- **Interpretability**: TreeSHAP on the tuned XGB merged model, computed on held-out folds only.

### Key findings

- Structured-only models provide strong, stable discrimination (best: GB, ROC–AUC 0.8193 [0.8156, 0.8230]).
- Text-only models are close to chance (ROC–AUC ≈ 0.50–0.51) across all four representations.
- Merging text with structured inputs yields limited average gains; tuning and ensembling add only marginal improvements.
- SHAP attribution: structured variables account for 97.3% of total importance; TF–IDF text features 2.7%.

## Repository structure

```
├── code/
│   ├── config.py                        # Paths, 13 structured variables, protocol constants, tuning grids
│   ├── run_repeated_cv_benchmark.py     # ★ Canonical evaluation (paper protocol; Tables 4–6)
│   ├── run_shap_analysis.py             # SHAP group attribution (Table 11 / Fig 2)
│   ├── create_tfidf_preprocessed_data.py
│   ├── phase0_structured_baseline.py    # Model training/persistence per phase
│   ├── phase1_text_only.py              # Stages 1–4 text representations
│   ├── phase2_merged_models.py
│   ├── phase3_hyperparameter_tuning.py  # Grid search (RF/GB/XGB; grids in config.py)
│   ├── phase4_ensemble_models.py        # Voting-S / Voting-W / Stacking
│   └── utils/                           # Data loading & evaluation helpers
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── DATA_DICTIONARY.md
└── requirements.txt
```

## Data availability

The raw loan-level data were obtained from a Korean P2P platform **under a
confidentiality agreement and are not distributed with this repository** (the
`data/` directory is intentionally excluded). The full data-processing
pipeline, feature construction, and analysis code are provided so that the
methodology is fully transparent and can be applied to comparable datasets.

## Quick start

```bash
pip install -r requirements.txt
# place the (restricted) raw file at data/raw/sentiment_scoring.25.12.30.xlsx
python code/run_repeated_cv_benchmark.py --setting structured
python code/run_repeated_cv_benchmark.py --setting text
python code/run_repeated_cv_benchmark.py --setting merged
python code/run_shap_analysis.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for details.

## License

MIT
