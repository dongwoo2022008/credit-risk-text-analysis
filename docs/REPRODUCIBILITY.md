# Reproducibility Guide

This document provides detailed instructions for reproducing all results presented in the paper **"Conditional Value of Borrower Narratives in Credit Risk Prediction: Evidence from a Korean P2P Platform"**.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Execution Instructions](#execution-instructions)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Expected Results](#expected-results)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 10 GB free space
- **CPU**: Multi-core processor recommended for faster execution

### Recommended Environment

- **Operating System**: Ubuntu 20.04 LTS or macOS 12+
- **Python**: 3.9 or 3.10
- **RAM**: 16 GB or more
- **Storage**: 20 GB free space
- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores)
- **GPU**: Optional, but recommended for faster text embedding (CUDA-compatible)

## Environment Setup

### Option 1: Using pip (Recommended)

1. **Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Verify installation**

```bash
python -c "import sklearn, xgboost, transformers; print('All packages installed successfully')"
```

### Option 2: Using conda

1. **Create conda environment**

```bash
conda env create -f environment.yml
conda activate credit-risk
```

2. **Verify installation**

```bash
python -c "import sklearn, xgboost, transformers; print('All packages installed successfully')"
```

### Option 3: Using Docker (Most Reproducible)

1. **Build Docker image**

```bash
docker build -t credit-risk-analysis .
```

2. **Run container**

```bash
docker run -it -v $(pwd)/results:/app/results credit-risk-analysis
```

## Data Preparation

### Step 1: Verify Raw Data

Ensure the raw data file exists:

```bash
ls -lh data/raw/sentiment_scoring.25.12.30.xlsx
```

Expected output:
```
-rw-r--r-- 1 user user 4.4M [date] data/raw/sentiment_scoring.25.12.30.xlsx
```

### Step 2: Data Integrity Check

Verify data integrity using checksums (optional):

```bash
md5sum data/raw/sentiment_scoring.25.12.30.xlsx
```

Expected MD5 checksum: `[to be filled after data finalization]`

### Step 3: Understand Data Structure

The raw data contains:
- **6,057 rows** (loan applications)
- **11 structured variables** (demographics, credit, loan terms)
- **3 text fields** (title, loan purpose, repayment plan)
- **1 target variable** (binary default indicator)

For detailed variable definitions, see [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

## Execution Instructions

### Sequential Execution (Recommended)

Execute each phase in order to reproduce all paper results.

#### Phase 0: Structured-Only Baseline

```bash
python code/phase0_structured_baseline.py
```

**What it does:**
- Loads raw data
- Preprocesses 11 structured variables
- Trains 8 classification models (LR, SVM, KNN, DT, NB, RF, GB, XGB)
- Evaluates performance (ROC-AUC, Recall, F1-score)
- Generates Table 4-1

**Expected output:**
- `results/phase0/table_4_1_structured_only_performance.csv`
- `results/phase0/phase0_model_comparison.png`

**Expected runtime:** ~5 minutes

#### Phase 1: Text-Only Models

```bash
python code/phase1_text_only.py
```

**What it does:**
- Extracts text features across 4 stages (TF-IDF, Subword, MiniLM, KoSimCSE)
- Trains 8 models per stage using text features only
- Evaluates performance
- Generates Table 4-2

**Expected output:**
- `results/phase1/table_4_2_text_only_performance.csv`
- `results/phase1/stage_wise_comparison.png`

**Expected runtime:** ~30 minutes (text embedding is time-consuming)

#### Phase 2: Merged Models (Structured + Text)

```bash
python code/phase2_merged_models.py
```

**What it does:**
- Combines structured variables with text features (Stage 1-4)
- Trains 8 models per stage
- Compares against Phase 0 baseline
- Generates Table 4-3

**Expected output:**
- `results/phase2/table_4_3_merged_performance.csv`
- `results/phase2/stage_comparison_vs_baseline.png`

**Expected runtime:** ~30 minutes

#### Phase 3: Hyperparameter Tuning

```bash
python code/phase3_hyperparameter_tuning.py
```

**What it does:**
- Performs grid search cross-validation for RF, GB, XGB
- Compares tuned vs. baseline performance
- Generates Table 4-4

**Expected output:**
- `results/phase3/table_4_4_tuning_comparison.csv`
- `results/phase3/tuning_results.json`

**Expected runtime:** ~60 minutes (grid search is computationally intensive)

#### Phase 4: Ensemble Models

```bash
python code/phase4_ensemble_models.py
```

**What it does:**
- Applies ensemble strategies (Hard/Soft/Weighted Voting, Blending, Stacking)
- Compares ensemble vs. single model performance
- Generates Table 4-5

**Expected output:**
- `results/phase4/table_4_5_ensemble_performance.csv`
- `results/phase4/ensemble_comparison.png`

**Expected runtime:** ~15 minutes

#### Phase 5: Conditional Effect Analysis

Execute all Phase 5 sub-analyses:

```bash
# Phase 5.1: Uncertainty-based marginal case analysis
python code/phase5_1_uncertainty_analysis.py

# Phase 5.2: Error case analysis (FN recovery)
python code/phase5_2_error_case_analysis.py

# Phase 5.3: Threshold sensitivity analysis
python code/phase5_3_threshold_sensitivity.py

# Phase 5.4: Text length/richness interaction analysis
python code/phase5_4_text_length_analysis.py
```

**Expected outputs:**
- Tables 4-6 to 4-13
- Figures 4-2 to 4-7

**Expected runtime:** ~20 minutes total

### Batch Execution

Run all phases at once using the provided shell script:

```bash
bash run_all.sh
```

This script executes all phases sequentially and logs output to `run_all.log`.

**Total expected runtime:** ~2.5–3 hours

## Verification

### Step 1: Check Output Files

Verify that all expected output files are generated:

```bash
# Check Phase 0 outputs
ls results/phase0/

# Check Phase 1 outputs
ls results/phase1/

# Check Phase 2 outputs
ls results/phase2/

# Check Phase 3 outputs
ls results/phase3/

# Check Phase 4 outputs
ls results/phase4/

# Check Phase 5 outputs
ls results/phase5/
```

### Step 2: Validate Key Results

Compare your results against the paper's reported values:

#### Table 4-1: Structured-Only Performance

| Model | ROC-AUC (Paper) | ROC-AUC (Your Result) | Match? |
|-------|-----------------|------------------------|--------|
| GB    | 0.812           | [Check your CSV]       | ✓/✗    |
| RF    | 0.799           | [Check your CSV]       | ✓/✗    |
| XGB   | 0.794           | [Check your CSV]       | ✓/✗    |

#### Table 4-6: Marginal Cases (GB+Text)

| Metric | Paper | Your Result | Match? |
|--------|-------|-------------|--------|
| ROC-AUC Change | +9.32% | [Check your CSV] | ✓/✗ |
| F1 Change | +12.78% | [Check your CSV] | ✓/✗ |

### Step 3: Visual Inspection

Compare generated figures with paper figures:

```bash
# View Phase 5 figures
ls figures/
```

Open and visually compare:
- `figure_4_2_marginal_vs_clear.png`
- `figure_4_3_fn_recovery_rate.png`
- `figure_4_4_f1_gap.png`
- `figure_4_5_recall_gap.png`
- `figure_4_6_default_by_length.png`
- `figure_4_7_length_sensitivity.png`

### Step 4: Statistical Tests (Optional)

Run statistical tests to verify reproducibility:

```bash
python code/utils/verify_reproducibility.py
```

This script compares your results against reference results and reports discrepancies.

## Troubleshooting

### Issue 1: Package Installation Errors

**Problem:** `pip install -r requirements.txt` fails

**Solution:**
1. Upgrade pip: `pip install --upgrade pip`
2. Install packages one by one to identify the problematic package
3. Check Python version: `python --version` (must be 3.8+)

### Issue 2: Out of Memory (OOM) Errors

**Problem:** Script crashes with "MemoryError" or "Killed"

**Solution:**
1. Reduce batch size in text embedding (edit `code/utils/text_features.py`)
2. Use a machine with more RAM (16 GB recommended)
3. Process data in chunks (modify scripts to use `chunksize` parameter)

### Issue 3: CUDA/GPU Errors

**Problem:** Text embedding fails with CUDA errors

**Solution:**
1. Force CPU mode: Set `device='cpu'` in `code/utils/text_features.py`
2. Update PyTorch: `pip install --upgrade torch`
3. Check CUDA compatibility: `nvidia-smi`

### Issue 4: Different Results

**Problem:** Your results don't match the paper exactly

**Possible causes:**
1. **Random seed not set correctly**: Verify `RANDOM_SEED = 42` in `code/utils/config.py`
2. **Different package versions**: Check `pip list` against `requirements.txt`
3. **Data preprocessing differences**: Ensure you're using the exact raw data file
4. **Floating-point precision**: Minor differences (<0.001) are acceptable due to numerical precision

**Solution:**
1. Re-run with fresh environment
2. Check random seed in all scripts
3. Verify data integrity (MD5 checksum)

### Issue 5: Slow Execution

**Problem:** Scripts take much longer than expected

**Solution:**
1. Use multi-core processing: Set `n_jobs=-1` in scikit-learn models
2. Use GPU for text embedding (if available)
3. Reduce hyperparameter search space in Phase 3
4. Skip Phase 1 and Phase 3 if only interested in final results

## Expected Results

### Performance Benchmarks

| Phase | Key Metric | Expected Value | Tolerance |
|-------|------------|----------------|-----------|
| Phase 0 | GB ROC-AUC | 0.812 | ±0.005 |
| Phase 2 | GB+Text ROC-AUC (Stage 1) | 0.811 | ±0.005 |
| Phase 5.1 | Marginal ROC-AUC improvement | +9.32% | ±1% |
| Phase 5.2 | FN Recovery Rate (RF+Text) | 41.30% | ±2% |

### File Size Benchmarks

| File | Expected Size |
|------|---------------|
| `data/raw/sentiment_scoring.25.12.30.xlsx` | ~4.4 MB |
| `results/phase0/table_4_1_*.csv` | ~1 KB |
| `results/phase5/figure_4_2_*.png` | ~50-100 KB |

### Runtime Benchmarks

| Phase | Expected Runtime (i7, 16GB RAM) |
|-------|----------------------------------|
| Phase 0 | 3-7 minutes |
| Phase 1 | 25-35 minutes |
| Phase 2 | 25-35 minutes |
| Phase 3 | 50-70 minutes |
| Phase 4 | 10-20 minutes |
| Phase 5 | 15-25 minutes |
| **Total** | **2.5-3 hours** |

## Additional Notes

### Reproducibility Guarantees

This repository guarantees reproducibility under the following conditions:

1. **Exact environment**: Same Python version and package versions (use `requirements.txt`)
2. **Fixed random seed**: All scripts use `RANDOM_SEED = 42`
3. **Fixed data split**: Train-test split is deterministic with `random_state=42`
4. **Deterministic algorithms**: All models use deterministic settings where possible

### Known Limitations

1. **Floating-point precision**: Results may vary slightly across different hardware/OS due to floating-point arithmetic
2. **Text embedding models**: Pre-trained models (MiniLM, KoSimCSE) may produce slightly different embeddings across versions
3. **Parallel processing**: Using `n_jobs=-1` may introduce minor non-determinism in some scikit-learn models

### Reporting Issues

If you encounter reproducibility issues:

1. Check this guide thoroughly
2. Verify environment setup
3. Open an issue on GitHub with:
   - Your environment details (`pip list`, `python --version`)
   - Error messages or logs
   - Observed vs. expected results

## Contact

For reproducibility questions, contact:

**Dongwoo Kim**  
Email: dongwoo.kim@bu.ac.kr  
GitHub: https://github.com/dongwoo2022008/credit-risk-text-analysis

---

Last updated: 2026-02-01
