# Conditional Value of Borrower Narratives in Credit Risk Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete reproducible research code and data for the paper **"Conditional Value of Borrower Narratives in Credit Risk Prediction: Evidence from a Korean P2P Platform"** by Dongwoo Kim (Baekseok University).

## Overview

This study systematically evaluates the incremental value of integrating borrower narrative text with structured variables for credit-risk prediction using long-term operational data (2006–2016) from a Korean peer-to-peer lending platform. Rather than assuming that text uniformly improves average predictive accuracy, we examine the operational conditions under which text provides value, including uncertainty regions, error-sensitive situations, threshold sensitivity, and information richness.

### Key Findings

- **Structured variables alone** deliver strong baseline discrimination (ROC-AUC: 0.812 with Gradient Boosting)
- **Text-only models** perform near random ranking (ROC-AUC ≈ 0.49–0.50)
- **Conditional value of text**: Narrative text adds significant value in specific contexts:
  - **Marginal cases** (predicted risk 0.3–0.7): +9.32% ROC-AUC, +12.78% F1-score
  - **False negative recovery**: 41.30% overall, 52.63% in high-risk group
  - **Longer narratives**: Modest performance gains
  - **Shorter narratives**: Can reduce performance

## Repository Structure

```
credit-risk-text-analysis/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory
│   ├── raw/                           # Original data
│   │   └── sentiment_scoring.25.12.30.xlsx
│   ├── processed/                     # Preprocessed data (generated)
│   └── README.md                      # Data documentation
│
├── code/                              # Analysis code
│   ├── phase0_structured_baseline.py  # Phase 0: Structured-only benchmark
│   ├── phase1_text_only.py            # Phase 1: Text-only models
│   ├── phase2_merged_models.py        # Phase 2: Structured + text integration
│   ├── phase3_hyperparameter_tuning.py # Phase 3: Hyperparameter tuning
│   ├── phase4_ensemble_models.py      # Phase 4: Ensemble strategies
│   ├── phase5_1_uncertainty_analysis.py    # Phase 5.1: Uncertainty analysis
│   ├── phase5_2_error_case_analysis.py     # Phase 5.2: Error case analysis
│   ├── phase5_3_threshold_sensitivity.py   # Phase 5.3: Threshold sensitivity
│   ├── phase5_4_text_length_analysis.py    # Phase 5.4: Text length analysis
│   │
│   └── utils/                         # Utility modules
│       ├── __init__.py
│       ├── data_preprocessing.py      # Data preprocessing functions
│       ├── text_preprocessing.py      # Text preprocessing functions
│       ├── text_features.py           # Text feature extraction (Stage 1-4)
│       ├── model_training.py          # Model training and evaluation
│       ├── visualization.py           # Visualization functions
│       └── config.py                  # Configuration (random seed, paths)
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 00_data_exploration.ipynb
│   ├── 01_phase0_structured_baseline.ipynb
│   ├── 02_phase1_text_only.ipynb
│   ├── 03_phase2_merged_models.ipynb
│   ├── 04_phase3_hyperparameter_tuning.ipynb
│   ├── 05_phase4_ensemble_models.ipynb
│   └── 06_phase5_conditional_analysis.ipynb
│
├── results/                           # Analysis results
│   ├── phase0/                        # Phase 0 results (Table 4-1)
│   ├── phase1/                        # Phase 1 results (Table 4-2)
│   ├── phase2/                        # Phase 2 results (Table 4-3)
│   ├── phase3/                        # Phase 3 results (Table 4-4)
│   ├── phase4/                        # Phase 4 results (Table 4-5)
│   └── phase5/                        # Phase 5 results (Tables 4-6 to 4-13, Figures 4-2 to 4-7)
│
├── figures/                           # Figures used in the paper
│   ├── figure_1_1_research_framework.png
│   ├── figure_4_1_phase5_framework.png
│   └── phase5_detail_flowchart.png
│
├── models/                            # Trained models (optional)
│
├── docs/                              # Documentation
│   ├── REPRODUCIBILITY.md             # Reproducibility guide
│   ├── DATA_DICTIONARY.md             # Data dictionary
│   ├── CODE_PAPER_MAPPING.md          # Code-to-paper mapping
│   └── METHODOLOGY.md                 # Detailed methodology
│
└── paper/                             # Paper files
    └── manuscript.docx                # Paper manuscript
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/dongwoo2022008/credit-risk-text-analysis.git
cd credit-risk-text-analysis
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate credit-risk
```

3. **Verify data availability**

```bash
ls data/raw/sentiment_scoring.25.12.30.xlsx
```

### Running the Analysis

Execute the analysis phases sequentially:

```bash
# Phase 0: Structured-only baseline
python code/phase0_structured_baseline.py

# Phase 1: Text-only models
python code/phase1_text_only.py

# Phase 2: Merged models (structured + text)
python code/phase2_merged_models.py

# Phase 3: Hyperparameter tuning
python code/phase3_hyperparameter_tuning.py

# Phase 4: Ensemble models
python code/phase4_ensemble_models.py

# Phase 5: Conditional effect analysis
python code/phase5_1_uncertainty_analysis.py
python code/phase5_2_error_case_analysis.py
python code/phase5_3_threshold_sensitivity.py
python code/phase5_4_text_length_analysis.py
```

Or run all phases at once:

```bash
bash run_all.sh
```

### Expected Runtime

- Phase 0: ~5 minutes
- Phase 1: ~30 minutes (includes text embedding)
- Phase 2: ~30 minutes
- Phase 3: ~60 minutes (hyperparameter tuning)
- Phase 4: ~15 minutes
- Phase 5: ~20 minutes

**Total**: Approximately 2.5–3 hours on a standard machine (Intel i7, 16GB RAM)

## Reproducibility

This repository is designed for full reproducibility. All experiments use:

- **Fixed random seed**: 42 (set in `code/utils/config.py`)
- **Fixed train-test split**: 80/20, stratified sampling
- **Version-pinned dependencies**: See `requirements.txt`
- **Self-contained code**: Each script runs from raw data to results

For detailed reproducibility instructions, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data

The dataset consists of 6,057 loan applications from a Korean P2P lending platform (2006–2016).

### Variables

- **Structured variables (11)**: Gender, Region, Age, Credit Score, Monthly Income, Loan Amount, Loan Interest Rate, Monthly DTI, Loan Term, Months of Service, Number of Investors
- **Text variables (3 fields)**: Title, Loan Purpose, Repayment Plan
- **Target variable**: Binary default indicator (1 = default, 0 = repayment)

For detailed variable definitions, see [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Methodology

The study follows a staged experimental design:

- **Phase 0**: Structured-only baseline (8 models: LR, SVM, KNN, DT, NB, RF, GB, XGB)
- **Phase 1**: Text-only models across 4 representation stages
  - Stage 1: TF-IDF
  - Stage 2: Subword features
  - Stage 3: MiniLM embeddings
  - Stage 4: KoSimCSE embeddings
- **Phase 2**: Structured + text integration
- **Phase 3**: Hyperparameter tuning (RF, GB, XGB)
- **Phase 4**: Ensemble strategies (Voting, Blending, Stacking)
- **Phase 5**: Conditional effect analysis
  - Uncertainty-based marginal case analysis
  - Error case analysis (FN recovery)
  - Threshold sensitivity analysis
  - Text length/richness interaction analysis

For detailed methodology, see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Code-to-Paper Mapping

Each table and figure in the paper corresponds to specific code:

| Paper Element | Code File | Output Location |
|---------------|-----------|-----------------|
| Table 4-1 | `phase0_structured_baseline.py` | `results/phase0/table_4_1_structured_only_performance.csv` |
| Table 4-2 | `phase1_text_only.py` | `results/phase1/table_4_2_text_only_performance.csv` |
| Table 4-3 | `phase2_merged_models.py` | `results/phase2/table_4_3_merged_performance.csv` |
| Table 4-4 | `phase3_hyperparameter_tuning.py` | `results/phase3/table_4_4_tuning_comparison.csv` |
| Table 4-5 | `phase4_ensemble_models.py` | `results/phase4/table_4_5_ensemble_performance.csv` |
| Tables 4-6, 4-7 | `phase5_1_uncertainty_analysis.py` | `results/phase5/table_4_6_marginal_cases.csv` |
| Tables 4-8, 4-9 | `phase5_2_error_case_analysis.py` | `results/phase5/table_4_8_confusion_matrix.csv` |
| Table 4-10 | `phase5_3_threshold_sensitivity.py` | `results/phase5/table_4_10_threshold_sensitivity.csv` |
| Tables 4-11 to 4-13 | `phase5_4_text_length_analysis.py` | `results/phase5/table_4_11_length_decile.csv` |

For complete mapping, see [`docs/CODE_PAPER_MAPPING.md`](docs/CODE_PAPER_MAPPING.md).

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{kim2026conditional,
  title={Conditional Value of Borrower Narratives in Credit Risk Prediction: Evidence from a Korean P2P Platform},
  author={Kim, Dongwoo},
  journal={[Journal Name]},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Dongwoo Kim**  
Baekseok University  
Email: dongwoo.kim@bu.ac.kr  
Tel: +82-10-3936-6650

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback and suggestions.

## Version History

- **v1.0** (2026-02-01): Initial release with full reproducible code and data
