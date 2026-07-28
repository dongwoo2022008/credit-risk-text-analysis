"""
Configuration file for credit risk analysis
Defines all global constants, paths, and parameters for reproducibility
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Raw data file
RAW_DATA_FILE = RAW_DATA_DIR / "sentiment_scoring.25.12.30.xlsx"

# ============================================================================
# REPRODUCIBILITY SETTINGS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Train/test split ratio (80/20 as per paper)
TEST_SIZE = 0.2

# Evaluation protocol used in the paper: repeated stratified 5-fold cross-validation
# (RepeatedStratifiedKFold; n_splits=5, n_repeats=5, random_state=42 -> 25 test folds)
CV_FOLDS = 5
CV_REPEATS = 5

# ============================================================================
# TARGET VARIABLE DEFINITION
# ============================================================================

# Target variable name in raw data
TARGET_COLUMN = "상환결과"

# Target encoding: 채무불이행 = 1, others = 0
TARGET_DEFAULT_VALUE = "채무불이행"
TARGET_POSITIVE_CLASS = 1
TARGET_NEGATIVE_CLASS = 0

# ============================================================================
# STRUCTURED VARIABLES
# ============================================================================

# Structured feature columns (13 variables, as in the paper)
STRUCTURED_FEATURES = [
    "성공횟수",
    "신용평점",
    "신청금액(만원)",
    "투자인원",
    "신청금리",
    "총횟수",
    "성공률",
    "대출용도(대출상환0)",
    "4대보험(가입0)",
    "근무개월",
    "대출시기",
    "나이",
    "대출(은행보험)"
]

# ============================================================================
# TEXT PROCESSING SETTINGS
# ============================================================================

# Text column names (title, purpose, repayment plan)
TEXT_COLUMNS = ["제목", "신청목적", "상환계획"]

# TF-IDF parameters
TFIDF_MAX_FEATURES = 100
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# Korean tokenizers
TOKENIZER_OKT = "okt"
TOKENIZER_MECAB = "mecab"

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# The six classifiers used in the paper (LR, DT, NB, RF, GB, XGB)
PHASE0_MODELS = {
    "LogisticRegression": {"C": 1.0, "max_iter": 1000, "random_state": RANDOM_SEED},
    "DecisionTree": {"random_state": RANDOM_SEED},
    "NaiveBayes": {},
    "RandomForest": {"n_estimators": 100, "random_state": RANDOM_SEED},
    "GradientBoosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": RANDOM_SEED},
    "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": RANDOM_SEED}
}

# Phase 3: Hyperparameter tuning ranges (RF/GB/XGB, as in the paper)
TUNING_PARAM_GRID = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7]
    }}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Probability threshold for classification (tau)
DEFAULT_THRESHOLD = 0.5

# Metrics to calculate
METRICS = [
    "roc_auc",
    "pr_auc",
    "h_measure",
    "recall",
    "f1_score"
]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_FONT = "Arial"
FIGURE_STYLE = "grayscale"  # Black and white

# Figure sizes (inches)
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_MEDIUM = (10, 8)
FIGURE_SIZE_LARGE = (12, 10)

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Logging level
LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_model_path(phase, model_name):
    """Get model save path"""
    ensure_dir(MODELS_DIR / phase)
    return MODELS_DIR / phase / f"{model_name}.joblib"

def get_table_path(table_name):
    """Get table save path"""
    ensure_dir(TABLES_DIR)
    return TABLES_DIR / f"{table_name}.csv"

def get_figure_path(figure_name):
    """Get figure save path"""
    ensure_dir(FIGURES_DIR)
    return FIGURES_DIR / f"{figure_name}.{FIGURE_FORMAT}"

# ============================================================================
# VALIDATION
# ============================================================================

# Validate paths on import
if not RAW_DATA_FILE.exists():
    print(f"Warning: Raw data file not found at {RAW_DATA_FILE}")
    print("Please place sentiment_scoring.25.12.30.xlsx in data/raw/ (restricted; see README)")

# Create necessary directories
for directory in [PROCESSED_DATA_DIR, SPLITS_DIR, TABLES_DIR, FIGURES_DIR]:
    ensure_dir(directory)

print(f"Configuration loaded successfully")
print(f"Project root: {PROJECT_ROOT}")
print(f"Random seed: {RANDOM_SEED}")
