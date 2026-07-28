"""
Phase 0: Structured-only Baseline Performance
===============================================

This script implements Phase 0 of the research: benchmarking the six
classification models used in the paper with structured variables only
(13 features) to establish a baseline for credit risk prediction.

Models evaluated (paper roster): LR, DT, NB, RF, GB, XGB
- Logistic Regression (LR)
- Decision Tree (DT)
- Naive Bayes (NB)
- Random Forest (RF)
- Gradient Boosting (GB)
- XGBoost (XGB)

Corresponds to: Table 4 in the paper (canonical repeated-CV figures are
produced by run_repeated_cv_benchmark.py; this script trains and persists
model artifacts on a fixed 80/20 split for inspection).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
import config
from utils import load_and_prepare_data, evaluate_multiple_models, print_evaluation_summary

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

def train_structured_models(X_train, y_train, random_seed=42):
    """
    Train the six classification models used in the paper (LR, DT, NB, RF, GB, XGB) with structured features only
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        random_seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary of trained models
    """
    print("Training 6 classification models with structured features only...")
    print(f"Training set size: {len(X_train)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Default rate: {y_train.mean():.2%}\n")
    
    models = {}
    
    # 1. Logistic Regression
    print("1/6 Training Logistic Regression...")
    models['LR'] = LogisticRegression(
        random_state=random_seed,
        max_iter=1000,
        solver='lbfgs'
    )
    models['LR'].fit(X_train, y_train)
            
    # 2. Decision Tree
    print("2/6 Training Decision Tree...")
    models['DT'] = DecisionTreeClassifier(
        random_state=random_seed,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    models['DT'].fit(X_train, y_train)
    
    # 3. Naive Bayes
    print("3/6 Training Naive Bayes...")
    models['NB'] = GaussianNB()
    models['NB'].fit(X_train, y_train)
    
    # 4. Random Forest
    print("4/6 Training Random Forest...")
    models['RF'] = RandomForestClassifier(
        random_state=random_seed,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )
    models['RF'].fit(X_train, y_train)
    
    # 5. Gradient Boosting
    print("5/6 Training Gradient Boosting...")
    models['GB'] = GradientBoostingClassifier(
        random_state=random_seed,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1
    )
    models['GB'].fit(X_train, y_train)
    
    # 6. XGBoost
    print("6/6 Training XGBoost...")
    models['XGB'] = XGBClassifier(
        random_state=random_seed,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    models['XGB'].fit(X_train, y_train)
    
    print("\nAll models trained successfully!\n")
    
    return models

def main():
    """
    Main execution function for Phase 0
    """
    print("="*80)
    print("Phase 0: Structured-only Baseline Performance")
    print("="*80)
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Test size: {config.TEST_SIZE}")
    print(f"Data path: {config.RAW_DATA_FILE}\n")
    
    # Load and prepare data
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}\n")
    
    # Train models
    models = train_structured_models(X_train, y_train, config.RANDOM_SEED)
    
    # Evaluate models
    print("="*80)
    print("Evaluating models on test set...")
    print("="*80)
    results_df = evaluate_multiple_models(models, X_test, y_test)
    
    # Display results
    print("\n" + "="*80)
    print("Phase 0 Results: Structured-only Baseline Performance")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Find best model
    best_model_name = results_df.loc[results_df['roc_auc'].idxmax(), 'model']
    best_roc_auc = results_df['roc_auc'].max()
    
    print(f"\nBest model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    # Save results
    output_dir = Path(config.RESULTS_DIR) / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'phase0_structured_only_performance.csv'
    
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_path}")
    
    # Save models
    model_dir = Path(config.MODELS_DIR) / 'phase0'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    for model_name, model in models.items():
        model_path = model_dir / f'{model_name.lower()}_model.joblib'
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")
    
    print("\n" + "="*80)
    print("Phase 0 completed successfully!")
    print("="*80)
    
    return results_df, models

if __name__ == "__main__":
    results_df, models = main()
