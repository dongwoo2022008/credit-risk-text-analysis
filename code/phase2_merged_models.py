"""
Phase 2: Merged Models (Structured + Text)
Uses preprocessed data files to train and evaluate merged models
across 4 text representation stages
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

import config
from utils.evaluator import evaluate_model

def load_preprocessed_data(data_path):
    """
    Load preprocessed data from pickle file
    """
    print(f"Loading preprocessed data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print(f"y_test shape: {data['y_test'].shape}")
    
    return data

def get_classifiers():
    """
    Initialize the six classification models used in the paper (LR, DT, NB, RF, GB, XGB)
    """
    models = {
        'LR': LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_SEED
        ),
        'DT': DecisionTreeClassifier(
            random_state=config.RANDOM_SEED
        ),
        'NB': GaussianNB(),
        'RF': RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        ),
        'GB': GradientBoostingClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED
        ),
        'XGB': XGBClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss',
            n_jobs=-1
        )
    }
    return models

def train_and_evaluate_stage(X_train, X_test, y_train, y_test, stage_name):
    """
    Train and evaluate all models for a given stage
    """
    print(f"\n{'='*80}")
    print(f"Training models for {stage_name}")
    print(f"{'='*80}")
    
    models = get_classifiers()
    results = []
    trained_models = {}
    
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"{i}/{len(models)} Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None
        
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        metrics['model'] = name
        results.append(metrics)
        
        # Store trained model
        trained_models[name] = model
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
                              'true_negative', 'false_positive', 'false_negative', 'true_positive']]
    
    return results_df, trained_models

def main():
    print(f"{'='*80}")
    print("Phase 2: Merged Models (Structured + Text)")
    print(f"{'='*80}")
    print(f"Random seed: {config.RANDOM_SEED}")
    
    # Define preprocessed data files (created by create_tfidf_preprocessed_data.py etc.)
    pre = config.DATA_DIR / "preprocessed"
    data_files = {
        'stage1_tfidf': pre / 'preprocessed_merged_struct_tfidf_binary.pkl',
        'stage2_subword': pre / 'preprocessed_merged_struct_subword_binary.pkl',
        'stage3_minilm': pre / 'preprocessed_merged_struct_minilm_binary.pkl',
        'stage4_kosimcse': pre / 'preprocessed_merged_struct_kosimcse_binary.pkl'
    }
    
    # Dictionary to store all results
    all_results = {}
    all_models = {}
    
    # Process each stage
    for stage_name, data_path in data_files.items():
        if not Path(data_path).exists():
            print(f"\nSkipping {stage_name}: {data_path} not found")
            continue
        print(f"\n{'='*80}")
        print(f"Processing {stage_name.upper()}")
        print(f"{'='*80}")
        
        # Load preprocessed data
        data = load_preprocessed_data(data_path)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Train and evaluate
        results_df, models_dict = train_and_evaluate_stage(
            X_train, X_test, y_train, y_test, stage_name
        )
        
        all_results[stage_name] = results_df
        all_models[stage_name] = models_dict
    
    # Print summary
    print(f"\n{'='*80}")
    print("Phase 2 Summary: Merged Models Across 4 Stages")
    print(f"{'='*80}")
    
    for stage_name, results_df in all_results.items():
        print(f"\n{stage_name.upper()}:")
        print(results_df.to_string(index=False))
        best_model = results_df.loc[results_df['roc_auc'].idxmax()]
        print(f"Best model: {best_model['model']} (ROC-AUC: {best_model['roc_auc']:.4f})")
    
    # Save results
    config.ensure_dir(config.TABLES_DIR)
    config.ensure_dir(config.MODELS_DIR / "phase2")
    
    for stage_name, results_df in all_results.items():
        output_file = config.TABLES_DIR / f"phase2_{stage_name}_performance.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_file}")
    
    # Save models
    for stage_name, models_dict in all_models.items():
        for model_name, model in models_dict.items():
            model_file = config.MODELS_DIR / "phase2" / f"{stage_name}_{model_name.lower()}_model.joblib"
            joblib.dump(model, model_file)
    
    print(f"\n{'='*80}")
    print("Phase 2 completed successfully!")
    print(f"{'='*80}")
    
    return all_results, all_models

if __name__ == "__main__":
    results, models = main()
