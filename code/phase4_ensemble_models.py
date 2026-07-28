"""
Phase 4: Ensemble Models (Voting, Blending, Stacking) - Merged Data
Combines multiple models to improve prediction performance using merged data (structured + TF-IDF)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

import config
from utils.evaluator import evaluate_model

def load_merged_data():
    """
    Load preprocessed merged data (structured + TF-IDF)
    """
    print("\nLoading preprocessed merged data (structured + TF-IDF)...")
    
    merged_data_path = config.DATA_DIR / 'preprocessed' / 'preprocessed_merged_struct_tfidf_binary.pkl'  # Stage 1 (TF-IDF): best merged setting in the paper
    with open(merged_data_path, 'rb') as f:
        merged_data = pickle.load(f)
    
    X_train = merged_data['X_train']
    X_test = merged_data['X_test']
    y_train = merged_data['y_train']
    y_test = merged_data['y_test']
    
    print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
    print(f"Test samples: {len(X_test)}, Features: {X_test.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def load_base_models():
    """
    Load trained base models from Phase 3 (tuned) or Phase 2 Stage 1 (TF-IDF merged models)
    """
    print("\nLoading base models...")
    
    models = {}
    
    # Prefer Phase 3 tuned models; fall back to Phase 2 Stage 1 models
    phase3_dir = config.MODELS_DIR / "phase3"
    phase2_dir = config.MODELS_DIR / "phase2"
    
    for model_name in ['rf', 'gb', 'xgb']:
        tuned_path = phase3_dir / f"{model_name}_tuned_merged_model.joblib"
        base_path = phase2_dir / f"stage1_tfidf_{model_name}_model.joblib"
        if tuned_path.exists():
            models[model_name] = joblib.load(tuned_path)
            print(f"  Loaded Phase 3 tuned {model_name.upper()} model")
        elif base_path.exists():
            models[model_name] = joblib.load(base_path)
            print(f"  Loaded Phase 2 Stage 1 {model_name.upper()} model")
        else:
            print(f"  Warning: {model_name.upper()} model not found (run Phase 2/3 first)")
    
    return models

def voting_ensemble(base_models, X_train, y_train, X_test, y_test, voting_type='soft'):
    """
    Create and evaluate voting ensemble
    """
    print(f"\n{'='*80}")
    print(f"Voting Ensemble ({voting_type.capitalize()} Voting)")
    print(f"{'='*80}")
    
    # Create voting classifier
    estimators = [(name, model) for name, model in base_models.items()]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=voting_type,
        n_jobs=-1
    )
    
    print(f"\nTraining voting ensemble ({voting_type} voting)...")
    voting_clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = voting_clf.predict(X_test)
    if voting_type == 'soft':
        y_proba = voting_clf.predict_proba(X_test)[:, 1]
    else:
        # For hard voting, use average of base model probabilities
        y_proba = np.mean([model.predict_proba(X_test)[:, 1] for model in base_models.values()], axis=0)
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    print(f"\nVoting Ensemble ({voting_type.capitalize()}) Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save model
    config.ensure_dir(config.MODELS_DIR / "phase4")
    model_path = config.MODELS_DIR / "phase4" / f"voting_{voting_type}_ensemble_merged.joblib"
    joblib.dump(voting_clf, model_path)
    print(f"\nVoting ensemble saved to: {model_path}")
    
    return voting_clf, metrics

def blending_ensemble(base_models, X_train, y_train, X_test, y_test):
    """
    Create and evaluate blending ensemble (simple averaging)
    """
    print(f"\n{'='*80}")
    print("Blending Ensemble (Simple Averaging)")
    print(f"{'='*80}")
    
    # Get predictions from all base models
    print("\nGetting predictions from base models...")
    predictions = []
    
    for name, model in base_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        predictions.append(y_proba)
        print(f"  {name.upper()} predictions obtained")
    
    # Average predictions
    blended_proba = np.mean(predictions, axis=0)
    blended_pred = (blended_proba >= 0.5).astype(int)
    
    # Evaluate
    metrics = evaluate_model(y_test, blended_pred, blended_proba)
    
    print(f"\nBlending Ensemble Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return blended_proba, metrics

def stacking_ensemble(base_models, X_train, y_train, X_test, y_test):
    """
    Create and evaluate stacking ensemble
    """
    print(f"\n{'='*80}")
    print("Stacking Ensemble")
    print(f"{'='*80}")
    
    # Create stacking classifier with logistic regression as meta-learner
    estimators = [(name, model) for name, model in base_models.items()]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    print("\nTraining stacking ensemble...")
    print("  Using 5-fold CV for meta-features...")
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = stacking_clf.predict(X_test)
    y_proba = stacking_clf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    print(f"\nStacking Ensemble Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save model
    model_path = config.MODELS_DIR / "phase4" / "stacking_ensemble_merged.joblib"
    joblib.dump(stacking_clf, model_path)
    print(f"\nStacking ensemble saved to: {model_path}")
    
    return stacking_clf, metrics

def compare_with_base_models(base_models, ensemble_results, X_test, y_test):
    """
    Compare ensemble models with base models
    """
    print(f"\n{'='*80}")
    print("Comparison: Base Models vs Ensemble Models (Merged Data)")
    print(f"{'='*80}")
    
    results = []
    
    # Base models
    for name, model in base_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)
        
        results.append({
            'model': f'{name.upper()} (tuned)',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        })
    
    # Ensemble models
    for ensemble_name, metrics in ensemble_results.items():
        results.append({
            'model': ensemble_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    print("\nAll Models Ranked by ROC-AUC:")
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['model']}")
    print(f"  ROC-AUC: {best_model['roc_auc']:.4f}")
    
    return results_df

def main():
    print(f"{'='*80}")
    print("Phase 4: Ensemble Models (Merged Data: Structured + TF-IDF)")
    print(f"{'='*80}")
    
    # Load merged data
    X_train, X_test, y_train, y_test = load_merged_data()
    
    # Load base models
    base_models = load_base_models()
    
    if len(base_models) == 0:
        print("\nError: No base models found. Please run Phase 2/3 first.")
        return
    
    print(f"\nLoaded {len(base_models)} base models: {list(base_models.keys())}")
    
    # Store ensemble results
    ensemble_results = {}
    
    # 1. Voting Ensemble (Hard)
    voting_hard_clf, voting_hard_metrics = voting_ensemble(base_models, X_train, y_train, X_test, y_test, voting_type='hard')
    ensemble_results['Voting (Hard)'] = voting_hard_metrics
    
    # 2. Voting Ensemble (Soft)
    voting_soft_clf, voting_soft_metrics = voting_ensemble(base_models, X_train, y_train, X_test, y_test, voting_type='soft')
    ensemble_results['Voting (Soft)'] = voting_soft_metrics
    
    # 3. Voting Ensemble (Weighted) - used in the paper
    # Weighted voting with weights based on validation performance
    print(f"\n{'='*80}")
    print("Voting Ensemble (Weighted)")
    print(f"{'='*80}")
    
    # Use ROC-AUC as weights (normalized)
    weights = []
    for name, model in base_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, y_proba)
        weights.append(metrics['roc_auc'])
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    print(f"\nModel weights: {dict(zip(base_models.keys(), weights))}")
    
    estimators = [(name, model) for name, model in base_models.items()]
    voting_weighted_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )
    
    voting_weighted_clf.fit(X_train, y_train)
    y_pred = voting_weighted_clf.predict(X_test)
    y_proba = voting_weighted_clf.predict_proba(X_test)[:, 1]
    voting_weighted_metrics = evaluate_model(y_test, y_pred, y_proba)
    
    print(f"\nVoting Ensemble (Weighted) Performance:")
    print(f"  Accuracy:  {voting_weighted_metrics['accuracy']:.4f}")
    print(f"  Precision: {voting_weighted_metrics['precision']:.4f}")
    print(f"  Recall:    {voting_weighted_metrics['recall']:.4f}")
    print(f"  F1-Score:  {voting_weighted_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {voting_weighted_metrics['roc_auc']:.4f}")
    
    model_path = config.MODELS_DIR / "phase4" / "voting_weighted_ensemble_merged.joblib"
    joblib.dump(voting_weighted_clf, model_path)
    
    ensemble_results['Voting (Weighted)'] = voting_weighted_metrics
    
    # 4. Blending Ensemble
    blended_proba, blending_metrics = blending_ensemble(base_models, X_train, y_train, X_test, y_test)
    ensemble_results['Blending'] = blending_metrics
    
    # 5. Stacking Ensemble
    stacking_clf, stacking_metrics = stacking_ensemble(base_models, X_train, y_train, X_test, y_test)
    ensemble_results['Stacking'] = stacking_metrics
    
    # Compare all models
    results_df = compare_with_base_models(base_models, ensemble_results, X_test, y_test)
    
    # Save results
    config.ensure_dir(config.TABLES_DIR)
    output_file = config.TABLES_DIR / "phase4_ensemble_performance_merged.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("Phase 4 completed successfully!")
    print(f"{'='*80}")
    
    return results_df, ensemble_results

if __name__ == "__main__":
    results_df, ensemble_results = main()
