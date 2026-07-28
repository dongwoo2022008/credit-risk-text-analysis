"""
Model evaluation utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Predicted probabilities (optional)
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC (requires probabilities)
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
    
    return metrics

def evaluate_multiple_models(models, X_test, y_test):
    """
    Evaluate multiple models
    
    Args:
        models (dict): Dictionary of trained models
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
    
    Returns:
        pd.DataFrame: Evaluation results for all models
    """
    results = []
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities (if available)
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        metrics['model'] = model_name
        
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    cols += [c for c in results_df.columns if c not in cols]
    results_df = results_df[cols]
    
    return results_df

def compare_models(results_df, baseline_model=None, sort_by='roc_auc'):
    """
    Compare models and calculate improvements
    
    Args:
        results_df (pd.DataFrame): Results from evaluate_multiple_models
        baseline_model (str): Name of baseline model for comparison
        sort_by (str): Metric to sort by
    
    Returns:
        pd.DataFrame: Comparison results
    """
    df = results_df.copy()
    
    # Sort by metric
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    
    # Calculate improvement over baseline
    if baseline_model and baseline_model in df['model'].values:
        baseline_row = df[df['model'] == baseline_model].iloc[0]
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric in df.columns:
                baseline_value = baseline_row[metric]
                df[f'{metric}_improvement'] = ((df[metric] - baseline_value) / baseline_value * 100).round(2)
    
    return df

def calculate_threshold_metrics(y_true, y_pred_proba, thresholds=None):
    """
    Calculate metrics at different probability thresholds
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        thresholds (list): List of thresholds to evaluate
    
    Returns:
        pd.DataFrame: Metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = evaluate_model(y_true, y_pred, y_pred_proba)
        metrics['threshold'] = threshold
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['threshold', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    cols += [c for c in results_df.columns if c not in cols]
    results_df = results_df[cols]
    
    return results_df

def calculate_uncertainty_metrics(y_true, y_pred_proba, uncertainty_threshold=0.1):
    """
    Analyze model performance in uncertain vs certain predictions
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        uncertainty_threshold (float): Distance from 0.5 to define uncertainty
    
    Returns:
        dict: Metrics for uncertain and certain predictions
    """
    # Define uncertain predictions (close to 0.5)
    uncertain_mask = np.abs(y_pred_proba - 0.5) <= uncertainty_threshold
    certain_mask = ~uncertain_mask
    
    results = {}
    
    # Uncertain predictions
    if uncertain_mask.sum() > 0:
        y_true_uncertain = y_true[uncertain_mask]
        y_pred_uncertain = (y_pred_proba[uncertain_mask] >= 0.5).astype(int)
        y_pred_proba_uncertain = y_pred_proba[uncertain_mask]
        
        results['uncertain'] = evaluate_model(
            y_true_uncertain,
            y_pred_uncertain,
            y_pred_proba_uncertain
        )
        results['uncertain']['count'] = uncertain_mask.sum()
        results['uncertain']['percentage'] = (uncertain_mask.sum() / len(y_true) * 100)
    
    # Certain predictions
    if certain_mask.sum() > 0:
        y_true_certain = y_true[certain_mask]
        y_pred_certain = (y_pred_proba[certain_mask] >= 0.5).astype(int)
        y_pred_proba_certain = y_pred_proba[certain_mask]
        
        results['certain'] = evaluate_model(
            y_true_certain,
            y_pred_certain,
            y_pred_proba_certain
        )
        results['certain']['count'] = certain_mask.sum()
        results['certain']['percentage'] = (certain_mask.sum() / len(y_true) * 100)
    
    return results

def analyze_error_cases(y_true, y_pred, y_pred_proba, X_test=None, feature_names=None):
    """
    Analyze false positive and false negative cases
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Predicted probabilities
        X_test (np.ndarray): Test features (optional)
        feature_names (list): Feature names (optional)
    
    Returns:
        dict: Analysis of error cases
    """
    # Identify error cases
    fp_mask = (y_true == 0) & (y_pred == 1)  # False positives
    fn_mask = (y_true == 1) & (y_pred == 0)  # False negatives
    
    results = {
        'false_positive': {
            'count': fp_mask.sum(),
            'percentage': (fp_mask.sum() / len(y_true) * 100),
            'avg_probability': y_pred_proba[fp_mask].mean() if fp_mask.sum() > 0 else np.nan
        },
        'false_negative': {
            'count': fn_mask.sum(),
            'percentage': (fn_mask.sum() / len(y_true) * 100),
            'avg_probability': y_pred_proba[fn_mask].mean() if fn_mask.sum() > 0 else np.nan
        }
    }
    
    return results

def print_evaluation_summary(metrics, model_name="Model"):
    """
    Print formatted evaluation summary
    
    Args:
        metrics (dict): Evaluation metrics
        model_name (str): Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if 'true_positive' in metrics:
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negative']:>5}  FP: {metrics['false_positive']:>5}")
        print(f"  FN: {metrics['false_negative']:>5}  TP: {metrics['true_positive']:>5}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test evaluation functions
    np.random.seed(config.RANDOM_SEED)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    
    metrics = evaluate_model(y_true, y_pred, y_pred_proba)
    print_evaluation_summary(metrics, "Test Model")
