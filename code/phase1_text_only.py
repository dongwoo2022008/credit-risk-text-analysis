"""
Phase 1: Text-only Models
Evaluates text-only models across 4 text representation stages:
- Stage 1: TF-IDF (sparse)
- Stage 2: Subword-based features (sparse)
- Stage 3: MiniLM sentence embeddings (dense)
- Stage 4: KoSimCSE sentence embeddings (dense)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
import re

import config
from utils.data_loader import load_raw_data, encode_target, split_data
from utils.evaluator import evaluate_model

def preprocess_text(text):
    """
    Simple text preprocessing for Korean text
    """
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove special characters but keep Korean, numbers, and spaces
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def prepare_text_data(df):
    """
    Prepare text data by concatenating all text columns
    """
    df = df.copy()
    
    # Concatenate all text columns
    text_cols = config.TEXT_COLUMNS
    df['combined_text'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    
    return df

def extract_tfidf_features(train_df, test_df, max_features=100):
    """
    Stage 1: Extract TF-IDF features
    """
    print(f"\n{'='*80}")
    print("Stage 1: TF-IDF Feature Extraction")
    print(f"{'='*80}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    X_train = vectorizer.fit_transform(train_df['combined_text']).toarray()
    X_test = vectorizer.transform(test_df['combined_text']).toarray()
    
    print(f"TF-IDF features: {X_train.shape[1]}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, vectorizer

def extract_subword_features(train_df, test_df, max_features=100):
    """
    Stage 2: Extract subword-based features (character n-grams)
    """
    print(f"\n{'='*80}")
    print("Stage 2: Subword-based Feature Extraction")
    print(f"{'='*80}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        analyzer='char',
        ngram_range=(2, 4)
    )
    
    X_train = vectorizer.fit_transform(train_df['combined_text']).toarray()
    X_test = vectorizer.transform(test_df['combined_text']).toarray()
    
    print(f"Subword features: {X_train.shape[1]}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, vectorizer

def extract_minilm_features(train_df, test_df):
    """
    Stage 3: MiniLM sentence embeddings (dense vectors).
    Uses sentence-transformers/all-MiniLM-L6-v2 as in the paper.
    Falls back to a TF-IDF proxy with a warning if sentence-transformers
    is not installed (paper results are based on the real embeddings).
    """
    print()
    print('='*80)
    print("Stage 3: MiniLM Sentence Embeddings")
    print(f"{'='*80}")
    try:
        X_train = _encode_with_sentence_transformer(train_df['combined_text'], 'sentence-transformers/all-MiniLM-L6-v2')
        X_test = _encode_with_sentence_transformer(test_df['combined_text'], 'sentence-transformers/all-MiniLM-L6-v2')
        print(f"MiniLM features: {X_train.shape[1]}")
        return X_train, X_test, None
    except ImportError:
        print("WARNING: sentence-transformers not installed; using a TF-IDF proxy. Install sentence-transformers to reproduce the paper's Stage 3.")
    vectorizer = TfidfVectorizer(max_features=384, min_df=2, max_df=0.95, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['combined_text']).toarray()
    X_test = vectorizer.transform(test_df['combined_text']).toarray()
    return X_train, X_test, vectorizer

def extract_kosimcse_features(train_df, test_df):
    """
    Stage 4: KoSimCSE sentence embeddings (dense vectors).
    Uses BM-K/KoSimCSE-roberta as in the paper.
    Falls back to a TF-IDF proxy with a warning if sentence-transformers
    is not installed (paper results are based on the real embeddings).
    """
    print()
    print('='*80)
    print("Stage 4: KoSimCSE Sentence Embeddings")
    print(f"{'='*80}")
    try:
        X_train = _encode_with_sentence_transformer(train_df['combined_text'], 'BM-K/KoSimCSE-roberta')
        X_test = _encode_with_sentence_transformer(test_df['combined_text'], 'BM-K/KoSimCSE-roberta')
        print(f"KoSimCSE features: {X_train.shape[1]}")
        return X_train, X_test, None
    except ImportError:
        print("WARNING: sentence-transformers not installed; using a TF-IDF proxy. Install sentence-transformers to reproduce the paper's Stage 4.")
    vectorizer = TfidfVectorizer(max_features=768, min_df=2, max_df=0.95, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['combined_text']).toarray()
    X_test = vectorizer.transform(test_df['combined_text']).toarray()
    return X_train, X_test, vectorizer



def _encode_with_sentence_transformer(texts, model_name):
    """Encode texts with a sentence-transformers model; raise ImportError if unavailable."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), show_progress_bar=False)

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
            random_state=config.RANDOM_SEED
        ),
        'GB': GradientBoostingClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED
        ),
        'XGB': XGBClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss'
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
    print("Phase 1: Text-only Models")
    print(f"{'='*80}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Test size: {config.TEST_SIZE}")
    
    # Load data
    print("\nLoading data...")
    df = load_raw_data()
    df = encode_target(df)
    df = prepare_text_data(df)
    
    # Split data (use existing split if available)
    train_df, test_df, train_idx, test_idx = split_data(df, save_indices=True)
    
    y_train = train_df['target'].values
    y_test = test_df['target'].values
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Train default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")
    
    # Dictionary to store all results
    all_results = {}
    all_models = {}
    
    # Stage 1: TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer_tfidf = extract_tfidf_features(train_df, test_df)
    results_tfidf, models_tfidf = train_and_evaluate_stage(
        X_train_tfidf, X_test_tfidf, y_train, y_test, "Stage 1: TF-IDF"
    )
    all_results['stage1_tfidf'] = results_tfidf
    all_models['stage1_tfidf'] = models_tfidf
    
    # Stage 2: Subword
    X_train_subword, X_test_subword, vectorizer_subword = extract_subword_features(train_df, test_df)
    results_subword, models_subword = train_and_evaluate_stage(
        X_train_subword, X_test_subword, y_train, y_test, "Stage 2: Subword"
    )
    all_results['stage2_subword'] = results_subword
    all_models['stage2_subword'] = models_subword
    
    # Stage 3: MiniLM
    X_train_minilm, X_test_minilm, vectorizer_minilm = extract_minilm_features(train_df, test_df)
    results_minilm, models_minilm = train_and_evaluate_stage(
        X_train_minilm, X_test_minilm, y_train, y_test, "Stage 3: MiniLM"
    )
    all_results['stage3_minilm'] = results_minilm
    all_models['stage3_minilm'] = models_minilm
    
    # Stage 4: KoSimCSE
    X_train_kosimcse, X_test_kosimcse, vectorizer_kosimcse = extract_kosimcse_features(train_df, test_df)
    results_kosimcse, models_kosimcse = train_and_evaluate_stage(
        X_train_kosimcse, X_test_kosimcse, y_train, y_test, "Stage 4: KoSimCSE"
    )
    all_results['stage4_kosimcse'] = results_kosimcse
    all_models['stage4_kosimcse'] = models_kosimcse
    
    # Print summary
    print(f"\n{'='*80}")
    print("Phase 1 Summary: Text-only Models Across 4 Stages")
    print(f"{'='*80}")
    
    for stage_name, results_df in all_results.items():
        print(f"\n{stage_name.upper()}:")
        print(results_df.to_string(index=False))
        best_model = results_df.loc[results_df['roc_auc'].idxmax()]
        print(f"Best model: {best_model['model']} (ROC-AUC: {best_model['roc_auc']:.4f})")
    
    # Save results
    config.ensure_dir(config.TABLES_DIR)
    config.ensure_dir(config.MODELS_DIR / "phase1")
    
    for stage_name, results_df in all_results.items():
        output_file = config.TABLES_DIR / f"phase1_{stage_name}_performance.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_file}")
    
    # Save models
    for stage_name, models_dict in all_models.items():
        for model_name, model in models_dict.items():
            model_file = config.MODELS_DIR / "phase1" / f"{stage_name}_{model_name.lower()}_model.joblib"
            joblib.dump(model, model_file)
    
    print(f"\n{'='*80}")
    print("Phase 1 completed successfully!")
    print(f"{'='*80}")
    
    return all_results, all_models

if __name__ == "__main__":
    results, models = main()
