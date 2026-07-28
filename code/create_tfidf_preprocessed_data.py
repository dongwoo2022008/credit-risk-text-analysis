"""
Create TF-IDF preprocessed data for Phase 3 and Phase 4
Loads original data and creates TF-IDF features merged with structured variables
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config

def main():
    print("="*80)
    print("Creating TF-IDF Preprocessed Data")
    print("="*80)
    
    # Load original data
    print("\nLoading original data...")
    data_path = config.DATA_DIR / "raw" / "sentiment_scoring.25.12.30.xlsx"
    df = pd.read_excel(data_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare structured variables (13 variables)
    structured_vars = [
        '성공횟수', '신용평점', '신청금액(만원)', '투자인원', '신청금리',
        '총횟수', '성공률', '대출용도', '4대보험', '근무개월',
        '대출시기', '나이', '대출(은행보험)'
    ]
    
    print(f"\nUsing {len(structured_vars)} structured variables")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    df_encoded = df[structured_vars].copy()
    
    categorical_cols = ['대출용도', '4대보험']
    for col in categorical_cols:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            print(f"  Encoded {col}: {len(le.classes_)} classes")
    
    X_structured = df_encoded.values
    # Create binary target (1=Default/채무불이행, 0=Repayment/기타)
    y = (df['상환결과'] == '채무불이행').astype(int).values
    texts = df['신청목적'].fillna('').values
    
    print(f"\nTarget distribution:")
    print(f"  Default (1): {y.sum()}")
    print(f"  Repayment (0): {len(y) - y.sum()}")
    print(f"  Default rate: {y.mean():.2%}")
    
    # Train-test split
    print("\nSplitting data (80/20)...")
    X_struct_train, X_struct_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X_structured, y, texts, 
        test_size=0.2, 
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    print(f"Train samples: {len(X_struct_train)}")
    print(f"Test samples: {len(X_struct_test)}")
    print(f"Train default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")
    
    # Create TF-IDF features
    print("\nCreating TF-IDF features (max 100 features)...")
    tfidf = TfidfVectorizer(
        max_features=100,
        min_df=2,
        ngram_range=(1, 2)
    )
    
    X_tfidf_train = tfidf.fit_transform(texts_train).toarray()
    X_tfidf_test = tfidf.transform(texts_test).toarray()
    
    print(f"TF-IDF features: {X_tfidf_train.shape[1]}")
    
    # Merge structured + TF-IDF
    print("\nMerging structured + TF-IDF features...")
    X_train = np.hstack([X_struct_train, X_tfidf_train])
    X_test = np.hstack([X_struct_test, X_tfidf_test])
    
    print(f"Final train shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")
    
    # Save preprocessed data
    output_path = config.DATA_DIR / "preprocessed" / "preprocessed_merged_struct_tfidf_binary.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    preprocessed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': structured_vars + [f'tfidf_{i}' for i in range(X_tfidf_train.shape[1])],
        'tfidf_vectorizer': tfidf
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"\nPreprocessed data saved to: {output_path}")
    print("="*80)
    print("TF-IDF preprocessed data created successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
