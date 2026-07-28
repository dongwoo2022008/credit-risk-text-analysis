"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def load_raw_data():
    """
    Load raw data from Excel file
    
    Returns:
        pd.DataFrame: Raw data
    """
    print(f"Loading raw data from {config.RAW_DATA_FILE}")
    df = pd.read_excel(config.RAW_DATA_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def encode_target(df):
    """
    Encode target variable: 채무불이행 = 1, others = 0
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with encoded target
    """
    df = df.copy()
    df['target'] = (df[config.TARGET_COLUMN] == config.TARGET_DEFAULT_VALUE).astype(int)
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    return df

def prepare_structured_features(df):
    """
    Prepare structured features for modeling
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with processed structured features
    """
    df = df.copy()
    
    # Select structured features (13 variables, as in the paper)
    feature_cols = config.STRUCTURED_FEATURES.copy()
    
    # Handle missing values
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    return df, feature_cols

def split_data(df, save_indices=True):
    """
    Split data into train and test sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_indices (bool): Whether to save train/test indices
    
    Returns:
        tuple: (train_df, test_df, train_idx, test_idx)
    """
    # Check if split indices already exist
    train_idx_file = config.SPLITS_DIR / "train_indices.npy"
    test_idx_file = config.SPLITS_DIR / "test_indices.npy"
    
    if train_idx_file.exists() and test_idx_file.exists():
        print("Loading existing train/test split...")
        train_idx = np.load(train_idx_file)
        test_idx = np.load(test_idx_file)
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
    else:
        print("Creating new train/test split...")
        train_df, test_df = train_test_split(
            df,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=df['target']
        )
        
        train_idx = train_df.index.values
        test_idx = test_df.index.values
        
        if save_indices:
            config.ensure_dir(config.SPLITS_DIR)
            np.save(train_idx_file, train_idx)
            np.save(test_idx_file, test_idx)
            print(f"Saved train/test indices to {config.SPLITS_DIR}")
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print(f"Train target distribution: {train_df['target'].value_counts().to_dict()}")
    print(f"Test target distribution: {test_df['target'].value_counts().to_dict()}")
    
    return train_df, test_df, train_idx, test_idx

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Args:
        X_train (np.ndarray or pd.DataFrame): Training features
        X_test (np.ndarray or pd.DataFrame): Test features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def get_credit_score_groups(df):
    """
    Divide data into credit score risk groups
    - High risk: Bottom 30% (lowest credit scores)
    - Low risk: Top 30% (highest credit scores)
    
    Args:
        df (pd.DataFrame): Input dataframe with credit scores
    
    Returns:
        tuple: (high_risk_df, low_risk_df)
    """
    credit_score_col = '신용평점'
    
    # Calculate percentiles
    low_threshold = df[credit_score_col].quantile(config.HIGH_RISK_PERCENTILE / 100)
    high_threshold = df[credit_score_col].quantile(config.LOW_RISK_PERCENTILE / 100)
    
    # High risk: bottom 30%
    high_risk_df = df[df[credit_score_col] <= low_threshold].copy()
    
    # Low risk: top 30%
    low_risk_df = df[df[credit_score_col] >= high_threshold].copy()
    
    print(f"High risk group (bottom 30%): {len(high_risk_df)} rows")
    print(f"  Credit score <= {low_threshold:.2f}")
    print(f"  Default rate: {high_risk_df['target'].mean():.4f}")
    
    print(f"Low risk group (top 30%): {len(low_risk_df)} rows")
    print(f"  Credit score >= {high_threshold:.2f}")
    print(f"  Default rate: {low_risk_df['target'].mean():.4f}")
    
    return high_risk_df, low_risk_df

def get_text_length_groups(df):
    """
    Divide data into text length deciles
    
    Args:
        df (pd.DataFrame): Input dataframe with text column
    
    Returns:
        dict: Dictionary mapping decile to dataframe
    """
    df = df.copy()
    # Concatenate all text columns
    text_combined = df[config.TEXT_COLUMNS].fillna('').astype(str).agg(' '.join, axis=1)
    df['text_length'] = text_combined.str.len()
    
    # Calculate deciles
    df['text_length_decile'] = pd.qcut(
        df['text_length'],
        q=config.TEXT_LENGTH_DECILES,
        labels=False,
        duplicates='drop'
    ) + 1  # 1-indexed
    
    groups = {}
    for decile in range(1, config.TEXT_LENGTH_DECILES + 1):
        groups[decile] = df[df['text_length_decile'] == decile].copy()
        print(f"Decile {decile}: {len(groups[decile])} rows, "
              f"length range: {groups[decile]['text_length'].min()}-{groups[decile]['text_length'].max()}")
    
    return groups

def load_and_prepare_data():
    """
    Main function to load and prepare all data
    
    Returns:
        dict: Dictionary containing all prepared data
    """
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    # Load raw data
    df = load_raw_data()
    
    # Encode target
    df = encode_target(df)
    
    # Prepare structured features
    df, feature_cols = prepare_structured_features(df)
    
    # Split data
    train_df, test_df, train_idx, test_idx = split_data(df)
    
    # Prepare feature matrices
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df['target'].values
    y_test = test_df['target'].values
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    
    return {
        'df': df,
        'train_df': train_df,
        'test_df': test_df,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'feature_cols': feature_cols,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }

if __name__ == "__main__":
    # Test data loading
    data = load_and_prepare_data()
    print(f"\nData keys: {list(data.keys())}")
    print(f"Feature columns: {data['feature_cols']}")
