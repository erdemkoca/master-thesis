#!/usr/bin/env python3
"""
Boston Housing Dataset Preprocessing
Converts the Boston housing regression dataset to binary classification format
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

def download_boston():
    """Load the Boston dataset from the raw CSV file."""
    raw_path = Path("data/real/boston/raw/boston.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Boston dataset not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    print(f"Loaded Boston dataset: {df.shape}")
    return df

def clean_boston_data(X):
    """Clean the Boston dataset - handle missing values and data types."""
    # Convert 'NA' strings to actual NaN
    X = X.replace('NA', np.nan)
    
    # Convert numeric columns to proper types
    numeric_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("Warning: Found missing values, imputing with median")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Convert all to float for consistency
    X = X.astype(float)
    
    print(f"Missing values after cleaning: {X.isnull().sum().sum()}")
    return X

def create_boston_binary_target(df):
    """Create binary target from Boston housing prices (high vs low value)."""
    # Use median house value as threshold
    # High value = positive class (1), Low value = negative class (0)
    
    medv_values = pd.to_numeric(df['MEDV'], errors='coerce')
    median_value = medv_values.median()
    y_binary = (medv_values >= median_value).astype(int)
    
    print(f"Boston housing classification:")
    print(f"  Median MEDV: {median_value:.2f}")
    print(f"  High value (≥{median_value:.2f}): {y_binary.sum()} samples ({y_binary.mean()*100:.1f}%)")
    print(f"  Low value (<{median_value:.2f}): {(1-y_binary).sum()} samples ({(1-y_binary).mean()*100:.1f}%)")
    
    return y_binary

# Configuration
CFG = {
    "name": "boston",
    "description": "Boston Housing (binary classification)",
    "raw_csv": "data/real/boston/raw/boston.csv",
    "target": "MEDV",
    "positive_label": "High value (≥median)",
    "drop_cols": [],  # No columns to drop
    "categorical": [],  # No categorical variables
    "numeric": ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
    "test_frac": 0.2,
    "random_state": 42,
    "out_dir": "data/real/boston/processed",
    "cleaning_func": clean_boston_data,
    "custom_load_func": download_boston,
    "custom_target_func": create_boston_binary_target
}

def prepare_dataset(cfg):
    """Prepare the Boston dataset for binary classification."""
    print(f"\n=== Preparing {cfg['name']} dataset ===")
    
    # Load data
    df = cfg["custom_load_func"]()
    
    # Create binary target
    y_binary = cfg["custom_target_func"](df)
    
    # Prepare features
    feature_cols = cfg["numeric"] + cfg["categorical"]
    X = df[feature_cols].copy()
    
    # Clean data
    X = cfg["cleaning_func"](X)
    
    print(f"Features: {X.shape[1]} ({', '.join(X.columns)})")
    print(f"Target distribution: {np.bincount(y_binary)}")
    
    # Create stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=cfg["test_frac"], 
        random_state=cfg["random_state"]
    )
    
    train_idx, test_idx = next(splitter.split(X, y_binary))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Create output directory
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    np.save(out_dir / "X_full.npy", X.values)
    np.save(out_dir / "y_full.npy", y_binary.values)
    np.save(out_dir / "X_train.npy", X_train.values)
    np.save(out_dir / "X_test.npy", X_test.values)
    np.save(out_dir / "y_train.npy", y_train.values)
    np.save(out_dir / "y_test.npy", y_test.values)
    
    # Save feature names and metadata
    feature_names = X.columns.tolist()
    metadata = {
        "n_samples": len(X),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_name": "MEDV",
        "positive_class": cfg["positive_label"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_frac": cfg["test_frac"],
        "random_state": cfg["random_state"]
    }
    
    import json
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved processed data to {out_dir}")
    print(f"Metadata: {metadata}")
    
    return X, y_binary, metadata

if __name__ == "__main__":
    prepare_dataset(CFG)