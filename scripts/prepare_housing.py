#!/usr/bin/env python3
"""
California Housing Dataset Preprocessing
Converts the housing price regression dataset to binary classification format
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def download_housing():
    """Load the housing dataset from the raw CSV file."""
    raw_path = Path("data/real/raw/housing/housing.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Housing dataset not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    print(f"Loaded housing dataset: {df.shape}")
    return df

def clean_housing_data(X):
    """Clean the housing dataset - handle missing values in total_bedrooms."""
    # Convert to float to ensure consistency
    X = X.astype(float)
    
    # Handle missing values in total_bedrooms (207 missing values)
    if X.isnull().sum().sum() > 0:
        print(f"Found {X.isnull().sum().sum()} missing values, imputing with median")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    return X

def create_housing_binary_target(df):
    """Create binary target from median house value (expensive vs cheap housing)."""
    # Use median house value as the target variable
    # Higher value = expensive housing = positive class (1)
    # Lower value = cheap housing = negative class (0)
    
    house_value = df['median_house_value'].copy()
    median_value = house_value.median()
    
    # Binary classification: 1 if expensive (above median), 0 if cheap
    y_binary = (house_value >= median_value).astype(int)
    
    print(f"Housing value classification:")
    print(f"  Median value: ${median_value:,.0f}")
    print(f"  Expensive (≥${median_value:,.0f}): {y_binary.sum()} samples ({y_binary.mean()*100:.1f}%)")
    print(f"  Cheap (<${median_value:,.0f}): {(1-y_binary).sum()} samples ({(1-y_binary).mean()*100:.1f}%)")
    
    return y_binary

# Configuration
CFG = {
    "name": "housing",
    "description": "California Housing (binary classification)",
    "raw_csv": "data/real/raw/housing/housing.csv",
    "target": "median_house_value",
    "positive_label": "Expensive housing (≥median value)",
    "drop_cols": ["ocean_proximity"],  # Drop categorical for now, could encode later
    "categorical": [],  # No categorical variables (dropped ocean_proximity)
    "numeric": [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income"
    ],
    "test_frac": 0.2,
    "random_state": 42,
    "out_dir": "data/real/housing/processed",
    "cleaning_func": clean_housing_data,
    "custom_load_func": download_housing,
    "custom_target_func": create_housing_binary_target
}

def prepare_dataset(cfg):
    """Prepare the housing dataset for binary classification."""
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
        "target_name": "housing_value",
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
