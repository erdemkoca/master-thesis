#!/usr/bin/env python3
"""
Generic Real Dataset Preparation Script
Prepares real datasets for the unified experiment pipeline
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def load_df(cfg):
    """Load and preprocess raw dataset based on configuration."""
    print(f"Loading {cfg['name']} from {cfg['raw_csv']}")
    
    # Use custom load function if provided
    if "custom_load_func" in cfg and cfg["custom_load_func"]:
        X, y = cfg["custom_load_func"](cfg)
        print(f"Final data shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        return X, y
    
    # Load CSV with appropriate separator
    sep = cfg.get("separator", ",")
    df = pd.read_csv(cfg["raw_csv"], sep=sep)
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Standardize target to 0/1
    y = (df[cfg["target"]] == cfg["positive_label"]).astype(int).values
    X = df.drop(columns=[cfg["target"]]).copy()
    
    # Drop known leakage / id columns
    if cfg.get("drop_cols"):
        X = X.drop(columns=[c for c in cfg["drop_cols"] if c in X.columns], errors="ignore")
        print(f"Dropped columns: {cfg['drop_cols']}")
    
    # Apply dataset-specific cleaning
    if "cleaning_func" in cfg and cfg["cleaning_func"]:
        X = cfg["cleaning_func"](X)
        print("Applied dataset-specific cleaning")
    
    print(f"Final data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    return X, y

def make_fixed_split(y, test_frac, random_state):
    """Create fixed stratified test/pool split."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    idx = np.arange(len(y))
    test_idx, pool_idx = next(sss.split(idx, y))
    return test_idx, pool_idx

def build_transformer(categorical, numeric):
    """Build preprocessing transformer for mixed data types."""
    ct = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat",  # impute + OHE
             Pipeline([
                 ("imp", SimpleImputer(strategy="most_frequent")),
                 ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
             ]),
             categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return ct

def prepare_dataset(cfg):
    """Prepare a single dataset based on configuration."""
    print(f"\n{'='*60}")
    print(f"PREPARING DATASET: {cfg['name'].upper()}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(cfg["out_dir"], exist_ok=True)
    
    # Load data
    X_df, y = load_df(cfg)
    
    # Create fixed big test split
    print(f"\nCreating fixed test/pool split (test_frac={cfg['test_frac']})")
    idx_test, idx_pool = make_fixed_split(y, cfg["test_frac"], cfg["random_state"])
    
    print(f"Test set: {len(idx_test)} samples")
    print(f"Pool: {len(idx_pool)} samples")
    
    # Fit transformer on POOL ONLY (avoid test leakage)
    print(f"\nFitting preprocessing transformer on pool only...")
    ct = build_transformer(cfg["categorical"], cfg["numeric"])
    ct.fit(X_df.iloc[idx_pool])
    
    # Transform full set with the pool-fitted transformer
    print("Transforming full dataset...")
    X_full = ct.transform(X_df)
    
    # Build feature names
    try:
        feat_names = ct.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"feature_{i}" for i in range(X_full.shape[1])]
    
    print(f"Final feature count: {X_full.shape[1]}")
    print(f"Feature names: {feat_names[:5]}..." if len(feat_names) > 5 else f"Feature names: {feat_names}")
    
    # Save outputs
    print(f"\nSaving processed data to {cfg['out_dir']}...")
    np.save(os.path.join(cfg["out_dir"], "X_full.npy"), X_full)
    np.save(os.path.join(cfg["out_dir"], "y_full.npy"), y.astype(int))
    np.save(os.path.join(cfg["out_dir"], "idx_test_big.npy"), idx_test)
    np.save(os.path.join(cfg["out_dir"], "idx_pool.npy"), idx_pool)
    
    with open(os.path.join(cfg["out_dir"], "feature_names.json"), "w") as f:
        json.dump(feat_names, f, indent=2)
    
    # Create metadata
    meta = {
        "dataset": cfg["name"],
        "description": cfg.get("description", f"Real dataset: {cfg['name']}"),
        "n_samples": int(X_full.shape[0]),
        "n_features": int(X_full.shape[1]),
        "class_dist_full": np.bincount(y).tolist(),
        "test_size": int(idx_test.size),
        "pool_size": int(idx_pool.size),
        "drop_cols": cfg.get("drop_cols", []),
        "target": cfg["target"],
        "positive_label": cfg["positive_label"],
        "categorical": cfg["categorical"],
        "numeric": cfg["numeric"],
        "separator": cfg.get("separator", ","),
        "note": "Transformer fitted on POOL only; no scaling here.",
    }
    
    with open(os.path.join(cfg["out_dir"], "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ Dataset {cfg['name']} prepared successfully!")
    print(f"   Samples: {X_full.shape[0]}, Features: {X_full.shape[1]}")
    print(f"   Test: {len(idx_test)}, Pool: {len(idx_pool)}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    return True

# Dataset-specific cleaning functions
def clean_pima(X):
    """Clean Pima Indians Diabetes dataset."""
    # Replace zeros in specific columns with NaN (they represent missing values)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
    return X

def clean_credit_default(X):
    """Clean Credit Card Default dataset."""
    # Cast categorical columns to string to ensure proper encoding
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
    return X

def clean_bank_marketing(df):
    """Clean Bank Marketing dataset - fix header issue."""
    # The header is split across two lines, fix it
    if len(df.columns) == 1 and 'age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome' in str(df.columns[0]):
        # Re-read with proper header handling
        df = pd.read_csv(df.name, sep=";", skiprows=1)
        # Set proper column names
        df.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    return df

if __name__ == "__main__":
    # Example usage - you can run this with different configurations
    print("Real Dataset Preparation Script")
    print("=" * 60)
    print("This script prepares real datasets for the unified experiment pipeline.")
    print("Configure the CFG dictionary below and run to prepare a dataset.")
    print("\nExample configurations are provided in the comments below.")
    
    # Example configuration (uncomment and modify as needed)
    """
    CFG = dict(
        name="bank_marketing",
        description="Bank Marketing Dataset (UCI)",
        raw_csv="../data/real/bank_marketing/raw/bank-full.csv",
        target="y",
        positive_label="yes",
        drop_cols=["duration"],  # leakage in Bank Marketing
        categorical=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
        numeric=["age", "balance", "day", "campaign", "pdays", "previous"],
        separator=";",
        test_frac=0.50,
        random_state=42,
        out_dir="../data/real/bank_marketing/processed"
    )
    
    prepare_dataset(CFG)
    """
