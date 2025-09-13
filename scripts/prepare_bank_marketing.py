#!/usr/bin/env python3
"""
Prepare Bank Marketing Dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_real_dataset import prepare_dataset
import pandas as pd

def load_bank_marketing_data(cfg):
    """Load and fix bank marketing data."""
    # Load with proper handling of the split header
    df = pd.read_csv(cfg["raw_csv"], sep=",", skiprows=1)
    # Set proper column names
    df.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Standardize target to 0/1
    y = (df[cfg["target"]] == cfg["positive_label"]).astype(int).values
    X = df.drop(columns=[cfg["target"]]).copy()
    
    # Drop known leakage / id columns
    if cfg.get("drop_cols"):
        X = X.drop(columns=[c for c in cfg["drop_cols"] if c in X.columns], errors="ignore")
        print(f"Dropped columns: {cfg['drop_cols']}")
    
    return X, y

if __name__ == "__main__":
    # Configuration for Bank Marketing
    CFG = dict(
        name="bank_marketing",
        description="Bank Marketing Dataset (UCI)",
        raw_csv="../data/real/bank_marketing/raw/bank_marketing.csv",
        target="y",
        positive_label="yes",
        drop_cols=["duration"],  # leakage in Bank Marketing
        categorical=["job", "marital", "education", "default", "housing", 
                    "loan", "contact", "month", "poutcome"],
        numeric=["age", "balance", "day", "campaign", "pdays", "previous"],
        separator=";",
        test_frac=0.50,
        random_state=42,
        out_dir="../data/real/bank_marketing/processed",
        custom_load_func=load_bank_marketing_data
    )
    
    print("Bank Marketing Dataset Preparation")
    print("=" * 50)
    print("Note: Download the dataset from UCI ML Repository:")
    print("https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
    print("Save as: data/real/bank_marketing/raw/bank-full.csv")
    print("Note: Duration column will be dropped (leakage)")
    print()
    
    # Check if file exists
    if not os.path.exists(CFG["raw_csv"]):
        print(f"❌ Raw data file not found: {CFG['raw_csv']}")
        print("Please download the dataset and place it in the correct location.")
        sys.exit(1)
    
    prepare_dataset(CFG)
