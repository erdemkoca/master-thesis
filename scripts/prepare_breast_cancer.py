#!/usr/bin/env python3
"""
Prepare Breast Cancer Wisconsin Dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_real_dataset import prepare_dataset
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

def download_breast_cancer():
    """Download breast cancer dataset from sklearn."""
    print("Downloading Breast Cancer dataset from sklearn...")
    data = load_breast_cancer(as_frame=True)
    X_df = data.data
    y_series = data.target
    
    # Convert target: 0=malignant, 1=benign -> 0=benign, 1=malignant
    y = (y_series == 0).astype(int)  # malignant=1, benign=0
    
    # Save to raw directory
    raw_dir = "../data/real/breast_cancer/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Combine X and y for CSV
    df = X_df.copy()
    df['target'] = y
    df.to_csv(os.path.join(raw_dir, "breast_cancer.csv"), index=False)
    
    print(f"Saved to {raw_dir}/breast_cancer.csv")
    return os.path.join(raw_dir, "breast_cancer.csv")

if __name__ == "__main__":
    # Download dataset
    csv_path = download_breast_cancer()
    
    # Configuration for Breast Cancer
    CFG = dict(
        name="breast_cancer",
        description="Breast Cancer Wisconsin Dataset (sklearn)",
        raw_csv=csv_path,
        target="target",
        positive_label=1,  # malignant
        drop_cols=[],  # No leakage columns
        categorical=[],  # All numeric
        numeric=list(range(30)),  # All 30 features are numeric
        separator=",",
        test_frac=0.50,
        random_state=42,
        out_dir="../data/real/breast_cancer/processed"
    )
    
    prepare_dataset(CFG)
