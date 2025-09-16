#!/usr/bin/env python3
"""
Prepare Adult Income Dataset
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_real_dataset import prepare_dataset

def clean_adult_income(X):
    """Clean adult income dataset."""
    print("Applying Adult Income specific cleaning...")
    
    # Replace '?' with NaN for missing values
    X = X.replace('?', np.nan)
    
    # Remove leading/trailing whitespace from string columns
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].str.strip()
    
    return X

if __name__ == "__main__":
    # Configuration for Adult Income
    CFG = dict(
        name="adult_income",
        description="Adult Income Dataset (UCI)",
        raw_csv="../data/real/adult_income/raw/adult_income.csv",
        target="income",
        positive_label=">50K",
        drop_cols=["fnlwgt"],  # Drop fnlwgt as it's a sampling weight
        categorical=["workclass", "education", "marital.status", "occupation", 
                    "relationship", "race", "sex", "native.country"],
        numeric=["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
        separator=",",
        test_frac=0.50,
        random_state=42,
        out_dir="../data/real/adult_income/processed",
        cleaning_func=clean_adult_income
    )
    
    print("Adult Income Dataset Preparation")
    print("=" * 50)
    print("Note: The dataset should be downloaded from UCI ML Repository:")
    print("https://archive.ics.uci.edu/ml/datasets/adult")
    print("Save as: data/real/adult_income/raw/adult_income.csv")
    print()
    
    # Check if file exists
    if not os.path.exists(CFG["raw_csv"]):
        print(f"❌ Raw data file not found: {CFG['raw_csv']}")
        print("Please download the dataset and place it in the correct location.")
        sys.exit(1)
    
    prepare_dataset(CFG)
