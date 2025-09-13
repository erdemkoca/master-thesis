#!/usr/bin/env python3
"""
Prepare Credit Card Default Dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_real_dataset import prepare_dataset, clean_credit_default

if __name__ == "__main__":
    # Configuration for Credit Card Default
    CFG = dict(
        name="credit_default",
        description="Credit Card Default Dataset (Taiwan)",
        raw_csv="../data/real/credit_default/raw/credit_default.csv",
        target="default.payment.next.month",
        positive_label=1,
        drop_cols=["ID"],  # ID column
        categorical=["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", 
                    "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
        numeric=["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", 
                "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", 
                "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
        separator=",",
        test_frac=0.50,
        random_state=42,
        out_dir="../data/real/credit_default/processed",
        cleaning_func=clean_credit_default  # Apply credit-specific cleaning
    )
    
    print("Credit Card Default Dataset Preparation")
    print("=" * 50)
    print("Note: Download the dataset from UCI ML Repository:")
    print("https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
    print("Save as: data/real/credit_default/raw/UCI_Credit_Card.csv")
    print()
    
    # Check if file exists
    if not os.path.exists(CFG["raw_csv"]):
        print(f"❌ Raw data file not found: {CFG['raw_csv']}")
        print("Please download the dataset and place it in the correct location.")
        sys.exit(1)
    
    prepare_dataset(CFG)
