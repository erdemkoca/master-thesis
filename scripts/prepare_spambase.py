#!/usr/bin/env python3
"""
Prepare Spambase Dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_real_dataset import prepare_dataset

if __name__ == "__main__":
    # Configuration for Spambase
    CFG = dict(
        name="spambase",
        description="Spambase Dataset (UCI)",
        raw_csv="../data/real/spambase/raw/spambase.data",
        target="class",
        positive_label=1,
        drop_cols=[],  # No leakage columns
        categorical=[],  # All numeric
        numeric=[f"word_freq_{i}" for i in range(48)] + 
                [f"char_freq_{i}" for i in range(6)] + 
                ["capital_run_length_average", "capital_run_length_longest", 
                 "capital_run_length_total"],
        separator=",",
        test_frac=0.50,
        random_state=42,
        out_dir="../data/real/spambase/processed"
    )
    
    print("Spambase Dataset Preparation")
    print("=" * 50)
    print("Note: Download the dataset from UCI ML Repository:")
    print("https://archive.ics.uci.edu/ml/datasets/Spambase")
    print("Save as: data/real/spambase/raw/spambase.data")
    print("Note: This dataset has no header row")
    print()
    
    # Check if file exists
    if not os.path.exists(CFG["raw_csv"]):
        print(f"❌ Raw data file not found: {CFG['raw_csv']}")
        print("Please download the dataset and place it in the correct location.")
        sys.exit(1)
    
    prepare_dataset(CFG)
