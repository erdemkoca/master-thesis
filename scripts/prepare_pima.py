#!/usr/bin/env python3
"""
Prepare Pima Indians Diabetes Dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_real_dataset import prepare_dataset, clean_pima

if __name__ == "__main__":
    # Configuration for Pima Indians Diabetes
    CFG = dict(
        name="pima",
        description="Pima Indians Diabetes Dataset (UCI)",
        raw_csv="../data/real/pima/raw/pima.csv",
        target="Outcome",
        positive_label=1,
        drop_cols=[],  # No leakage columns
        categorical=[],  # All numeric after cleaning
        numeric=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        separator=",",
        test_frac=0.50,
        random_state=42,
        out_dir="../data/real/pima/processed",
        cleaning_func=clean_pima  # Apply Pima-specific cleaning
    )
    
    print("Pima Indians Diabetes Dataset Preparation")
    print("=" * 50)
    print("Note: Download the dataset from UCI ML Repository:")
    print("https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes")
    print("Save as: data/real/pima/raw/pima-indians-diabetes.csv")
    print()
    
    # Check if file exists
    if not os.path.exists(CFG["raw_csv"]):
        print(f"❌ Raw data file not found: {CFG['raw_csv']}")
        print("Please download the dataset and place it in the correct location.")
        sys.exit(1)
    
    prepare_dataset(CFG)
