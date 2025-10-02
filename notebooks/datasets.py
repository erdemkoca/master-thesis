#!/usr/bin/env python3
"""
Dataset Registry - Configuration for all datasets
Defines synthetic and real datasets with their properties
"""

DATASETS = [
    # Synthetic datasets
    {"kind": "synthetic", "id": "A", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Linear (low-dim) → Lasso optimal"},
    {"kind": "synthetic", "id": "B", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Linear (high-dim) → Lasso optimal"},
    {"kind": "synthetic", "id": "C", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Strong univariate nonlinearities (low-dim) → RF/NN sollten Vorteile zeigen"},
    {"kind": "synthetic", "id": "D", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Nonlinear interactions (high-dim) → NIMO/NN/RF sollten besser als Lasso sein"},
    {"kind": "synthetic", "id": "E", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Purely nonlinear (medium-dim) → Lasso scheitert, nur RF/NN/NIMO gewinnen"},
    
    # Real datasets
    {"kind": "real", "id": "boston", "path": "../data/real/boston/processed", "n_train": 140, "n_val": 60, "desc": "Boston Housing (binary classification)"},
    {"kind": "real", "id": "housing", "path": "../data/real/housing/processed", "n_train": 140, "n_val": 60, "desc": "California Housing (binary classification)"},
    {"kind": "real", "id": "diabetes", "path": "../data/real/diabetes/processed", "n_train": 140, "n_val": 60, "desc": "Diabetes Progression (binary classification)"},
    {"kind": "real", "id": "moon", "path": "../data/real/moon/processed", "n_train": 140, "n_val": 60, "desc": "Two-Moon Dataset (binary classification)"},
]

def get_dataset_by_id(dataset_id):
    """Get dataset configuration by ID."""
    for dataset in DATASETS:
        if dataset["id"] == dataset_id:
            return dataset
    raise ValueError(f"Dataset {dataset_id} not found in registry")

def get_synthetic_datasets():
    """Get all synthetic datasets."""
    return [d for d in DATASETS if d["kind"] == "synthetic"]

def get_real_datasets():
    """Get all real datasets."""
    return [d for d in DATASETS if d["kind"] == "real"]

def list_all_datasets():
    """List all available datasets."""
    print("Available datasets:")
    for dataset in DATASETS:
        print(f"  {dataset['kind']}: {dataset['id']} - {dataset['desc']}")
    return DATASETS

if __name__ == "__main__":
    list_all_datasets()
