#!/usr/bin/env python3
"""
Dataset Registry - Configuration for all datasets
Defines synthetic and real datasets with their properties
"""

DATASETS = [
    # Synthetic datasets
    {"kind": "synthetic", "id": "A", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Linear (low-dim)"},
    {"kind": "synthetic", "id": "B", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Linear (high-dim)"},
    {"kind": "synthetic", "id": "C", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Linear + univariate nonlinearity (low-dim)"},
    {"kind": "synthetic", "id": "D", "path": "../data/synthetic", "n_train": 140, "n_val": 60, "desc": "Linear + interactions + nonlinearity (high-dim)"},
    
    # Real datasets (add your real datasets here)
    # {"kind": "real", "id": "breast_cancer", "path": "../data/real/bc", "n_train": 140, "n_val": 60, "desc": "Breast Cancer Wisconsin"},
    # {"kind": "real", "id": "diabetes", "path": "../data/real/diabetes", "n_train": 140, "n_val": 60, "desc": "Pima Indians Diabetes"},
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
