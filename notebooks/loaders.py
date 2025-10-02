#!/usr/bin/env python3
"""
Unified Data Loaders
Provides consistent interface for loading synthetic and real datasets
"""

import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

def load_synth(sid, path):
    """
    Load synthetic dataset by scenario ID.
    
    Args:
        sid: Scenario ID (e.g., "A", "B", "C", "D")
        path: Path to synthetic data directory
        
    Returns:
        X, y, idx_test, idx_pool, meta
    """
    X = np.load(f"{path}/scenario_{sid}_X_full.npy")
    y = np.load(f"{path}/scenario_{sid}_y_full.npy")
    idx_test = np.load(f"{path}/scenario_{sid}_idx_test_big.npy")
    idx_pool = np.load(f"{path}/scenario_{sid}_idx_pool.npy")
    
    with open(f"{path}/scenario_{sid}_metadata.json", "r") as f:
        meta = json.load(f)
    
    # Add feature names
    feat_names = [f"feature_{i}" for i in range(X.shape[1])]
    meta["feature_names"] = feat_names
    
    return X, y, idx_test, idx_pool, meta

def load_real(did, path):
    """
    Load real dataset by dataset ID.
    Assumes the dataset has been preprocessed and saved in the expected format.
    
    Args:
        did: Dataset ID (e.g.,)
        path: Path to processed real data directory
        
    Returns:
        X, y, idx_test, idx_pool, meta
    """
    # Load preprocessed data - try .npy first, then .pkl
    import pickle
    
    try:
        X = np.load(f"{path}/X_full.npy")
        y = np.load(f"{path}/y_full.npy")
    except (FileNotFoundError, ValueError):
        # Fall back to pickle files for mixed data types
        with open(f"{path}/X_full.pkl", "rb") as f:
            X = pickle.load(f)
        with open(f"{path}/y_full.pkl", "rb") as f:
            y = pickle.load(f)
        
        # Convert pandas objects to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
    
    # Load metadata
    with open(f"{path}/metadata.json", "r") as f:
        meta = json.load(f)
    
    # For real datasets, we need to create idx_test and idx_pool
    # Use the pre-split train/test data to create these indices
    n_samples = len(y)
    
    # Load the pre-split indices if they exist, otherwise create them
    if (os.path.exists(f"{path}/X_train.npy") and os.path.exists(f"{path}/X_test.npy")) or \
       (os.path.exists(f"{path}/X_train.pkl") and os.path.exists(f"{path}/X_test.pkl")):
        
        # Try .npy first, then .pkl
        try:
            X_train = np.load(f"{path}/X_train.npy")
            X_test = np.load(f"{path}/X_test.npy")
            y_train = np.load(f"{path}/y_train.npy")
            y_test = np.load(f"{path}/y_test.npy")
        except (FileNotFoundError, ValueError):
            with open(f"{path}/X_train.pkl", "rb") as f:
                X_train = pickle.load(f)
            with open(f"{path}/X_test.pkl", "rb") as f:
                X_test = pickle.load(f)
            with open(f"{path}/y_train.pkl", "rb") as f:
                y_train = pickle.load(f)
            with open(f"{path}/y_test.pkl", "rb") as f:
                y_test = pickle.load(f)
            
            # Convert pandas objects to numpy arrays
            if hasattr(X_train, 'values'):
                X_train = X_train.values
            if hasattr(X_test, 'values'):
                X_test = X_test.values
            if hasattr(y_train, 'values'):
                y_train = y_train.values
            if hasattr(y_test, 'values'):
                y_test = y_test.values
        
        # Find the indices in the full dataset that correspond to train/test
        # This is a bit hacky but works for our use case
        n_train = len(X_train)
        n_test = len(X_test)
        
        # Create indices: first n_train are train, next n_test are test
        # This assumes the data was split in order
        idx_pool = np.arange(n_train)  # Training indices (pool for sampling)
        idx_test = np.arange(n_train, n_train + n_test)  # Test indices
        
    else:
        # Fallback: create a random split
        from sklearn.model_selection import train_test_split
        idx_all = np.arange(n_samples)
        idx_pool, idx_test = train_test_split(
            idx_all, test_size=0.2, random_state=42, stratify=y
        )
    
    # Ensure feature_names is in meta
    if "feature_names" not in meta:
        meta["feature_names"] = [f"feature_{i}" for i in range(X.shape[1])]
    
    return X, y, idx_test, idx_pool, meta

def load_any(entry):
    """
    Load any dataset (synthetic or real) based on entry configuration.
    
    Args:
        entry: Dataset configuration dict with 'kind', 'id', 'path'
        
    Returns:
        X, y, idx_test, idx_pool, meta
    """
    if entry["kind"] == "synthetic":
        return load_synth(entry["id"], entry["path"])
    elif entry["kind"] == "real":
        return load_real(entry["id"], entry["path"])
    else:
        raise ValueError(f"Unknown dataset kind: {entry['kind']}")

def validate_dataset(entry):
    """
    Validate that a dataset entry is properly configured.
    
    Args:
        entry: Dataset configuration dict
        
    Returns:
        bool: True if valid
    """
    required_fields = ["kind", "id", "path", "n_train", "n_val"]
    for field in required_fields:
        if field not in entry:
            print(f"Missing required field: {field}")
            return False
    
    if entry["kind"] not in ["synthetic", "real"]:
        print(f"Invalid kind: {entry['kind']}")
        return False
    
    if not os.path.exists(entry["path"]):
        print(f"Path does not exist: {entry['path']}")
        return False
    
    return True

def get_dataset_info(entry):
    """
    Get information about a dataset without loading the full data.
    
    Args:
        entry: Dataset configuration dict
        
    Returns:
        dict: Dataset information
    """
    try:
        X, y, idx_test, idx_pool, meta = load_any(entry)
        return {
            "id": entry["id"],
            "kind": entry["kind"],
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_test": len(idx_test),
            "n_pool": len(idx_pool),
            "class_distribution": {
                "class_0": int(np.sum(y == 0)),
                "class_1": int(np.sum(y == 1))
            },
            "description": meta.get("desc", entry.get("desc", "No description"))
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    from datasets import DATASETS
    
    print("Testing data loaders...")
    for entry in DATASETS:
        print(f"\n--- {entry['id']} ({entry['kind']}) ---")
        if validate_dataset(entry):
            info = get_dataset_info(entry)
            if "error" not in info:
                print(f"✓ {info['n_samples']} samples, {info['n_features']} features")
                print(f"  Test: {info['n_test']}, Pool: {info['n_pool']}")
                print(f"  Classes: {info['class_distribution']}")
            else:
                print(f"✗ Error: {info['error']}")
        else:
            print("✗ Invalid configuration")
