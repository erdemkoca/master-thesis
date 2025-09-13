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
    Creates fixed test/pool split on first run and saves indices.
    
    Args:
        did: Dataset ID (e.g., "breast_cancer", "diabetes")
        path: Path to real data directory
        
    Returns:
        X, y, idx_test, idx_pool, meta
    """
    # Load data
    X = np.load(f"{path}/X.npy")
    y = np.load(f"{path}/y.npy")
    
    # Create fixed test/pool split if it doesn't exist
    test_path = f"{path}/idx_test_big.npy"
    pool_path = f"{path}/idx_pool.npy"
    
    if not os.path.exists(test_path) or not os.path.exists(pool_path):
        print(f"Creating fixed test/pool split for {did}...")
        idx = np.arange(len(y))
        idx_pool, idx_test = train_test_split(
            idx, 
            test_size=0.5, 
            stratify=y, 
            random_state=42
        )
        np.save(test_path, idx_test)
        np.save(pool_path, idx_pool)
        print(f"Saved test indices: {len(idx_test)} samples")
        print(f"Saved pool indices: {len(idx_pool)} samples")
    else:
        idx_test = np.load(test_path)
        idx_pool = np.load(pool_path)
    
    # Create metadata
    feat_names = [f"feature_{i}" for i in range(X.shape[1])]
    meta = {
        "feature_names": feat_names,
        "desc": did,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "class_distribution": {
            "class_0": int(np.sum(y == 0)),
            "class_1": int(np.sum(y == 1))
        }
    }
    
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
