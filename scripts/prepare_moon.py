#!/usr/bin/env python3
"""
Two Moons Dataset Preprocessing
Generates the two-moons dataset directly with sklearn and saves in the expected format
"""

from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
from pathlib import Path
import json

def prepare_moons(n_samples=1000, noise=0.2, test_frac=0.2, random_state=42, out_dir="data/real/moon/processed"):
    """Generate and prepare the two-moons dataset."""
    print("=== Preparing Two Moons dataset ===")
    
    # Generate directly with sklearn
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = pd.DataFrame(X, columns=["X1", "X2"])
    
    print(f"Generated dataset: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Class 0: {(y == 0).sum()} samples ({(y == 0).mean()*100:.1f}%)")
    print(f"Class 1: {(y == 1).sum()} samples ({(y == 1).mean()*100:.1f}%)")
    
    # Split into train/test
    n_test = int(test_frac * n_samples)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_samples)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Save processed data
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    np.save(out_path / "X_full.npy", X.values)
    np.save(out_path / "y_full.npy", y)
    np.save(out_path / "X_train.npy", X_train.values)
    np.save(out_path / "X_test.npy", X_test.values)
    np.save(out_path / "y_train.npy", y_train)
    np.save(out_path / "y_test.npy", y_test)
    
    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "n_features": 2,
        "feature_names": ["X1", "X2"],
        "target_name": "moon_class",
        "positive_class": "Moon class 1",
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "test_frac": test_frac,
        "random_state": random_state,
        "noise": noise
    }
    
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved dataset to {out_path}")
    print(f"Metadata: {metadata}")
    
    return X, y, metadata

if __name__ == "__main__":
    prepare_moons()