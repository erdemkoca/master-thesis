#!/usr/bin/env python3
"""
Sampling Utilities
Provides stratified sampling with replacement and train set rebalancing
"""

import numpy as np

def stratified_with_replacement(y, idx_pool, n_train, n_val, seed):
    """
    Stratified sampling with replacement from pool for train/val splits.
    
    Args:
        y: Full target array
        idx_pool: Indices available for sampling
        n_train: Number of training samples to draw
        n_val: Number of validation samples to draw
        seed: Random seed for reproducibility
        
    Returns:
        idx_train, idx_val: Training and validation indices
    """
    rng = np.random.default_rng(seed)
    y_pool = y[idx_pool]
    
    # Get indices for each class
    idx0 = idx_pool[y_pool == 0]
    idx1 = idx_pool[y_pool == 1]
    
    def draw(k, arr):
        """Draw k samples with replacement from array."""
        if len(arr) == 0:
            return np.array([], dtype=int)
        return rng.choice(arr, size=k, replace=True)
    
    # Calculate target per-class counts (proportional split)
    pi = y_pool.mean() if len(y_pool) > 0 else 0.5
    n_tr1 = int(round(n_train * pi))
    n_tr0 = n_train - n_tr1
    n_va1 = int(round(n_val * pi))
    n_va0 = n_val - n_va1
    
    # Draw samples
    idx_tr0 = draw(n_tr0, idx0)
    idx_tr1 = draw(n_tr1, idx1)
    idx_va0 = draw(n_va0, idx0)
    idx_va1 = draw(n_va1, idx1)
    
    # Combine and shuffle
    idx_train = np.concatenate([idx_tr0, idx_tr1])
    idx_val = np.concatenate([idx_va0, idx_va1])
    
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    
    return idx_train, idx_val

def rebalance_train_indices(y, idx_tr, mode="oversample", target_pos=0.5, seed=0):
    """
    Rebalance training indices by oversampling or undersampling.
    Only affects training set; validation and test remain unchanged.
    
    Args:
        y: Full target array
        idx_tr: Training indices to rebalance
        mode: "oversample", "undersample", or "none"
        target_pos: Target proportion of positive class
        seed: Random seed for reproducibility
        
    Returns:
        idx_tr_rebalanced: Rebalanced training indices
    """
    if mode == "none":
        return idx_tr
    
    rng = np.random.default_rng(seed)
    y_tr = y[idx_tr]
    
    # Get indices for each class
    idx_pos = idx_tr[y_tr == 1]
    idx_neg = idx_tr[y_tr == 0]
    
    def draw(k, arr, replace):
        """Draw k samples with or without replacement."""
        if len(arr) == 0:
            return np.array([], dtype=int)
        return rng.choice(arr, size=k, replace=replace)
    
    if mode == "oversample":
        # Oversample minority class - target counts based on total training set
        n_target_pos = int(round(target_pos * len(idx_tr)))
        n_target_neg = len(idx_tr) - n_target_pos
        new_pos = draw(n_target_pos, idx_pos, replace=True)
        new_neg = draw(n_target_neg, idx_neg, replace=True)
    elif mode == "undersample":
        # Undersample majority class - target counts based on available samples
        n_available_pos = len(idx_pos)
        n_available_neg = len(idx_neg)
        
        # Calculate how many samples we can take while maintaining target_pos
        # If we take all positive samples, how many negative samples do we need?
        n_target_pos = n_available_pos
        n_target_neg = int(round(n_available_pos * (1 - target_pos) / target_pos))
        
        # If we don't have enough negative samples, take all and adjust positive
        if n_target_neg > n_available_neg:
            n_target_neg = n_available_neg
            n_target_pos = int(round(n_available_neg * target_pos / (1 - target_pos)))
        
        new_pos = draw(n_target_pos, idx_pos, replace=False)
        new_neg = draw(n_target_neg, idx_neg, replace=False)
    else:
        raise ValueError(f"Unknown rebalancing mode: {mode}")
    
    # Combine and shuffle
    idx_tr_rebalanced = np.concatenate([new_pos, new_neg])
    rng.shuffle(idx_tr_rebalanced)
    
    return idx_tr_rebalanced

def get_class_distribution(y, idx):
    """
    Get class distribution for given indices.
    
    Args:
        y: Full target array
        idx: Indices to analyze
        
    Returns:
        dict: Class distribution statistics
    """
    y_subset = y[idx]
    n_total = len(y_subset)
    n_pos = np.sum(y_subset == 1)
    n_neg = np.sum(y_subset == 0)
    
    return {
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pos_proportion": n_pos / n_total if n_total > 0 else 0.0,
        "neg_proportion": n_neg / n_total if n_total > 0 else 0.0
    }

def validate_sampling(y, idx_train, idx_val, expected_n_train, expected_n_val):
    """
    Validate that sampling produced expected results.
    
    Args:
        y: Full target array
        idx_train: Training indices
        idx_val: Validation indices
        expected_n_train: Expected number of training samples
        expected_n_val: Expected number of validation samples
        
    Returns:
        dict: Validation results
    """
    results = {
        "valid": True,
        "errors": []
    }
    
    # Check sizes
    if len(idx_train) != expected_n_train:
        results["valid"] = False
        results["errors"].append(f"Expected {expected_n_train} train samples, got {len(idx_train)}")
    
    if len(idx_val) != expected_n_val:
        results["valid"] = False
        results["errors"].append(f"Expected {expected_n_val} val samples, got {len(idx_val)}")
    
    # Check for overlap
    overlap = np.intersect1d(idx_train, idx_val)
    if len(overlap) > 0:
        results["valid"] = False
        results["errors"].append(f"Overlap between train and val: {len(overlap)} samples")
    
    # Check class distributions
    train_dist = get_class_distribution(y, idx_train)
    val_dist = get_class_distribution(y, idx_val)
    
    results["train_distribution"] = train_dist
    results["val_distribution"] = val_dist
    
    return results

if __name__ == "__main__":
    # Test sampling functions
    print("Testing sampling functions...")
    
    # Create test data
    np.random.seed(42)
    n_total = 1000
    y = np.random.binomial(1, 0.3, n_total)  # 30% positive class
    idx_pool = np.arange(n_total)
    
    print(f"Test data: {n_total} samples, {np.sum(y)} positive ({np.mean(y):.1%})")
    
    # Test stratified sampling
    idx_train, idx_val = stratified_with_replacement(y, idx_pool, n_train=100, n_val=50, seed=42)
    
    print(f"\nSampling results:")
    print(f"Train: {len(idx_train)} samples")
    print(f"Val: {len(idx_val)} samples")
    
    # Check distributions
    train_dist = get_class_distribution(y, idx_train)
    val_dist = get_class_distribution(y, idx_val)
    
    print(f"\nTrain distribution: {train_dist['n_positive']} pos, {train_dist['n_negative']} neg ({train_dist['pos_proportion']:.1%})")
    print(f"Val distribution: {val_dist['n_positive']} pos, {val_dist['n_negative']} neg ({val_dist['pos_proportion']:.1%})")
    
    # Test rebalancing
    print(f"\nTesting rebalancing...")
    idx_tr_rebalanced = rebalance_train_indices(y, idx_train, mode="oversample", target_pos=0.5, seed=42)
    rebalanced_dist = get_class_distribution(y, idx_tr_rebalanced)
    print(f"Rebalanced train: {rebalanced_dist['n_positive']} pos, {rebalanced_dist['n_negative']} neg ({rebalanced_dist['pos_proportion']:.1%})")
    
    # Validate
    validation = validate_sampling(y, idx_train, idx_val, 100, 50)
    print(f"\nValidation: {'✓' if validation['valid'] else '✗'}")
    if not validation['valid']:
        for error in validation['errors']:
            print(f"  Error: {error}")
