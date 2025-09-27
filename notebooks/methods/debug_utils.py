x# methods/debug_utils.py
"""
Utility functions for debugging and validating synthetic experiments
"""

import numpy as np
import pandas as pd
from hashlib import md5
from sklearn.metrics import f1_score, confusion_matrix
import json
from typing import Tuple, Dict, Any

def md5_array(a: np.ndarray) -> str:
    """
    Calculate MD5 hash of numpy array for debugging purposes
    
    Args:x
        a: numpy array
    
    Returns:
        MD5 hash string (first 8 characters)
    """
    a = np.ascontiguousarray(a)
    return md5(a.view(np.uint8)).hexdigest()[:8]

def best_threshold_f1(y_true: np.ndarray, proba: np.ndarray, step: float = 0.001) -> float:
    """
    Find optimal threshold for F1 score using fine-grained search
    
    Args:
        y_true: True binary labels
        proba: Predicted probabilities
        step: Step size for threshold search
    
    Returns:
        Optimal threshold value
    """
    thrs = np.arange(0.0, 1.0 + step, step)
    f1s = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thrs]
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx])

def confusion_counts(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix counts at given threshold
    
    Args:
        y_true: True binary labels
        proba: Predicted probabilities  
        thr: Threshold value
    
    Returns:
        Tuple of (tp, fp, fn, tn) counts
    """
    yhat = (proba >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (yhat == 1)))
    fp = int(np.sum((y_true == 0) & (yhat == 1)))
    fn = int(np.sum((y_true == 1) & (yhat == 0)))
    tn = int(np.sum((y_true == 0) & (yhat == 0)))
    return tp, fp, fn, tn

def calculate_comprehensive_hashes(X: np.ndarray, y: np.ndarray, 
                                 train_idx: np.ndarray, val_idx: np.ndarray, 
                                 test_X: np.ndarray, test_y: np.ndarray) -> Dict[str, str]:
    """
    Calculate comprehensive hash set for debugging data variation
    
    Args:
        X: Training feature matrix
        y: Training target vector
        train_idx: Training indices
        val_idx: Validation indices
        test_X: Test feature matrix
        test_y: Test target vector
    
    Returns:
        Dictionary of hash values
    """
    return {
        'dataset_hash': md5_array(X) + md5_array(y),
        'train_idx_hash': md5_array(train_idx),
        'val_idx_hash': md5_array(val_idx),
        'test_hash': md5_array(test_X) + md5_array(test_y),
        'train_data_hash': md5_array(X[train_idx]) + md5_array(y[train_idx]),
        'val_data_hash': md5_array(X[val_idx]) + md5_array(y[val_idx])
    }

def validate_variation(results_df: pd.DataFrame, method_name: str, min_variation_pct: float = 50.0) -> None:
    """
    Assert that results show sufficient variation (fail-fast debugging)
    
    Args:
        results_df: DataFrame with experimental results
        method_name: Name of the method being validated
        min_variation_pct: Minimum percentage of unique values required
    
    Raises:
        AssertionError: If variation is insufficient
    """
    method_results = results_df[results_df['method'] == method_name]
    
    # Check F1 variation
    f1_var = method_results['f1_test'].var()
    assert f1_var > 0, f"❌ {method_name}: F1 variance is 0! (identical values detected)"
    
    # Check uniqueness (adjust threshold for small sample sizes)
    unique_f1 = method_results['f1_test'].nunique()
    total_f1 = len(method_results)
    variation_pct = unique_f1 / total_f1 * 100
    
    # Adaptive threshold: for small samples, require at least 2 unique values
    if total_f1 <= 5:
        min_unique = 2
        assert unique_f1 >= min_unique, \
            f"❌ {method_name}: Only {unique_f1} unique F1 values (< {min_unique} for small sample)"
    else:
        assert variation_pct >= min_variation_pct, \
            f"❌ {method_name}: Only {variation_pct:.1f}% unique F1 values (< {min_variation_pct}%)"
    
    # Check hash variation for dataset (more lenient for small samples)
    unique_datasets = method_results['dataset_hash'].nunique()
    total_datasets = len(method_results)
    dataset_variation_pct = unique_datasets / total_datasets * 100
    
    min_dataset_pct = 70 if total_datasets <= 5 else 90
    assert dataset_variation_pct >= min_dataset_pct, \
        f"❌ {method_name}: Only {dataset_variation_pct:.1f}% unique datasets (< {min_dataset_pct}%)"
    
    print(f"✅ {method_name}: Variation checks passed!")
    print(f"   F1 uniqueness: {variation_pct:.1f}% ({unique_f1}/{total_f1})")
    print(f"   F1 variance: {f1_var:.6f}")
    print(f"   Dataset uniqueness: {dataset_variation_pct:.1f}%")

def validate_lasso_specific(results_df: pd.DataFrame, min_coef_variation_pct: float = 50.0) -> None:
    """
    Additional validation for Lasso-specific metrics
    
    Args:
        results_df: DataFrame with Lasso results
        min_coef_variation_pct: Minimum percentage of unique coefficient hashes
    
    Raises:
        AssertionError: If Lasso-specific variation is insufficient
    """
    lasso_results = results_df[results_df['method'] == 'lasso']
    
    if len(lasso_results) == 0:
        return
    
    # Check coefficient hash variation
    if 'coef_hash' in lasso_results.columns:
        unique_coef = lasso_results['coef_hash'].nunique()
        total_coef = len(lasso_results)
        coef_variation_pct = unique_coef / total_coef * 100
        
        assert coef_variation_pct >= min_coef_variation_pct, \
            f"❌ Lasso: Only {coef_variation_pct:.1f}% unique coefficient hashes (< {min_coef_variation_pct}%)"
    
    # Check nnz (number of non-zero coefficients) variation
    if 'nnz' in lasso_results.columns:
        nnz_var = lasso_results['nnz'].var()
        assert nnz_var > 0, f"❌ Lasso: No variation in number of non-zero coefficients!"
    
    print("✅ Lasso-specific validation passed!")

def convert_to_native_types(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    else:
        return obj

def create_experiment_log_entry(iter_id: int, method: str, scenario: str, seed_iter: int,
                               hashes: Dict[str, str], metrics: Dict[str, Any],
                               hyperparams: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized log entry for experiment results
    
    Args:
        iter_id: Iteration number
        method: Method name
        scenario: Scenario name
        seed_iter: Iteration-specific seed
        hashes: Dictionary of hash values
        metrics: Dictionary of performance metrics
        hyperparams: Dictionary of hyperparameters (optional)
    
    Returns:
        Standardized log entry dictionary
    """
    # Convert all values to native types
    metrics_native = convert_to_native_types(metrics)
    hashes_native = convert_to_native_types(hashes)
    
    log_entry = {
        'iter_id': int(iter_id),
        'method': str(method),
        'scenario': str(scenario),
        'seed_iter': int(seed_iter),
        **hashes_native,
        **metrics_native
    }
    
    if hyperparams:
        log_entry['hp_json'] = json.dumps(convert_to_native_types(hyperparams))
    else:
        log_entry['hp_json'] = None
    
    return log_entry

def smoke_test_results(results_df: pd.DataFrame, n_iterations: int) -> None:
    """
    Comprehensive smoke test for experimental results
    
    Args:
        results_df: DataFrame with all experimental results
        n_iterations: Expected number of iterations
    
    Raises:
        AssertionError: If any smoke test fails
    """
    print("🔍 Running smoke tests...")
    
    # Basic data integrity
    assert len(results_df) > 0, "❌ No results found!"
    
    methods = results_df['method'].unique()
    scenarios = results_df['scenario'].unique()
    
    print(f"   Methods: {list(methods)}")
    print(f"   Scenarios: {list(scenarios)}")
    
    # Check each method
    for method in methods:
        validate_variation(results_df, method)
    
    # Lasso-specific checks
    if 'lasso' in methods:
        validate_lasso_specific(results_df)
    
    # Overall variation check
    overall_f1_var = results_df['f1_test'].var()
    assert overall_f1_var > 0, "❌ No overall F1 variation detected!"
    
    print(f"✅ All smoke tests passed!")
    print(f"   Overall F1 variance: {overall_f1_var:.6f}")
    print(f"   Total results: {len(results_df)}")

def summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics
    
    Args:
        results_df: DataFrame with experimental results
    
    Returns:
        Summary statistics DataFrame
    """
    summary_stats = []
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        
        f1_stats = method_data['f1_test'].agg(['count', 'mean', 'std', 'min', 'max', 'var'])
        unique_f1 = method_data['f1_test'].nunique()
        variation_pct = unique_f1 / len(method_data) * 100
        
        stats_dict = {
            'method': method,
            'n_iterations': f1_stats['count'],
            'f1_mean': f1_stats['mean'],
            'f1_std': f1_stats['std'],
            'f1_min': f1_stats['min'],
            'f1_max': f1_stats['max'],
            'f1_range': f1_stats['max'] - f1_stats['min'],
            'f1_var': f1_stats['var'],
            'unique_f1_count': unique_f1,
            'variation_pct': variation_pct
        }
        
        # Add method-specific stats
        if method == 'lasso' and 'nnz' in method_data.columns:
            stats_dict['nnz_mean'] = method_data['nnz'].mean()
            stats_dict['nnz_std'] = method_data['nnz'].std()
            stats_dict['unique_coef_hashes'] = method_data['coef_hash'].nunique() if 'coef_hash' in method_data.columns else None
        
        summary_stats.append(stats_dict)
    
    return pd.DataFrame(summary_stats).round(4)
