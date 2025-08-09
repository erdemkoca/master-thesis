#!/usr/bin/env python3
"""
DEBUG VERSION: Synthetic Data Experiment Runner - Phase 1: L1-LogReg Only
Focus: Identify and fix identical F1 scores and discrete score issues
"""

import numpy as np
import pandas as pd
import os
import json
import hashlib
from scipy.special import expit
from scipy.stats import multivariate_normal, t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# Create results directories
os.makedirs("../results/synthetic_debug/", exist_ok=True)
os.makedirs("../results/synthetic_debug/logs/", exist_ok=True)

def generate_toeplitz_covariance(n_features, rho):
    """Generate Toeplitz covariance matrix with correlation rho"""
    if rho == 0:
        return np.eye(n_features)
    
    cov_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            cov_matrix[i, j] = rho ** abs(i - j)
    
    return cov_matrix

def generate_synthetic_data(n_samples=200, n_features=20, n_true_features=5,
                            interactions=None, nonlinear=None, rho=0.0,
                            noise='gaussian', custom=None, seed=42):
    """
    Generate synthetic data with various characteristics.
    CRITICAL: Uses provided seed for reproducible but varying data per iteration
    """
    np.random.seed(seed)  # Use the provided seed - this will vary per iteration

    # 1) Feature matrix X
    if rho > 0:
        cov = generate_toeplitz_covariance(n_features, rho)
        X = multivariate_normal.rvs(
            mean=np.zeros(n_features), cov=cov, size=n_samples
        )
    else:
        X = np.random.randn(n_samples, n_features)

    # 2) Ground-truth linear predictor
    linear_predictor = np.zeros(n_samples)

    # Standard linear case
    beta_true = np.zeros(n_features)
    support = np.random.choice(n_features, size=n_true_features, replace=False)
    beta_true[support] = np.random.uniform(1.0, 3.0, size=n_true_features)
    linear_predictor = X.dot(beta_true)

    # 3) Add interactions if specified
    if interactions:
        coeffs_int = np.random.uniform(0.5, 1.5, size=len(interactions))
        for inter, coeff in zip(interactions, coeffs_int):
            if len(inter) == 2:
                i, j = inter
                linear_predictor += coeff * X[:, i] * X[:, j]
            elif len(inter) == 3:
                i, j, k = inter
                linear_predictor += coeff * X[:, i] * X[:, j] * X[:, k]

    # 4) Add nonlinear terms if specified
    if nonlinear is not None and isinstance(nonlinear, (list, tuple)):
        coeffs_nonlin = np.random.uniform(0.5, 1.5, size=len(nonlinear))
        for spec, coeff in zip(nonlinear, coeffs_nonlin):
            if len(spec) == 2:
                transform, idx = spec
            elif len(spec) == 3:
                transform, idx, extra = spec
            else:
                continue
                
            col = X[:, idx]
            if transform == 'sin':
                linear_predictor += coeff * np.sin(col)
            elif transform == 'square':
                linear_predictor += coeff * (col ** 2)
            elif transform == 'cube':
                linear_predictor += coeff * (col ** 3)

    # 5) Generate target variable with noise
    if noise == 'gaussian':
        p = expit(linear_predictor)
        y = np.random.binomial(1, p)
    elif noise == 'student_t':
        noise_term = t.rvs(df=3, size=n_samples) * 0.1
        p = expit(linear_predictor + noise_term)
        y = np.random.binomial(1, p)
    elif noise == 'gaussian_heavy':
        p = expit(linear_predictor + np.random.randn(n_samples) * 2.0)
        y = np.random.binomial(1, p)
    else:
        raise ValueError(f"Unknown noise type: {noise!r}")

    return X, y, support, beta_true

def calculate_hash(arr):
    """Calculate MD5 hash of array for debugging"""
    if isinstance(arr, np.ndarray):
        return hashlib.md5(arr.tobytes()).hexdigest()[:8]
    else:
        return hashlib.md5(str(arr).encode()).hexdigest()[:8]

def run_l1_logreg_debug(X_train, y_train, X_val, y_val, X_test, y_test, seed_iter, iteration, X_columns):
    """
    L1 Logistic Regression with comprehensive debugging and proper randomness
    """
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            max_iter=1000,
            random_state=seed_iter  # Use iteration-specific seed
        ))
    ])
    
    # Hyperparameter grid - extensive C values
    param_grid = {
        'classifier__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                         4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0,
                         25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 100.0]
    }
    
    # Grid search with validation set
    # Combine train and val for CV, then use separate test set
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])
    
    # Create custom CV split that uses our train/val split
    cv_splits = [(np.arange(len(X_train)), np.arange(len(X_train), len(X_train_val)))]
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1',
        cv=cv_splits,  # Use our predefined split
        n_jobs=1,  # Single job for debugging
        verbose=0
    )
    
    grid_search.fit(X_train_val, y_train_val)
    
    # Get best model and refit on training data only
    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(X_train, y_train)
    
    # Get validation predictions for threshold optimization
    y_val_probs = best_pipeline.predict_proba(X_val)[:, 1]
    
    # Fine-grained threshold optimization on validation set
    thresholds = np.linspace(0.000, 1.000, 1001)  # 0.001 step size
    f1_scores_val = []
    
    for t in thresholds:
        y_val_pred = (y_val_probs >= t).astype(int)
        f1_val = f1_score(y_val, y_val_pred, zero_division=0)
        f1_scores_val.append(f1_val)
    
    best_threshold_idx = np.argmax(f1_scores_val)
    best_threshold = thresholds[best_threshold_idx]
    best_f1_val = f1_scores_val[best_threshold_idx]
    
    # Apply fixed threshold to test set
    y_test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= best_threshold).astype(int)
    
    # Calculate test metrics
    f1_test = f1_score(y_test, y_test_pred, zero_division=0)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    # Feature selection from L1 coefficients
    coefficients = best_pipeline.named_steps['classifier'].coef_.flatten()
    selected_features = [X_columns[i] for i, c in enumerate(coefficients) if abs(c) > 1e-8]
    n_nonzero = np.sum(np.abs(coefficients) > 1e-8)
    
    # Calculate coefficient hash for debugging
    coef_hash = calculate_hash(coefficients)
    
    return {
        'iteration': iteration,
        'seed_iter': seed_iter,
        'best_C': grid_search.best_params_['classifier__C'],
        'threshold_val_opt': best_threshold,
        'f1_val': best_f1_val,
        'f1_test': f1_test,
        'accuracy_test': accuracy_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'n_nonzero': int(n_nonzero),
        'coef_hash': coef_hash,
        'coefficients': coefficients.tolist(),
        'y_test_pred': y_test_pred.tolist(),
        'y_test_probs': y_test_probs.tolist()
    }

# Define Scenario A only for Phase 1
scenario_A = {
    'n_samples': 200,
    'n_features': 20,
    'n_true_features': 5,
    'interactions': None,
    'nonlinear': None,
    'rho': 0.0,
    'noise': 'gaussian',
    'description': 'Independent linear',
    'description_long': 'Basic linear model with independent features'
}

# Main experiment parameters
n_iterations = 20
n_test_large = 50000  # Large test set to avoid quantization

print("="*60)
print("DEBUG EXPERIMENT: PHASE 1 - L1 LOGISTIC REGRESSION ONLY")
print("="*60)
print(f"Scenario: A - {scenario_A['description']}")
print(f"Iterations: {n_iterations}")
print(f"Large test set size: {n_test_large}")
print()

# Initialize master RNG
master_rng = np.random.default_rng(42)

# Results storage
all_results = []
debug_logs = []

print("Starting experiment...")
for iteration in range(n_iterations):
    print(f"\n--- Iteration {iteration} ---")
    
    # Generate iteration-specific seed from master RNG
    seed_iter = master_rng.integers(0, 2**31-1)
    print(f"  seed_iter: {seed_iter}")
    
    # Generate NEW data for this iteration
    data_params = {k: v for k, v in scenario_A.items() if not k.startswith('description')}
    X_full, y_full, true_support, beta_true = generate_synthetic_data(**data_params, seed=seed_iter)
    
    # Create NEW large test set directly (not from 80/20 split)
    test_params = data_params.copy()
    test_params['n_samples'] = n_test_large
    X_test_large, y_test_large, _, _ = generate_synthetic_data(**test_params, seed=seed_iter + 1000)
    
    # Split remaining data into train/val
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed_iter)
    train_idx, val_idx = next(splitter.split(X_full, y_full))
    
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    X_test, y_test = X_test_large, y_test_large
    
    # Calculate hashes for debugging
    dataset_hash = calculate_hash(X_full) + calculate_hash(y_full)
    train_hash = calculate_hash(train_idx)
    val_hash = calculate_hash(val_idx)
    test_hash = calculate_hash(X_test) + calculate_hash(y_test)
    
    print(f"  dataset_hash: {dataset_hash}")
    print(f"  train_hash: {train_hash}, val_hash: {val_hash}, test_hash: {test_hash}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Class dist - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    # Create feature column names
    X_columns = [f'feature_{i}' for i in range(X_full.shape[1])]
    
    # Run L1 Logistic Regression
    try:
        result = run_l1_logreg_debug(X_train, y_train, X_val, y_val, X_test, y_test, 
                                   seed_iter, iteration, X_columns)
        
        # Add debug information
        result.update({
            'dataset_hash': dataset_hash,
            'train_hash': train_hash,
            'val_hash': val_hash,
            'test_hash': test_hash,
            'true_support': sorted(true_support),
            'n_true_features': len(true_support),
            'beta_true': beta_true.tolist(),
            'scenario': 'A'
        })
        
        all_results.append(result)
        
        # Log key metrics
        print(f"  F1_test: {result['f1_test']:.4f}, Accuracy: {result['accuracy_test']:.4f}")
        print(f"  Threshold: {result['threshold_val_opt']:.3f}, C: {result['best_C']}")
        print(f"  Features selected: {result['n_selected']}/{len(X_columns)}")
        print(f"  Coef hash: {result['coef_hash']}")
        
        # Store debug log
        debug_log = {
            'iteration': iteration,
            'seed_iter': seed_iter,
            'dataset_hash': dataset_hash,
            'f1_test': result['f1_test'],
            'coef_hash': result['coef_hash'],
            'n_nonzero': result['n_nonzero'],
            'threshold': result['threshold_val_opt']
        }
        debug_logs.append(debug_log)
        
    except Exception as e:
        print(f"  ERROR in iteration {iteration}: {e}")
        import traceback
        traceback.print_exc()

# Save results
results_df = pd.DataFrame(all_results)
debug_df = pd.DataFrame(debug_logs)

results_df.to_csv("../results/synthetic_debug/phase1_l1logreg_results.csv", index=False)
debug_df.to_csv("../results/synthetic_debug/phase1_debug_logs.csv", index=False)

print("\n" + "="*60)
print("PHASE 1 RESULTS SUMMARY")
print("="*60)
print(f"Total successful iterations: {len(all_results)}")

if len(all_results) > 0:
    print(f"\nF1 Test Score Statistics:")
    print(f"  Mean: {results_df['f1_test'].mean():.4f}")
    print(f"  Std:  {results_df['f1_test'].std():.4f}")
    print(f"  Min:  {results_df['f1_test'].min():.4f}")
    print(f"  Max:  {results_df['f1_test'].max():.4f}")
    print(f"  Range: {results_df['f1_test'].max() - results_df['f1_test'].min():.4f}")
    
    print(f"\nCoefficient Hash Variation:")
    unique_hashes = debug_df['coef_hash'].nunique()
    total_hashes = len(debug_df)
    print(f"  Unique hashes: {unique_hashes}/{total_hashes}")
    print(f"  Hash variation: {unique_hashes/total_hashes*100:.1f}%")
    
    print(f"\nDataset Hash Variation:")
    unique_dataset_hashes = debug_df['dataset_hash'].nunique()
    print(f"  Unique dataset hashes: {unique_dataset_hashes}/{total_hashes}")
    print(f"  Dataset variation: {unique_dataset_hashes/total_hashes*100:.1f}%")
    
    print(f"\nFeature Selection:")
    print(f"  Mean features selected: {results_df['n_selected'].mean():.1f}")
    print(f"  Std features selected: {results_df['n_selected'].std():.1f}")
    print(f"  Range: {results_df['n_selected'].min()}-{results_df['n_selected'].max()}")

print(f"\nResults saved to:")
print(f"  ../results/synthetic_debug/phase1_l1logreg_results.csv")
print(f"  ../results/synthetic_debug/phase1_debug_logs.csv")
print("\nPhase 1 completed!")
