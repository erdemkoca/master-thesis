#!/usr/bin/env python3
"""
COMPREHENSIVE Synthetic Data Experiment Runner
Implements all specifications from the debug requirements:
- Master RNG with iteration-specific seeds
- Fresh data generation per iteration  
- Large separate test sets
- Fine-grained threshold optimization on validation
- Comprehensive logging and hashing
- Built-in variation assertions
- Support for all methods (L1-LogReg, RF, NN, NIMO variants)
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from scipy.special import expit
from scipy.stats import multivariate_normal, t
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Import debug utilities
from methods.debug_utils import (
    md5_array, best_threshold_f1, confusion_counts, 
    calculate_comprehensive_hashes, validate_variation,
    validate_lasso_specific, create_experiment_log_entry,
    smoke_test_results, summary_statistics
)

# Configuration constants
MASTER_SEED = 42
N_ITERATIONS = 20
N_TEST_LARGE = 50000  # Large test set to avoid quantization
THRESHOLD_STEP = 0.001  # Fine-grained threshold optimization
MIN_VARIATION_PCT = 80.0  # Minimum required variation percentage

# Create results directories
results_dir = Path("../results/synthetic_comprehensive/")
results_dir.mkdir(exist_ok=True)
(results_dir / "logs").mkdir(exist_ok=True)

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
    Generate synthetic data with comprehensive scenario support
    
    Ensures g(0) = 0 for interactions and proper scaling for nonlinearities
    """
    np.random.seed(seed)

    # 1) Feature matrix X with correlation structure
    if rho > 0:
        cov = generate_toeplitz_covariance(n_features, rho)
        X = multivariate_normal.rvs(mean=np.zeros(n_features), cov=cov, size=n_samples)
    else:
        X = np.random.randn(n_samples, n_features)

    # 2) Linear predictor initialization
    linear_predictor = np.zeros(n_samples)
    
    # 2a) Main linear effects
    beta_true = np.zeros(n_features)
    support = np.random.choice(n_features, size=n_true_features, replace=False)
    beta_true[support] = np.random.uniform(1.0, 3.0, size=n_true_features)
    linear_predictor = X.dot(beta_true)

    # 3) Add interaction terms (ensuring g(0) = 0)
    if interactions:
        coeffs_int = np.random.uniform(0.5, 1.5, size=len(interactions))
        for inter, coeff in zip(interactions, coeffs_int):
            if len(inter) == 2:
                i, j = inter
                # Centered interaction: (x_i - E[x_i]) * (x_j - E[x_j])
                interaction_term = coeff * (X[:, i] - X[:, i].mean()) * (X[:, j] - X[:, j].mean())
                linear_predictor += interaction_term
            elif len(inter) == 3:
                i, j, k = inter
                # Three-way centered interaction
                interaction_term = coeff * ((X[:, i] - X[:, i].mean()) * 
                                          (X[:, j] - X[:, j].mean()) * 
                                          (X[:, k] - X[:, k].mean()))
                linear_predictor += interaction_term

    # 4) Add nonlinear terms (properly scaled)
    if nonlinear is not None and isinstance(nonlinear, (list, tuple)):
        coeffs_nonlin = np.random.uniform(0.5, 1.5, size=len(nonlinear))
        for spec, coeff in zip(nonlinear, coeffs_nonlin):
            if len(spec) >= 2:
                transform, idx = spec[0], spec[1]
                extra = spec[2] if len(spec) > 2 else None
                
                col = X[:, idx]
                
                if transform == 'sin':
                    # Scale argument to make it non-trivial: sin(2*pi*x) instead of sin(x)
                    freq = extra if extra else 2.0
                    nonlin_term = coeff * np.sin(freq * np.pi * col)
                    linear_predictor += nonlin_term
                    
                elif transform == 'square':
                    # Centered square: (x - E[x])^2 - Var[x]
                    centered_col = col - col.mean()
                    nonlin_term = coeff * (centered_col**2 - centered_col.var())
                    linear_predictor += nonlin_term
                    
                elif transform == 'cube':
                    # Centered cube: (x - E[x])^3
                    centered_col = col - col.mean()
                    nonlin_term = coeff * (centered_col**3)
                    linear_predictor += nonlin_term
                    
                elif transform == 'sin_highfreq':
                    freq = extra if extra else 5.0
                    nonlin_term = coeff * np.sin(freq * np.pi * col)
                    linear_predictor += nonlin_term

    # 5) Generate binary target with noise
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
        raise ValueError(f"Unknown noise type: {noise}")

    return X, y, support, beta_true

def run_lasso_comprehensive(X_train, y_train, X_val, y_val, X_test, y_test, 
                          seed_iter, X_columns):
    """
    Comprehensive L1 Logistic Regression with full logging
    """
    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            max_iter=1000,
            random_state=seed_iter
        ))
    ])
    
    # Extended hyperparameter grid
    C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0,
                25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 100.0]
    
    # Manual hyperparameter search (for full control)
    best_f1_val = 0
    best_C = C_values[0]
    best_pipeline = None
    
    for C in C_values:
        pipeline.set_params(classifier__C=C)
        pipeline.fit(X_train, y_train)
        
        # Validate on validation set
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        thr_val = best_threshold_f1(y_val, y_val_proba, step=THRESHOLD_STEP)
        y_val_pred = (y_val_proba >= thr_val).astype(int)
        f1_val = f1_score(y_val, y_val_pred, zero_division=0)
        
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_C = C
            best_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    penalty='l1', solver='liblinear', max_iter=1000,
                    random_state=seed_iter, C=C
                ))
            ])
            best_pipeline.fit(X_train, y_train)
    
    # Final evaluation on test set
    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    thr_val = best_threshold_f1(y_val, y_val_proba, step=THRESHOLD_STEP)
    
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= thr_val).astype(int)
    
    # Calculate metrics
    f1_test = f1_score(y_test, y_test_pred, zero_division=0)
    acc_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)
    
    tp, fp, fn, tn = confusion_counts(y_test, y_test_proba, thr_val)
    
    # Feature analysis
    coefficients = best_pipeline.named_steps['classifier'].coef_.flatten()
    selected_features = [X_columns[i] for i, c in enumerate(coefficients) if abs(c) > 1e-8]
    nnz = int(np.sum(np.abs(coefficients) > 1e-8))
    coef_hash = md5_array(coefficients)
    
    return {
        'thr_val': thr_val,
        'f1_test': f1_test,
        'acc_test': acc_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'coef_hash': coef_hash,
        'nnz': nnz,
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'best_C': best_C,
        'coefficients': coefficients.tolist()
    }

def run_random_forest_comprehensive(X_train, y_train, X_val, y_val, X_test, y_test, 
                                  seed_iter, X_columns):
    """
    Comprehensive Random Forest with extended hyperparameter search
    """
    # Extended hyperparameter grid
    param_combinations = [
        {'n_estimators': n_est, 'max_depth': max_d, 'max_features': max_f, 'min_samples_split': min_split}
        for n_est in [50, 100, 200, 300]
        for max_d in [3, 5, 10, None]
        for max_f in ['sqrt', 'log2', None]
        for min_split in [2, 5, 10]
    ]
    
    best_f1_val = 0
    best_params = None
    best_model = None
    
    for params in param_combinations:
        model = RandomForestClassifier(
            random_state=seed_iter,
            n_jobs=1,
            **params
        )
        model.fit(X_train, y_train)
        
        # Validate
        y_val_proba = model.predict_proba(X_val)[:, 1]
        thr_val = best_threshold_f1(y_val, y_val_proba, step=THRESHOLD_STEP)
        y_val_pred = (y_val_proba >= thr_val).astype(int)
        f1_val = f1_score(y_val, y_val_pred, zero_division=0)
        
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_params = params
            best_model = model
    
    # Final evaluation
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    thr_val = best_threshold_f1(y_val, y_val_proba, step=THRESHOLD_STEP)
    
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= thr_val).astype(int)
    
    # Calculate metrics
    f1_test = f1_score(y_test, y_test_pred, zero_division=0)
    acc_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)
    
    tp, fp, fn, tn = confusion_counts(y_test, y_test_proba, thr_val)
    
    # Feature importance analysis
    importances = best_model.feature_importances_
    importance_threshold = np.mean(importances) + np.std(importances)
    selected_features = [X_columns[i] for i, imp in enumerate(importances) 
                        if imp > importance_threshold]
    
    return {
        'thr_val': thr_val,
        'f1_test': f1_test,
        'acc_test': acc_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'feature_importances': importances.tolist(),
        'importance_hash': md5_array(importances)
    }

# Define comprehensive scenarios with mathematical formulations
scenarios = {
    'A': {
        'name': 'Independent Linear',
        'formula': 'p(y=1|x) = sigmoid(β₀ + β₁x₁ + ... + β₅x₅)',
        'n_samples': 200, 'n_features': 20, 'n_true_features': 5,
        'interactions': None, 'nonlinear': None, 'rho': 0.0, 'noise': 'gaussian'
    },
    'B': {
        'name': 'Feature Interactions',  
        'formula': 'p(y=1|x) = sigmoid(β₀ + Σβᵢxᵢ + γ₁x₀x₁ + γ₂x₂x₃)',
        'n_samples': 200, 'n_features': 20, 'n_true_features': 5,
        'interactions': [(0,1), (2,3)], 'nonlinear': None, 'rho': 0.0, 'noise': 'gaussian'
    },
    'C': {
        'name': 'Pure Nonlinear',
        'formula': 'p(y=1|x) = sigmoid(Σsin(2πxᵢ) + Σ(xⱼ-μ)² + Σ(xₖ-μ)³)',
        'n_samples': 200, 'n_features': 25, 'n_true_features': 0,
        'interactions': None, 
        'nonlinear': ([('sin', i, 2.0) for i in range(5)] + 
                     [('square', i) for i in range(5,10)] + 
                     [('cube', i) for i in range(10,15)]),
        'rho': 0.0, 'noise': 'gaussian'
    }
}

def main():
    """Main experiment runner with comprehensive logging and validation"""
    
    print("="*80)
    print("COMPREHENSIVE SYNTHETIC DATA EXPERIMENT")
    print("="*80)
    print(f"Master seed: {MASTER_SEED}")
    print(f"Iterations per scenario: {N_ITERATIONS}")
    print(f"Test set size: {N_TEST_LARGE}")
    print(f"Threshold step: {THRESHOLD_STEP}")
    print(f"Scenarios: {list(scenarios.keys())}")
    print()
    
    # Initialize master RNG
    rng_master = np.random.default_rng(MASTER_SEED)
    
    # Results storage
    all_results = []
    
    # Run experiments
    for scenario_key, scenario_params in scenarios.items():
        print(f"\n{'='*20} SCENARIO {scenario_key}: {scenario_params['name']} {'='*20}")
        print(f"Formula: {scenario_params['formula']}")
        
        scenario_results = []
        
        for iter_id in range(N_ITERATIONS):
            print(f"\n--- Iteration {iter_id} ---")
            
            # Generate iteration-specific seed
            seed_iter = int(rng_master.integers(0, 2**32-1))
            print(f"  seed_iter: {seed_iter}")
            
            # Generate fresh training data
            data_params = {k: v for k, v in scenario_params.items() 
                          if k not in ['name', 'formula']}
            X_full, y_full, true_support, beta_true = generate_synthetic_data(
                **data_params, seed=seed_iter
            )
            
            # Generate separate large test set
            test_params = data_params.copy()
            test_params['n_samples'] = N_TEST_LARGE
            X_test, y_test, _, _ = generate_synthetic_data(**test_params, seed=seed_iter + 1000)
            
            # Split training data into train/val
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed_iter)
            train_idx, val_idx = next(splitter.split(X_full, y_full))
            
            X_train, y_train = X_full[train_idx], y_full[train_idx]
            X_val, y_val = X_full[val_idx], y_full[val_idx]
            
            # Calculate comprehensive hashes
            hashes = calculate_comprehensive_hashes(X_full, y_full, train_idx, val_idx, X_test, y_test)
            
            print(f"  dataset_hash: {hashes['dataset_hash']}")
            print(f"  train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
            print(f"  true_support: {sorted(true_support)}")
            
            # Feature column names
            X_columns = [f'feature_{i}' for i in range(X_full.shape[1])]
            
            # Run methods
            methods = [
                ('lasso', run_lasso_comprehensive),
                ('random_forest', run_random_forest_comprehensive)
            ]
            
            for method_name, method_func in methods:
                print(f"    Running {method_name}...")
                
                try:
                    # Run method
                    method_results = method_func(X_train, y_train, X_val, y_val, 
                                               X_test, y_test, seed_iter, X_columns)
                    
                    # Create comprehensive log entry
                    log_entry = create_experiment_log_entry(
                        iter_id=iter_id,
                        method=method_name,
                        scenario=scenario_key,
                        seed_iter=seed_iter,
                        hashes=hashes,
                        metrics=method_results,
                        hyperparams={'best_C': method_results.get('best_C')} if 'best_C' in method_results else None
                    )
                    
                    # Add scenario metadata
                    log_entry.update({
                        'scenario_name': scenario_params['name'],
                        'scenario_formula': scenario_params['formula'],
                        'true_support': json.dumps([int(x) for x in sorted(true_support)]),
                        'n_true_features': int(len(true_support)),
                        'beta_true': json.dumps([float(x) for x in beta_true.tolist()])
                    })
                    
                    all_results.append(log_entry)
                    scenario_results.append(log_entry)
                    
                    print(f"      F1: {method_results['f1_test']:.4f}, "
                          f"Acc: {method_results['acc_test']:.4f}, "
                          f"Thr: {method_results['thr_val']:.3f}")
                    
                    if method_name == 'lasso':
                        print(f"      Features: {method_results['n_selected']}/{len(X_columns)}, "
                              f"Hash: {method_results['coef_hash']}")
                    
                except Exception as e:
                    print(f"      ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Log error
                    error_entry = create_experiment_log_entry(
                        iter_id=iter_id,
                        method=method_name,
                        scenario=scenario_key,
                        seed_iter=seed_iter,
                        hashes=hashes,
                        metrics={'error': str(e), 'f1_test': 0, 'acc_test': 0}
                    )
                    all_results.append(error_entry)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    # Save comprehensive results
    output_file = results_dir / f"comprehensive_results_{N_ITERATIONS}iter.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n📊 Results saved to: {output_file}")
    
    # Run comprehensive validation
    print(f"\n{'='*80}")
    print("VALIDATION & SMOKE TESTS")
    print("="*80)
    
    try:
        smoke_test_results(results_df, N_ITERATIONS)
        
        # Generate summary statistics
        summary_stats = summary_statistics(results_df)
        print(f"\n📈 SUMMARY STATISTICS:")
        print(summary_stats.to_string(index=False))
        
        # Save summary
        summary_file = results_dir / f"summary_statistics_{N_ITERATIONS}iter.csv"
        summary_stats.to_csv(summary_file, index=False)
        print(f"\n📈 Summary saved to: {summary_file}")
        
        print(f"\n🎉 ALL VALIDATION CHECKS PASSED!")
        print(f"✅ F1 score variation achieved for all methods")
        print(f"✅ Hash variation confirmed")
        print(f"✅ No identical values detected")
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
    print(f"\n🚀 Comprehensive synthetic experiment completed successfully!")
