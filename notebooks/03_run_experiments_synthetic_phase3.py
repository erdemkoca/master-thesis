#!/usr/bin/env python3
"""
DEBUG VERSION: Synthetic Data Experiment Runner - Phase 3: All Fixed Methods
Focus: Test L1-LogReg, Random Forest, and Neural Network with fixes
"""

import numpy as np
import pandas as pd
import os
import json
import hashlib
from scipy.special import expit
from scipy.stats import multivariate_normal, t

# Import fixed methods
from methods.lasso_fixed import run_lasso
from methods.random_forest_fixed import run_random_forest  
from methods.neural_net_fixed import run_neural_net

# Create results directories
os.makedirs("../results/synthetic_debug/", exist_ok=True)
os.makedirs("../results/synthetic_debug/phase3/", exist_ok=True)

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
    """Generate synthetic data with various characteristics."""
    np.random.seed(seed)

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

# Define test scenarios
scenarios = {
    'A': {
        'n_samples': 200,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': None,
        'nonlinear': None,
        'rho': 0.0,
        'noise': 'gaussian',
        'description': 'Independent linear'
    },
    'B': {
        'n_samples': 200,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': [(0,1), (2,3)],
        'nonlinear': None,
        'rho': 0.0,
        'noise': 'gaussian',
        'description': 'Feature interactions'
    }
}

# Define methods to test
methods = [
    ('lasso', run_lasso),
    ('RandomForest', run_random_forest),
    ('NeuralNet', run_neural_net)
]

# Main experiment parameters
n_iterations = 10  # Reduced for faster testing
n_test_large = 10000  # Large test set to avoid quantization

print("="*60)
print("DEBUG EXPERIMENT: PHASE 3 - ALL FIXED METHODS")
print("="*60)
print(f"Scenarios: {list(scenarios.keys())}")
print(f"Methods: {[name for name, _ in methods]}")
print(f"Iterations per scenario: {n_iterations}")
print(f"Large test set size: {n_test_large}")
print()

# Initialize master RNG
master_rng = np.random.default_rng(42)

# Results storage
all_results = []

# Loop over scenarios
for scenario_name, scenario_params in scenarios.items():
    print(f"\n{'='*20} SCENARIO {scenario_name} {'='*20}")
    print(f"Description: {scenario_params['description']}")
    
    scenario_results = []
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration} ---")
        
        # Generate iteration-specific seed from master RNG
        seed_iter = master_rng.integers(0, 2**31-1)
        print(f"  seed_iter: {seed_iter}")
        
        # Generate NEW data for this iteration
        data_params = {k: v for k, v in scenario_params.items() if not k.startswith('description')}
        X_full, y_full, true_support, beta_true = generate_synthetic_data(**data_params, seed=seed_iter)
        
        # Create NEW large test set directly
        test_params = data_params.copy()
        test_params['n_samples'] = n_test_large
        X_test_large, y_test_large, _, _ = generate_synthetic_data(**test_params, seed=seed_iter + 1000)
        
        X_test, y_test = X_test_large, y_test_large
        
        # Calculate hashes for debugging
        dataset_hash = calculate_hash(X_full) + calculate_hash(y_full)
        test_hash = calculate_hash(X_test) + calculate_hash(y_test)
        
        print(f"  dataset_hash: {dataset_hash}, test_hash: {test_hash}")
        print(f"  Train samples: {len(X_full)}, Test samples: {len(X_test)}")
        print(f"  True support: {sorted(true_support)}")
        
        # Create feature column names
        X_columns = [f'feature_{i}' for i in range(X_full.shape[1])]
        
        # Test each method
        for method_name, method_fn in methods:
            print(f"    Running {method_name}...")
            
            try:
                # Create a separate RNG for this method call
                method_rng = np.random.default_rng(seed_iter)
                
                result = method_fn(
                    X_full, y_full,  # Use full data as "training" (method will split internally)
                    X_test, y_test,
                    method_rng,
                    iteration,
                    seed_iter,  # randomState
                    X_columns
                )
                
                # Add scenario metadata
                result.update({
                    'scenario': scenario_name,
                    'scenario_description': scenario_params['description'],
                    'dataset_hash': dataset_hash,
                    'test_hash': test_hash,
                    'true_support': sorted(true_support),
                    'n_true_features': len(true_support),
                    'beta_true': beta_true.tolist()
                })
                
                all_results.append(result)
                scenario_results.append(result)
                
                print(f"      F1_test: {result['best_f1']:.4f}, Accuracy: {result.get('accuracy', 0):.4f}")
                print(f"      Threshold: {result['best_threshold']:.3f}")
                print(f"      Features selected: {result['n_selected']}/{len(X_columns)}")
                
            except Exception as e:
                print(f"      ERROR in {method_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add error result
                error_result = {
                    'model_name': method_name,
                    'iteration': iteration,
                    'scenario': scenario_name,
                    'seed_iter': seed_iter,
                    'error': str(e),
                    'best_f1': 0.0,
                    'accuracy': 0.0
                }
                all_results.append(error_result)

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/synthetic_debug/phase3_all_methods_results.csv", index=False)

print("\n" + "="*60)
print("PHASE 3 RESULTS SUMMARY")
print("="*60)
print(f"Total results: {len(all_results)}")

# Filter successful results (no errors)
success_df = results_df[results_df['error'].isna()] if 'error' in results_df.columns else results_df

if len(success_df) > 0:
    print(f"Successful results: {len(success_df)}")
    
    print(f"\nF1 Test Score Statistics by Method:")
    for method_name in success_df['model_name'].unique():
        method_results = success_df[success_df['model_name'] == method_name]
        print(f"  {method_name}:")
        print(f"    Mean: {method_results['best_f1'].mean():.4f}")
        print(f"    Std:  {method_results['best_f1'].std():.4f}")
        print(f"    Min:  {method_results['best_f1'].min():.4f}")
        print(f"    Max:  {method_results['best_f1'].max():.4f}")
        print(f"    Range: {method_results['best_f1'].max() - method_results['best_f1'].min():.4f}")
    
    print(f"\nF1 Test Score Statistics by Scenario:")
    for scenario in success_df['scenario'].unique():
        scenario_results = success_df[success_df['scenario'] == scenario]
        print(f"  Scenario {scenario}:")
        print(f"    Mean: {scenario_results['best_f1'].mean():.4f}")
        print(f"    Std:  {scenario_results['best_f1'].std():.4f}")
        print(f"    Range: {scenario_results['best_f1'].max() - scenario_results['best_f1'].min():.4f}")

    # Check for variation within methods
    print(f"\nVariation Check:")
    for method_name in success_df['model_name'].unique():
        method_results = success_df[success_df['model_name'] == method_name]
        unique_f1 = len(method_results['best_f1'].unique())
        total_results = len(method_results)
        variation_pct = unique_f1 / total_results * 100
        print(f"  {method_name}: {unique_f1}/{total_results} unique F1 scores ({variation_pct:.1f}% variation)")

print(f"\nResults saved to:")
print(f"  ../results/synthetic_debug/phase3_all_methods_results.csv")
print("\nPhase 3 completed!")
