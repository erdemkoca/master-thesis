#!/usr/bin/env python3
"""
Synthetic Data Experiment Runner
Similar to 03_run_experiments.py but uses synthetic data instead of real Framingham data.
"""

import numpy as np
import pandas as pd
import os
import json
from scipy.special import expit

# Import all methods
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net
from methods.nimo_variants.baseline import run_nimo_baseline
from methods.nimo_variants.variant import run_nimo_variant
from notebooks.methods.nimo_variants.nimoNew import run_nimoNew

# Create results directories
os.makedirs("../results/synthetic/", exist_ok=True)
os.makedirs("../results/synthetic/preds/", exist_ok=True)
os.makedirs("../results/synthetic/features/", exist_ok=True)

def generate_synthetic_data(n_samples=200, n_features=20, n_true_features=5, seed=42):
    """
    Generate synthetic data with known true support.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_true_features: Number of truly important features
        seed: Random seed for reproducibility
    
    Returns:
        X: Feature matrix
        y: Target vector
        support: True support indices
        beta_true: True coefficients
    """
    np.random.seed(seed)
    
    # Generate true coefficients with sparse support
    beta_true = np.zeros(n_features)
    support = np.random.choice(n_features, size=n_true_features, replace=False)
    beta_true[support] = np.random.uniform(1.0, 3.0, size=n_true_features)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets using logistic model
    p = expit(X.dot(beta_true))
    y = np.random.binomial(1, p)
    
    return X, y, support, beta_true

def split_data(X, y, test_size=0.3, seed=42):
    """
    Split data into train and test sets.
    """
    np.random.seed(seed)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, y_train, X_test, y_test

# Generate synthetic data
print("Generating synthetic data...")
X_full, y_full, true_support, beta_true = generate_synthetic_data(
    n_samples=200, 
    n_features=20, 
    n_true_features=5, 
    seed=42
)

# Split into train/test
X_train, y_train, X_test, y_test = split_data(X_full, y_full, test_size=0.3, seed=42)

print(f"Generated synthetic data:")
print(f"  Total samples: {len(X_full)}")
print(f"  Features: {X_full.shape[1]}")
print(f"  True support features: {sorted(true_support)}")
print(f"  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Class distribution (train): {np.bincount(y_train)}")
print(f"  Class distribution (test): {np.bincount(y_test)}")
print()

# Create feature column names for synthetic data
X_columns = [f'feature_{i}' for i in range(X_full.shape[1])]

all_results = []

# Outer loop: multiple iterations with different seeds
n_iterations = 5
print(f"Running {n_iterations} iterations...")

for iteration in range(n_iterations):
    print(f"\n=== Iteration {iteration} ===")
    
    # Use iteration as seed for reproducibility
    rng = np.random.default_rng(iteration)
    
    # For synthetic data, we can either:
    # 1. Use the same data for all iterations (deterministic)
    # 2. Generate new data for each iteration (stochastic)
    # Here we'll use option 1 for consistency, but you can change this
    
    # Option 1: Use same data (deterministic)
    X_sub = X_train.copy()
    y_sub = y_train.copy()
    
    # Option 2: Generate new data each iteration (uncomment to use)
    # X_sub, y_sub, _, _ = generate_synthetic_data(
    #     n_samples=len(X_train), 
    #     n_features=X_train.shape[1], 
    #     n_true_features=5, 
    #     seed=iteration
    # )
    
    # Define methods to run
    methods = [
        run_lasso,
        run_lassonet,
        #run_random_forest,
        run_neural_net,
        run_nimo_baseline,
        run_nimo_variant,
        run_nimoNew
    ]
    
    # Use iteration as random state for reproducibility
    randomState = iteration
    
    for method_fn in methods:
        print(f"Running {method_fn.__name__}...")
        
        try:
            result = method_fn(
                X_sub, y_sub, X_test, y_test,
                rng,
                iteration,
                randomState,
                X_columns
            )
            
            # Convert numpy arrays to lists for JSON serialization
            preds = result['y_pred']
            if isinstance(preds, np.ndarray):
                preds = preds.tolist()
            elif isinstance(preds, list):
                # Convert any numpy types in the list to Python types
                preds = [int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x for x in preds]
            # Ensure all elements are native Python types
            preds = [int(x) if isinstance(x, (np.integer, int)) else float(x) if isinstance(x, (np.floating, float)) else x for x in preds]
            result['y_pred'] = json.dumps(preds)
            
            probs = result['y_prob']
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
            elif isinstance(probs, list):
                # Convert any numpy types in the list to Python types
                probs = [float(x) if isinstance(x, np.floating) else x for x in probs]
            # Ensure all elements are native Python types
            probs = [float(x) if isinstance(x, (np.floating, float)) else x for x in probs]
            result['y_prob'] = json.dumps(probs)
            
            if 'selected_features' in result:
                selected_features = result['selected_features']
                if isinstance(selected_features, list):
                    # Convert any numpy types in the list to Python types
                    selected_features = [int(x) if isinstance(x, np.integer) else str(x) if isinstance(x, str) else x for x in selected_features]
                # Ensure all elements are native Python types
                selected_features = [int(x) if isinstance(x, (np.integer, int)) else str(x) if isinstance(x, str) else x for x in selected_features]
                result['selected_features'] = json.dumps(selected_features)
            
            # Convert all numpy types to Python types for JSON serialization
            for key, value in result.items():
                if isinstance(value, np.integer):
                    result[key] = int(value)
                elif isinstance(value, np.floating):
                    result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, list):
                    # Convert any numpy types in lists
                    result[key] = [
                        int(x) if isinstance(x, np.integer) else 
                        float(x) if isinstance(x, np.floating) else 
                        str(x) if isinstance(x, (np.str_, str)) else x 
                        for x in value
                    ]
            
            # Add synthetic data metadata
            # Convert numpy int64 to Python int for JSON serialization
            true_support_list = [int(x) for x in sorted(true_support)]
            result['true_support'] = json.dumps(true_support_list)
            result['n_true_features'] = len(true_support)
            result['data_type'] = 'synthetic'
            
            # Debug: Print result keys and types
            # print(f"Result keys: {list(result.keys())}")
            # for key, value in result.items():
            #     print(f"  {key}: {type(value)} = {value}")
            
            all_results.append(result)
            
            # Save raw arrays
            np.save(f"../results/synthetic/preds/{result['model_name']}_iteration{iteration}_probs.npy",
                    np.array(json.loads(result['y_prob'])))
            np.save(f"../results/synthetic/preds/{result['model_name']}_iteration{iteration}_preds.npy",
                    np.array(json.loads(result['y_pred'])))
            
            if 'selected_features' in result:
                with open(f"../results/synthetic/features/{result['model_name']}_iteration{iteration}.json", "w") as f:
                    f.write(result['selected_features'])
                    
        except Exception as e:
            print(f"Error running {method_fn.__name__}: {e}")
            # Add error result
            error_result = {
                'model_name': method_fn.__name__,
                'iteration': iteration,
                'error': str(e),
                'data_type': 'synthetic'
            }
            all_results.append(error_result)

# Save all results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/synthetic/all_model_results_synthetic.csv", index=False)
print(f"\nAll experiment results saved to ../results/synthetic/all_model_results_synthetic.csv")

# Print summary statistics
print("\n" + "="*60)
print("SYNTHETIC DATA EXPERIMENT SUMMARY")
print("="*60)
print(f"True support features: {sorted(true_support)}")
print(f"Number of iterations: {n_iterations}")
print(f"Methods tested: {len(methods)}")
print(f"Total results: {len(all_results)}")

# Calculate average F1 scores
if 'best_f1' in results_df.columns:
    print("\nAverage F1 Scores:")
    f1_summary = results_df.groupby('model_name')['best_f1'].agg(['mean', 'std']).round(4)
    print(f1_summary)

# Calculate feature selection accuracy
if 'selected_features' in results_df.columns:
    print("\nFeature Selection Summary:")
    for method_name in results_df['model_name'].unique():
        method_results = results_df[results_df['model_name'] == method_name]
        if 'selected_features' in method_results.columns:
            avg_selected = method_results['n_selected'].mean()
            print(f"{method_name}: Average {avg_selected:.1f} features selected")

print("\nSynthetic data experiment completed!") 