#!/usr/bin/env python3
"""
Synthetic Data Experiment Runner - Enhanced Version
Supports multiple scenarios with different data characteristics.
"""

import numpy as np
import pandas as pd
import os
import json
from scipy.special import expit
from scipy.stats import multivariate_normal, t

# Import all methods
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net
from methods.nimo_variants.baseline import run_nimo_baseline
from methods.nimo_variants.variant import run_nimo_variant
from methods.nimo_variants.nimoNew import run_nimoNew
from methods.utils import to_native, standardize_method_output
from methods.neural_net2 import run_neural_net2


# Create results directories
os.makedirs("../results/synthetic/", exist_ok=True)
os.makedirs("../results/synthetic/preds/", exist_ok=True)
os.makedirs("../results/synthetic/features/", exist_ok=True)
os.makedirs("../results/synthetic/convergence/", exist_ok=True)

# 1) Define scenarios with detailed descriptions
scenarios = {
    'A': {
        'n_samples': 200,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': None,
        'nonlinear': None,
        'rho': 0.0,
        'noise': 'gaussian',
        'description':       'Independent linear',
        'description_long':  'Basic linear model with independent features'
    },
    'B': {
        'n_samples': 200,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': [(0,1), (2,3)],
        'nonlinear': None,
        'rho': 0.0,
        'noise': 'gaussian',
        'description':       'Feature interactions',
        'description_long':  'Linear model with feature interactions'
    },
    'C': {
        'n_samples': 200,
        'n_features': 25,
        'n_true_features': 0,
        'interactions': None,
        'nonlinear': (
            [('sin', i) for i in range(5)]
          + [('square', i) for i in range(5,10)]
          + [('cube', i) for i in range(10,15)]
        ),
        'rho': 0.0,
        'noise': 'gaussian',
        'description':      'Pure nonlinear',
        'description_long': 'No linear β·X. 5×sin(X), 5×X², 5×X³ + Gaussian noise'
    },
    'D': {
        'n_samples': 100,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': [(0,1), (2,3), (4,5,6)],
        'nonlinear': [('sin', 2)],
        'rho': 0.95,
        'noise': 'gaussian_heavy',
        'description':       'Hard Interactions + corr.',
        'description_long':  'Complex model w/ interactions, ρ=0.95, two interactions + 3‑way, heavy noise '
    },
    'E': {
        'n_samples': 200,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': [(0,1)],
        'nonlinear': None,
        'rho': 0.0,
        'noise': 'student_t',
        'description':       'Heavy‑tailed noise',
        'description_long':  'Linear model with heavy‑tailed (Student‑t) noise'
    },
    'F': {
        'n_samples': 300,
        'n_features': 20,           # hier ziehen wir 20 rohe X
        'n_true_features': 5,
        'interactions': None,
        'nonlinear': [ # Wir hängen 5 sehr hochfrequente Sinus‑Basisfunktionen an
            ('sin_highfreq', k, freq)
            for k, freq in zip(range(5), [5, 7, 9, 11, 13])
        ],
        'rho': 0.0,
        'noise': 'gaussian_heavy',
        'description':       'High‑freq sin (hidden)',
        'description_long':  (
            'Draw X ~ N(0,I), then add 5 columns sin(freq·X_k) '
            'with frequencies [5,7,9,11,13] + heavy Gaussian noise'
        )
    },
    'G': {
        'n_samples': 500,
        'n_features': 10,
        'n_true_features': 2,
        'interactions': [(0,1), (2,3)],
        'nonlinear': None,
        'rho': 0.0,
        'noise': 'gaussian',
        'description':       'XOR interactions',
        'description_long':  'Signal = x₀⊕x₁ + x₂⊕x₃ + Gaussian noise'
    },
    'H': {
        'n_samples': 300,
        'n_features': 15,
        'n_true_features': 3,
        'interactions': None,
        'nonlinear': [('square', 0), ('cube', 1)],
        'rho': 0.9,
        'noise': 'gaussian',
        'description':       'Polynom + corr.',
        'description_long':  'ρ=0.9, Signal = β₀ x₀² + β₁ x₁³ + Gaussian noise'
    },
    'I': {
        'n_samples': 300,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': None,
        'nonlinear': None, 
        'custom': 'rbf',             # RBF kernel features
        'rho': 0.0,
        'noise': 'gaussian',
        'description':      'RBF-Kernel',
        'description_long': 'y = sigmoid( exp(-||x - mu||^2 / sigma^2) ) + noise'
    },
    'J': {
        'n_samples': 800,
        'n_features': 15,
        'n_true_features': 5,
        'interactions': None,
        'nonlinear': 'sawtooth',
        'rho': 0.0,
        'noise': 'gaussian',
        'description':      'Piecewise Sägezahn',
        'description_long': 'y = sawtooth(x_k) + sum_sin(...) + noise'
    },
    'K': {
        'n_samples': 2000,
        'n_features': 12,
        'n_true_features': 4,
        'interactions': [(0,1,2), (3,4,5)],  # 3-way interactions
        'nonlinear': None,
        'rho': 0.5,
        'noise': 'gaussian',
        'description':      '3‑Wege Interaktionen',
        'description_long': 'y = x0*x1*x2 + x3*x4*x5 + corr + noise'
    },
    'L': {
        'n_samples': 50,
        'n_features': 500,
        'n_true_features': 10,
        'interactions': None,
        'nonlinear': [('sin', i) for i in range(10)],
        'rho': 0.0,
        'noise': 'gaussian_heavy',
        'description':      'High-dim + noise',
        'description_long': 'n < p, high noise, nonlinear signal'
    },
    'M': {
        'n_samples': 300,
        'n_features': 20,
        'n_true_features': 5,
        'interactions': [(0,1), (2,3)],
        'nonlinear': [('square', 0), ('cube', 1), ('sin', 2)],
        'rho': 0.8,
        'noise': 'student_t',
        'description':      'Complex + heavy noise',
        'description_long': 'Interactions + polynomials + correlations + heavy-tailed noise'
    },
}




def ensure_core_fields(result):
    """Ensure all core fields exist in the result dictionary"""
    # Use the standardize_method_output function which already handles this
    result = standardize_method_output(result)
    
    # Ensure specific defaults for list fields
    if result.get('y_pred') is None:
        result['y_pred'] = []
    if result.get('y_prob') is None:
        result['y_prob'] = []
    if result.get('selected_features') is None:
        result['selected_features'] = []
    
    return result

def generate_toeplitz_covariance(n_features, rho):
    """Generate Toeplitz covariance matrix with correlation rho"""
    if rho == 0:
        return np.eye(n_features)
    
    # Create Toeplitz matrix: Σ_ij = ρ^|i-j|
    cov_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            cov_matrix[i, j] = rho ** abs(i - j)
    
    return cov_matrix

def calculate_support_recovery_metrics(selected_features, true_support, n_features):
    """
    Calculate support recovery metrics.
    
    Args:
        selected_features: List of selected feature indices
        true_support: List of true feature indices
        n_features: Total number of features
    
    Returns:
        dict: Support recovery metrics
    """
    if not selected_features or not true_support:
        return {
            'true_positive_rate': 0.0,
            'false_positive_rate': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_recovery': 0.0
        }
    
    # Convert feature names to indices if needed
    if isinstance(selected_features[0], str):
        selected_indices = [int(f.split('_')[-1]) for f in selected_features if '_' in f]
    else:
        selected_indices = selected_features
    
    selected_set = set(selected_indices)
    true_set = set(true_support)
    
    # Calculate metrics
    intersection = selected_set.intersection(true_set)
    
    # True positive rate (recall): |S_pred ∩ S_true| / |S_true|
    true_positive_rate = len(intersection) / len(true_set) if true_set else 0.0
    
    # False positive rate: |S_pred \ S_true| / (d - |S_true|)
    false_positives = selected_set - true_set
    false_positive_rate = len(false_positives) / (n_features - len(true_set)) if (n_features - len(true_set)) > 0 else 0.0
    
    # Precision: |S_pred ∩ S_true| / |S_pred|
    precision = len(intersection) / len(selected_set) if selected_set else 0.0
    
    # Recall (same as true_positive_rate)
    recall = true_positive_rate
    
    # F1 score for recovery
    f1_recovery = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'true_positive_rate': true_positive_rate,
        'false_positive_rate': false_positive_rate,
        'precision': precision,
        'recall': recall,
        'f1_recovery': f1_recovery
    }


import numpy as np
from scipy.stats import multivariate_normal, t
from scipy.special import expit

def generate_synthetic_data(n_samples=200, n_features=20, n_true_features=5,
                            interactions=None, nonlinear=None, rho=0.0,
                            noise='gaussian', custom=None, seed=42):
    """
    Generate synthetic data with various characteristics.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_true_features: Number of truly important features
        interactions: List of tuples (i,j) or (i,j,k) for interaction terms
        nonlinear: List of tuples
                   - (type, feature_idx)
                   - (type, feature_idx, extra)
                   or the string 'sawtooth'
        rho: Correlation parameter for feature covariance
        noise: 'gaussian', 'student_t', or 'gaussian_heavy'
        custom: 'rbf' for RBF kernel features
        seed: Random seed for reproducibility

    Returns:
        X: Feature matrix
        y: Target vector
        support: True support indices
        beta_true: True linear coefficients (only for standard linear part)
    """
    np.random.seed(seed)

    # 1) Merkmalsmatrix X
    if rho > 0:
        cov = generate_toeplitz_covariance(n_features, rho)
        X = multivariate_normal.rvs(
            mean=np.zeros(n_features), cov=cov, size=n_samples
        )
    else:
        X = np.random.randn(n_samples, n_features)

    # 2) Ground‐truth‐Linear‐Predictor
    linear_predictor = np.zeros(n_samples)

    # 2a) RBF‐Szenario
    if custom == 'rbf':
        # centers und sigma
        mus = np.random.randn(n_true_features, n_features)
        sigma = 1.0
        # φ(x)
        phi = np.exp(
            -np.sum((X[:, None, :] - mus[None, :, :]) ** 2, axis=2)
             / sigma ** 2
        )
        coeffs       = np.random.uniform(1.0, 3.0, size=n_true_features)
        linear_predictor = phi.dot(coeffs)
        support      = np.arange(n_true_features)
        beta_true    = np.zeros(n_features)

    else:
        # 2b) Standard‐linear
        beta_true = np.zeros(n_features)
        support   = np.random.choice(n_features, size=n_true_features, replace=False)
        beta_true[support] = np.random.uniform(1.0, 3.0, size=n_true_features)
        linear_predictor   = X.dot(beta_true)

        # 3) Interaktionen
        if interactions:
            coeffs_int = np.random.uniform(0.5, 1.5, size=len(interactions))
            for inter, coeff in zip(interactions, coeffs_int):
                if len(inter) == 2:
                    i, j = inter
                    linear_predictor += coeff * X[:, i] * X[:, j]
                elif len(inter) == 3:
                    i, j, k = inter
                    linear_predictor += coeff * X[:, i] * X[:, j] * X[:, k]

        # 4) Nicht‐Linearitäten
        if nonlinear is not None:
            # a) String‑Spec 'sawtooth'
            if isinstance(nonlinear, str) and nonlinear.lower() == 'sawtooth':
                import scipy.signal as sg
                # wende Sawtooth pro Feature an (Periodenfaktor 5)
                linear_predictor += np.sum(sg.sawtooth(5 * X, width=0.5), axis=1)

            # b) Liste/Tupel‑Spec
            elif isinstance(nonlinear, (list, tuple)):
                coeffs_nonlin = np.random.uniform(0.5, 1.5, size=len(nonlinear))
                for spec, coeff in zip(nonlinear, coeffs_nonlin):
                    # je nach Länge 2 oder 3 entpacken
                    if len(spec) == 2:
                        transform, idx = spec
                        extra = None
                    elif len(spec) == 3:
                        transform, idx, extra = spec
                    else:
                        raise ValueError(f"Ungültiges nonlinear‑Spec: {spec!r}")

                    col = X[:, idx]
                    if transform == 'sin':
                        linear_predictor += coeff * np.sin(col)
                    elif transform == 'square':
                        linear_predictor += coeff * (col ** 2)
                    elif transform == 'cube':
                        linear_predictor += coeff * (col ** 3)
                    elif transform == 'exp':
                        linear_predictor += coeff * np.exp(col)
                    elif transform == 'log':
                        linear_predictor += coeff * np.log(np.abs(col) + 1e-8)
                    elif transform == 'sin_highfreq':
                        # hier extra = Frequenz
                        linear_predictor += coeff * np.sin(extra * col)
                    elif transform == 'sawtooth':
                        import scipy.signal as sg
                        linear_predictor += coeff * sg.sawtooth(
                            extra * col, width=0.5
                        )
                    else:
                        raise ValueError(f"Unbekannter Transform‑Typ: {transform!r}")
            else:
                raise ValueError(f"Ungültiges nonlinear‑Spec: {nonlinear!r}")

    # 5) Rauschen & Zielvariable y
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



def split_data(X, y, test_size=0.3, seed=42):
    """Split data into train and test sets."""
    np.random.seed(seed)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, y_train, X_test, y_test

# Main experiment loop
all_results = []
n_iterations = 20

print("="*60)
print("SYNTHETIC DATA EXPERIMENT RUNNER")
print("="*60)
print(f"Scenarios: {list(scenarios.keys())}")
print(f"Iterations per scenario: {n_iterations}")
print()

# Define methods to run
methods = [
    run_lasso,
    run_lassonet,
    run_random_forest,
    run_neural_net,
    run_nimo_baseline,
    run_nimo_variant,
    run_nimoNew,
    run_neural_net2
]

print(f"Methods: {[m.__name__ for m in methods]}")
print()

# 2) Loop over scenarios
for scenario_name, params in scenarios.items():
    print(f"\n{'='*20} SCENARIO {scenario_name} {'='*20}")
    print(f"Description: {params['description']}")
    print(f"Parameters: {params}")
    
    # Generate data for this scenario
    # Remove description from params for function call
    data_params = {
        k: v
        for k, v in params.items()
        if not k.startswith('description')
    }
    X_full, y_full, true_support, beta_true = generate_synthetic_data(**data_params)
    X_train, y_train, X_test, y_test = split_data(X_full, y_full, test_size=0.3, seed=42)
    
    print(f"Generated data:")
    print(f"  Total samples: {len(X_full)}")
    print(f"  Features: {X_full.shape[1]}")
    print(f"  True support features: {sorted(true_support)}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")
    print(f"  Class distribution (test): {np.bincount(y_test)}")
    
    # Create feature column names for synthetic data
    X_columns = [f'feature_{i}' for i in range(X_full.shape[1])]
    
    # Inner loop: iterations
    for iteration in range(n_iterations):
        print(f"\n  --- Iteration {iteration} ---")
        
        # Use iteration as seed for reproducibility
        rng = np.random.default_rng(iteration)
        
        # Use same data for all iterations (deterministic)
        X_sub = X_train.copy()
        y_sub = y_train.copy()
        
        # Use iteration as random state for reproducibility
        randomState = iteration
        
        # Method loop
        for method_fn in methods:
            print(f"    Running {method_fn.__name__}...")
            
            try:
                print(f"      Calling {method_fn.__name__}...")
                result = method_fn(
                    X_sub, y_sub, X_test, y_test,
                    rng,
                    iteration,
                    randomState,
                    X_columns
                )
                print(f"      {method_fn.__name__} returned successfully")
                
                # Ensure all core fields exist and convert to native types
                result = ensure_core_fields(result)
                
                # Calculate support recovery metrics
                if result['selected_features']:
                    recovery_metrics = calculate_support_recovery_metrics(
                        result['selected_features'], 
                        sorted(true_support), 
                        X_full.shape[1]
                    )
                    result.update(recovery_metrics)
                
                # 4) Add scenario metadata with detailed descriptions
                scenario_description = f"{scenario_name}: {params['description']}"
                if params['interactions']:
                    scenario_description += f", interactions={params['interactions']}"
                if params['nonlinear']:
                    scenario_description += f", nonlinear={params['nonlinear']}"
                if params.get('custom'):
                    scenario_description += f", custom={params['custom']}"
                scenario_description += f", ρ={params['rho']}, noise={params['noise']}"
                
                # Convert metadata to native types before adding
                metadata = {
                    'scenario': scenario_name,
                    'scenario_short': params['description'],
                    'scenario_long': params.get('description_long', params['description']),
                    'scenario_description': scenario_description,
                    'rho': float(params['rho']),
                    'noise_type': params['noise'],
                    'interactions': json.dumps(params['interactions']) if params['interactions'] else None,
                    'nonlinear_terms': json.dumps(params['nonlinear']) if params['nonlinear'] else None,
                    'custom_function': params.get('custom', None),
                    'true_support': json.dumps([int(x) for x in sorted(true_support)]),
                    'n_true_features': int(len(true_support)),
                    'beta_true': json.dumps(beta_true.tolist()),
                    'data_type': 'synthetic',
                    'n_features_total': int(X_full.shape[1])
                }
                result.update(metadata)
                
                # Add early-stopping and convergence information if available
                if 'convergence_history' in result:
                    try:
                        # Save convergence history to separate file
                        convergence_file = f"../results/synthetic/convergence/{result['model_name']}_scenario{scenario_name}_iteration{iteration}.json"
                        with open(convergence_file, 'w') as f:
                            json.dump(result['convergence_history'], f)
                        
                        # Add convergence metadata
                        result['convergence_file'] = convergence_file
                        result['n_iterations_run'] = int(len(result['convergence_history']))
                        result['stopped_early'] = bool(result.get('stopped_early', False))
                        
                        # Remove the full history from main result to keep CSV manageable
                        del result['convergence_history']
                    except Exception as e:
                        print(f"    Warning: Could not save convergence history: {e}")
                        # Remove convergence history if it can't be serialized
                        if 'convergence_history' in result:
                            del result['convergence_history']
                
                # Add group regularization CV results if available
                if 'best_group_reg' in result:
                    result['group_reg_cv_performed'] = True
                else:
                    result['group_reg_cv_performed'] = False
                    result['best_group_reg'] = None
                
                # Final check: ensure all values are native types
                try:
                    # Test JSON serialization to catch any remaining numpy types
                    json.dumps(result)
                    all_results.append(result)
                except Exception as e:
                    print(f"    Warning: JSON serialization failed for {result.get('model_name', 'unknown')}: {e}")
                    # Try to fix the issue by converting all values to native types
                    fixed_result = {}
                    for k, v in result.items():
                        try:
                            fixed_result[k] = to_native(v)
                        except:
                            fixed_result[k] = str(v)  # Convert to string as last resort
                    all_results.append(fixed_result)
                
                # 3) And safe to write raw arrays straight from Python lists
                proba_arr = np.array(result.get('y_prob', []), dtype=float)
                pred_arr = np.array(result.get('y_pred', []), dtype=int)
                np.save(f"../results/synthetic/preds/{result['model_name']}_scenario{scenario_name}_iteration{iteration}_probs.npy", proba_arr)
                np.save(f"../results/synthetic/preds/{result['model_name']}_scenario{scenario_name}_iteration{iteration}_preds.npy", pred_arr)
                
                # 4) If you wrote out selected_features, do the same:
                if result.get('selected_features'):
                    with open(f"../results/synthetic/features/{result['model_name']}_scenario{scenario_name}_iteration{iteration}.json", "w") as f:
                        json.dump(result['selected_features'], f)
                        
            except Exception as e:
                print(f"    Error running {method_fn.__name__}: {e}")
                import traceback
                print(f"    Full traceback:")
                traceback.print_exc()
                # Add error result with core fields
                error_result = ensure_core_fields({
                    'model_name': method_fn.__name__,
                    'iteration': iteration,
                    'scenario': scenario_name,
                    'scenario_description': f"{scenario_name}: {params['description']}",
                    'error': str(e),
                    'data_type': 'synthetic'
                })
                
                all_results.append(error_result)

# 5) Write single CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/synthetic/all_model_results_synthetic_5Iterations.csv", index=False)
print(f"\nAll experiment results saved to ../results/synthetic/all_model_results_synthetic_5Iterations.csv")

# Print summary statistics
print("\n" + "="*60)
print("SYNTHETIC DATA EXPERIMENT SUMMARY")
print("="*60)
print(f"Total results: {len(all_results)}")
print(f"Scenarios tested: {len(scenarios)}")
print(f"Methods tested: {len(methods)}")
print(f"Iterations per scenario: {n_iterations}")

# Calculate average F1 scores by scenario and method
if 'best_f1' in results_df.columns:
    print("\nAverage F1 Scores by Scenario and Method:")
    f1_summary = results_df.groupby(['scenario', 'model_name'])['best_f1'].agg(['mean', 'std']).round(4)
    print(f1_summary)

# Calculate feature selection accuracy by scenario
if 'selected_features' in results_df.columns:
    print("\nFeature Selection Summary by Scenario:")
    for scenario in results_df['scenario'].unique():
        scenario_df = results_df[results_df['scenario'] == scenario]
        print(f"\nScenario {scenario}:")
        for method_name in scenario_df['model_name'].unique():
            method_results = scenario_df[scenario_df['model_name'] == method_name]
            if 'n_selected' in method_results.columns:
                avg_selected = method_results['n_selected'].mean()
                print(f"  {method_name}: Average {avg_selected:.1f} features selected")

# Calculate support recovery metrics summary
if 'f1_recovery' in results_df.columns:
    print("\nSupport Recovery Summary by Scenario:")
    for scenario in results_df['scenario'].unique():
        scenario_df = results_df[results_df['scenario'] == scenario]
        print(f"\nScenario {scenario}:")
        for method_name in scenario_df['model_name'].unique():
            method_results = scenario_df[scenario_df['model_name'] == method_name]
            if 'f1_recovery' in method_results.columns:
                avg_f1_recovery = method_results['f1_recovery'].mean()
                avg_precision = method_results['precision'].mean()
                avg_recall = method_results['recall'].mean()
                print(f"  {method_name}: F1={avg_f1_recovery:.3f}, Precision={avg_precision:.3f}, Recall={avg_recall:.3f}")

print("\nSynthetic data experiment completed!") 