#!/usr/bin/env python3
"""
Synthetic Data Experiment Runner - Enhanced Version
Supports multiple scenarios with different data characteristics.
"""

import numpy as np
import pandas as pd
import os
import json
import hashlib
from scipy.special import expit
from scipy.stats import multivariate_normal, t

# Import all methods
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net
from methods.nimo_variants.baseline import run_nimo_baseline
from methods.nimo_variants.variant import run_nimo_variant
from methods.nimo_variants.nimo import run_nimo
from methods.utils import to_native, standardize_method_output
from methods.neural_net2 import run_neural_net2

def _md5_bytes(*arrays):
    """Generate stable MD5 hash from numpy arrays"""
    m = hashlib.md5()
    for a in arrays:
        a_c = np.ascontiguousarray(a)
        m.update(a_c.view(np.uint8))
    return m.hexdigest()


# Create results directories
os.makedirs("../../results/synthetic/", exist_ok=True)
os.makedirs("../../results/synthetic/preds/", exist_ok=True)
os.makedirs("../../results/synthetic/features/", exist_ok=True)
os.makedirs("../../results/synthetic/convergence/", exist_ok=True)

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
        'logit_noise_std': 0.1,  # kontrolliertes Rauschen
        'beta_scale': 5.0,        # Skalierung für β
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
        'logit_noise_std': 0.15,  # etwas mehr Rauschen für Interaktionen
        'beta_scale': 4.0,        # Skalierung für β
        'description':       'Feature interactions',
        'description_long':  'Linear model with feature interactions'
    },
    'C': {
        'n_samples': 200,
        'n_features': 25,
        'n_true_features': 0,     # Keine linearen Features
        'interactions': None,
        'nonlinear': (
            [('sin', i) for i in range(5)]
          + [('square', i) for i in range(5,10)]
          + [('cube', i) for i in range(10,15)]
        ),
        'rho': 0.0,
        'noise': 'gaussian',
        'logit_noise_std': 0.2,   # mehr Rauschen für reine Nichtlinearität
        'beta_scale': 0.0,        # Keine linearen β (da n_true_features=0)
        'description':      'Pure nonlinear',
        'description_long': 'No linear β·X. 5×sin(X), 5×X², 5×X³ + Gaussian noise'
    },
    # 'D': {
    #     'n_samples': 100,
    #     'n_features': 20,
    #     'n_true_features': 5,
    #     'interactions': [(0,1), (2,3), (4,5,6)],
    #     'nonlinear': [('sin', 2)],
    #     'rho': 0.95,
    #     'noise': 'gaussian_heavy',
    #     'description':       'Hard Interactions + corr.',
    #     'description_long':  'Complex model w/ interactions, ρ=0.95, two interactions + 3‑way, heavy noise '
    # },
    # 'E': {
    #     'n_samples': 200,
    #     'n_features': 20,
    #     'n_true_features': 5,
    #     'interactions': [(0,1)],
    #     'nonlinear': None,
    #     'rho': 0.0,
    #     'noise': 'student_t',
    #     'description':       'Heavy‑tailed noise',
    #     'description_long':  'Linear model with heavy‑tailed (Student‑t) noise'
    # },
    # 'F': {
    #     'n_samples': 300,
    #     'n_features': 20,           # hier ziehen wir 20 rohe X
    #     'n_true_features': 5,
    #     'interactions': None,
    #     'nonlinear': [ # Wir hängen 5 sehr hochfrequente Sinus‑Basisfunktionen an
    #         ('sin_highfreq', k, freq)
    #         for k, freq in zip(range(5), [5, 7, 9, 11, 13])
    #     ],
    #     'rho': 0.0,
    #     'noise': 'gaussian_heavy',
    #     'description':       'High‑freq sin (hidden)',
    #     'description_long':  (
    #         'Draw X ~ N(0,I), then add 5 columns sin(freq·X_k) '
    #         'with frequencies [5,7,9,11,13] + heavy Gaussian noise'
    #     )
    # },
    # 'G': {
    #     'n_samples': 500,
    #     'n_features': 10,
    #     'n_true_features': 2,
    #     'interactions': [(0,1), (2,3)],
    #     'nonlinear': None,
    #     'rho': 0.0,
    #     'noise': 'gaussian',
    #     'description':       'XOR interactions',
    #     'description_long':  'Signal = x₀⊕x₁ + x₂⊕x₃ + Gaussian noise'
    # },
    # 'H': {
    #     'n_samples': 300,
    #     'n_features': 15,
    #     'n_true_features': 3,
    #     'interactions': None,
    #     'nonlinear': [('square', 0), ('cube', 1)],
    #     'rho': 0.9,
    #     'noise': 'gaussian',
    #     'description':       'Polynom + corr.',
    #     'description_long':  'ρ=0.9, Signal = β₀ x₀² + β₁ x₁³ + Gaussian noise'
    # },
    # 'I': {
    #     'n_samples': 300,
    #     'n_features': 20,
    #     'n_true_features': 5,
    #     'interactions': None,
    #     'nonlinear': None,
    #     'custom': 'rbf',             # RBF kernel features
    #     'rho': 0.0,
    #     'noise': 'gaussian',
    #     'description':      'RBF-Kernel',
    #     'description_long': 'y = sigmoid( exp(-||x - mu||^2 / sigma^2) ) + noise'
    # },
    # 'J': {
    #     'n_samples': 800,
    #     'n_features': 15,
    #     'n_true_features': 5,
    #     'interactions': None,
    #     'nonlinear': 'sawtooth',
    #     'rho': 0.0,
    #     'noise': 'gaussian',
    #     'description':      'Piecewise Sägezahn',
    #     'description_long': 'y = sawtooth(x_k) + sum_sin(...) + noise'
    # },
    # 'K': {
    #     'n_samples': 2000,
    #     'n_features': 12,
    #     'n_true_features': 4,
    #     'interactions': [(0,1,2), (3,4,5)],  # 3-way interactions
    #     'nonlinear': None,
    #     'rho': 0.5,
    #     'noise': 'gaussian',
    #     'description':      '3‑Wege Interaktionen',
    #     'description_long': 'y = x0*x1*x2 + x3*x4*x5 + corr + noise'
    # },
    # 'L': {
    #     'n_samples': 50,
    #     'n_features': 500,
    #     'n_true_features': 10,
    #     'interactions': None,
    #     'nonlinear': [('sin', i) for i in range(10)],
    #     'rho': 0.0,
    #     'noise': 'gaussian_heavy',
    #     'description':      'High-dim + noise',
    #     'description_long': 'n < p, high noise, nonlinear signal'
    # },
    # 'M': {
    #     'n_samples': 300,
    #     'n_features': 20,
    #     'n_true_features': 5,
    #     'interactions': [(0,1), (2,3)],
    #     'nonlinear': [('square', 0), ('cube', 1), ('sin', 2)],
    #     'rho': 0.8,
    #     'noise': 'student_t',
    #     'description':      'Complex + heavy noise',
    #     'description_long': 'Interactions + polynomials + correlations + heavy-tailed noise'
    # },
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
                            noise='gaussian', custom=None, seed=42,
                            fixed_support=None, fixed_beta=None, logit_noise_std=0.0):
    """
    Generate synthetic data with various characteristics.

    NEW:
      - fixed_support: array/list der wahren linearen Feature-Indizes (oder None)
      - fixed_beta:    Länge n_features, die wahren linearen Koeffizienten (oder None)
      - logit_noise_std: σ für additiven Gauss-Rauschterm auf den Logits (vor Sigmoid)
    """
    np.random.seed(seed)

    # 1) Merkmalsmatrix X
    if rho > 0:
        cov = generate_toeplitz_covariance(n_features, rho)
        X = multivariate_normal.rvs(
            mean=np.zeros(n_features), cov=cov, size=n_samples
        )
    else:
        X = np.random.randn(n_samples, n_features) * 10.0

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
        # 2b) Standard linearer Anteil: jetzt mit optional fixem Support/Beta
        if fixed_beta is not None:
            beta_true = np.asarray(fixed_beta, dtype=float)
            assert beta_true.shape[0] == n_features
            support = np.nonzero(beta_true)[0]
        else:
            beta_true = np.zeros(n_features)
            if fixed_support is not None:
                support = np.asarray(fixed_support, dtype=int)
            else:
                support = np.random.choice(n_features, size=n_true_features, replace=False)
            beta_true[support] = np.random.uniform(1.0, 3.0, size=len(support))
        linear_predictor = X.dot(beta_true)

        # 3) Interaktionen (nur zwischen wahren Support-Features)
        if interactions:
            coeffs_int = np.random.uniform(0.5, 1.5, size=len(interactions))
            for inter, coeff in zip(interactions, coeffs_int):
                if len(inter) == 2:
                    i, j = inter
                    # Prüfe, ob beide Features im wahren Support sind
                    if i in support and j in support:
                        linear_predictor += coeff * X[:, i] * X[:, j]
                elif len(inter) == 3:
                    i, j, k = inter
                    # Prüfe, ob alle drei Features im wahren Support sind
                    if i in support and j in support and k in support:
                        linear_predictor += coeff * X[:, i] * X[:, j] * X[:, k]

        # 4) Nicht‐Linearitäten (nur auf wahren Support-Features)
        if nonlinear is not None:
            # a) String‑Spec 'sawtooth'
            if isinstance(nonlinear, str) and nonlinear.lower() == 'sawtooth':
                import scipy.signal as sg
                # Sawtooth nur auf Support-Features anwenden
                support_X = X[:, support] if len(support) > 0 else X[:, :0]
                if support_X.size > 0:
                    linear_predictor += np.sum(sg.sawtooth(5 * support_X, width=0.5), axis=1)

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
                        raise ValueError(f"Ungültiges nonlinear‑Spec: {nonlinear!r}")

                    # Prüfe, ob das Feature im wahren Support ist
                    if int(idx) not in support:
                        continue  # Überspringe Features außerhalb des Supports
                    
                    col = X[:, int(idx)]
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

    # 5) Logit-Rauschen (NEU): additiver Gauss-Term vor dem Sigmoid
    if logit_noise_std and logit_noise_std > 0:
        linear_predictor = linear_predictor + np.random.normal(0.0, float(logit_noise_std), size=n_samples)

    # 6) Zielvariable y
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
n_iterations = 4

print("="*60)
print("SYNTHETIC DATA EXPERIMENT RUNNER")
print("="*60)
print(f"Scenarios: {list(scenarios.keys())}")
print(f"Iterations per scenario: {n_iterations}")
print()

# Define methods to run
methods = [
    run_lasso,
    #run_lassonet,
    #run_random_forest,
    #run_neural_net,
    #run_nimo_baseline,
    #run_nimo_variant,
    run_nimo,
    #run_neural_net2
]

print(f"Methods: {[m.__name__ for m in methods]}")
print()

# 2) Loop over scenarios
for scenario_name, params in scenarios.items():
    print(f"\n{'='*20} SCENARIO {scenario_name} {'='*20}")
    print(f"Description: {params['description']}")
    print(f"Parameters: {params}")
    
    # --- GLOBALE Ground-Truth je Szenario festlegen (fix & konstant) ---
    n_features = params['n_features']
    k_true     = params.get('n_true_features', 0)
    beta_scale = params.get('beta_scale', 5.0)  # ggf. in den Szenario-Parametern ergänzbar

    if k_true > 0:
        support_fixed = np.arange(k_true, dtype=int)  # z. B. die ersten k_true
        beta_fixed    = np.zeros(n_features, dtype=float)
        # alternierende Vorzeichen für Klarheit (+ - + - +)
        signs = np.array([1 if i % 2 == 0 else -1 for i in range(k_true)], dtype=float)
        beta_fixed[support_fixed] = beta_scale * signs
    else:
        support_fixed = np.array([], dtype=int)
        beta_fixed    = np.zeros(n_features, dtype=float)

    # kontrollierte Logit-Rauschstärke (klein halten in A)
    logit_noise_std = float(params.get('logit_noise_std', 0.1))
    
    # Generate data for this scenario
    # Remove description, beta_scale, and logit_noise_std from params for function call
    data_params = {
        k: v
        for k, v in params.items()
        if not k.startswith('description') and k != 'beta_scale' and k != 'logit_noise_std'
    }
    # Generate initial data for scenario description (mit festen β)
    X_full_init, y_full_init, true_support, beta_true = generate_synthetic_data(
        **data_params,
        seed=np.random.randint(0, 2**31-1),
        fixed_support=support_fixed,
        fixed_beta=beta_fixed,
        logit_noise_std=logit_noise_std
    )
    X_train_init, y_train_init, X_test_init, y_test_init = split_data(X_full_init, y_full_init, test_size=0.3, seed=np.random.randint(0, 2**31-1))
    
    print(f"Generated data:")
    print(f"  Total samples: {len(X_full_init)}")
    print(f"  Features: {X_full_init.shape[1]}")
    print(f"  True support features (FIXED): {sorted(true_support)}")
    print(f"  Fixed support: {support_fixed.tolist()}")
    print(f"  Train samples: {len(X_train_init)}")
    print(f"  Test samples: {len(X_test_init)}")
    print(f"  Class distribution (train): {np.bincount(y_train_init)}")
    print(f"  Class distribution (test): {np.bincount(y_test_init)}")
    
    # Create feature column names for synthetic data
    X_columns = [f'feature_{i}' for i in range(X_full_init.shape[1])]
    
    # Datenparameter fuer das Szenario (ohne description, beta_scale, logit_noise_std)
    data_params = {k:v for k,v in params.items() if not k.startswith('description') and k != 'beta_scale' and k != 'logit_noise_std'}

    # Fixes Testset erzeugen und cachen
    seed_test  = int(np.random.randint(0, 2**31-1))
    test_params = {k: v for k, v in params.items() if not k.startswith('description') and k != 'beta_scale' and k != 'logit_noise_std'}
    test_params['n_samples'] = 50_000
    X_test_fixed, y_test_fixed, true_support_test, beta_true_test = generate_synthetic_data(
        **test_params,
        seed=seed_test,
        fixed_support=support_fixed,
        fixed_beta=beta_fixed,
        logit_noise_std=logit_noise_std
    )

    test_hash = hash(X_test_fixed.tobytes()) ^ hash(y_test_fixed.tobytes())

    # Inner loop: iterations
    for iteration in range(n_iterations):
        print(f"\n  --- Iteration {iteration} ---")
        
        # Generate completely random seed for this iteration (no reproducibility)
        seed_iter = int(np.random.randint(0, 2**31-1))
        
        # Pro Iteration: Train/Val auch mit festen β generieren
        X_full_iter, y_full_iter, true_support_iter, beta_true_iter = generate_synthetic_data(
            **data_params,
            seed=seed_iter,
            fixed_support=support_fixed,
            fixed_beta=beta_fixed,
            logit_noise_std=logit_noise_std
        )

        # Stratifizierter Train/Val-Split (z.B. 70/30)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed_iter)
        (train_idx, val_idx), = sss.split(X_full_iter, y_full_iter)

        X_train_iter, y_train_iter = X_full_iter[train_idx], y_full_iter[train_idx]
        X_val_iter,   y_val_iter   = X_full_iter[val_idx],   y_full_iter[val_idx]

        # Hashes fuer Debug/Variation
        train_idx_hash = hash(train_idx.tobytes())
        val_idx_hash   = hash(val_idx.tobytes())
        
        # True-Support pro Iteration aufbauen (linear & total)
        # Linearer Support ist true_support_iter (kommt vom Generator)
        true_linear_iter = sorted(map(int, true_support_iter))

        # Totaler Support = linear ∪ alle Indizes aus interactions ∪ alle Indizes aus nonlinear
        true_total_iter = set(true_linear_iter)
        if params.get('interactions'):
            for inter in params['interactions']:
                for idx in inter: true_total_iter.add(int(idx))
        if params.get('nonlinear'):
            for spec in params['nonlinear']:
                # spec: (typ, feature_idx[, extra])
                idx = int(spec[1])
                true_total_iter.add(idx)
        true_total_iter = sorted(true_total_iter)
        
        # Use fresh data for this iteration
        X_sub = X_train_iter.copy()
        y_sub = y_train_iter.copy()
        
        # Use iteration-specific seed for reproducibility
        randomState = seed_iter
        
        # Method loop
        for method_fn in methods:
            print(f"    Running {method_fn.__name__}...")
            
            try:
                print(f"      Calling {method_fn.__name__}...")
                # Erweitere den Call nur für LASSO um optionale Debug-Flags
                if method_fn is run_lasso:
                    result = method_fn(
                        X_train_iter, y_train_iter,
                        X_test_fixed, y_test_fixed,
                        iteration,
                        seed_iter,
                        X_columns,
                        X_val=X_val_iter, y_val=y_val_iter,
                        no_cv=False,            # True setzen, wenn du den Debug-Pfad testen willst
                        C_fixed=None,           # z.B. 1e6 im Debug
                        return_train_fit=True
                    )
                else:
                    # unverändert für andere Methoden
                    result = method_fn(
                        X_train_iter, y_train_iter,
                        X_test_fixed, y_test_fixed,                 # fixes Testset fuer alle
                        iteration,
                        seed_iter,
                        X_columns,
                        X_val=X_val_iter, y_val=y_val_iter          # NEU: Validation fuer Threshold
                    )
                print(f"      {method_fn.__name__} returned successfully")
                
                # Ensure all core fields exist and convert to native types
                result = ensure_core_fields(result)
                
                # Add feature names for this iteration
                result['feature_names'] = json.dumps([f'feature_{i}' for i in range(X_full_iter.shape[1])])
                
                # Seeds & Hashes
                result['seed_iter'] = int(seed_iter)
                result['train_idx_hash'] = int(train_idx_hash)
                result['val_idx_hash'] = int(val_idx_hash)
                result['test_hash'] = int(test_hash)
                
                # Add debug columns for randomness verification
                result['dataset_hash'] = _md5_bytes(X_sub, y_sub)
                
                # Calculate support recovery metrics
                if result['selected_features']:
                    recovery_metrics = calculate_support_recovery_metrics(
                        result['selected_features'], 
                        true_linear_iter, 
                        X_full_iter.shape[1]
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
                    'true_support': json.dumps(true_linear_iter),          # linear
                    'true_support_total': json.dumps(true_total_iter),     # linear ∪ inter ∪ nonlin
                    'n_true_features': int(len(true_linear_iter)),
                    'n_true_features_total': int(len(true_total_iter)),
                    'beta_true': json.dumps([float(b) for b in beta_true_iter.tolist()]),
                    'data_type': 'synthetic',
                    'n_features_total': int(X_full_iter.shape[1])
                }
                
                # Add detailed truth metadata: linear, interaction, nonlinear and union
                true_interaction_idx = set()
                if params.get('interactions'):
                    for tpl in params['interactions']:
                        if isinstance(tpl, (list, tuple)):
                            for i in tpl:
                                true_interaction_idx.add(int(i))

                true_nonlinear_idx = set()
                if params.get('nonlinear') and isinstance(params['nonlinear'], (list, tuple)):
                    for spec in params['nonlinear']:
                        # ('sin', i) bzw. ('sin_highfreq', i, freq) etc.
                        if isinstance(spec, (list, tuple)) and len(spec) >= 2:
                            true_nonlinear_idx.add(int(spec[1]))

                true_union_idx = sorted(set(true_support_iter) | true_interaction_idx | true_nonlinear_idx)

                metadata.update({
                    'true_support_linear': json.dumps(sorted(int(i) for i in true_support_iter)),
                    'true_support_interaction': json.dumps(sorted(int(i) for i in true_interaction_idx)) if true_interaction_idx else None,
                    'true_support_nonlinear': json.dumps(sorted(int(i) for i in true_nonlinear_idx)) if true_nonlinear_idx else None,
                    'true_support_total': json.dumps([int(i) for i in true_union_idx]),
                })
                
                # Logging erweitern (Wahrheiten & Noise)
                metadata.update({
                    'true_support_fixed': json.dumps([int(i) for i in support_fixed.tolist()]),
                    'beta_true_fixed': json.dumps([float(b) for b in beta_fixed.tolist()]),
                    'logit_noise_std': float(logit_noise_std)
                })
                
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
                
                # Mini-Guards
                assert 'best_f1' in result
                assert 'selected_features' in result or 'coef_all' in result
                assert 'true_support' in result and 'true_support_total' in result
                
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
                            # Simple conversion to native types
                            if isinstance(v, np.ndarray):
                                fixed_result[k] = v.tolist()
                            elif isinstance(v, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                                fixed_result[k] = int(v)
                            elif isinstance(v, (np.floating, np.float64, np.float32, np.float16)):
                                fixed_result[k] = float(v)
                            else:
                                fixed_result[k] = v
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
results_df.to_csv("../results/synthetic/lasso_nimo_coefficients.csv", index=False)
print(f"\nAll experiment results saved to ../results/synthetic/lasso_nimo_coefficients.csv")

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