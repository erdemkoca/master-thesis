#!/usr/bin/env python3
"""
Clean Synthetic Data Generator - NIMO-inspired
Generates simple, focused synthetic datasets for LASSO vs NIMO comparison.
Uses fixed big test sets and stratified sampling.
"""

import numpy as np
import json
import os
import shutil
from scipy.special import expit

# ===== Distribution-based scenarios for method comparison =====
SCENARIOS = {
    "A": {  # Linear baseline
        "p": 5, "sigma": 0.1, "b0": 1.0,
        "beta": {0: 2.0, 1: -3.0, 2: 1.5, 3: -2.0},
        "nl": [],
        "dist": ("normal", 0, 1),
        "desc": "Scenario A: Purely linear baseline (N(0,1))"
    },
    "B": {  # Main effect + strong interaction
        "p": 5, "sigma": 0.1, "b0": 0.5,
        # added x2 baseline as well
        "beta": {0: 1.5, 1: 0.5, 2: 0.5, 3: -0.5},
        "nl": [
            ("main_int_linear", 1, 2, 2.0, 15.0)
        ],
        "dist": ("normal", 0, 1),
        "desc": "Scenario C: Main effect + strong x1*x2 interaction"
    },
    "C": {
      "p": 3, "sigma": 0.1, "b0": 1.0,
      "beta": {0: 2.0, 1: -2.0},
      "nl": [
        ("main_int_tanh", 0, 1, 0.0,  4.0, 1.0),   # 4*x0*tanh(x1)
        ("main_int_sin",  1, 0, 0.0, -6.0, 2.0),   # -6*x1*sin(2*x0)
        ("main_int_tanh", 1, 0, 0.0, -2.0, 2.0)    # -2*x1*tanh(2*x0)
      ],
      "dist": ("uniform", -3, 3),
      "desc": "Scenario C: matches screenshot equation"
    },
    "D": {
        "p": 10, "sigma": 0.1, "b0": -10.0,  # constant -10
        "beta": {3: 10.0},  # +10 * x4
        "nl": [
            # 10 * x1 * (1 + 2*tanh(2*x2))
            ("main_int_tanh", 0, 1, 1.0, 2.0, 2.0),

            # 10 * x1 * sin(x4)
            ("main_int_sin", 0, 3, 0.0, 10.0, 1.0),

            # 20 * x2 * (1 + 2*cos(2*x1))
            ("main_int_cos", 1, 0, 1.0, 2.0, 2.0),  # you'd need to add main_int_cos handler

            # -20 * x3 * (1 + 2*arctan(x2*x4))
            ("main_int_arctan_prod", 2, 1, 3, -20.0, 1.0, 2.0, 1.0)

        ],
        "dist": ("uniform", -3, 3),
        "desc": "Scenario G: Complex nonlinear interactions with tanh, sin, cos, arctan"
    },
    "E": {
        "p": 3, "sigma": 0.1, "b0": 1.0,  # b0=1 gives the +1 constant
        "beta": {},  # no extra linear terms
        "nl": [
            ("main_int_tanh", 0, 1, 1.0, 2.0, 1.0),      # 2 * x1 * (1 + 2*tanh(x2))
            ("main_int_sin", 1, 0, -2.0, 6.0, 2.0),      # -2 * x2 * (3*sin(2*x1))
            ("main_int_tanh", 1, 0, 0.0, -2.0, 2.0)      # -2 * x2 * (tanh(2*x1))
        ],
        "dist": ("uniform", -3, 3),
        "desc": "Scenario F: matches screenshot equation"
    },
    "F": {
      "p": 50, "sigma": 0.1, "b0": 1.0,
      # linear parts pulled out explicitly:
      "beta": {0: -20.0, 1:  20.0, 3:  30.0, 4: -10.0},

      # set base=0.0 to avoid double-counting; keep the interaction weights:
      "nl": [
        # -20 * x0 * [1 + tanh(x1*x3)]  ->  -20*x0  +  (-20)*x0*tanh(...)
        ("main_int_tanh_prod",   0, 1, 3, 0.0, -20.0, 1.0),

        #  20 * x1 * [1 + (2/pi) * arctan(x3 - x4)]
        # interaction weight = 20*(2/pi) = 40/pi
        ("main_int_arctan_diff", 1, 3, 4, 0.0,  40.0/np.pi, 1.0),

        #  30 * x3 * [1 + tanh(x1 + sin(x4))]
        ("main_int_tanh_x_plus_sin", 3, 1, 4, 0.0, 30.0, 1.0),

        # -10 * x4 * [1 + tanh(0.5 * x0 * x3)]
        ("main_int_tanh_prod",   4, 0, 3, 0.0, -10.0, 0.5)
      ],
      "dist": ("normal", 0, 1),
      "desc": "Scenario F: NIMO Setting 3 (classification)"
    }
}


def gen_data(n, spec):
    """Generate synthetic data according to specification."""
    rng = np.random.default_rng(42)
    p, sigma, b0 = spec["p"], spec["sigma"], spec.get("b0", 0.0)

    # Beta vector
    beta = np.zeros(p)
    for j, v in spec["beta"].items():
        beta[j] = float(v)

    # ---- Feature distribution ----
    dist = spec.get("dist", ("normal", 0, 1))
    if dist[0] == "normal":
        mu, std = dist[1], dist[2]
        X = rng.normal(loc=mu, scale=std, size=(n, p))
    elif dist[0] == "uniform":
        low, high = dist[1], dist[2]
        X = rng.uniform(low, high, size=(n, p))
    elif dist[0] == "t":
        df = dist[1]
        X = rng.standard_t(df, size=(n, p))
    else:
        raise ValueError(f"Unknown distribution {dist[0]}")

    # ---- Linear predictor ----
    eta = b0 + X @ beta

    # ---- Nonlinear terms ----
    for term in spec["nl"]:
        kind = term[0]
        if kind == "int":
            _, i, j, w = term
            eta += float(w) * (X[:, i] * X[:, j])
        elif kind == "int_tanh":
            _, i, j, w, scale = term
            eta += float(w) * (X[:, i] * np.tanh(scale * X[:, j]))
        elif kind == "int_sin":
            _, i, j, w, scale = term
            eta += float(w) * (X[:, i] * np.sin(scale * X[:, j]))

        # ---- deine neuen "main+nonlinear" Varianten ----
        elif kind == "main_int_tanh":
            # (kind, i, j, base, w, scale)
            _, i, j, base, w, scale = term
            eta += X[:, i] * (base + w * np.tanh(scale * X[:, j]))

        elif kind == "main_int_linear":
            # (kind, i, j, base, w)
            _, i, j, base, w = term
            eta += X[:, i] * (base + w * X[:, j])

        elif kind == "main_int_sin":
            # (kind, i, j, base, w, scale)
            _, i, j, base, w, scale = term
            eta += X[:, i] * (base + w * np.sin(scale * X[:, j]))

        # ---- Rest unverändert ----
        elif kind == "sin":
            _, j, w = term
            eta += float(w) * np.sin(X[:, j])
        elif kind == "sin_scaled":
            _, j, w, scale = term
            eta += float(w) * np.sin(scale * X[:, j])
        elif kind == "tanh":
            _, j, w = term
            eta += float(w) * np.tanh(X[:, j])
        elif kind == "int_cos":
            _, i, j, w, scale = term
            eta += float(w) * (X[:, i] * np.cos(scale * X[:, j]))
        elif kind == "sqc":
            _, j, w = term
            xj = X[:, j]
            eta += float(w) * (xj**2 - 1.0)
        elif kind == "main_int_cos":
            # (kind, i, j, base, w, scale)
            _, i, j, base, w, scale = term
            eta += X[:, i] * (base + w * np.cos(scale * X[:, j]))
        elif kind == "main_int_arctan_prod":

            # (kind, i, j, k, factor, base, w, scale)

            _, i, j, k, factor, base, w, scale = term

            eta += factor * X[:, i] * (base + w * np.arctan(scale * (X[:, j] * X[:, k])))

        elif kind == "main_int_tanh_prod":
            # (kind, i, j, k, base, w, scale)
            _, i, j, k, base, w, scale = term
            eta += X[:, i] * (base + w * np.tanh(scale * (X[:, j] * X[:, k])))

        elif kind == "main_int_arctan_diff":
            # (kind, i, j, k, base, w, scale)
            _, i, j, k, base, w, scale = term
            eta += X[:, i] * (base + w * np.arctan(scale * (X[:, j] - X[:, k])))

        elif kind == "main_int_tanh_x_plus_sin":
            # (kind, i, j, k, base, w, scale)  # scale applies to sin
            _, i, j, k, base, w, scale = term
            eta += X[:, i] * (base + w * np.tanh(X[:, j] + np.sin(scale * X[:, k])))

        else:
            raise ValueError(f"Unknown nonlinear term type: {kind}")

    # ---- Noise + Labels ----
    eta += rng.normal(0.0, sigma, size=n)
    y = rng.binomial(1, expit(eta))

    true_support = np.where(beta != 0)[0].tolist()
    return X, y, beta, true_support

def make_fixed_test_indices(y, test_frac=0.5, seed=123):
    """Create stratified fixed big test set; returns (idx_test, idx_pool)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    
    # Get indices for each class
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    # Stratified split
    k0 = int(round(test_frac * len(idx0)))
    k1 = int(round(test_frac * len(idx1)))
    
    idx_test = np.concatenate([idx0[:k0], idx1[:k1]])
    idx_pool = np.concatenate([idx0[k0:], idx1[k1:]])
    
    # Shuffle final indices
    rng.shuffle(idx_test)
    rng.shuffle(idx_pool)
    
    return idx_test, idx_pool

def main():
    """Generate and save all synthetic datasets."""
    
    # Clean output directory
    output_dir = "../data/synthetic"
    if os.path.exists(output_dir):
        print(f"🗑️  Cleaning existing data in {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("CLEAN SYNTHETIC DATA GENERATOR - NIMO-INSPIRED")
    print("="*60)
    print(f"Scenarios: {list(SCENARIOS.keys())}")
    print(f"Output directory: {output_dir}")
    print()

    # Fixed parameters
    n_full = 10000   # Large pool to carve out a big fixed test set
    seed = 42

    for scenario_name, spec in SCENARIOS.items():
        print(f"Generating Scenario {scenario_name}: {spec['desc']}")
        
        # Generate data
        X, y, beta, true_support = gen_data(n_full, spec)
        
        # Create fixed big test set (stratified) + pool for train/val sampling
        idx_test, idx_pool = make_fixed_test_indices(y, test_frac=0.5, seed=seed+7)
        
        # Save arrays and indices
        np.save(f"{output_dir}/scenario_{scenario_name}_X_full.npy", X)
        np.save(f"{output_dir}/scenario_{scenario_name}_y_full.npy", y.astype(int))
        np.save(f"{output_dir}/scenario_{scenario_name}_idx_test_big.npy", idx_test)
        np.save(f"{output_dir}/scenario_{scenario_name}_idx_pool.npy", idx_pool)
        
        # Minimal metadata
        metadata = {
            "scenario": scenario_name,
            "desc": spec["desc"],
            "p": spec["p"],
            "sigma": spec["sigma"],
            "b0": spec.get("b0", 0.0),
            "beta_nonzero": {int(k): float(v) for k, v in spec["beta"].items()},
            "nl": spec["nl"],
            "true_support": true_support,
            "n_full": n_full,
            "sizes": {"test_big": len(idx_test), "pool": len(idx_pool)},
            "class_dist_full": np.bincount(y).tolist()
        }
        
        with open(f"{output_dir}/scenario_{scenario_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved data and metadata for scenario {scenario_name}")
        print(f"    Full: {X.shape[0]} samples, Features: {X.shape[1]}")
        print(f"    Test set: {len(idx_test)} samples, Pool: {len(idx_pool)} samples")
        print(f"    True support: {true_support}")
        print(f"    Beta: {[f'{beta[i]:.1f}' for i in true_support]}")
        print(f"    Class distribution: {np.bincount(y)}")
        print()

    print("="*60)
    print("CLEAN DATA GENERATION COMPLETED")
    print("="*60)
    print(f"All datasets saved to: {output_dir}")
    print("Ready for experiments!")
    print()
    print("Usage in experiment loop:")
    print("  X = np.load('.../scenario_A_X_full.npy')")
    print("  y = np.load('.../scenario_A_y_full.npy')")
    print("  idx_test = np.load('.../scenario_A_idx_test_big.npy')")
    print("  idx_pool = np.load('.../scenario_A_idx_pool.npy')")
    print("  # Sample train/val from idx_pool for each run")

if __name__ == "__main__":
    main()