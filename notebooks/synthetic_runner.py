#!/usr/bin/env python3
"""
Synthetic Experiment Runner
Runs all methods on synthetic datasets only (scenarios A, B, C, D, E, etc.)
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Import our modules
from datasets import DATASETS
from loaders import load_any
from sampling import stratified_with_replacement, rebalance_train_indices, get_class_distribution

# Import methods
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.nimo_variants import run_nimo, run_nimo_baseline, run_nimo_transformer
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net


def discover_synthetic_scenarios(synthetic_data_path="../data/synthetic"):
    """
    Dynamically discover all available synthetic scenarios by scanning the data directory.

    Args:
        synthetic_data_path: Path to synthetic data directory

    Returns:
        list: List of discovered synthetic dataset configurations
    """
    synthetic_scenarios = []
    data_path = Path(synthetic_data_path)

    if not data_path.exists():
        print(f"Warning: Synthetic data path {data_path} does not exist")
        return []

    # Look for scenario_* directories or files
    pattern_files = list(data_path.glob("scenario_*_X_full.npy"))

    for pattern_file in pattern_files:
        # Extract scenario ID from filename (e.g., "scenario_A_X_full.npy" -> "A")
        scenario_id = pattern_file.stem.split("_")[1]  # scenario_A_X_full -> A

        # Check if all required files exist for this scenario
        required_files = [
            f"scenario_{scenario_id}_X_full.npy",
            f"scenario_{scenario_id}_y_full.npy",
            f"scenario_{scenario_id}_idx_pool.npy",
            f"scenario_{scenario_id}_idx_test_big.npy",
            f"scenario_{scenario_id}_metadata.json"
        ]

        all_files_exist = all((data_path / f).exists() for f in required_files)

        if all_files_exist:
            # Create dataset configuration
            scenario_config = {
                "kind": "synthetic",
                "id": scenario_id,
                "path": str(data_path),
                "n_train": 1400,  # Default values
                "n_val": 600,
                "desc": f"Synthetic scenario {scenario_id} (auto-discovered)"
            }
            synthetic_scenarios.append(scenario_config)
            print(f"  ✓ Discovered synthetic scenario: {scenario_id}")
        else:
            print(f"  ✗ Incomplete scenario {scenario_id}: missing required files")

    # Sort by scenario ID for consistent ordering
    synthetic_scenarios.sort(key=lambda x: x["id"])

    print(f"Discovered {len(synthetic_scenarios)} synthetic scenarios: {[s['id'] for s in synthetic_scenarios]}")
    return synthetic_scenarios


def run_all_methods(X_tr, y_tr, X_va, y_va, X_te, y_te, seed, feature_names, dataset_info=None):
    """
    Run all methods on the given data splits.

    Args:
        X_tr, y_tr: Training data
        X_va, y_va: Validation data
        X_te, y_te: Test data
        seed: Random seed
        feature_names: List of feature names
        dataset_info: Optional dataset metadata

    Returns:
        list: Results from all methods
    """
    methods = [
        ("Lasso", run_lasso),
        #("LassoNet", run_lassonet),
        #("nimo_transformer", run_nimo),
        ("NIMO", run_nimo_transformer),
        ("RF", run_random_forest),
        #("nimo_baseline", run_nimo_baseline),
        #("NN", run_neural_net),
        # ("sparse_neural_net", run_sparse_neural_net),
        # ("sparse_linear_baseline", run_sparse_linear_baseline)
    ]

    results = []

    for method_name, method_func in methods:
        try:
            print(f"    Running {method_name}...")
            start_time = time.time()

            # Run method with consistent interface
            result = method_func(
                X_tr, y_tr, X_te, y_te,
                iteration=0,  # Will be set by caller
                randomState=seed,
                X_columns=feature_names,
                X_val=X_va,
                y_val=y_va
            )

            # Add timing info
            result["training_time"] = time.time() - start_time

            # Add dataset info if provided
            if dataset_info:
                result.update(dataset_info)

            results.append(result)
            print(f"      ✓ {method_name} completed in {result['training_time']:.2f}s")

        except Exception as e:
            print(f"      ✗ Error running {method_name}: {e}")
            # Create error result
            error_result = {
                "model_name": method_name,
                "iteration": 0,  # Will be set by caller
                "random_seed": seed,
                "error": str(e),
                "training_time": 0.0
            }
            if dataset_info:
                error_result.update(dataset_info)
            results.append(error_result)

    return results


def main(n_iterations=30, rebalance_config=None, output_dir="../results/synthetic"):
    """
    Main experiment runner for synthetic datasets only.

    Args:
        n_iterations: Number of iterations per dataset
        rebalance_config: Rebalancing configuration dict
        output_dir: Output directory for results
    """
    # Default rebalancing config
    if rebalance_config is None:
        rebalance_config = {"mode": "undersample", "target_pos": 0.5}

    # Get only synthetic datasets
    all_datasets = discover_synthetic_scenarios()

    print("=" * 80)
    print("SYNTHETIC EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Datasets: {len(all_datasets)} (synthetic only)")
    print(f"Iterations per dataset: {n_iterations}")
    print(f"Rebalancing: {rebalance_config}")
    print(f"NIMO variants: nimo_transformer (updated), nimo_transformer_old (original)")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    start_time = time.time()

    for dataset_idx, entry in enumerate(all_datasets):
        print(f"\n{'=' * 20} DATASET {entry['id']} ({entry['kind']}) {'=' * 20}")
        print(f"Description: {entry.get('desc', 'No description')}")

        try:
            # Load dataset
            X, y, idx_test, idx_pool, meta = load_any(entry)
            X_te, y_te = X[idx_test], y[idx_test]
            feature_names = meta["feature_names"]

            print(f"Data shape: {X.shape}")
            print(f"Test set: {len(idx_test)} samples")
            print(f"Pool: {len(idx_pool)} samples")
            print(f"Features: {len(feature_names)}")

            # Dataset info for results
            dataset_info = {
                "data_type": entry["kind"],
                "dataset_id": entry["id"],
                "dataset_description": entry.get("desc", ""),
                "n_features_total": X.shape[1],
                "n_test_samples": len(idx_test),
                "n_pool_samples": len(idx_pool)
            }

            # Add unified scenario-like metadata for plotting compatibility
            scenario_descriptions = {
                'A': 'Linear (low-dim, 20 features)',
                'B': 'Linear (high-dim, 200 features)',
                'C': 'Linear + univariate nonlinearity (low-dim, 20 features)',
                'D': 'Linear + interactions + nonlinearity (high-dim, 200 features)',
                'E': 'Purely nonlinear (medium-dim, 50 features)',
                'F': 'High-dimensional with four interactions',
                'G': 'Medium-dimensional with complex interactions',
                'H': 'Low-dimensional with noise',
                'I': 'High-dimensional with sparsity'
            }

            # Add scenario-like fields for unified plotting
            dataset_info.update({
                "scenario": entry["id"],  # Use dataset_id as scenario
                "scenario_description": scenario_descriptions.get(entry["id"], entry.get("desc", "")),
                "scenario_title": f"Scenario {entry['id']}: {scenario_descriptions.get(entry['id'], entry.get('desc', ''))}"
            })

            # Add true support info for synthetic data
            true_support = meta.get("true_support", [])
            beta_nonzero = meta.get("beta_nonzero", {})
            n_features = meta.get("p", len(true_support))
            
            # Create full-length beta_true vector
            beta_true_full = [0.0] * n_features
            for idx, val in beta_nonzero.items():
                if 0 <= int(idx) < n_features:
                    beta_true_full[int(idx)] = float(val)
            
            dataset_info.update({
                "n_true_features": len(true_support),
                "true_support": json.dumps(true_support),
                "beta_true": json.dumps(beta_true_full),
                "b0_true": meta.get("b0", 0.0)
            })

            # Run iterations
            for it in range(n_iterations):
                print(f"\n  --- Iteration {it + 1}/{n_iterations} ---")

                seed = 42 + 1000 * it

                # Sample train/val with replacement
                idx_tr, idx_va = stratified_with_replacement(
                    y, idx_pool,
                    entry["n_train"], entry["n_val"],
                    seed
                )

                # Rebalance training set if requested
                if rebalance_config["mode"] != "none":
                    idx_tr = rebalance_train_indices(
                        y, idx_tr,
                        mode=rebalance_config["mode"],
                        target_pos=rebalance_config["target_pos"],
                        seed=seed
                    )

                # Get data splits
                X_tr, y_tr = X[idx_tr], y[idx_tr]
                X_va, y_va = X[idx_va], y[idx_va]

                # Print class distributions
                train_dist = get_class_distribution(y, idx_tr)
                val_dist = get_class_distribution(y, idx_va)
                print(f"    Train: {train_dist['n_total']} samples ({train_dist['pos_proportion']:.1%} pos)")
                print(f"    Val: {val_dist['n_total']} samples ({val_dist['pos_proportion']:.1%} pos)")

                # Run all methods
                iteration_results = run_all_methods(
                    X_tr, y_tr, X_va, y_va, X_te, y_te,
                    seed, feature_names, dataset_info
                )

                # Add iteration info to all results
                for result in iteration_results:
                    result["iteration"] = it
                    result["random_seed"] = seed

                all_results.extend(iteration_results)

        except Exception as e:
            print(f"✗ Error loading dataset {entry['id']}: {e}")
            continue

    # Save results
    df = pd.DataFrame(all_results)
    output_file = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(output_file, index=False)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_iterations": n_iterations,
        "rebalance_config": rebalance_config,
        "n_datasets": len(all_datasets),
        "n_results": len(all_results),
        "datasets": all_datasets,
        "total_time": time.time() - start_time
    }

    with open(os.path.join(output_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SYNTHETIC EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total results: {len(all_results)}")
    print(f"Datasets processed: {len(all_datasets)}")
    print(f"Iterations per dataset: {n_iterations}")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print(f"Results saved to: {output_file}")

    # Print per-dataset summary
    if len(all_results) > 0:
        print(f"\nPer-dataset summary:")
        for dataset_id in df['dataset_id'].unique():
            dataset_results = df[df['dataset_id'] == dataset_id]
            if 'error' in df.columns:
                n_success = len(dataset_results[dataset_results['error'].isna()])
            else:
                n_success = len(dataset_results)
            n_total = len(dataset_results)
            print(f"  {dataset_id}: {n_success}/{n_total} successful runs")

    # Print per-method summary
    print(f"\nPer-method summary:")
    for method_name in df['model_name'].unique():
        method_results = df[df['model_name'] == method_name]
        if 'error' in df.columns:
            n_success = len(method_results[method_results['error'].isna()])
            successful_results = method_results[method_results['error'].isna()]
        else:
            n_success = len(method_results)
            successful_results = method_results
        n_total = len(method_results)
        avg_f1 = successful_results['f1'].mean() if n_success > 0 and 'f1' in successful_results.columns else 0
        print(f"  {method_name}: {n_success}/{n_total} successful runs, avg F1: {avg_f1:.3f}")

    print(f"\n✓ Synthetic experiment completed successfully!")
    return df


if __name__ == "__main__":
    # Run with default settings
    df = main(n_iterations=2)  # Start with 1 iteration for testing
