#!/usr/bin/env python3
"""
Clean Synthetic Data Experiment Runner
Loads pre-generated synthetic data and runs experiments.
Uses fixed big test set and stratified sampling from pool.
"""

import numpy as np
import pandas as pd
import os
import json
import glob

# Import methods
from methods.lasso import run_lasso
from methods.nimo_variants.nimo import run_nimo
from methods.lasso_Net import run_lassonet
from methods.utils import standardize_method_output

# Constants for easy tuning
N_ITERATIONS = 2
N_TRAIN = 140
N_VAL = 60

def harmonize_result(res):
    """Unify method outputs to common field structure and remove redundant fields."""
    # unify metrics - ensure f1 and accuracy are present
    if "f1" in res:
        res["f1"] = float(res["f1"])
    elif "metrics" in res and "f1" in res["metrics"]:
        res["f1"] = float(res["metrics"]["f1"])
    
    if "accuracy" in res:
        res["accuracy"] = float(res["accuracy"])
    elif "metrics" in res and "accuracy" in res["metrics"]:
        res["accuracy"] = float(res["metrics"]["accuracy"])
    
    # ensure threshold is present
    if "threshold" in res:
        res["threshold"] = float(res["threshold"])

    # unify selection - ensure selected_features and n_selected are present
    if (not res.get("selected_features") or res.get("selected_features") == []) and "selection" in res and "features" in res["selection"]:
        res["selected_features"] = res["selection"]["features"]
    if res.get("n_selected") is None:
        res["n_selected"] = len(res.get("selected_features", []))

    # Remove redundant/unnecessary fields to clean up the output
    fields_to_remove = [
        "best_f1", "best_threshold", "metrics", "selection", 
        "decomposition_val", "decomposition_test", "correction_stats_val", 
        "no_harm_val", "training", "method_has_selection",
        "best_lambda", "best_cv_score", "coefficients_std"
    ]
    
    for field in fields_to_remove:
        res.pop(field, None)

    return res

def load_synthetic_data(scenario, data_dir="../data/synthetic"):
    """Load pre-generated synthetic data for a scenario."""
    X = np.load(f"{data_dir}/scenario_{scenario}_X_full.npy")
    y = np.load(f"{data_dir}/scenario_{scenario}_y_full.npy")
    idx_test = np.load(f"{data_dir}/scenario_{scenario}_idx_test_big.npy")
    idx_pool = np.load(f"{data_dir}/scenario_{scenario}_idx_pool.npy")
    
    with open(f"{data_dir}/scenario_{scenario}_metadata.json", "r") as f:
        meta = json.load(f)

    # Handle old/new field names
    desc = meta.get("description", meta.get("desc", f"Scenario {scenario}"))
    p = meta.get("p", X.shape[1])

    # Reconstruct full beta if needed
    beta_full = np.zeros(p)
    if "beta_true" in meta:
        beta_full = np.array(meta["beta_true"], dtype=float)
    elif "beta_nonzero" in meta:
        for j, v in meta["beta_nonzero"].items():
            beta_full[int(j)] = float(v)

    true_support = meta.get("true_support", np.where(beta_full != 0)[0].tolist())
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    return X, y, idx_test, idx_pool, desc, feature_names, true_support, beta_full.tolist()

def stratified_sample_from_pool(y, idx_pool, n_train, n_val, seed):
    """Stratified sampling from pool for train/val splits."""
    rng = np.random.default_rng(seed)
    idx_pool = np.asarray(idx_pool)
    y_pool = y[idx_pool]

    idx0 = idx_pool[y_pool == 0]
    idx1 = idx_pool[y_pool == 1]

    def take(k, arr):
        k = min(k, len(arr))
        sel = rng.choice(arr, size=k, replace=False)
        rest = np.setdiff1d(arr, sel, assume_unique=False)
        return sel, rest

    # target per-class counts (proportional split)
    pi = (y_pool.mean() if len(y_pool) else 0.5)
    n_tr1 = int(round(n_train * pi))
    n_tr0 = n_train - n_tr1
    n_va1 = int(round(n_val * pi))
    n_va0 = n_val - n_va1

    tr1, rem1 = take(n_tr1, idx1)
    tr0, rem0 = take(n_tr0, idx0)
    va1, rem1 = take(n_va1, rem1)
    va0, rem0 = take(n_va0, rem0)

    idx_train = np.concatenate([tr0, tr1])
    idx_val = np.concatenate([va0, va1])

    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    return idx_train, idx_val

def calculate_support_recovery_metrics(selected_features, true_support, n_features):
    """Calculate support recovery metrics."""
    if selected_features is None:
        return dict(true_positive_rate=0.0, false_positive_rate=0.0, precision=0.0, recall=0.0, f1_recovery=0.0)

    # allow either names "feature_k" or integer indices
    if len(selected_features) == 0:
        return dict(true_positive_rate=0.0, false_positive_rate=0.0, precision=0.0, recall=0.0, f1_recovery=0.0)

    if isinstance(selected_features[0], str):
        sel_idx = [int(s.split('_')[-1]) for s in selected_features if '_' in s]
    else:
        sel_idx = [int(i) for i in selected_features]

    S_pred, S_true = set(sel_idx), set(int(i) for i in true_support)
    tp = len(S_pred & S_true)
    fp = len(S_pred - S_true)
    fn = len(S_true - S_pred)
    tn_denom = max(n_features - len(S_true), 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_rec = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / tn_denom

    return dict(true_positive_rate=recall, false_positive_rate=fpr,
                precision=precision, recall=recall, f1_recovery=f1_rec)

def main():
    """Run experiments on pre-generated synthetic data."""
    
    # Create results directories
    os.makedirs("../results/synthetic/preds", exist_ok=True)
    os.makedirs("../results/synthetic/features", exist_ok=True)
    os.makedirs("../results/synthetic", exist_ok=True)

    data_dir = "../data/synthetic"
    meta_files = glob.glob(f"{data_dir}/scenario_*_metadata.json")
    scenarios = sorted([os.path.basename(f).split("_")[1] for f in meta_files])

    methods = [run_lasso, run_nimo, run_lassonet] #

    print("="*60)
    print("SYNTHETIC RUNNER (fixed big test set + pool)")
    print("="*60)
    print("Scenarios:", scenarios)
    print("Methods:", [m.__name__ for m in methods])
    print(f"Train/val sizes: {N_TRAIN}/{N_VAL}")
    print()

    all_results = []

    for sc in scenarios:
        print(f"\n{'='*20} SCENARIO {sc} {'='*20}")
        
        # Load data for this scenario
        X, y, idx_test, idx_pool, desc, feat_names, true_support, beta_true = load_synthetic_data(sc, data_dir)
        X_test, y_test = X[idx_test], y[idx_test]
        
        print(f"Description: {desc}")
        print(f"Full data shape: {X.shape}")
        print(f"Test set size: {len(idx_test)}, Pool size: {len(idx_pool)}")
        print(f"True support: {true_support}")
        print()

        for it in range(N_ITERATIONS):
            print(f"  --- Iteration {it} ---")
            
            seed = 42 + 1000*it
            idx_train, idx_val = stratified_sample_from_pool(y, idx_pool, N_TRAIN, N_VAL, seed)
            X_train, y_train = X[idx_train], y[idx_train]
            X_val, y_val = X[idx_val], y[idx_val]

            print(f"    Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
            print(f"    Class distribution (train): {np.bincount(y_train)}")

            for method in methods:
                print(f"    Running {method.__name__}...")
                
                try:
                    res = method(X_train, y_train, X_test, y_test,
                                 iteration=it, randomState=seed, X_columns=feat_names,
                                 X_val=X_val, y_val=y_val)

                    res = standardize_method_output(res)
                    
                    # harmonize first to clean up field names
                    res = harmonize_result(res)
                    
                    # Add only essential fields
                    res.update({
                        "scenario": sc,
                        "scenario_description": desc,
                        "data_type": "synthetic",
                        "iteration": it,
                        "random_seed": seed,
                        "n_features_total": X.shape[1],
                        "n_true_features": len(true_support),
                        "true_support": json.dumps(true_support),
                        "beta_true": json.dumps(beta_true),
                    })

                    # support-recovery metrics (only f1_recovery is essential)
                    rec = calculate_support_recovery_metrics(
                        res.get("selected_features"), true_support, X.shape[1]
                    )
                    res["f1_recovery"] = rec["f1_recovery"]  # Only keep the essential metric

                    # persist preds/features if present
                    if "y_prob" in res:
                        np.save(f"../results/synthetic/preds/{res['model_name']}_S{sc}_it{it}_probs.npy",
                                np.array(res["y_prob"]))
                    if "y_pred" in res:
                        np.save(f"../results/synthetic/preds/{res['model_name']}_S{sc}_it{it}_preds.npy",
                                np.array(res["y_pred"]))
                    if "selected_features" in res:
                        with open(f"../results/synthetic/features/{res['model_name']}_S{sc}_it{it}.json", "w") as f:
                            json.dump(res["selected_features"], f)
                    
                    # save coefficients if present
                    coefs = res.get("coef_all") or (res.get("coefficients", {}).get("values"))
                    if coefs is not None:
                        with open(f"../results/synthetic/features/{res['model_name']}_S{sc}_it{it}_coefs.json", "w") as f:
                            json.dump(coefs, f)

                    all_results.append(res)
                    print(f"      {method.__name__} completed successfully")

                except Exception as e:
                    print(f"      Error running {method.__name__}: {e}")
                    all_results.append({
                        "model_name": method.__name__,
                        "scenario": sc, 
                        "iteration": it, 
                        "error": str(e),
                        "scenario_description": desc, 
                        "data_type": "synthetic",
                        "random_seed": seed,
                    })

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv("../results/synthetic/clean_experiment_results.csv", index=False)
    
    print("\n" + "="*60)
    print("CLEAN EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total results: {len(all_results)}")
    print(f"Scenarios tested: {len(scenarios)}")
    print(f"Methods tested: {len(methods)}")
    print(f"Iterations per scenario: {N_ITERATIONS}")
    print(f"Results saved to: ../results/synthetic/clean_experiment_results.csv")
    
    # Print summary statistics
    if "best_f1" in df.columns:
        print("\nAverage F1 Scores by Scenario and Method:")
        f1_summary = df.groupby(["scenario", "model_name"])["best_f1"].agg(["mean", "std"]).round(4)
        print(f1_summary)
    
    if "accuracy" in df.columns:
        print("\nAverage Accuracy by Scenario and Method:")
        acc_summary = df.groupby(["scenario", "model_name"])["accuracy"].agg(["mean", "std"]).round(4)
        print(acc_summary)
    
    if "n_selected" in df.columns:
        print("\nAverage Features Selected by Scenario and Method:")
        selection_summary = df.groupby(["scenario", "model_name"])["n_selected"].agg(["mean", "std"]).round(2)
        print(selection_summary)
    
    if "f1_recovery" in df.columns:
        print("\nSupport Recovery F1 by Scenario and Method:")
        recovery_summary = df.groupby(["scenario", "model_name"])["f1_recovery"].agg(["mean", "std"]).round(4)
        print(recovery_summary)
    
    print("\nClean synthetic data experiment completed!")

if __name__ == "__main__":
    main()