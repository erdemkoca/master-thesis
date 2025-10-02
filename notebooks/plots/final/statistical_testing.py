#!/usr/bin/env python3
"""
Statistical Testing: Paired Wilcoxon Tests per Scenario
======================================================

This script performs paired Wilcoxon signed-rank tests for all model pairs
within each scenario (A-E), providing p-values and effect sizes.

Usage:
    python statistical_testing.py

Output:
    - Console tables with p-values and interpretations
    - CSV files for each scenario
    - LaTeX tables for thesis inclusion
    - Summary across scenarios
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats

# --- CONFIG ---
PRIMARY_METRIC = "f1"
SAVE_TABLES = True
OUT_DIR = Path("wilcoxon_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- HELPERS ---
def significance_stars(p):
    """Convert p-value to significance stars."""
    if pd.isna(p): 
        return "N/A"
    if p < 1e-3:   
        return "***"
    if p < 1e-2:   
        return "**"
    if p < 5e-2:   
        return "*"
    return "ns"

def paired_wilcoxon_table(df_scenario, metric="f1"):
    """
    Returns a DataFrame with Wilcoxon signed-rank results for all model pairs
    within one scenario (dataset_id). Uses only iterations present for both models.
    """
    # Ensure we have iteration, model_name, metric
    assert {"iteration", "model_name", metric}.issubset(df_scenario.columns)

    # Build iteration x model pivot to align pairs
    piv = df_scenario.pivot_table(index="iteration", columns="model_name", values=metric)
    models = list(piv.columns)

    rows = []
    for m1, m2 in combinations(models, 2):
        # Align and drop rows with NaNs in either model
        pair_df = piv[[m1, m2]].dropna()
        if pair_df.empty:
            continue
        x = pair_df[m1].values
        y = pair_df[m2].values
        diffs = x - y

        # Wilcoxon needs non-zero diffs; if all zeros -> p = 1.0, N=0 (or skip)
        nonzero = diffs != 0
        N_nonzero = int(nonzero.sum())
        if N_nonzero == 0:
            p_w = 1.0
            med = 0.0
        else:
            # two-sided Wilcoxon on paired samples
            try:
                stat, p_w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                # Fallback if something weird (e.g., constant); treat as no diff
                p_w = 1.0
            med = float(np.median(diffs))

        rows.append({
            "pair": f"{m1} vs {m2}",
            "N_nonzero": N_nonzero,
            "median_delta_f1": med,
            "wilcoxon_p": p_w,
            "sig": significance_stars(p_w),
            "interpretation": f"{m1} {'>' if med>0 else '<' if med<0 else '='} {m2}"
        })

    if not rows:
        return pd.DataFrame(columns=["pair","N_nonzero","median_delta_f1","wilcoxon_p","sig","interpretation"])
    res = pd.DataFrame(rows).sort_values("wilcoxon_p")
    return res

def print_table(df_results, scenario_id):
    """Print a formatted table of results for one scenario."""
    print("\n" + "="*80)
    print(f"WILCOXON P-VALUES — Scenario {scenario_id}")
    print("="*80)
    if df_results.empty:
        print("No comparable pairs (after aligning iterations).")
        return
    print(f"{'Comparison':<28} {'N':>3} {'Median ΔF1':>12} {'p':>10} {'sig':>5}   Interpretation")
    print("-"*80)
    for _, r in df_results.iterrows():
        print(f"{r['pair']:<28} {int(r['N_nonzero']):>3} {r['median_delta_f1']:>12.4f} {r['wilcoxon_p']:>10.6f} {r['sig']:>5}   {r['interpretation']}")

def main():
    """Main function to run statistical tests."""
    print("="*80)
    print("STATISTICAL TESTING: PAIRED WILCOXON TESTS PER SCENARIO")
    print("="*80)
    print(f"Metric: {PRIMARY_METRIC}")
    print(f"Output directory: {OUT_DIR}")
    print("="*80)
    
    # Load data
    results_path = '../../../results/all/experiment_results.csv'
    df = pd.read_csv(results_path)
    
    # Filter dynamically from whatever is in the results file
    synthetic_ids = sorted(df['dataset_id'].unique().tolist())
    df_synthetic = df[df['dataset_id'].isin(synthetic_ids)].copy()
    
    print(f"Loaded {len(df_synthetic)} synthetic experiments")
    print(f"Methods: {df_synthetic['model_name'].unique()}")
    print(f"Scenarios: {synthetic_ids}")
    
    # --- RUN PER SCENARIO (DYNAMIC) ---
    all_summaries = []
    for scen in synthetic_ids:
        df_scen = df_synthetic[df_synthetic["dataset_id"] == scen].copy()
        res = paired_wilcoxon_table(df_scen, metric=PRIMARY_METRIC)
        print_table(res, scen)

        if SAVE_TABLES and not res.empty:
            # Save CSV
            res.to_csv(OUT_DIR / f"wilcoxon_{scen}.csv", index=False)
            
            # Optional LaTeX export (for direct paste into thesis)
            latex = res[["pair","N_nonzero","median_delta_f1","wilcoxon_p","sig","interpretation"]].to_latex(
                index=False, float_format="%.6f",
                caption=f"Paired Wilcoxon (two-sided) on {PRIMARY_METRIC} for scenario {scen}.",
                label=f"tab:wilcoxon_{scen}"
            )
            with open(OUT_DIR / f"wilcoxon_{scen}.tex", "w") as f:
                f.write(latex)
            
            print(f"  → Saved: {OUT_DIR / f'wilcoxon_{scen}.csv'}")
            print(f"  → Saved: {OUT_DIR / f'wilcoxon_{scen}.tex'}")

        # For a tiny across-scenarios summary (descriptive)
        if not res.empty:
            # Count wins by sign of median ΔF1 for each pair
            wins = (res["median_delta_f1"] > 0).sum()
            losses = (res["median_delta_f1"] < 0).sum()
            ties = (res["median_delta_f1"] == 0).sum()
            all_summaries.append({
                "scenario": scen, 
                "pairs": len(res), 
                "wins": wins, 
                "ties": ties, 
                "losses": losses
            })

    # Optional: quick descriptive summary across scenarios (no pooling p-values)
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        print("\n" + "="*80)
        print("ACROSS-SCENARIOS (DESCRIPTIVE) — win/tie/loss counts by pair (sign of median ΔF1)")
        print("="*80)
        print(summary_df.to_string(index=False))
        if SAVE_TABLES:
            summary_df.to_csv(OUT_DIR / "wilcoxon_summary_across_scenarios.csv", index=False)
            print(f"  → Saved: {OUT_DIR / 'wilcoxon_summary_across_scenarios.csv'}")
    
    print("\n" + "="*80)
    print("STATISTICAL TESTING COMPLETE")
    print("="*80)
    print("Files saved in:", OUT_DIR.absolute())
    print("Use the LaTeX files for direct inclusion in your thesis!")

if __name__ == "__main__":
    main()
