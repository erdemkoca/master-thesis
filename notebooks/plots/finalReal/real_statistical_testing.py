    #!/usr/bin/env python3
"""
Real Dataset Statistical Testing
Performs paired Wilcoxon signed-rank tests for all model pairs within each real dataset
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os

def significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def paired_wilcoxon_table(df, dataset_id):
    """Create paired Wilcoxon test table for a specific dataset."""
    print(f"\n=== Statistical Testing for Dataset {dataset_id} ===")
    
    # Filter data for this dataset
    dataset_data = df[df['dataset_id'] == dataset_id].copy()
    
    if dataset_data.empty:
        print(f"No data found for dataset {dataset_id}")
        return None
    
    # Get unique methods
    methods = sorted(dataset_data['model_name'].unique())
    n_methods = len(methods)
    
    if n_methods < 2:
        print(f"Need at least 2 methods for comparison, found {n_methods}")
        return None
    
    # Create results matrix
    results = []
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i >= j:  # Only upper triangle + diagonal
                continue
                
            # Get F1 scores for both methods
            scores1 = dataset_data[dataset_data['model_name'] == method1]['f1'].values
            scores2 = dataset_data[dataset_data['model_name'] == method2]['f1'].values
            
            # Ensure same length (should be same iterations)
            min_len = min(len(scores1), len(scores2))
            scores1 = scores1[:min_len]
            scores2 = scores2[:min_len]
            
            if len(scores1) < 2:
                print(f"Not enough data for {method1} vs {method2}")
                continue
            
            # Calculate differences
            diff = scores1 - scores2
            
            # Remove zero differences (required for Wilcoxon test)
            non_zero_diff = diff[diff != 0]
            
            if len(non_zero_diff) < 2:
                print(f"No non-zero differences for {method1} vs {method2}")
                continue
            
            # Perform Wilcoxon test
            try:
                if len(non_zero_diff) >= 2:
                    statistic, p_value = wilcoxon(non_zero_diff, alternative='two-sided')
                else:
                    statistic, p_value = np.nan, np.nan
            except Exception as e:
                print(f"Error in Wilcoxon test for {method1} vs {method2}: {e}")
                statistic, p_value = np.nan, np.nan
            
            # Calculate median difference
            median_diff = np.median(diff)
            
            # Store results
            results.append({
                'Method1': method1,
                'Method2': method2,
                'Median_Delta_F1': f"{median_diff:.4f}",
                'P_Value': f"{p_value:.6f}" if not np.isnan(p_value) else "N/A",
                'Significance': significance_stars(p_value) if not np.isnan(p_value) else "N/A",
                'N_Comparisons': len(non_zero_diff)
            })
    
    if not results:
        print("No valid comparisons found")
        return None
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Print table
    print(f"\nPaired Wilcoxon Signed-Rank Test Results for Dataset {dataset_id}:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    return results_df

def print_table(results_df, dataset_id):
    """Print formatted table."""
    if results_df is None or results_df.empty:
        return
    
    print(f"\n{'='*20} Dataset {dataset_id} Results {'='*20}")
    print(results_df.to_string(index=False))
    print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")

def main():
    """Main function to run statistical tests for all real datasets."""
    print("=== Real Dataset Statistical Testing ===")
    print("Performing paired Wilcoxon signed-rank tests for all model pairs")
    
    # Load data
    results_path = '../../../results/real/experiment_results.csv'
    df = pd.read_csv(results_path)
    
    # Get unique real datasets
    real_datasets = sorted(df['dataset_id'].unique().tolist())
    df_real = df[df['dataset_id'].isin(real_datasets)].copy()
    
    print(f"Loaded {len(df_real)} real dataset experiments")
    print(f"Datasets: {real_datasets}")
    print(f"Methods: {df_real['model_name'].unique()}")
    
    # Create output directory
    os.makedirs("wilcoxon_tables", exist_ok=True)
    
    all_results = []
    
    # Run tests for each dataset
    for dataset in real_datasets:
        try:
            results_df = paired_wilcoxon_table(df_real, dataset)
            if results_df is not None:
                all_results.append(results_df)
                
                # Save individual dataset results
                output_file = f"wilcoxon_tables/real_wilcoxon_{dataset}.csv"
                results_df.to_csv(output_file, index=False)
                print(f"✓ Saved results to {output_file}")
                
        except Exception as e:
            print(f"✗ Error processing dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_file = "wilcoxon_tables/real_wilcoxon_combined.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Saved combined results to {combined_file}")
        
        # Create LaTeX table
        latex_file = "wilcoxon_tables/real_wilcoxon_combined.tex"
        with open(latex_file, 'w') as f:
            f.write(combined_df.to_latex(index=False, escape=False))
        print(f"✓ Saved LaTeX table to {latex_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total comparisons: {len(combined_df)}")
        print(f"Datasets processed: {len(real_datasets)}")
        
        # Count significant results
        sig_results = combined_df[combined_df['Significance'] != 'ns']
        print(f"Significant results: {len(sig_results)}/{len(combined_df)} ({len(sig_results)/len(combined_df)*100:.1f}%)")
        
        if len(sig_results) > 0:
            print("\nSignificant comparisons:")
            for _, row in sig_results.iterrows():
                print(f"  {row['Method1']} vs {row['Method2']}: {row['Significance']} (p={row['P_Value']})")
    
    print(f"\n✓ Real dataset statistical testing completed!")
    print("Results saved in wilcoxon_tables/ directory")

if __name__ == "__main__":
    main()
