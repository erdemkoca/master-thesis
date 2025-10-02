#!/usr/bin/env python3
"""
Real Dataset Perfect Plots Script
Creates publication-ready plots for real datasets (no ground truth coefficients)
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import re

# Canonical dataset mapping (use short keys internally, map to titles for display)
DATASET_TITLES = {
    "moon": "Two-Moons Dataset",
    "boston": "Boston Housing Dataset", 
    "housing": "California Housing Dataset",
    "diabetes": "Diabetes Progression Dataset",
}

# Method lists for consistency
F1_METHODS = ["Lasso", "LassoNet", "NN", "NIMO_T", "NIMO_MLP", "RF"]
COEFF_METHODS = ['Lasso', 'NIMO_T', 'NIMO_MLP']

def get_canonical_dataset_key(dataset_description):
    """Map dataset description to canonical short key."""
    mapping = {
        "Two-Moon Dataset (binary classification)": "moon",
        "Boston Housing (binary classification)": "boston", 
        "California Housing (binary classification)": "housing",
        "Diabetes Progression (binary classification)": "diabetes",
    }
    return mapping.get(dataset_description, dataset_description.lower())

# Force consistent settings for perfect plots
matplotlib.use('Agg')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 18,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 1.0,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'xtick.major.size': 6,
    'xtick.minor.size': 4,
    'ytick.major.size': 6,
    'ytick.minor.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': False,
    'axes.axisbelow': True,
})

# Typography: smaller titles, serif look, CM-style math
plt.rcParams.update({
    "axes.titlesize": 14,          # was 18
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

def parse_json_safe(x):
    """Safely parse JSON-like strings."""
    if pd.isna(x) or x == "":
        return None
    try:
        import ast
        return ast.literal_eval(x)
    except Exception:
        return None

def destring_coeff(row):
    """Extract coefficient information from a row."""
    co = parse_json_safe(row.get("coefficients", "{}"))
    if not isinstance(co, dict):
        return None, None
    
    vals = np.array(co.get("values", []), dtype=float)
    mean = np.array(co.get("mean", []), dtype=float) if "mean" in co else None
    scale = np.array(co.get("scale", []), dtype=float) if "scale" in co else None
    space = co.get("space", "raw")
    
    return {
        "values": vals, 
        "mean": mean, 
        "scale": scale, 
        "space": space,
        "intercept": co.get("intercept", 0.0)
    }, co.get("feature_names", None)

def to_raw_beta(coeff_info):
    """Convert standardized coefficients to raw scale."""
    vals = coeff_info["values"]
    if coeff_info["space"] == "standardized":
        scale = coeff_info["scale"]
        if scale is None:
            raise ValueError("Need scale to de-standardize Lasso coefficients")
        return vals / scale
    return vals

def mean_ci(x, axis=0, alpha=0.05):
    """Calculate mean and confidence interval."""
    mean = np.mean(x, axis=axis)
    n = x.shape[axis]
    se = np.std(x, axis=axis) / np.sqrt(n)
    ci = se * 1.96  # 95% CI
    return mean, ci

def standard_boxplot(ax, df, x="Method", y="F1", order=None, colors=None,
                     title="Dataset — F1 over iterations", whis=1.5, width=0.85):
    """
    Standard boxplot with visible outliers.
    Uses Matplotlib's boxplot for full control.
    """
    cats = order or sorted(df[x].unique().tolist())
    data = [df.loc[df[x] == c, y].to_numpy() for c in cats]

    bp = ax.boxplot(
        data,
        notch=False,              # ← turn off notches
        widths=width,
        whis=1.5,                 # ← standard Tukey definition
        showmeans=False,
        showfliers=True,
        patch_artist=True,
        flierprops=dict(marker="o", markersize=4, markerfacecolor="black",
                        markeredgewidth=0, alpha=0.7),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Color each box
    if colors:
        for box, cat in zip(bp["boxes"], cats):
            c = colors.get(cat, "#cccccc")
            box.set_facecolor(c)
            box.set_edgecolor("black")
            box.set_linewidth(1.0)

    # X ticks/labels
    ax.set_xticks(np.arange(1, len(cats) + 1))
    ax.set_xticklabels(cats, rotation=0, ha="center")

    # Title/labels
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1", fontsize=12)

    # Grid + frame + tickmarks
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=1.0, color="#999999", alpha=0.7)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.2)

    # Slight margin to reduce whitespace
    ax.margins(x=0.02)

def get_small_coefficients_indices(dd, dataset_key, feature_names):
    """
    Get small coefficients indices based on NIMO_T best F1 iteration.
    Returns indices of features with small coefficients.
    """
    if dataset_key not in ["boston", "diabetes", "housing"]:
        return []
    
    # Get NIMO_T best F1 iteration
    nimo_data = dd[dd['model_name'] == 'NIMO_T']
    if nimo_data.empty:
        return []
    
    best_row = nimo_data.loc[nimo_data['f1'].idxmax()]
    info, _ = destring_coeff(best_row)
    
    if info is None or info["values"] is None:
        return []
    
    try:
        beta_raw = to_raw_beta(info)
        abs_coefs = np.abs(beta_raw)
        
        if dataset_key == "diabetes":
            # Take bottom 6 by magnitude
            smallest_indices = np.argsort(abs_coefs)[:6]
            small_coef_indices = smallest_indices.tolist()
            print(f"  - Diabetes: using 6 smallest coefficients from NIMO_T best F1 iteration")
            print(f"  - Small coefficient features: {[feature_names[i] for i in small_coef_indices]}")
            return small_coef_indices
            
        elif dataset_key == "boston":
            # τ_abs = 0.1 and bottom 50% union
            threshold_abs = 0.1
            small_indices_abs = np.where(abs_coefs < threshold_abs)[0]
            
            threshold_percentile = np.percentile(abs_coefs, 50)
            small_indices_percentile = np.where(abs_coefs < threshold_percentile)[0]
            
            small_indices = np.union1d(small_indices_abs, small_indices_percentile)
            small_coef_indices = small_indices.tolist()
            print(f"  - Boston: using combined approach (abs<0.1 OR bottom 50%)")
            print(f"  - Small coefficient features: {[feature_names[i] for i in small_coef_indices]}")
            return small_coef_indices
            
        elif dataset_key == "housing":
            # τ_abs = 0.02 and bottom 50% union
            threshold_abs = 0.02
            small_indices_abs = np.where(abs_coefs < threshold_abs)[0]
            
            threshold_percentile = np.percentile(abs_coefs, 50)
            small_indices_percentile = np.where(abs_coefs < threshold_percentile)[0]
            
            small_indices = np.union1d(small_indices_abs, small_indices_percentile)
            small_coef_indices = small_indices.tolist()
            print(f"  - Housing: using combined approach (abs<0.02 OR bottom 50%)")
            print(f"  - Small coefficient features: {[feature_names[i] for i in small_coef_indices]}")
            return small_coef_indices
            
    except Exception as e:
        print(f"Warning: Could not process NIMO_T coefficients for small coefficients: {e}")
        return []
    
    return []

def create_coefficients_variance_plot(ax, dd, feature_names, dataset_key, feature_colors):
    """Create coefficients variance plot (mean ± 95% CI across iterations)."""
    if len(feature_names) == 0:
        ax.text(0.5, 0.5, "No coefficient data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    
    # Collect coefficients for each method across iterations
    method_betas = {}
    for method in COEFF_METHODS:
        try:
            betas = []
            for _, row in dd[dd['model_name'] == method].iterrows():
                info, _ = destring_coeff(row)
                if info is not None and info["values"] is not None:
                    try:
                        beta_raw = to_raw_beta(info)
                        betas.append(beta_raw)
                    except Exception as e:
                        print(f"Warning: Could not process coefficients for {method}: {e}")
                        continue
            if betas:
                method_betas[method] = np.array(betas)
        except Exception as e:
            print(f"Warning: Could not collect coefficients for {method}")
    
    if not method_betas:
        ax.text(0.5, 0.5, "No coefficient data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    
    # Calculate mean coefficients with confidence intervals
    coef_data = []
    for method, betas in method_betas.items():
        mean_coefs, ci_coefs = mean_ci(betas, axis=0)
        for i, (mean_coef, ci_coef, name) in enumerate(zip(mean_coefs, ci_coefs, feature_names)):
            coef_data.append({
                'feature': name,
                'method': method,
                'mean': mean_coef,
                'ci_low': mean_coef - ci_coef,
                'ci_high': mean_coef + ci_coef
            })
    
    if coef_data:
        coef_df = pd.DataFrame(coef_data)
        
        # Check if NIMO_MLP normalization is needed
        lasso_coefs = coef_df[coef_df['method'] == 'Lasso']['mean'].values
        nimo_mlp_coefs = coef_df[coef_df['method'] == 'NIMO_MLP']['mean'].values
        
        normalization_applied = False
        if len(lasso_coefs) > 0 and len(nimo_mlp_coefs) > 0:
            nz_lasso = np.abs(lasso_coefs[lasso_coefs != 0])
            nz_mlp = np.abs(nimo_mlp_coefs[nimo_mlp_coefs != 0])
            
            if nz_lasso.size and nz_mlp.size:
                scale_factor = np.median(nz_lasso) / np.median(nz_mlp)
                if np.isfinite(scale_factor) and scale_factor > 0:
                    print(f"  - Scaling NIMO_MLP coefficients by factor: {scale_factor:.6f}")
                    
                    # Apply scaling to NIMO_MLP coefficients
                    nimo_mlp_mask = coef_df['method'] == 'NIMO_MLP'
                    coef_df.loc[nimo_mlp_mask, 'mean'] *= scale_factor
                    coef_df.loc[nimo_mlp_mask, 'ci_low'] *= scale_factor
                    coef_df.loc[nimo_mlp_mask, 'ci_high'] *= scale_factor
                    normalization_applied = True
        
        # Sort features by average absolute coefficient across methods
        feature_avg_abs = coef_df.groupby('feature')['mean'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
        top_features = feature_avg_abs.index.tolist()
        
        # Filter to top features
        plot_df = coef_df[coef_df['feature'].isin(top_features)].copy()
        plot_df['feature'] = pd.Categorical(plot_df['feature'], top_features, ordered=True)
        plot_df = plot_df.sort_values('feature')
        
        # Create grouped bar plot with error bars
        x = np.arange(len(top_features))
        w = 0.35
        
        for i, method in enumerate(COEFF_METHODS):
            method_data = plot_df[plot_df['method'] == method]
            if not method_data.empty:
                # Draw bars
                ax.bar(x + i*w, method_data['mean'], width=w,
                       color=feature_colors[method], edgecolor="black", linewidth=0.5, label=method)
                
                # Draw error bars
                ax.errorbar(
                    x + i*w, method_data['mean'],
                    yerr=[method_data['mean'] - method_data['ci_low'], 
                          method_data['ci_high'] - method_data['mean']],
                    fmt='none', ecolor='black', elinewidth=0.7, capsize=2
                )
        
        # Dynamic y-axis scaling
        all_values = np.concatenate([plot_df['ci_low'].values, plot_df['ci_high'].values])
        min_val, max_val = np.min(all_values), np.max(all_values)
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(min_val - margin, max_val + margin)
        
        # Style the coefficient plot
        title = "Feature Coefficients (variance)"
        if normalization_applied:
            title += ", NIMO_MLP normalized"
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Coefficient Value", fontsize=12)
        ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, labelsize=12)
        
        # Set y-axis ticks with dataset-specific intervals
        if dataset_key == "housing":
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
        elif dataset_key == "diabetes":
            ax.yaxis.set_major_locator(MultipleLocator(8))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.2)
        else:
            ax.yaxis.set_major_locator(MultipleLocator(4))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
        ax.xaxis.grid(False)
        
        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        ax.set_xticks(x + w/2)
        ax.set_xticklabels(top_features, rotation=0, ha='center')
        ax.set_facecolor((1,1,1,0))
        
        # Black frame
        for s in ax.spines.values():
            s.set_color('black')
            s.set_linewidth(1.2)
        
        # Legend
        ax.legend(
            frameon=True, fancybox=True, framealpha=0.9,
            borderpad=0.4, handlelength=1.5, labelspacing=0.5,
            prop={"size": 12},
            loc="upper right"
        )
    else:
        ax.text(0.5, 0.5, "No feature importance data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

def create_best_iteration_coefficients_plot(ax, dd, feature_names, small_coef_indices, dataset_key, feature_colors):
    """Create best iteration coefficients plot (exact values, no error bars)."""
    if len(feature_names) == 0:
        ax.text(0.5, 0.5, "No coefficient data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    
    # Get best F1 iteration for each method
    coef_data = []
    
    for method in COEFF_METHODS:
        method_rows = dd[dd['model_name'] == method]
        if method_rows.empty or method_rows['f1'].dropna().empty:
            continue
            
        best_idx = method_rows['f1'].idxmax()
        if pd.isna(best_idx):
            continue
        best_row = method_rows.loc[best_idx]
        info, _ = destring_coeff(best_row)
        
        if info is not None and info["values"] is not None:
            try:
                beta_raw = to_raw_beta(info)
                for i, (coef_val, name) in enumerate(zip(beta_raw, feature_names)):
                    coef_data.append({
                        'feature': name,
                        'method': method,
                        'value': coef_val
                    })
            except Exception as e:
                print(f"Warning: Could not process coefficients for {method} in best iteration: {e}")
                continue
    
    if coef_data:
        coef_df = pd.DataFrame(coef_data)
        
        # Check if NIMO_MLP normalization is needed
        lasso_coefs = coef_df[coef_df['method'] == 'Lasso']['value'].values
        nimo_mlp_coefs = coef_df[coef_df['method'] == 'NIMO_MLP']['value'].values
        
        normalization_applied = False
        if len(lasso_coefs) > 0 and len(nimo_mlp_coefs) > 0:
            nz_lasso = np.abs(lasso_coefs[lasso_coefs != 0])
            nz_mlp = np.abs(nimo_mlp_coefs[nimo_mlp_coefs != 0])
            
            if nz_lasso.size and nz_mlp.size:
                scale_factor = np.median(nz_lasso) / np.median(nz_mlp)
                if np.isfinite(scale_factor) and scale_factor > 0:
                    print(f"  - Scaling NIMO_MLP coefficients by factor: {scale_factor:.6f}")
                    
                    # Apply scaling to NIMO_MLP coefficients
                    nimo_mlp_mask = coef_df['method'] == 'NIMO_MLP'
                    coef_df.loc[nimo_mlp_mask, 'value'] *= scale_factor
                    normalization_applied = True
        
        # Sort features by average absolute coefficient across methods
        feature_avg_abs = coef_df.groupby('feature')['value'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
        top_features = feature_avg_abs.index.tolist()
        
        # Filter out small coefficients from main plot (they will be shown in heatmap)
        if small_coef_indices:
            small_feature_names = [feature_names[i] for i in small_coef_indices]
            top_features = [f for f in top_features if f not in small_feature_names]
            print(f"  - Excluding small coefficients from main plot: {small_feature_names}")
            print(f"  - Main plot will show: {top_features}")
        
        # Filter to top features (excluding small ones)
        plot_df = coef_df[coef_df['feature'].isin(top_features)].copy()
        plot_df['feature'] = pd.Categorical(plot_df['feature'], top_features, ordered=True)
        plot_df = plot_df.sort_values('feature')
        
        # Create grouped bar plot WITHOUT error bars (exact values)
        x = np.arange(len(top_features))
        w = 0.35
        
        for i, method in enumerate(COEFF_METHODS):
            method_data = plot_df[plot_df['method'] == method]
            if not method_data.empty:
                # Draw bars (no error bars)
                ax.bar(x + i*w, method_data['value'], width=w,
                       color=feature_colors[method], edgecolor="black", linewidth=0.5, label=method)
        
        # Dynamic y-axis scaling
        all_values = plot_df['value'].values
        min_val, max_val = np.min(all_values), np.max(all_values)
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(min_val - margin, max_val + margin)
        
        # Style the coefficient plot
        title = "Feature Coefficients (best iteration)"
        if normalization_applied:
            title += ", NIMO_MLP normalized"
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Coefficient Value", fontsize=12)
        ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, labelsize=12)
        
        # Set y-axis ticks with dataset-specific intervals
        if dataset_key == "housing":
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
        elif dataset_key == "diabetes":
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.2)
        elif dataset_key == "boston":
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
        else:
            ax.yaxis.set_major_locator(MultipleLocator(4))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
        ax.xaxis.grid(False)
        
        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        ax.set_xticks(x + w/2)
        ax.set_xticklabels(top_features, rotation=0, ha='center')
        ax.set_facecolor((1,1,1,0))
        
        # Black frame
        for s in ax.spines.values():
            s.set_color('black')
            s.set_linewidth(1.2)
        
        # Legend
        ax.legend(
            frameon=True, fancybox=True, framealpha=0.9,
            borderpad=0.4, handlelength=1.5, labelspacing=0.5,
            prop={"size": 12},
            loc="upper right"
        )
    else:
        ax.text(0.5, 0.5, "No feature importance data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

def create_small_coefficients_heatmap(ax, dd, feature_names, small_coef_indices):
    """Create small coefficients heatmap (best iteration)."""
    if not small_coef_indices:
        ax.text(0.5, 0.5, "No small coefficient data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    
    heatmap_data = []
    for method in COEFF_METHODS:
        # pick the best F1 iteration for this method
        method_rows = dd[dd['model_name'] == method]
        if method_rows.empty or method_rows['f1'].dropna().empty:
            continue
        best_idx = method_rows['f1'].idxmax()
        if pd.isna(best_idx):
            continue
        best_row = method_rows.loc[best_idx]

        info, _ = destring_coeff(best_row)
        if info is not None and info["values"] is not None:
            try:
                beta_raw = to_raw_beta(info)
                for j in small_coef_indices:
                    heatmap_data.append({
                        'method': method,
                        'feature': feature_names[j],
                        'value': beta_raw[j]
                    })
            except Exception as e:
                print(f"Warning: Could not process coefficients for {method} in best iteration: {e}")
                continue

    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index='method', columns='feature', values='value')

        # Transpose the heatmap for horizontal arrangement (features as rows, methods as columns)
        heatmap_pivot_t = heatmap_pivot.T

        sns.heatmap(
            heatmap_pivot_t, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            ax=ax, cbar_kws={'shrink': 0.8}
        )
        ax.set_title('Small Coefficients (best iteration)', fontsize=14)
        ax.set_xlabel('Method')
        ax.set_ylabel('Feature')
    else:
        ax.text(0.5, 0.5, "No small coefficient data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

def create_real_plot(dataset_description, df_real, save_path=None):
    """Create the final perfect per-dataset figure for real datasets."""
    
    # Get canonical dataset key
    dataset_key = get_canonical_dataset_key(dataset_description)
    print(f"Creating real dataset plot for {dataset_key}...")
    
    # Filter data for this dataset
    dd = df_real[df_real['dataset_description'] == dataset_description].copy()
    
    if dd.empty:
        print(f"No data found for dataset {dataset_key}")
        return None
    
    # Get feature names
    feature_names = None
    for _, row in dd.iterrows():
        _, names = destring_coeff(row)
        if names is not None:
            feature_names = names
            break
    
    if feature_names is None:
        # Fallback: create generic feature names
        co = parse_json_safe(dd.iloc[0].get('coefficients', '{}')) or {}
        n_features = len(co.get('values', []))
        feature_names = [f"β{i+1}" for i in range(n_features)]
    
    # Refactor feature names to beta notation (β₁, β₂, β₃, etc.) or shorten long names for specific datasets
    def refactor_feature_names(names, dataset_key):
        """Convert feature_0, feature_1, etc. to β₁, β₂, etc., or shorten long names for specific datasets."""
        refactored = []
        
        # Dataset-specific feature name mappings
        if dataset_key == "housing":
            name_mapping = {
                "longitude": "long",
                "latitude": "lat", 
                "housing_median_age": "age",
                "total_rooms": "rooms",
                "total_bedrooms": "bedrooms",
                "population": "pop",
                "households": "households",
                "median_income": "income"
            }
            for name in names:
                refactored.append(name_mapping.get(name, name))
        elif dataset_key == "boston":
            name_mapping = {
                "LSTAT": "LS",
                "INDUS": "IN",
                "PTRATIO": "PTR"
            }
            for name in names:
                refactored.append(name_mapping.get(name, name))
        else:
            # For other datasets, use beta notation for generic features
            for name in names:
                if name.startswith("feature_"):
                    try:
                        num = int(name.split("_")[1])
                        refactored.append(f"β{num+1}")
                    except (ValueError, IndexError):
                        refactored.append(name)
                else:
                    refactored.append(name)
        return refactored
    
    feature_names = refactor_feature_names(feature_names, dataset_key)
    
    # Extract F1 scores for boxplot
    f1_tbl = dd.pivot_table(index="iteration", columns="model_name", values="f1")
    
    # Define consistent, distinct palettes
    BOX_COLORS = {
        "Lasso":        "#396AB1",   # blue
        "LassoNet":     "#00A0A0",   # teal
        "NN":           "#B07AA1",   # purple
        "NIMO_T":       "#DA7C30",   # orange
        "NIMO_MLP":     "#FF6B35",   # red-orange
        "RF":           "#9C755F",   # brown
    }
    
    # Feature importance colors
    FEATURE_COLORS = {
        "Lasso": "#396AB1",          # same blue as boxplot Lasso
        "NIMO_T": "#DA7C30",         # same orange as boxplot NIMO_T
        "NIMO_MLP": "#FF6B35",       # red-orange
    }
    
    # Determine layout based on dataset
    if dataset_key == "moon":
        # Moon: 2-panel horizontal layout (no heatmap) - F1 wider, coefficients narrower
        fig = plt.figure(figsize=(12, 6))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=1, ncols=2, figure=fig, 
                     width_ratios=[1.5, 1.0])
        
        # Global figure title
        fig.suptitle(DATASET_TITLES[dataset_key], fontsize=16, y=0.98)
        
        # Left: F1 boxplot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Right: feature coefficients
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = None  # No heatmap panel
        ax4 = None  # No additional plot
    elif dataset_key in ["boston", "housing", "diabetes"]:
        # Boston, Housing, Diabetes: 2x2 grid layout
        fig = plt.figure(figsize=(16, 12))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=2, ncols=2, figure=fig, 
                     width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])
        
        # Global figure title
        fig.suptitle(DATASET_TITLES[dataset_key], fontsize=16, y=0.98)
        
        # Top Left: F1 across iterations
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Top Right: Coefficients across iterations
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Bottom Left: Best iteration coefficients
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Bottom Right: Small coefficients heatmap
        ax4 = fig.add_subplot(gs[1, 1])
    else:
        # Other datasets: 3-panel horizontal layout
        fig = plt.figure(figsize=(18, 6))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=1, ncols=3, figure=fig, 
                     width_ratios=[1.0, 1.5, 1.0])
        
        # Global figure title
        fig.suptitle(f"Dataset {dataset_key}", fontsize=16, y=0.98)
        
        # Left: F1 boxplot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Middle: feature coefficients
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Right: small coefficients heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = None  # No additional plot
    
    # --- (A) F1 boxplot with custom styling ---
    f1_long = f1_tbl.reset_index().melt(id_vars="iteration", var_name="Method", value_name="F1")
    present = [m for m in F1_METHODS if m in set(f1_long['Method'])]
    df_plot = f1_long[f1_long['Method'].isin(present)].copy()
    
    # Create standard boxplot with visible outliers
    standard_boxplot(
        ax1, df_plot, x="Method", y="F1", order=present,
        colors={k: BOX_COLORS[k] for k in present},
        title="F1 (across iterations)",
        whis=1.5,          # standard Tukey definition
        width=0.9          # make boxes a bit wider
    )
    
    # Make method labels horizontal for better readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center')
    
    # --- (B) Feature coefficients comparison (signed values) ---
    if ax2 is not None:
        create_coefficients_variance_plot(ax2, dd, feature_names, dataset_key, FEATURE_COLORS)
    
    # --- (C) Best iteration coefficients plot ---
    if ax3 is not None and dataset_key in ["boston", "housing", "diabetes"]:
        # Get small coefficients indices
        small_coef_indices = get_small_coefficients_indices(dd, dataset_key, feature_names)
        create_best_iteration_coefficients_plot(ax3, dd, feature_names, small_coef_indices, dataset_key, FEATURE_COLORS)
    
    # --- (D) Small coefficients heatmap (best-F1 run) ---
    if ax4 is not None and dataset_key in ["boston", "housing", "diabetes"]:
        # Get small coefficients indices
        small_coef_indices = get_small_coefficients_indices(dd, dataset_key, feature_names)
        create_small_coefficients_heatmap(ax4, dd, feature_names, small_coef_indices)
    
    # Adjust layout
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.12, wspace=0.25)
    
    # Always save the plot when run
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ Real dataset plot saved to {save_path}")
    else:
        # Sanitize dataset name for filename
        safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', dataset_key)
        default_path = f"dataset_{safe_name}_real.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ Real dataset plot saved to {default_path}")
    
    plt.close()
    return fig

def main():
    """Main function to generate all real dataset plots."""
    print("=== Generating Real Dataset Perfect Plots ===")
    print("This will create publication-ready plots for real datasets!")
    
    # Load data
    results_path = '../../../results/real/experiment_results.csv'
    df = pd.read_csv(results_path)
    
    # Filter to real datasets only - use dataset_description for actual dataset names
    # The data has both dataset_name='moon' and dataset_name='real' with dataset_description='moon'
    # We want to group by the actual dataset names in dataset_description
    real_datasets = sorted(df['dataset_description'].unique().tolist())
    df_real = df[df['dataset_description'].isin(real_datasets)].copy()
    
    print(f"Loaded {len(df_real)} real dataset experiments")
    print(f"Methods: {df_real['model_name'].unique()}")
    print(f"Datasets: {real_datasets}")
    
    # Generate all plots dynamically
    saved_plots = []
    
    for dataset in real_datasets:
        try:
            save_path = f"dataset_{get_canonical_dataset_key(dataset)}_real.png"
            fig = create_real_plot(dataset, df_real, save_path=save_path)
            saved_plots.append(save_path)
        except Exception as e:
            print(f"✗ Error creating plot for dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Successfully generated {len(saved_plots)} real dataset plots!")
    print("Files saved:")
    for plot_path in saved_plots:
        print(f"  - {plot_path}")
    print("\nThese are publication-ready plots for real datasets!")

if __name__ == "__main__":
    main()
