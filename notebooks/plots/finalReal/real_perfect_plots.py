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
from matplotlib.ticker import MultipleLocator, FixedLocator

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
    ax.set_xticklabels(cats, rotation=90, ha="right")

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

def count_outliers_by_group(df, x="Method", y="F1", order=None, whis=1.5):
    """Count outliers per group for diagnostic purposes."""
    cats = order or sorted(df[x].unique().tolist())
    out = {}
    for c in cats:
        v = df.loc[df[x]==c, y].to_numpy()
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if isinstance(whis, (int, float)):
            low, high = q1 - whis*iqr, q3 + whis*iqr
        else:
            low, high = np.percentile(v, whis[0]), np.percentile(v, whis[1])
        out[c] = int(((v < low) | (v > high)).sum())
    return out

def get_small_coefficients_threshold(method_betas, feature_names, dataset_id, df_real, percentile=25):
    """
    Dynamically determine which coefficients are 'small' based on absolute values.
    Returns indices of features with small coefficients.
    """
    if not method_betas:
        return []
    
    # Dataset-specific adjustments
    if dataset_id in ["diabetes", "boston"]:
        # For diabetes and boston, use only NIMO coefficients for threshold calculation
        if 'NIMO' in method_betas:
            nimo_betas = method_betas['NIMO']
            mean_coefs, _ = mean_ci(nimo_betas, axis=0)
            all_abs_coefs = np.abs(mean_coefs)
            print(f"  - Using NIMO coefficients for {dataset_id} threshold calculation")
            
            # Use absolute threshold instead of percentile for more control
            if dataset_id == "diabetes":
                # For diabetes: take only the 2 smallest coefficients for heatmap
                # Use best F1 iteration instead of averaged coefficients
                dd = df_real[df_real['dataset_id'] == dataset_id].copy()
                nimo_data = dd[dd['model_name'] == 'NIMO']
                best_row = nimo_data.loc[nimo_data['f1'].idxmax()]
                
                # Parse coefficients from best F1 iteration
                info, _ = destring_coeff(best_row)
                if info is not None and info["values"] is not None:
                    best_coefs = to_raw_beta(info)
                    abs_coefs = np.abs(best_coefs)
                    
                    # Get indices of the 2 smallest coefficients
                    smallest_indices = np.argsort(abs_coefs)[:2]
                    small_coef_indices = smallest_indices.tolist()
                    print(f"  - Diabetes: using only 2 smallest coefficients from best F1 iteration")
                    print(f"  - Small coefficient features: {[feature_names[i] for i in small_coef_indices]}")
                    return small_coef_indices
                else:
                    return []
            else:  # boston
                # For boston: use best F1 iteration instead of averaged coefficients
                dd = df_real[df_real['dataset_id'] == dataset_id].copy()
                nimo_data = dd[dd['model_name'] == 'NIMO']
                best_row = nimo_data.loc[nimo_data['f1'].idxmax()]
                
                # Parse coefficients from best F1 iteration
                info, _ = destring_coeff(best_row)
                if info is not None and info["values"] is not None:
                    best_coefs = to_raw_beta(info)
                    abs_coefs = np.abs(best_coefs)
                    
                    # Use threshold 0.5 on best F1 iteration (more features in heatmap)
                    threshold = 0.5
                    small_indices = np.where(abs_coefs < threshold)[0]
                    small_coef_indices = small_indices.tolist()
                    print(f"  - Boston: using best F1 iteration with threshold {threshold}")
                    print(f"  - Small coefficient features: {[feature_names[i] for i in small_coef_indices]}")
                    return small_coef_indices
                else:
                    return []
            print(f"  - Using absolute threshold: {threshold}")
        else:
            return []
    elif dataset_id == "housing":
        # For housing: use best F1 iteration instead of averaged coefficients
        dd = df_real[df_real['dataset_id'] == dataset_id].copy()
        nimo_data = dd[dd['model_name'] == 'NIMO']
        best_row = nimo_data.loc[nimo_data['f1'].idxmax()]
        
        # Parse coefficients from best F1 iteration
        info, _ = destring_coeff(best_row)
        if info is not None and info["values"] is not None:
            best_coefs = to_raw_beta(info)
            abs_coefs = np.abs(best_coefs)
            
            # Use threshold 0.01 on best F1 iteration (small coefficients)
            threshold = 0.01
            small_indices = np.where(abs_coefs < threshold)[0]
            small_coef_indices = small_indices.tolist()
            print(f"  - Housing: using best F1 iteration with threshold {threshold}")
            print(f"  - Small coefficient features: {[feature_names[i] for i in small_coef_indices]}")
            return small_coef_indices
        else:
            return []
    else:
        # For other datasets, use all methods
        all_abs_coefs = []
        for method, betas in method_betas.items():
            mean_coefs, _ = mean_ci(betas, axis=0)
            all_abs_coefs.extend(np.abs(mean_coefs))
        
        if len(all_abs_coefs) == 0:
            return []
        
        # Calculate threshold as percentile of absolute values
        threshold = np.percentile(all_abs_coefs, percentile)
    
    # Find features with mean absolute coefficient below threshold
    small_coef_indices = []
    for method, betas in method_betas.items():
        mean_coefs, _ = mean_ci(betas, axis=0)
        small_indices = np.where(np.abs(mean_coefs) <= threshold)[0]
        small_coef_indices.extend(small_indices)
    
    # Remove duplicates and return
    small_coef_indices = list(set(small_coef_indices))
    small_coef_indices.sort()
    
    print(f"  - Small coefficient threshold: {threshold:.4f}")
    print(f"  - Small coefficient features: {[feature_names[i] for i in small_coef_indices]}")
    
    return small_coef_indices

def create_coefficients_across_iterations_plot(ax, dd, method_betas, feature_names, small_coef_indices, dataset_id, feature_colors):
    """Create coefficients across iterations plot for 2x2 grid (top right) - original error bar plot."""
    if not method_betas or len(feature_names) == 0:
        ax.text(0.5, 0.5, "No coefficient data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    
    # Calculate mean coefficients with confidence intervals (signed values) - same as original
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
        
        # Sort features by average absolute coefficient across methods
        feature_avg_abs = coef_df.groupby('feature')['mean'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
        top_features = feature_avg_abs.index.tolist()  # All features, sorted by importance
        
        # Note: All custom ordering removed since shortened names avoid overlap
        # Features will be ordered by NIMO coefficient magnitude (descending) as intended
        
        # Filter to top features (excluding small ones)
        plot_df = coef_df[coef_df['feature'].isin(top_features)].copy()
        plot_df['feature'] = pd.Categorical(plot_df['feature'], top_features, ordered=True)
        plot_df = plot_df.sort_values('feature')
        
        # Create grouped bar plot with error bars (same as original)
        x = np.arange(len(top_features))
        w = 0.35
        
        for i, method in enumerate(['Lasso', 'NIMO']):
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
        ax.set_title("Feature Coefficients (variance)", fontsize=14)
        ax.set_ylabel("Coefficient Value", fontsize=12)
        ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, labelsize=12)
        
        # Set y-axis ticks with dataset-specific intervals
        from matplotlib.ticker import MultipleLocator
        if dataset_id == "housing":
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
        elif dataset_id == "diabetes":
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

def create_best_iteration_coefficients_plot(ax, dd, feature_names, small_coef_indices, dataset_id, feature_colors):
    """Create best iteration coefficients plot for 2x2 grid (bottom left) - exact values, no error bars."""
    if len(feature_names) == 0:
        ax.text(0.5, 0.5, "No coefficient data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    
    # Get best F1 iteration for each method
    coef_data = []
    
    for method in ['Lasso', 'NIMO']:
        method_rows = dd[dd['model_name'] == method]
        if method_rows.empty:
            continue
            
        best_row = method_rows.loc[method_rows['f1'].idxmax()]
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
        
        # Sort features by average absolute coefficient across methods
        feature_avg_abs = coef_df.groupby('feature')['value'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
        top_features = feature_avg_abs.index.tolist()
        
        # Filter out small coefficients from main plot (they will be shown in heatmap)
        if small_coef_indices:
            small_feature_names = [feature_names[i] for i in small_coef_indices]
            top_features = [f for f in top_features if f not in small_feature_names]
        
        # Filter to top features (excluding small ones)
        plot_df = coef_df[coef_df['feature'].isin(top_features)].copy()
        plot_df['feature'] = pd.Categorical(plot_df['feature'], top_features, ordered=True)
        plot_df = plot_df.sort_values('feature')
        
        # Create grouped bar plot WITHOUT error bars (exact values)
        x = np.arange(len(top_features))
        w = 0.35
        
        for i, method in enumerate(['Lasso', 'NIMO']):
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
        ax.set_title("Feature Coefficients (best iteration)", fontsize=14)
        ax.set_ylabel("Coefficient Value", fontsize=12)
        ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, labelsize=12)
        
        # Set y-axis ticks with dataset-specific intervals
        from matplotlib.ticker import MultipleLocator
        if dataset_id == "housing":
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
        elif dataset_id == "diabetes":
            ax.yaxis.set_major_locator(MultipleLocator(8))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.2)
        elif dataset_id == "boston":
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

def create_real_plot(dataset_id, df_real, save_path=None):
    """Create the final perfect per-dataset figure for real datasets."""
    
    print(f"Creating real dataset plot for {dataset_id}...")
    
    # Filter data for this dataset
    dd = df_real[df_real['dataset_id'] == dataset_id].copy()
    
    if dd.empty:
        print(f"No data found for dataset {dataset_id}")
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
        n_features = len(dd.iloc[0].get('coefficients', '{}'))
        feature_names = [f"feature_{j}" for j in range(n_features)]
    
    # Refactor feature names to beta notation (β₁, β₂, β₃, etc.) or shorten long names
    def refactor_feature_names(names, dataset_id):
        """Convert feature_0, feature_1, etc. to β₁, β₂, etc., or shorten long names for specific datasets."""
        refactored = []
        
        # Dataset-specific feature name mappings
        if dataset_id == "housing":
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
        elif dataset_id == "boston":
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
    
    feature_names = refactor_feature_names(feature_names, dataset_id)
    
    # Extract F1 scores for boxplot
    f1_tbl = dd.pivot_table(index="iteration", columns="model_name", values="f1")
    
    # Collect model coefficients for feature importance
    method_betas = {}
    for method in ['Lasso', 'NIMO']:
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
    
    # Determine small coefficients for heatmap (dataset-specific)
    # Moon dataset: always use 2-panel layout (no heatmap)
    if dataset_id == "moon":
        small_coef_indices = []
        print(f"  - Moon dataset: using 2-panel layout (no heatmap)")
    else:
        # Use ultra-conservative threshold - only the absolute smallest coefficients go to heatmap
        small_coef_indices = get_small_coefficients_threshold(method_betas, feature_names, dataset_id, df_real, percentile=1)
    
    # Define consistent, distinct palettes
    BOX_COLORS = {
        "Lasso":        "#396AB1",   # blue
        "LassoNet":     "#00A0A0",   # teal
        "NN":           "#B07AA1",   # purple
        "NIMO":         "#DA7C30",   # orange
        "RF":           "#9C755F",   # brown
    }
    
    # Feature importance colors
    FEATURE_COLORS = {
        "Lasso": "#396AB1",          # same blue as boxplot Lasso
        "NIMO":  "#DA7C30",          # same orange as boxplot NIMO
    }
    
    # Shared styling constants
    TITLE_SIZE = 14
    TICK_SIZE = 12
    
    # Determine layout based on dataset and small coefficients
    has_small_coefs = len(small_coef_indices) > 0
    
    if dataset_id == "moon":
        # Moon: 2-panel horizontal layout (no heatmap) - F1 wider, coefficients narrower
        fig = plt.figure(figsize=(12, 6))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=1, ncols=2, figure=fig, 
                     width_ratios=[1.5, 1.0])
        
        # Global figure title
        fig.suptitle("Two-Moons Dataset", fontsize=16, y=0.98)
        
        # Left: F1 boxplot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Right: feature coefficients
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = None  # No heatmap panel
        ax4 = None  # No additional plot
    elif has_small_coefs and dataset_id in ["boston", "housing", "diabetes"]:
        # Boston, Housing, Diabetes: 2x2 grid layout
        fig = plt.figure(figsize=(16, 12))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=2, ncols=2, figure=fig, 
                     width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])
        
        # Global figure title with proper dataset names
        dataset_titles = {
            "boston": "Boston Housing Dataset",
            "housing": "California Housing Dataset", 
            "diabetes": "Diabetes Progression Dataset"
        }
        fig.suptitle(dataset_titles[dataset_id], fontsize=16, y=0.98)
        
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
        fig.suptitle(f"Dataset {dataset_id}", fontsize=16, y=0.98)
        
        # Left: F1 boxplot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Middle: feature coefficients
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Right: small coefficients heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = None  # No additional plot
    
    # --- (A) F1 boxplot with custom styling ---
    f1_long = f1_tbl.reset_index().melt(id_vars="iteration", var_name="Method", value_name="F1")
    methods_order = ["Lasso", "LassoNet", "NN", "NIMO", "RF"]
    present = [m for m in methods_order if m in set(f1_long['Method'])]
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
    
    # Optional: Check outlier counts for diagnostic purposes
    outlier_counts = count_outliers_by_group(df_plot, order=present, whis=1.5)
    print(f"  - Outliers per method: {outlier_counts}")
    
    # For 2x2 grid: Create coefficients across iterations plot (top right)
    if ax2 is not None and dataset_id in ["boston", "housing", "diabetes"] and has_small_coefs:
        # Top Right: Coefficients across iterations (original error bar plot)
        create_coefficients_across_iterations_plot(ax2, dd, method_betas, feature_names, small_coef_indices, dataset_id, FEATURE_COLORS)
    
    # For 2x2 grid: Create best iteration coefficients plot (bottom left)
    if ax3 is not None and dataset_id in ["boston", "housing", "diabetes"] and has_small_coefs:
        # Bottom Left: Best iteration coefficients (exact values, no error bars)
        create_best_iteration_coefficients_plot(ax3, dd, feature_names, small_coef_indices, dataset_id, FEATURE_COLORS)
    
    # --- (B) Feature coefficients comparison (signed values) ---
    # For 2x2 grid: this is handled above, for other layouts use ax2
    target_ax = ax2 if not (dataset_id in ["boston", "housing", "diabetes"] and has_small_coefs) else None
    
    if target_ax is not None and method_betas and len(feature_names) > 0:
        # Calculate mean coefficients with confidence intervals (signed values)
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
            
            # Sort features by average absolute coefficient across methods
            feature_avg_abs = coef_df.groupby('feature')['mean'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
            top_features = feature_avg_abs.index.tolist()  # All features, sorted by importance
            
            # Filter out small coefficients from main plot (they will be shown in heatmap)
            if has_small_coefs and small_coef_indices:
                small_feature_names = [feature_names[i] for i in small_coef_indices]
                top_features = [f for f in top_features if f not in small_feature_names]
                print(f"  - Excluding small coefficients from main plot: {small_feature_names}")
                print(f"  - Main plot will show: {top_features}")
            
            # Filter to top features (excluding small ones)
            plot_df = coef_df[coef_df['feature'].isin(top_features)].copy()
            plot_df['feature'] = pd.Categorical(plot_df['feature'], top_features, ordered=True)
            plot_df = plot_df.sort_values('feature')
            
            # Create grouped bar plot with error bars
            x = np.arange(len(top_features))
            w = 0.35
            
            for i, method in enumerate(['Lasso', 'NIMO']):
                method_data = plot_df[plot_df['method'] == method]
                if not method_data.empty:
                    # Draw bars
                    target_ax.bar(x + i*w, method_data['mean'], width=w,
                           color=FEATURE_COLORS[method], edgecolor="black", linewidth=0.5, label=method)
                    
                    # Draw error bars
                    target_ax.errorbar(
                        x + i*w, method_data['mean'],
                        yerr=[method_data['mean'] - method_data['ci_low'], 
                              method_data['ci_high'] - method_data['mean']],
                        fmt='none', ecolor='black', elinewidth=0.7, capsize=2
                    )
            
            # Dynamic y-axis scaling
            all_values = np.concatenate([plot_df['ci_low'].values, plot_df['ci_high'].values])
            min_val, max_val = np.min(all_values), np.max(all_values)
            margin = (max_val - min_val) * 0.1
            target_ax.set_ylim(min_val - margin, max_val + margin)
            
            # Style the coefficient plot
            plot_title = "Feature Coefficients (best iteration)" if (dataset_id in ["boston", "housing", "diabetes"] and has_small_coefs) else "Feature Coefficients (variance)"
            target_ax.set_title(plot_title, fontsize=TITLE_SIZE)
            target_ax.set_ylabel("Coefficient Value", fontsize=TICK_SIZE)
            target_ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, labelsize=TICK_SIZE)
            
            # Set y-axis ticks with dataset-specific intervals
            from matplotlib.ticker import MultipleLocator
            if dataset_id == "housing":
                # Housing: 0.5-step grid lines
                target_ax.yaxis.set_major_locator(MultipleLocator(0.5))
                target_ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
            elif dataset_id == "diabetes":
                # Diabetes: less grid lines (2-unit intervals)
                target_ax.yaxis.set_major_locator(MultipleLocator(2))
                target_ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.2)
            else:
                # Default: 4-unit intervals
                target_ax.yaxis.set_major_locator(MultipleLocator(4))
                target_ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.3)
            target_ax.xaxis.grid(False)
            
            # Add horizontal reference lines at y=0 and ±2
            target_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            target_ax.axhline(y=2, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            target_ax.axhline(y=-2, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            
            target_ax.set_xticks(x + w/2)
            target_ax.set_xticklabels(top_features, rotation=0, ha='center')
            target_ax.set_facecolor((1,1,1,0))
            
            # Black frame
            for s in target_ax.spines.values():
                s.set_color('black')
                s.set_linewidth(1.2)
            
            # Legend
            target_ax.legend(
                frameon=True, fancybox=True, framealpha=0.9,
                borderpad=0.4, handlelength=1.5, labelspacing=0.5,
                prop={"size": 12},
                loc="upper right"
            )
        else:
            target_ax.text(0.5, 0.5, "No feature importance data available", 
                    ha="center", va="center", transform=target_ax.transAxes)
            target_ax.axis("off")
    elif target_ax is not None:
        target_ax.text(0.5, 0.5, "No feature importance data available", 
                ha="center", va="center", transform=target_ax.transAxes)
        target_ax.axis("off")
    
    # --- (C) Small coefficients heatmap (best-F1 run) ---
    # For 2x2 grid: this goes to bottom right (ax4), otherwise ax3
    heatmap_ax = ax4 if (dataset_id in ["boston", "housing", "diabetes"] and has_small_coefs) else ax3
    
    if heatmap_ax is not None and has_small_coefs and small_coef_indices:
        heatmap_data = []
        for method in ['Lasso', 'NIMO']:
            # pick the best F1 iteration for this method
            method_rows = dd[dd['model_name'] == method]
            if method_rows.empty:
                continue
            best_row = method_rows.loc[method_rows['f1'].idxmax()]

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
                ax=heatmap_ax, cbar_kws={'shrink': 0.8}
            )
            heatmap_ax.set_title('Small Coefficients (best iteration)', fontsize=TITLE_SIZE)
            heatmap_ax.set_xlabel('Method')
            heatmap_ax.set_ylabel('Feature')
        else:
            heatmap_ax.text(0.5, 0.5, "No small coefficient data available", 
                     ha="center", va="center", transform=heatmap_ax.transAxes)
            heatmap_ax.axis("off")
    elif heatmap_ax is not None:
        heatmap_ax.text(0.5, 0.5, "No small coefficient data available", 
                ha="center", va="center", transform=heatmap_ax.transAxes)
        heatmap_ax.axis("off")
    
    # Adjust layout based on whether we have heatmap
    if has_small_coefs:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.12, wspace=0.25)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.12, wspace=0.25)
    
    # Always save the plot when run
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ Real dataset plot saved to {save_path}")
    else:
        default_path = f"dataset_{dataset_id}_real.png"
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
    
    # Filter to real datasets only
    real_datasets = sorted(df['dataset_id'].unique().tolist())
    df_real = df[df['dataset_id'].isin(real_datasets)].copy()
    
    print(f"Loaded {len(df_real)} real dataset experiments")
    print(f"Methods: {df_real['model_name'].unique()}")
    print(f"Datasets: {real_datasets}")
    
    # Generate all plots dynamically
    saved_plots = []
    
    for dataset in real_datasets:
        try:
            save_path = f"dataset_{dataset}_real.png"
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
