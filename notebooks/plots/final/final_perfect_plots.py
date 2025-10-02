#!/usr/bin/env python3
"""
Final perfect plots script - combines working simple approach with full functionality.
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
                     title="Scenario — F1 over iterations", whis=1.5, width=0.85):
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

def create_final_plot(scenario_id, df_synthetic, save_path=None):
    """Create the final perfect per-scenario figure."""
    
    print(f"Creating final plot for Scenario {scenario_id}...")
    
    # Filter data for this scenario
    dd = df_synthetic[df_synthetic['dataset_id'] == scenario_id].copy()
    
    if dd.empty:
        print(f"No data found for scenario {scenario_id}")
        return None
    
    # Dynamically detect available methods in the data
    available_methods = sorted(dd['model_name'].unique().tolist())
    print(f"Available methods for scenario {scenario_id}: {available_methods}")
    
    # Get ground truth
    base_row = dd.iloc[0]
    beta_true = parse_json_safe(base_row.get('beta_true', '[]'))
    if beta_true is None:
        beta_true = []
    beta_true = np.array(beta_true, dtype=float)
    d = len(beta_true)
    
    # True support / zeros
    nz_idx = np.where(np.abs(beta_true) > 0)[0]
    z_idx = np.where(np.abs(beta_true) == 0)[0]
    
    # Get feature names and refactor to beta notation
    feature_names = None
    for _, row in dd.iterrows():
        _, names = destring_coeff(row)
        if names is not None:
            feature_names = names
            break
    
    if feature_names is None:
        feature_names = [f"x{j+1}" for j in range(d)]
    
    # Refactor feature names to beta notation (β₁, β₂, β₃, etc.)
    def refactor_feature_names(names):
        """Convert feature_0, feature_1, etc. to β₁, β₂, etc."""
        refactored = []
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
    
    feature_names = refactor_feature_names(feature_names)
    
    # Extract F1 scores for boxplot
    f1_tbl = dd.pivot_table(index="iteration", columns="model_name", values="f1")
    
    # Collect model coefficients
    method_betas = {}
    for method in available_methods:
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
            print(f"Warning: Could not collect coefficients for {method}: {e}")
    
    # Define consistent, distinct palettes
    # Boxplot colors (method names as in your CSV)
    BOX_COLORS = {}
    
    # Add colors for available methods dynamically
    method_colors = {
        "Lasso": "#396AB1",          # blue
        "LassoNet": "#00A0A0",       # teal
        "NN": "#B07AA1",             # purple
        "NIMO": "#DA7C30",           # orange
        "NIMO_T": "#DA7C30",         # orange (same as NIMO)
        "NIMO_MLP": "#FF6B35",       # red-orange
        "RF": "#9C755F",             # brown
    }
    
    for method in available_methods:
        if method in method_colors:
            BOX_COLORS[method] = method_colors[method]
        else:
            # Default color for unknown methods
            BOX_COLORS[method] = "#666666"
    
    # Coefficient panel colors (GT is neutral gray)
    COEF_COLORS = {
        "GT":    "#7F7F7F",          # gray
    }
    
    # Add colors for available methods dynamically
    method_colors = {
        "Lasso": "#396AB1",          # blue
        "NIMO":  "#DA7C30",          # orange
        "NIMO_T": "#DA7C30",         # orange (same as NIMO)
        "NIMO_MLP": "#FF6B35",       # red-orange
        "RF": "#9C755F",             # brown
        "NN": "#B07AA1",             # purple
        "LassoNet": "#00A0A0",       # teal
    }
    
    for method in available_methods:
        if method in method_colors:
            COEF_COLORS[method] = method_colors[method]
        else:
            # Default color for unknown methods
            COEF_COLORS[method] = "#666666"
    
    # Shared styling constants
    TITLE_SIZE = 14
    TICK_SIZE = 12
    
    # Create the plot
    fig = plt.figure(figsize=(15, 7))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=2, ncols=3, figure=fig, 
                 width_ratios=[1.1, 1.2, 1.2], height_ratios=[1.25, 0.95])
    
    # Global figure title
    fig.suptitle(f"Scenario {scenario_id}", fontsize=16, y=0.98)
    
    # Left: F1 boxplot spans both rows
    ax1 = fig.add_subplot(gs[:, 0])
    
    # Right–top: nonzeros
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # Right–bottom: zeros heatmap
    ax3 = fig.add_subplot(gs[1, 1:])
    
    # --- (A) F1 boxplot with custom styling ---
    f1_long = f1_tbl.reset_index().melt(id_vars="iteration", var_name="Method", value_name="F1")
    # Use dynamically detected methods instead of hardcoded list
    present = available_methods
    df_plot = f1_long[f1_long['Method'].isin(present)].copy()
    
    # Create standard boxplot with visible outliers
    standard_boxplot(
        ax1, df_plot, x="Method", y="F1", order=present,
        colors={k: BOX_COLORS[k] for k in present},
        title="F1 (across iterations)",
        whis=1.5,          # standard Tukey definition
        width=0.9          # make boxes a bit wider
    )
    
    # Optional: Check outlier counts for diagnostic purposes
    outlier_counts = count_outliers_by_group(df_plot, order=present, whis=1.5)
    print(f"  - Outliers per method: {outlier_counts}")
    
    # --- (B) Non-zero coefficients bar+CI vs ground truth ---
    if len(nz_idx) > 0 and method_betas:
        # Get ground truth intercept
        intercept_gt = base_row.get('b0_true', 0.0)
        if isinstance(intercept_gt, str):
            try:
                intercept_gt = float(intercept_gt)
            except:
                intercept_gt = 0.0
        
        # Create coefficient names with β₀ FIRST, then β₁, β₂, ... using mathtext
        labels_math = [r"$\beta_0$"] + [rf"$\beta_{j+1}$" for j in nz_idx]
        
        # Prepare data for plotting
        coef_data = []
        
        # Add intercept β₀ FIRST
        coef_data.append({
            'coef_name': r"$\beta_0$",
            'method': 'GT',
            'value': intercept_gt,
            'ci_low': intercept_gt,
            'ci_high': intercept_gt
        })
        
        # Add intercept data for each available method
        for method in available_methods:
            if method in method_betas:
                # Get intercept for this method
                intercept_value = 0.0
                for _, row in dd[dd['model_name'] == method].iterrows():
                    info, _ = destring_coeff(row)
                    if info is not None:
                        intercept_value = float(info.get('intercept', 0.0))
                        break
                
                coef_data.append({
                    'coef_name': r"$\beta_0$",
                    'method': method,
                    'value': intercept_value,
                    'ci_low': intercept_value,
                    'ci_high': intercept_value
                })
        
        # Add non-zero coefficients
        for i, j in enumerate(nz_idx):
            coef_data.append({
                'coef_name': rf"$\beta_{j+1}$",
                'method': 'GT',
                'value': beta_true[j],
                'ci_low': beta_true[j],
                'ci_high': beta_true[j]
            })
            
            # Add data for each available method
            for method in available_methods:
                if method in method_betas:
                    method_betas_array = method_betas[method]
                    mean_nz, ci_nz = mean_ci(method_betas_array[:, nz_idx], axis=0)
                    coef_data.append({
                        'coef_name': rf"$\beta_{j+1}$",
                        'method': method,
                        'value': mean_nz[i],
                        'ci_low': mean_nz[i] - ci_nz[i],
                        'ci_high': mean_nz[i] + ci_nz[i]
                    })
        
        # Create DataFrame and plot
        coef_df = pd.DataFrame(coef_data)
        coef_df['coef_name'] = pd.Categorical(coef_df['coef_name'], labels_math, ordered=True)
        coef_df.sort_values('coef_name', inplace=True)
        
        # Plot the coefficients
        methods = ['GT'] + available_methods
        w = 0.24
        x = np.arange(len(labels_math))
        
        # Draw bars + CI whiskers
        for i, m in enumerate(methods):
            sub = coef_df[coef_df["method"] == m]
            if not sub.empty:
                ax2.bar(x + (i-1)*w, sub["value"], width=w,
                        color=COEF_COLORS[m], edgecolor="black", linewidth=0.5, label=m)
                ax2.errorbar(
                    x + (i-1)*w, sub["value"],
                    yerr=[sub["value"]-sub["ci_low"], sub["ci_high"]-sub["value"]],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2
                )
        
        # Add thick red lines for best iteration values
        if method_betas:
            # Get best F1 iteration for each method
            best_iteration_values = {}
            for method in available_methods:
                if method in method_betas:
                    method_rows = dd[dd['model_name'] == method]
                    if not method_rows.empty:
                        best_row = method_rows.loc[method_rows['f1'].idxmax()]
                        info, _ = destring_coeff(best_row)
                        if info is not None and info["values"] is not None:
                            try:
                                beta_raw = to_raw_beta(info)
                                best_iteration_values[method] = beta_raw
                            except Exception as e:
                                print(f"Warning: Could not process best iteration coefficients for {method}: {e}")
                                continue
            
            # Add intercept values for best iteration
            if 'Lasso' in best_iteration_values:
                for _, row in dd[dd['model_name'] == 'Lasso'].iterrows():
                    info, _ = destring_coeff(row)
                    if info is not None:
                        best_iteration_values['Lasso'] = np.concatenate([[info.get('intercept', 0.0)], best_iteration_values['Lasso']])
                        break
            
            if 'NIMO' in best_iteration_values:
                for _, row in dd[dd['model_name'] == 'NIMO'].iterrows():
                    info, _ = destring_coeff(row)
                    if info is not None:
                        best_iteration_values['NIMO'] = np.concatenate([[info.get('intercept', 0.0)], best_iteration_values['NIMO']])
                        break
            
            # Add ground truth intercept
            best_iteration_values['GT'] = np.concatenate([[intercept_gt], beta_true[nz_idx]])
            
            # Draw red lines for best iteration values (excluding ground truth)
            for i, m in enumerate(["GT","Lasso","NIMO"]):
                if m in best_iteration_values and m != "GT":  # Skip ground truth
                    # Calculate x positions (same as bars)
                    x_positions = x + (i-1)*w
                    # Get best iteration values for non-zero coefficients (including intercept)
                    best_vals = best_iteration_values[m]
                    # Draw red horizontal lines at the actual coefficient values
                    for j, (x_pos, val) in enumerate(zip(x_positions, best_vals)):
                        # Add horizontal line at the actual value (thinner)
                        ax2.plot([x_pos-0.1, x_pos+0.1], [val, val], color='red', linewidth=2.5, alpha=0.9)
        
        # Style the coefficient plot
        ax2.set_title("Non-zero coefficients (across iterations)", fontsize=TITLE_SIZE)
        ax2.set_ylabel("Coefficient", fontsize=TICK_SIZE)
        ax2.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, labelsize=TICK_SIZE)
        
        # grey dotted y-grid, every 0.5 units
        from matplotlib.ticker import MultipleLocator
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.grid(True, linestyle="--", linewidth=1.0, color="#999999", alpha=0.7)
        ax2.xaxis.grid(False)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_math, rotation=25, ha='right')
        ax2.set_facecolor((1,1,1,0))
        
        # black frame
        for s in ax2.spines.values():
            s.set_color('black')
            s.set_linewidth(1.2)
        
        # Add red line explanation to the plot
        ax2.text(0.02, 0.98, "Red lines: best iteration values", 
                transform=ax2.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="red"),
                verticalalignment='top', horizontalalignment='left')
        
        # Scenario-specific legend positioning
        if scenario_id == "B":
            # Scenario B: legend in top right
            ax2.legend(
                frameon=True, fancybox=True, framealpha=0.9,
                borderpad=0.4, handlelength=1.5, labelspacing=0.5,
                prop={"size": 12},
                loc="upper right", bbox_to_anchor=(0.98, 0.98), bbox_transform=ax2.transAxes
            )
        elif scenario_id == "C":
            # Scenario C: legend in bottom right
            ax2.legend(
                frameon=True, fancybox=True, framealpha=0.9,
                borderpad=0.4, handlelength=1.5, labelspacing=0.5,
                prop={"size": 12},
                loc="lower right", bbox_to_anchor=(0.98, 0.02), bbox_transform=ax2.transAxes
            )
        else:
            # Default: legend in bottom left
            ax2.legend(
                frameon=True, fancybox=True, framealpha=0.9,
                borderpad=0.4, handlelength=1.5, labelspacing=0.5,
                prop={"size": 12},
                loc="lower left", bbox_to_anchor=(0.01, 0.02), bbox_transform=ax2.transAxes
            )
    else:
        ax2.text(0.5, 0.5, "No coefficient data available", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")
    
    # --- (C) Zero coefficients heatmap (best-F1 run) ---
    if len(z_idx) > 0:
        heatmap_data = []
        for method in available_methods:
            # pick the best F1 iteration for this method
            method_rows = dd[dd['model_name'] == method]
            if method_rows.empty:
                continue
            best_row = method_rows.loc[method_rows['f1'].idxmax()]

            info, _ = destring_coeff(best_row)
            if info is not None and info["values"] is not None:
                try:
                    beta_raw = to_raw_beta(info)
                    for j in z_idx:
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

            sns.heatmap(
                heatmap_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax3, cbar_kws={'shrink': 0.8}
            )
            ax3.set_title('Zero Coefficients (best-F1 run)', fontsize=TITLE_SIZE)
            ax3.set_xlabel('Feature')
            ax3.set_ylabel('Method')
        else:
            ax3.text(0.5, 0.5, "No zero coefficient data available", 
                     ha="center", va="center", transform=ax3.transAxes)
            ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "No zero coefficient data available", 
                ha="center", va="center", transform=ax3.transAxes)
        ax3.axis("off")
    
    # Adjust layout
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.12, wspace=0.35, hspace=0.42)
    
    # Always save the plot when run
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ Final plot saved to {save_path}")
    else:
        default_path = f"scenario_{scenario_id}_final.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ Final plot saved to {default_path}")
    
    plt.close()
    return fig

def main():
    """Main function to generate all final perfect plots."""
    print("=== Generating Final Perfect Plots ===")
    print("This will create the same perfect plots I see in windows!")
    
    # Load data
    results_path = '../../../results/synthetic/experiment_results.csv'
    df = pd.read_csv(results_path)
    
    # Filter dynamically from whatever is in the results file
    synthetic_datasets = sorted(df['dataset_id'].unique().tolist())
    df_synthetic = df[df['dataset_id'].isin(synthetic_datasets)].copy()
    
    print(f"Loaded {len(df_synthetic)} synthetic experiments")
    print(f"Methods: {df_synthetic['model_name'].unique()}")
    print(f"Scenarios: {synthetic_datasets}")
    
    # Generate all plots dynamically
    scenarios = synthetic_datasets
    saved_plots = []
    
    for scenario in scenarios:
        try:
            save_path = f"scenario_{scenario}_final.png"
            fig = create_final_plot(scenario, df_synthetic, save_path=save_path)
            saved_plots.append(save_path)
        except Exception as e:
            print(f"✗ Error creating plot for scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Successfully generated {len(saved_plots)} final plots!")
    print("Files saved:")
    for plot_path in saved_plots:
        print(f"  - {plot_path}")
    print("\nThese should look exactly like the perfect plots I see in windows!")

if __name__ == "__main__":
    main()
