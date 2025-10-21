#!/usr/bin/env python3
"""
Filtered Success Rate Correlation vs Circuit Depth Analysis
===========================================================

This module creates a scatter plot showing the correlation between circuit depth 
and success rate for lower depth ranges where correlation is observable.

Filters:
- IBM: Circuit depths 5-19 (where correlation is visible)
- Rigetti: Circuit depths 10-24 (where correlation is visible)

This focused view demonstrates the correlation at lower depths before
payload size dominates the success rate.

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from load_data import load_circuit_depth_datasets
from styles import COLORBREWER_PALETTE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, FIG_SIZE


# Output directory
OUTPUT_DIR = 'img'


def calculate_correlation_stats(depths, success_rates):
    """
    Calculate correlation statistics between circuit depth and success rate.
    
    Parameters:
    -----------
    depths : array-like
        Circuit depth values
    success_rates : array-like
        Success rate values
    
    Returns:
    --------
    dict
        Dictionary containing correlation statistics:
        - pearson_r: Pearson correlation coefficient
        - pearson_p: P-value for Pearson correlation
        - spearman_r: Spearman rank correlation coefficient
        - spearman_p: P-value for Spearman correlation
        - r_squared: R² from linear regression
        - slope: Slope from linear regression
        - intercept: Intercept from linear regression
    """
    # Pearson correlation (parametric - assumes linear relationship)
    pearson_r, pearson_p = stats.pearsonr(depths, success_rates)
    
    # Spearman correlation (non-parametric - for monotonic relationships)
    spearman_r, spearman_p = stats.spearmanr(depths, success_rates)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(depths, success_rates)
    r_squared = r_value ** 2
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'regression_p': p_value
    }


def format_correlation_stats(stats_dict, platform_name):
    """
    Format correlation statistics for display.
    
    Parameters:
    -----------
    stats_dict : dict
        Dictionary from calculate_correlation_stats
    platform_name : str
        Name of the platform (IBM or Rigetti)
    
    Returns:
    --------
    str
        Formatted string with statistics
    """
    lines = [
        f"\n{platform_name} Correlation Statistics:",
        f"  Pearson r = {stats_dict['pearson_r']:.4f} (p = {stats_dict['pearson_p']:.4e})",
        f"  Spearman ρ = {stats_dict['spearman_r']:.4f} (p = {stats_dict['spearman_p']:.4e})",
        f"  Linear Regression:",
        f"    R² = {stats_dict['r_squared']:.4f}",
        f"    Slope = {stats_dict['slope']:.4f}",
        f"    p-value = {stats_dict['regression_p']:.4e}"
    ]
    
    # Add significance interpretation
    if stats_dict['pearson_p'] < 0.001:
        lines.append(f"  → Highly significant correlation (p < 0.001)")
    elif stats_dict['pearson_p'] < 0.01:
        lines.append(f"  → Very significant correlation (p < 0.01)")
    elif stats_dict['pearson_p'] < 0.05:
        lines.append(f"  → Significant correlation (p < 0.05)")
    else:
        lines.append(f"  → Not statistically significant (p ≥ 0.05)")
    
    return '\n'.join(lines)



def group_circuit_depths(df, bin_size=5):
    """
    Group circuit depths into bins for better visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'circuit_depth' column
    bin_size : int
        Size of each bin (default: 5)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'depth_group' column
    """
    df = df.copy()
    # Create bins: 5-9, 10-14, 15-19, etc.
    df['depth_group'] = ((df['circuit_depth'] // bin_size) * bin_size)
    return df


def plot_filtered_success_rate_correlation(df1, df2, bin_size=5):
    """
    Create a scatter plot showing success rate correlation for lower circuit depths.
    
    Filters data to show only the ranges where correlation is visible:
    - IBM: depths 5-19
    - Rigetti: depths 10-24
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        IBM dataset
    df2 : pandas.DataFrame
        Rigetti dataset
    bin_size : int
        Size of circuit depth bins for grouping (default: 5)
    """
    # Set font sizes for IEEE format
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Work directly with success rate
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Filter to lower depth ranges where correlation is visible
    # IBM: 5-19 (includes 5-9, 10-14, 15-19)
    df1_filtered = df1[(df1['circuit_depth'] >= 5) & (df1['circuit_depth'] <= 19)]
    # Rigetti: 10-24 (includes 10-14, 15-19, 20-24)
    df2_filtered = df2[(df2['circuit_depth'] >= 10) & (df2['circuit_depth'] <= 24)]
    
    print(f"\nFiltered datasets:")
    print(f"IBM: {len(df1_filtered)} experiments (depths 5-19)")
    print(f"Rigetti: {len(df2_filtered)} experiments (depths 10-24)")
    
    # Calculate correlation statistics on raw data (all individual points)
    ibm_stats = calculate_correlation_stats(df1_filtered['circuit_depth'].values, 
                                           df1_filtered['success_rate'].values)
    rigetti_stats = calculate_correlation_stats(df2_filtered['circuit_depth'].values, 
                                               df2_filtered['success_rate'].values)
    
    # Print correlation statistics
    print(format_correlation_stats(ibm_stats, "IBM"))
    print(format_correlation_stats(rigetti_stats, "Rigetti"))
    
    # Group circuit depths into bins
    df1_grouped = group_circuit_depths(df1_filtered, bin_size)
    df2_grouped = group_circuit_depths(df2_filtered, bin_size)
    
    # Get unique depth groups (sorted)
    ibm_depth_groups = sorted(df1_grouped['depth_group'].unique())
    rigetti_depth_groups = sorted(df2_grouped['depth_group'].unique())
    
    print(f"\nIBM circuit depth groups: {ibm_depth_groups}")
    print(f"Rigetti circuit depth groups: {rigetti_depth_groups}")
    
    # Calculate statistics for IBM
    ibm_means = []
    ibm_mins = []
    ibm_maxs = []
    ibm_valid_depths = []
    
    for depth_group in ibm_depth_groups:
        subset = df1_grouped[df1_grouped['depth_group'] == depth_group]
        if len(subset) > 0:
            ibm_means.append(subset['success_rate'].mean())
            ibm_mins.append(subset['success_rate'].min())
            ibm_maxs.append(subset['success_rate'].max())
            ibm_valid_depths.append(depth_group)
            print(f"IBM Depth {depth_group}-{depth_group+bin_size-1}: mean={subset['success_rate'].mean():.3f}, "
                  f"min={subset['success_rate'].min():.3f}, max={subset['success_rate'].max():.3f}, n={len(subset)}")
    
    # Calculate statistics for Rigetti
    rigetti_means = []
    rigetti_mins = []
    rigetti_maxs = []
    rigetti_valid_depths = []
    
    for depth_group in rigetti_depth_groups:
        subset = df2_grouped[df2_grouped['depth_group'] == depth_group]
        if len(subset) > 0:
            rigetti_means.append(subset['success_rate'].mean())
            rigetti_mins.append(subset['success_rate'].min())
            rigetti_maxs.append(subset['success_rate'].max())
            rigetti_valid_depths.append(depth_group)
            print(f"Rigetti Depth {depth_group}-{depth_group+bin_size-1}: mean={subset['success_rate'].mean():.3f}, "
                  f"min={subset['success_rate'].min():.3f}, max={subset['success_rate'].max():.3f}, n={len(subset)}")
    
    # Convert to numpy arrays
    ibm_valid_depths = np.array(ibm_valid_depths)
    ibm_means = np.array(ibm_means)
    ibm_mins = np.array(ibm_mins)
    ibm_maxs = np.array(ibm_maxs)
    
    rigetti_valid_depths = np.array(rigetti_valid_depths)
    rigetti_means = np.array(rigetti_means)
    rigetti_mins = np.array(rigetti_mins)
    rigetti_maxs = np.array(rigetti_maxs)
    
    # Define colors from ColorBrewer palette
    ibm_color = COLORBREWER_PALETTE['IBM']
    rigetti_color = COLORBREWER_PALETTE['Rigetti']
    
    # Plot IBM - signal (mean), low (min), high (max)
    if len(ibm_valid_depths) > 0:
        ax.plot(ibm_valid_depths, ibm_means, 
                color=ibm_color, linewidth=2.5, 
                marker='o', markersize=10, label='IBM', zorder=5)
        ax.plot(ibm_valid_depths, ibm_mins, 
                color=ibm_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.plot(ibm_valid_depths, ibm_maxs, 
                color=ibm_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        # Fill between min and max to create envelope
        ax.fill_between(ibm_valid_depths, ibm_mins, ibm_maxs, 
                        color=ibm_color, alpha=0.2, zorder=1)
    
    # Plot Rigetti - signal (mean), low (min), high (max)
    if len(rigetti_valid_depths) > 0:
        ax.plot(rigetti_valid_depths, rigetti_means, 
                color=rigetti_color, linewidth=2.5, 
                marker='s', markersize=10, label='Rigetti', zorder=5)
        ax.plot(rigetti_valid_depths, rigetti_mins, 
                color=rigetti_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.plot(rigetti_valid_depths, rigetti_maxs, 
                color=rigetti_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        # Fill between min and max to create envelope
        ax.fill_between(rigetti_valid_depths, rigetti_mins, rigetti_maxs, 
                        color=rigetti_color, alpha=0.2, zorder=2)
    
    # Set axis labels
    ax.set_xlabel('Circuit Depth', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Mean Success Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels for the filtered range
    all_depths = sorted(set(ibm_depth_groups) | set(rigetti_depth_groups))
    tick_labels = [f"{d}-{d+bin_size-1}" for d in all_depths]
    ax.set_xticks(all_depths)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.15, 1.15)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='both')
    
    # Add correlation statistics as text box
    rigetti_p_text = '< 0.001' if rigetti_stats['pearson_p'] < 0.001 else f"= {rigetti_stats['pearson_p']:.3f}"
    stats_text = (
        f"IBM: r = {ibm_stats['pearson_r']:.3f}, R² = {ibm_stats['r_squared']:.3f}, p < 0.001\n"
        f"Rigetti: r = {rigetti_stats['pearson_r']:.3f}, R² = {rigetti_stats['r_squared']:.3f}, "
        f"p {rigetti_p_text}"
    )
    
    # Position text box in upper right (adjust if needed based on data)
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=legend_size-2,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            family='monospace')
    
    # Create legend
    ax.legend(fontsize=legend_size, loc='upper right', ncol=1, framealpha=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt, ibm_stats, rigetti_stats


def run_filtered_correlation_analysis():
    """
    Main function to run the filtered success rate correlation analysis.
    """
    print("=" * 60)
    print("FILTERED SUCCESS RATE CORRELATION vs CIRCUIT DEPTH")
    print("=" * 60)
    
    # Load combined datasets
    try:
        df1, df2 = load_circuit_depth_datasets()
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Print dataset statistics
    print("\n" + "=" * 40)
    print("FULL DATASET STATISTICS")
    print("=" * 40)
    
    print(f"\nIBM Dataset:")
    print(f"  Total experiments: {len(df1)}")
    print(f"  Circuit depth range: {df1['circuit_depth'].min()} - {df1['circuit_depth'].max()}")
    
    print(f"\nRigetti Dataset:")
    print(f"  Total experiments: {len(df2)}")
    print(f"  Circuit depth range: {df2['circuit_depth'].min()} - {df2['circuit_depth'].max()}")
    
    # Generate the filtered correlation plot
    print("\n" + "=" * 40)
    print("GENERATING FILTERED CORRELATION PLOT")
    print("=" * 40)
    
    plt_obj, ibm_stats, rigetti_stats = plot_filtered_success_rate_correlation(df1, df2, bin_size=5)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '3c_filtered_success_rate_correlation_circuit_depth.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✓ Saved: {output_file}")
    print("Analysis complete!")
    
    return df1, df2


if __name__ == "__main__":
    # Run the analysis
    run_filtered_correlation_analysis()

