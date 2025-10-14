#!/usr/bin/env python3
"""
Error Rate Envelope vs Circuit Depth Analysis
=============================================

This module creates a scatter plot with mean values and envelope (upper/lower bounds)
for error rate vs circuit depth, showing the range of variation.

Similar to plot 1c but for circuit depth instead of payload size.

Data Sources (via load_data module):
- IBM: All CSV files from ibm/ directory
- AWS/Rigetti: All CSV files from aws/ directory, plus root-level Rigetti files

Filters to circuit depth ≤ 50 for better visualization.

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from load_data import load_circuit_depth_datasets
from styles import COLORBREWER_PALETTE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, FIG_SIZE


# Output directory
OUTPUT_DIR = 'img'


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


def plot_error_rate_envelope_by_circuit_depth(df1, df2, bin_size=5):
    """
    Create a scatter plot with mean values and envelope lines showing
    the upper and lower bounds of error rate by circuit depth.
    
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
    
    # Calculate error rate for both datasets
    df1 = df1.copy()
    df2 = df2.copy()
    df1['error_rate'] = 1 - df1['success_rate']
    df2['error_rate'] = 1 - df2['success_rate']
    
    # Group circuit depths into bins
    df1_grouped = group_circuit_depths(df1, bin_size)
    df2_grouped = group_circuit_depths(df2, bin_size)
    
    # Get unique depth groups (sorted)
    depth_groups = sorted(set(df1_grouped['depth_group'].unique()) | set(df2_grouped['depth_group'].unique()))
    
    print(f"Circuit depth groups: {depth_groups}")
    
    # Calculate statistics for IBM
    ibm_means = []
    ibm_mins = []
    ibm_maxs = []
    ibm_valid_depths = []
    
    for depth_group in depth_groups:
        subset = df1_grouped[df1_grouped['depth_group'] == depth_group]
        if len(subset) > 0:
            ibm_means.append(subset['error_rate'].mean())
            ibm_mins.append(subset['error_rate'].min())
            ibm_maxs.append(subset['error_rate'].max())
            ibm_valid_depths.append(depth_group)
            print(f"IBM Depth {depth_group}-{depth_group+bin_size-1}: mean={subset['error_rate'].mean():.3f}, "
                  f"min={subset['error_rate'].min():.3f}, max={subset['error_rate'].max():.3f}, n={len(subset)}")
    
    # Calculate statistics for Rigetti
    rigetti_means = []
    rigetti_mins = []
    rigetti_maxs = []
    rigetti_valid_depths = []
    
    for depth_group in depth_groups:
        subset = df2_grouped[df2_grouped['depth_group'] == depth_group]
        if len(subset) > 0:
            rigetti_means.append(subset['error_rate'].mean())
            rigetti_mins.append(subset['error_rate'].min())
            rigetti_maxs.append(subset['error_rate'].max())
            rigetti_valid_depths.append(depth_group)
            print(f"Rigetti Depth {depth_group}-{depth_group+bin_size-1}: mean={subset['error_rate'].mean():.3f}, "
                  f"min={subset['error_rate'].min():.3f}, max={subset['error_rate'].max():.3f}, n={len(subset)}")
    
    # Convert to numpy arrays
    ibm_valid_depths = np.array(ibm_valid_depths)
    ibm_means = np.array(ibm_means)
    ibm_mins = np.array(ibm_mins)
    ibm_maxs = np.array(ibm_maxs)
    
    rigetti_valid_depths = np.array(rigetti_valid_depths)
    rigetti_means = np.array(rigetti_means)
    rigetti_mins = np.array(rigetti_mins)
    rigetti_maxs = np.array(rigetti_maxs)
    
    # Define better colors for differentiation (same as 1c)
    # IBM: Use a darker teal/green for better contrast
    ibm_color = '#1b7837'  # Dark green
    # Rigetti: Use a darker orange for better contrast
    rigetti_color = '#d95f02'  # Dark orange
    
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
    ax.set_ylabel('Mean Error Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels
    # Use the bin start values as tick positions
    tick_labels = [f"{d}-{d+bin_size-1}" for d in depth_groups]
    ax.set_xticks(depth_groups)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.15, 1.15)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='both')
    
    # Create legend - single column, positioned to avoid overlapping with data
    ax.legend(fontsize=legend_size, loc='upper left', ncol=1, framealpha=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def run_error_rate_envelope_circuit_depth_analysis():
    """
    Main function to run the error rate envelope circuit depth analysis.
    """
    print("=" * 60)
    print("ERROR RATE ENVELOPE vs CIRCUIT DEPTH ANALYSIS")
    print("=" * 60)
    
    # Load combined datasets
    try:
        df1, df2 = load_circuit_depth_datasets()
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Calculate error rates
    df1 = df1.copy()
    df2 = df2.copy()
    df1['error_rate'] = 1 - df1['success_rate']
    df2['error_rate'] = 1 - df2['success_rate']
    
    # Print dataset statistics
    print("\n" + "=" * 40)
    print("DATASET STATISTICS")
    print("=" * 40)
    
    print(f"\nIBM Dataset:")
    print(f"  Total experiments: {len(df1)}")
    print(f"  Circuit depth range: {df1['circuit_depth'].min()} - {df1['circuit_depth'].max()}")
    print(f"  Error rate range: {df1['error_rate'].min():.3f} - {df1['error_rate'].max():.3f}")
    print(f"  Mean error rate: {df1['error_rate'].mean():.3f}")
    
    print(f"\nRigetti Dataset:")
    print(f"  Total experiments: {len(df2)}")
    print(f"  Circuit depth range: {df2['circuit_depth'].min()} - {df2['circuit_depth'].max()}")
    print(f"  Error rate range: {df2['error_rate'].min():.3f} - {df2['error_rate'].max():.3f}")
    print(f"  Mean error rate: {df2['error_rate'].mean():.3f}")
    
    # Generate the envelope plot
    print("\n" + "=" * 40)
    print("GENERATING ENVELOPE PLOT")
    print("=" * 40)
    
    plt_obj = plot_error_rate_envelope_by_circuit_depth(df1, df2, bin_size=5)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '3c_error_rate_envelope_vs_circuit_depth.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Saved: {output_file}")
    print("Analysis complete!")
    
    return df1, df2


if __name__ == "__main__":
    # Run the analysis
    run_error_rate_envelope_circuit_depth_analysis()

