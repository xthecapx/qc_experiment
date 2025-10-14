#!/usr/bin/env python3
"""
Error Rate vs Circuit Depth Analysis
====================================

This module creates boxplot visualizations showing error rate distributions
by circuit depth for IBM and Rigetti platforms.

Complement to plot 3 (same data, showing error rate instead of success rate).

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
from matplotlib.patches import Patch
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


def plot_error_rate_vs_circuit_depth_boxplot(df1, df2, bin_size=5):
    """
    Create boxplots showing error rate distributions by circuit depth.
    Style matches plots 1b with side-by-side comparison.
    
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
    
    print(f"Found circuit depth groups: {depth_groups}")
    
    # Prepare data for boxplots (side by side like plot 1b)
    ibm_data = []
    rigetti_data = []
    valid_depth_groups = []
    
    for depth_group in depth_groups:
        # IBM data for this depth group
        ibm_subset = df1_grouped[df1_grouped['depth_group'] == depth_group]
        # Rigetti data for this depth group
        rigetti_subset = df2_grouped[df2_grouped['depth_group'] == depth_group]
        
        # Only include if at least one platform has data
        if len(ibm_subset) > 0 or len(rigetti_subset) > 0:
            ibm_data.append(ibm_subset['error_rate'].values if len(ibm_subset) > 0 else [])
            rigetti_data.append(rigetti_subset['error_rate'].values if len(rigetti_subset) > 0 else [])
            valid_depth_groups.append(depth_group)
            print(f"Depth {depth_group}-{depth_group+bin_size-1}: IBM {len(ibm_subset)} samples, Rigetti {len(rigetti_subset)} samples")
    
    if not ibm_data and not rigetti_data:
        print("No data to plot!")
        return plt
    
    # Create positions for boxplots (side by side like plot 1b)
    # Use the bin start as the position
    positions_ibm = np.array(valid_depth_groups) - 1
    positions_rigetti = np.array(valid_depth_groups) + 1
    
    # Create boxplots matching plot 1b style
    # Filter out empty data for IBM
    ibm_positions = [pos for pos, data in zip(positions_ibm, ibm_data) if len(data) > 0]
    ibm_data_filtered = [data for data in ibm_data if len(data) > 0]
    
    if ibm_data_filtered:
        bp1 = ax.boxplot(ibm_data_filtered, positions=ibm_positions, widths=1.5, 
                         patch_artist=True, 
                         boxprops=dict(facecolor=COLORBREWER_PALETTE['IBM'], alpha=0.7),
                         medianprops=dict(color='black', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='black', markersize=8, alpha=0.8, markeredgecolor='black'))
    
    # Filter out empty data for Rigetti
    rigetti_positions = [pos for pos, data in zip(positions_rigetti, rigetti_data) if len(data) > 0]
    rigetti_data_filtered = [data for data in rigetti_data if len(data) > 0]
    
    if rigetti_data_filtered:
        bp2 = ax.boxplot(rigetti_data_filtered, positions=rigetti_positions, widths=1.5, 
                         patch_artist=True,
                         boxprops=dict(facecolor=COLORBREWER_PALETTE['Rigetti'], alpha=0.7),
                         medianprops=dict(color='black', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='black', markersize=8, alpha=0.8, markeredgecolor='black'))
    
    # Set axis labels (matching plot 1b)
    ax.set_xlabel('Circuit Depth', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels
    # Create labels like "5-9", "10-14", etc.
    tick_labels = [f"{d}-{d+bin_size-1}" for d in valid_depth_groups]
    ax.set_xticks(valid_depth_groups)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid (matching plot 1b)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.15, 1.15)
    
    # Create simple custom legend (matching plot 1b style)
    legend_elements = [
        Patch(facecolor=COLORBREWER_PALETTE['IBM'], alpha=0.7, label='IBM'),
        Patch(facecolor=COLORBREWER_PALETTE['Rigetti'], alpha=0.7, label='Rigetti')
    ]
    
    ax.legend(handles=legend_elements, fontsize=legend_size, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def run_error_rate_circuit_depth_analysis():
    """
    Main function to run the error rate vs circuit depth analysis.
    """
    print("=" * 60)
    print("ERROR RATE vs CIRCUIT DEPTH ANALYSIS")
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
    
    # Generate the boxplot
    print("\n" + "=" * 40)
    print("GENERATING BOXPLOT")
    print("=" * 40)
    
    plt_obj = plot_error_rate_vs_circuit_depth_boxplot(df1, df2, bin_size=5)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '3b_error_rate_vs_circuit_depth_boxplot.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Saved: {output_file}")
    print("Analysis complete!")
    
    return df1, df2


if __name__ == "__main__":
    # Run the analysis
    run_error_rate_circuit_depth_analysis()

