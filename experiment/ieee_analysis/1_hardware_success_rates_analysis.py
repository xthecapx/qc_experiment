#!/usr/bin/env python3
"""
Hardware Success Rates Analysis
===============================

This module creates boxplot visualizations showing success rate distributions
by payload size and hardware platform (IBM and Rigetti).

Data Sources:
- IBM: 2_experiment_results_target_depth.csv, 3_experiment_results.csv
- Rigetti: experiment_results_target_depth_20250903_221822_updated.csv

This module was extracted from analysis.py to improve maintainability and
follow the same structure as success_rate_gates_analysis.py.

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from load_data import load_combined_hardware_data
from styles import COLORBREWER_PALETTE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, FIG_SIZE


# ColorBrewer palette for consistent colors
# COLORBREWER_PALETTE now imported from styles

# Output directory
OUTPUT_DIR = 'img'


def plot_success_rates_boxplot_by_hardware(df):
    """
    Create boxplots showing success rate distributions by payload size and hardware platform.
    Style matches plot 2 with side-by-side comparison and payload size labels only.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results with 'hardware' column
    """
    # Set font sizes for IEEE format
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get unique hardware platforms and payload sizes
    hardware_platforms = sorted(df['hardware'].unique())
    payload_sizes = sorted(df['payload_size'].unique())
    
    print(f"Found payload sizes: {payload_sizes}")
    print(f"Hardware platforms: {hardware_platforms}")
    
    # Prepare data for boxplots (side by side like plot 2)
    ibm_data = []
    rigetti_data = []
    
    for payload_size in payload_sizes:
        # IBM data for this payload size (convert from percentage to 0-1 scale)
        ibm_subset = df[(df['payload_size'] == payload_size) & (df['hardware'] == 'IBM')]
        ibm_data.append(ibm_subset['success_rate'].values / 100)
        
        # Rigetti data for this payload size (convert from percentage to 0-1 scale)
        rigetti_subset = df[(df['payload_size'] == payload_size) & (df['hardware'] == 'Rigetti')]
        rigetti_data.append(rigetti_subset['success_rate'].values / 100)
        
        print(f"Payload {payload_size}: IBM {len(ibm_subset)} samples, Rigetti {len(rigetti_subset)} samples")
    
    if not ibm_data and not rigetti_data:
        print("No data to plot!")
        return plt
    
    # Create positions for boxplots (side by side like plot 2)
    positions_ibm = np.array(payload_sizes) - 0.2
    positions_rigetti = np.array(payload_sizes) + 0.2
    
    # Create boxplots matching plot 2 style
    bp1 = ax.boxplot(ibm_data, positions=positions_ibm, widths=0.3, 
                     patch_artist=True, 
                     boxprops=dict(facecolor=COLORBREWER_PALETTE['IBM'], alpha=0.7),
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor='black', markersize=8, alpha=0.8, markeredgecolor='black'))
    
    bp2 = ax.boxplot(rigetti_data, positions=positions_rigetti, widths=0.3, 
                     patch_artist=True,
                     boxprops=dict(facecolor=COLORBREWER_PALETTE['Rigetti'], alpha=0.7),
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor='black', markersize=8, alpha=0.8, markeredgecolor='black'))
    
    # Set axis labels (matching plot 2)
    ax.set_xlabel('Payload Size', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels with padding (matching plot 2)
    ax.set_xticks(payload_sizes)
    ax.set_xticklabels(payload_sizes)
    ax.set_xlim(0.5, 5.5)  # Add padding around the payload sizes
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.15, 1.15)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid (matching plot 2)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Create simple custom legend (matching plot 2 style)
    legend_elements = [
        Patch(facecolor=COLORBREWER_PALETTE['IBM'], alpha=0.7, label='IBM'),
        Patch(facecolor=COLORBREWER_PALETTE['Rigetti'], alpha=0.7, label='Rigetti')
    ]
    
    ax.legend(handles=legend_elements, fontsize=legend_size, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def run_hardware_success_rates_analysis():
    """
    Main function to run the hardware success rates analysis.
    """
    print("=" * 60)
    print("HARDWARE SUCCESS RATES ANALYSIS")
    print("=" * 60)
    
    # Load combined data
    combined_df = load_combined_hardware_data()
    
    if combined_df.empty:
        print("No data available for analysis.")
        return
    
    # Print dataset statistics
    print("\n" + "=" * 40)
    print("DATASET STATISTICS BY HARDWARE AND PAYLOAD")
    print("=" * 40)
    
    for hardware in combined_df['hardware'].unique():
        hw_data = combined_df[combined_df['hardware'] == hardware]
        print(f"\n{hardware} Statistics:")
        print(f"  Total experiments: {len(hw_data)}")
        print(f"  Payload size range: {hw_data['payload_size'].min()} - {hw_data['payload_size'].max()}")
        print(f"  Success rate range: {hw_data['success_rate'].min():.2f}% - {hw_data['success_rate'].max():.2f}%")
        print(f"  Mean success rate: {hw_data['success_rate'].mean():.2f}%")
        
        # Show data distribution by payload sizes
        print(f"  Distribution by payload sizes:")
        for payload_size in sorted(hw_data['payload_size'].unique()):
            payload_data = hw_data[hw_data['payload_size'] == payload_size]
            print(f"    Payload {payload_size}: {len(payload_data)} experiments, "
                  f"mean: {payload_data['success_rate'].mean():.2f}%, "
                  f"std: {payload_data['success_rate'].std():.2f}%")
    
    # Generate the boxplot
    print("\n" + "=" * 40)
    print("GENERATING BOXPLOT")
    print("=" * 40)
    
    plt_obj = plot_success_rates_boxplot_by_hardware(combined_df)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '1_combined_hardware_success_rates_boxplot.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Plot saved to: {output_file}")
    print("Analysis complete!")
    
    return combined_df


if __name__ == "__main__":
    # Run the analysis
    run_hardware_success_rates_analysis()
