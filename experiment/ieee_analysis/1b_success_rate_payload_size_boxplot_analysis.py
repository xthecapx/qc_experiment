#!/usr/bin/env python3
"""
Payload Size Boxplot Analysis
=============================

This module creates boxplot visualizations showing success rate distribution by payload size
for combined datasets (IBM and AWS/Rigetti). Focuses on data distribution without circuit depth
coloring or trend analysis.

Data Sources (via load_data module):
- IBM: All CSV files from ibm/ directory
- AWS/Rigetti: All CSV files from aws/ directory, plus root-level Rigetti files

This module was extracted from target_depth_analysis.py to improve maintainability and
follow the same structure as other analysis modules.

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



def plot_success_rate_vs_payload_size_combined_boxplot(df1, df2):
    """
    Create boxplots showing success rate distribution vs payload size for both datasets
    Focuses on data distribution without circuit depth coloring or trend analysis
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First dataset (IBM Sherbrooke)
    df2 : pandas.DataFrame
        Second dataset (Rigetti Ankaa-3)
    """
    # IEEE format settings - optimized proportions
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
    
    # Filter out rows with 0 success rate (likely data errors)
    df1_filtered = df1[df1['success_rate'] > 0].copy()
    df2_filtered = df2[df2['success_rate'] > 0].copy()
    
    print(f"IBM data: {len(df1)} total, {len(df1_filtered)} after filtering zero success rates")
    print(f"Rigetti data: {len(df2)} total, {len(df2_filtered)} after filtering zero success rates")
    
    # Get unique payload sizes (assuming both datasets have the same range)
    payload_sizes = sorted(df1_filtered['payload_size'].unique())
    print(f"Payload sizes found: {payload_sizes}")
    
    # Prepare data for boxplots
    ibm_data = []
    rigetti_data = []
    
    for payload_size in payload_sizes:
        # IBM data for this payload size (filtered)
        ibm_subset = df1_filtered[df1_filtered['payload_size'] == payload_size]['success_rate']
        ibm_data.append(ibm_subset.values)
        
        # Rigetti data for this payload size (filtered)
        rigetti_subset = df2_filtered[df2_filtered['payload_size'] == payload_size]['success_rate']
        rigetti_data.append(rigetti_subset.values)
        
        print(f"Payload {payload_size}: IBM {len(ibm_subset)} samples, Rigetti {len(rigetti_subset)} samples")
    
    # Create positions for boxplots (side by side)
    positions_ibm = np.array(payload_sizes) - 0.2
    positions_rigetti = np.array(payload_sizes) + 0.2
    
    # Create boxplots
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
    
    # Set axis labels
    ax.set_xlabel('Payload Size', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels with padding
    ax.set_xticks(payload_sizes)
    ax.set_xticklabels(payload_sizes)
    ax.set_xlim(0.5, 5.5)  # Add padding around the payload sizes
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Create simple custom legend (matching 1_ analysis style)
    legend_elements = [
        Patch(facecolor=COLORBREWER_PALETTE['IBM'], alpha=0.7, label='IBM'),
        Patch(facecolor=COLORBREWER_PALETTE['Rigetti'], alpha=0.7, label='Rigetti')
    ]
    
    ax.legend(handles=legend_elements, fontsize=legend_size, loc='upper left')
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.15, 1.15)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt, {
        'payload_sizes': payload_sizes,
        'ibm_counts': [len(v) for v in ibm_data],
        'rigetti_counts': [len(v) for v in rigetti_data]
    }


def run_payload_size_boxplot_analysis():
    """
    Main function to run the payload size boxplot analysis.
    """
    print("=" * 60)
    print("PAYLOAD SIZE BOXPLOT ANALYSIS")
    print("=" * 60)
    
    # Load combined datasets
    try:
        df1, df2 = load_circuit_depth_datasets()
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Print dataset statistics
    print("\n" + "=" * 40)
    print("DATASET STATISTICS")
    print("=" * 40)
    
    print(f"\nIBM Dataset:")
    print(f"  Total experiments: {len(df1)}")
    print(f"  Payload size range: {df1['payload_size'].min()} - {df1['payload_size'].max()}")
    print(f"  Success rate range: {df1['success_rate'].min():.3f} - {df1['success_rate'].max():.3f}")
    print(f"  Mean success rate: {df1['success_rate'].mean():.3f}")
    
    print(f"\nRigetti Dataset:")
    print(f"  Total experiments: {len(df2)}")
    print(f"  Payload size range: {df2['payload_size'].min()} - {df2['payload_size'].max()}")
    print(f"  Success rate range: {df2['success_rate'].min():.3f} - {df2['success_rate'].max():.3f}")
    print(f"  Mean success rate: {df2['success_rate'].mean():.3f}")
    
    # Generate the boxplot
    print("\n" + "=" * 40)
    print("GENERATING BOXPLOT")
    print("=" * 40)
    
    plt_obj, data_meta = plot_success_rate_vs_payload_size_combined_boxplot(df1, df2)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '1b_success_rate_vs_payload_size_combined_boxplot.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Saved: {output_file}")
    # Optionally export sample counts per payload for appendix
    try:
        counts_df = pd.DataFrame({
            'payload_size': data_meta['payload_sizes'],
            'ibm_samples': data_meta['ibm_counts'],
            'rigetti_samples': data_meta['rigetti_counts']
        })
        counts_df.to_csv(os.path.join(output_dir, 'payload_boxplot_sample_counts.csv'), index=False)
        print('Exported payload-size sample counts CSV for appendix.')
    except Exception as e:
        print(f'Could not export payload-size counts: {e}')
    print("Analysis complete!")
    
    return df1, df2


if __name__ == "__main__":
    # Run the analysis
    run_payload_size_boxplot_analysis()

