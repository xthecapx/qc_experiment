#!/usr/bin/env python3
"""
Error Rate Envelope Analysis
============================

This module creates a scatter plot with mean values and envelope (upper/lower bounds)
for error rate vs payload size, showing the range of variation.

Similar to scipy.signal.envelope but using simple statistical bounds.

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


def plot_error_rate_envelope_by_payload_size(df1, df2):
    """
    Create a scatter plot with mean values and envelope lines showing
    the upper and lower bounds of error rate by payload size.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        IBM dataset
    df2 : pandas.DataFrame
        Rigetti dataset
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
    
    # Get unique payload sizes
    payload_sizes = sorted(set(df1['payload_size'].unique()) | set(df2['payload_size'].unique()))
    
    print(f"Payload sizes: {payload_sizes}")
    
    # Calculate statistics for IBM
    ibm_means = []
    ibm_mins = []
    ibm_maxs = []
    ibm_valid_payloads = []
    
    for payload_size in payload_sizes:
        subset = df1[df1['payload_size'] == payload_size]
        if len(subset) > 0:
            ibm_means.append(subset['error_rate'].mean())
            ibm_mins.append(subset['error_rate'].min())
            ibm_maxs.append(subset['error_rate'].max())
            ibm_valid_payloads.append(payload_size)
            print(f"IBM Payload {payload_size}: mean={subset['error_rate'].mean():.3f}, "
                  f"min={subset['error_rate'].min():.3f}, max={subset['error_rate'].max():.3f}, n={len(subset)}")
    
    # Calculate statistics for Rigetti
    rigetti_means = []
    rigetti_mins = []
    rigetti_maxs = []
    rigetti_valid_payloads = []
    
    for payload_size in payload_sizes:
        subset = df2[df2['payload_size'] == payload_size]
        if len(subset) > 0:
            rigetti_means.append(subset['error_rate'].mean())
            rigetti_mins.append(subset['error_rate'].min())
            rigetti_maxs.append(subset['error_rate'].max())
            rigetti_valid_payloads.append(payload_size)
            print(f"Rigetti Payload {payload_size}: mean={subset['error_rate'].mean():.3f}, "
                  f"min={subset['error_rate'].min():.3f}, max={subset['error_rate'].max():.3f}, n={len(subset)}")
    
    # Convert to numpy arrays
    ibm_valid_payloads = np.array(ibm_valid_payloads)
    ibm_means = np.array(ibm_means)
    ibm_mins = np.array(ibm_mins)
    ibm_maxs = np.array(ibm_maxs)
    
    rigetti_valid_payloads = np.array(rigetti_valid_payloads)
    rigetti_means = np.array(rigetti_means)
    rigetti_mins = np.array(rigetti_mins)
    rigetti_maxs = np.array(rigetti_maxs)
    
    # Define better colors for differentiation
    # IBM: Use a darker teal/green for better contrast
    ibm_color = '#1b7837'  # Dark green
    # Rigetti: Use a darker orange for better contrast
    rigetti_color = '#d95f02'  # Dark orange
    
    # Plot IBM - signal (mean), low (min), high (max)
    if len(ibm_valid_payloads) > 0:
        ax.plot(ibm_valid_payloads, ibm_means, 
                color=ibm_color, linewidth=2.5, 
                marker='o', markersize=10, label='IBM', zorder=5)
        ax.plot(ibm_valid_payloads, ibm_mins, 
                color=ibm_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.plot(ibm_valid_payloads, ibm_maxs, 
                color=ibm_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        # Fill between min and max to create envelope
        ax.fill_between(ibm_valid_payloads, ibm_mins, ibm_maxs, 
                        color=ibm_color, alpha=0.2, zorder=1)
    
    # Plot Rigetti - signal (mean), low (min), high (max)
    if len(rigetti_valid_payloads) > 0:
        ax.plot(rigetti_valid_payloads, rigetti_means, 
                color=rigetti_color, linewidth=2.5, 
                marker='s', markersize=10, label='Rigetti', zorder=5)
        ax.plot(rigetti_valid_payloads, rigetti_mins, 
                color=rigetti_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.plot(rigetti_valid_payloads, rigetti_maxs, 
                color=rigetti_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        # Fill between min and max to create envelope
        ax.fill_between(rigetti_valid_payloads, rigetti_mins, rigetti_maxs, 
                        color=rigetti_color, alpha=0.2, zorder=2)
    
    # Set axis labels
    ax.set_xlabel('Payload Size', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Mean Error Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels
    ax.set_xticks(payload_sizes)
    ax.set_xticklabels(payload_sizes)
    ax.set_xlim(0.5, 5.5)
    
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


def run_error_rate_envelope_analysis():
    """
    Main function to run the error rate envelope analysis.
    """
    print("=" * 60)
    print("ERROR RATE ENVELOPE ANALYSIS")
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
    print(f"  Payload size range: {df1['payload_size'].min()} - {df1['payload_size'].max()}")
    print(f"  Error rate range: {df1['error_rate'].min():.3f} - {df1['error_rate'].max():.3f}")
    print(f"  Mean error rate: {df1['error_rate'].mean():.3f}")
    
    print(f"\nRigetti Dataset:")
    print(f"  Total experiments: {len(df2)}")
    print(f"  Payload size range: {df2['payload_size'].min()} - {df2['payload_size'].max()}")
    print(f"  Error rate range: {df2['error_rate'].min():.3f} - {df2['error_rate'].max():.3f}")
    print(f"  Mean error rate: {df2['error_rate'].mean():.3f}")
    
    # Generate the envelope plot
    print("\n" + "=" * 40)
    print("GENERATING ENVELOPE PLOT")
    print("=" * 40)
    
    plt_obj = plot_error_rate_envelope_by_payload_size(df1, df2)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '1c_error_rate_envelope_vs_payload_size.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Saved: {output_file}")
    print("Analysis complete!")
    
    return df1, df2


if __name__ == "__main__":
    # Run the analysis
    run_error_rate_envelope_analysis()

