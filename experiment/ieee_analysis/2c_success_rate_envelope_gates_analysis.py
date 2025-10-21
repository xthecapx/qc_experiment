#!/usr/bin/env python3
"""
Success Rate Envelope vs Number of Gates Analysis
=================================================

This module creates a scatter plot with mean values and envelope (upper/lower bounds)
for success rate vs number of gates, showing the range of variation.

Similar to plots 1c and 3c but for gate count.

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from load_data import load_combined_hardware_data
from styles import COLORBREWER_PALETTE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, FIG_SIZE


# Output directory
OUTPUT_DIR = 'img'


def format_gate_count_scientific(gate_count):
    """Format gate count in scientific notation for better readability."""
    if gate_count >= 10000:
        return f"{gate_count//1000}E3"
    elif gate_count >= 1000:
        if gate_count == 1010:
            return "1E3"
        elif gate_count == 1510:
            return "1.5E3"
        elif gate_count == 3010:
            return "3E3"
        elif gate_count == 5010:
            return "5E3"
        elif gate_count == 7010:
            return "7E3"
        else:
            return f"{gate_count//1000}E3"
    elif gate_count >= 100:
        if gate_count == 210:
            return "2E2"
        elif gate_count == 510:
            return "5E2"
        else:
            return f"{gate_count//100}E2"
    else:
        return str(gate_count)


def group_gates_by_range(df, range_size=10):
    """Group gates into ranges (e.g., 200±10 becomes 190-210)."""
    df = df.copy()
    df['gate_group'] = ((df['num_gates'] // (range_size * 2)) * (range_size * 2)) + range_size
    df['gate_group_label'] = df['gate_group'].apply(format_gate_count_scientific)
    return df


def plot_success_rate_envelope_by_gates(df, min_samples=5):
    """
    Create a scatter plot with mean values and envelope lines showing
    the upper and lower bounds of success rate by number of gates.
    """
    # Set font sizes for IEEE format
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Use success rate directly
    df = df.copy()
    df['success_rate_scaled'] = df['success_rate'] / 100  # Convert to 0-1 scale
    
    # Group gates into ranges
    df_grouped = group_gates_by_range(df, range_size=10)
    
    # Get unique gate groups
    gate_groups = sorted(df_grouped['gate_group'].unique())
    
    print(f"Gate groups: {gate_groups}")
    
    # Calculate statistics for IBM
    ibm_means = []
    ibm_mins = []
    ibm_maxs = []
    ibm_valid_gates = []
    ibm_labels = []
    
    for gate_group in gate_groups:
        subset = df_grouped[(df_grouped['gate_group'] == gate_group) & (df_grouped['hardware'] == 'IBM')]
        if len(subset) >= min_samples:
            ibm_means.append(subset['success_rate_scaled'].mean())
            ibm_mins.append(subset['success_rate_scaled'].min())
            ibm_maxs.append(subset['success_rate_scaled'].max())
            ibm_valid_gates.append(gate_group)
            ibm_labels.append(subset['gate_group_label'].iloc[0])
            print(f"IBM Gates {subset['gate_group_label'].iloc[0]}: mean={subset['success_rate_scaled'].mean():.3f}, "
                  f"min={subset['success_rate_scaled'].min():.3f}, max={subset['success_rate_scaled'].max():.3f}, n={len(subset)}")
        elif len(subset) > 0:
            print(f"  ⚠️  IBM Gates {subset['gate_group_label'].iloc[0]}: Filtered (only {len(subset)} samples)")
    
    # Calculate statistics for Rigetti
    rigetti_means = []
    rigetti_mins = []
    rigetti_maxs = []
    rigetti_valid_gates = []
    rigetti_labels = []
    
    for gate_group in gate_groups:
        subset = df_grouped[(df_grouped['gate_group'] == gate_group) & (df_grouped['hardware'] == 'Rigetti')]
        if len(subset) >= min_samples:
            rigetti_means.append(subset['success_rate_scaled'].mean())
            rigetti_mins.append(subset['success_rate_scaled'].min())
            rigetti_maxs.append(subset['success_rate_scaled'].max())
            rigetti_valid_gates.append(gate_group)
            rigetti_labels.append(subset['gate_group_label'].iloc[0])
            print(f"Rigetti Gates {subset['gate_group_label'].iloc[0]}: mean={subset['success_rate_scaled'].mean():.3f}, "
                  f"min={subset['success_rate_scaled'].min():.3f}, max={subset['success_rate_scaled'].max():.3f}, n={len(subset)}")
        elif len(subset) > 0:
            print(f"  ⚠️  Rigetti Gates {subset['gate_group_label'].iloc[0]}: Filtered (only {len(subset)} samples)")
    
    # Convert to numpy arrays
    ibm_valid_gates = np.array(ibm_valid_gates)
    ibm_means = np.array(ibm_means)
    ibm_mins = np.array(ibm_mins)
    ibm_maxs = np.array(ibm_maxs)
    
    rigetti_valid_gates = np.array(rigetti_valid_gates)
    rigetti_means = np.array(rigetti_means)
    rigetti_mins = np.array(rigetti_mins)
    rigetti_maxs = np.array(rigetti_maxs)
    
    # Define better colors for differentiation (same as 1c and 3c)
    ibm_color = '#1b7837'  # Dark green
    rigetti_color = '#d95f02'  # Dark orange
    
    # Plot IBM
    if len(ibm_valid_gates) > 0:
        ax.plot(ibm_valid_gates, ibm_means, 
                color=ibm_color, linewidth=2.5, 
                marker='o', markersize=10, label='IBM', zorder=5)
        ax.plot(ibm_valid_gates, ibm_mins, 
                color=ibm_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.plot(ibm_valid_gates, ibm_maxs, 
                color=ibm_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.fill_between(ibm_valid_gates, ibm_mins, ibm_maxs, 
                        color=ibm_color, alpha=0.2, zorder=1)
    
    # Plot Rigetti
    if len(rigetti_valid_gates) > 0:
        ax.plot(rigetti_valid_gates, rigetti_means, 
                color=rigetti_color, linewidth=2.5, 
                marker='s', markersize=10, label='Rigetti', zorder=5)
        ax.plot(rigetti_valid_gates, rigetti_mins, 
                color=rigetti_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.plot(rigetti_valid_gates, rigetti_maxs, 
                color=rigetti_color, linewidth=1.5, 
                linestyle='--', alpha=0.8, zorder=4)
        ax.fill_between(rigetti_valid_gates, rigetti_mins, rigetti_maxs, 
                        color=rigetti_color, alpha=0.2, zorder=2)
    
    # Set axis labels
    ax.set_xlabel('Number of Gates', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Mean Success Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis to log scale for better visualization
    ax.set_xscale('log')
    
    # Set x-axis ticks
    all_gates = sorted(set(ibm_valid_gates.tolist() + rigetti_valid_gates.tolist()))
    if len(all_gates) > 0:
        ax.set_xticks(all_gates)
        # Get labels for ticks
        tick_labels = []
        for gate in all_gates:
            if gate in ibm_valid_gates:
                idx = np.where(ibm_valid_gates == gate)[0][0]
                tick_labels.append(ibm_labels[idx])
            elif gate in rigetti_valid_gates:
                idx = np.where(rigetti_valid_gates == gate)[0][0]
                tick_labels.append(rigetti_labels[idx])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.15, 1.15)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='both')
    
    # Create legend
    ax.legend(fontsize=legend_size, loc='upper left', ncol=1, framealpha=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def run_success_rate_envelope_gates_analysis():
    """Main function to run the success rate envelope gates analysis."""
    print("=" * 60)
    print("SUCCESS RATE ENVELOPE vs NUMBER OF GATES ANALYSIS")
    print("=" * 60)
    
    # Load combined data
    combined_df = load_combined_hardware_data()
    
    if combined_df.empty:
        print("No data available for analysis.")
        return
    
    # Print dataset statistics
    print("\n" + "=" * 40)
    print("DATASET STATISTICS")
    print("=" * 40)
    
    for hardware in combined_df['hardware'].unique():
        hw_data = combined_df[combined_df['hardware'] == hardware]
        print(f"\n{hardware} Statistics:")
        print(f"  Total experiments: {len(hw_data)}")
        print(f"  Gate range: {hw_data['num_gates'].min()} - {hw_data['num_gates'].max()}")
        print(f"  Success rate range: {hw_data['success_rate'].min():.2f}% - {hw_data['success_rate'].max():.2f}%")
        print(f"  Mean success rate: {hw_data['success_rate'].mean():.2f}%")
    
    # Generate the envelope plot
    print("\n" + "=" * 40)
    print("GENERATING ENVELOPE PLOT")
    print("=" * 40)
    
    plt_obj = plot_success_rate_envelope_by_gates(combined_df, min_samples=5)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '2c_success_rate_envelope_vs_gates.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Saved: {output_file}")
    print("Analysis complete!")
    
    return combined_df


if __name__ == "__main__":
    run_success_rate_envelope_gates_analysis()


