#!/usr/bin/env python3
"""
Error Rate vs Number of Gates Analysis
======================================

This module creates boxplot visualizations showing error rate distributions
by number of gates for different hardware platforms (IBM and Rigetti).

Complement to plot 2 (same data, showing error rate instead of success rate).

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


def plot_error_rate_vs_gates_boxplot(df, group_spacing=0.5, min_samples=5):
    """
    Create boxplots showing error rate distributions by number of gates and hardware platform.
    """
    # Set font sizes for IEEE format
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    
    # Calculate error rate
    df = df.copy()
    df['error_rate'] = (100 - df['success_rate']) / 100  # Convert to 0-1 scale
    
    # Group gates into ranges
    df_grouped = group_gates_by_range(df, range_size=10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get unique hardware platforms and gate groups
    hardware_platforms = sorted(df_grouped['hardware'].unique())
    gate_groups = sorted(df_grouped['gate_group'].unique())
    
    print(f"Found gate groups: {gate_groups}")
    print(f"Hardware platforms: {hardware_platforms}")
    
    # Create boxplot data structure with grouped positioning
    boxplot_data = []
    positions = []
    colors = []
    gate_group_positions = []
    gate_group_labels = []
    
    pos_counter = 0
    filtered_count = 0
    for gate_group in gate_groups:
        gate_data = df_grouped[df_grouped['gate_group'] == gate_group]
        gate_label = gate_data['gate_group_label'].iloc[0]
        
        if len(gate_data) == 0:
            continue
        
        group_start = pos_counter
        group_has_data = False
        
        for i, hardware in enumerate(hardware_platforms):
            hw_data = gate_data[gate_data['hardware'] == hardware]
            
            # Filter: only include if we have at least min_samples
            if len(hw_data) >= min_samples:
                boxplot_data.append(hw_data['error_rate'].values)
                positions.append(pos_counter)
                colors.append(COLORBREWER_PALETTE.get(hardware, '#gray'))
                pos_counter += 1
                group_has_data = True
            elif len(hw_data) > 0:
                filtered_count += 1
                print(f"  ⚠️  Filtered out {hardware} at {gate_label} gates (only {len(hw_data)} samples, need {min_samples})")
        
        # Only add the gate group label if it has data to show
        if group_has_data:
            group_end = pos_counter - 1
            gate_group_center = (group_start + group_end) / 2
            gate_group_positions.append(gate_group_center)
            gate_group_labels.append(gate_label)
            pos_counter += group_spacing
    
    if not boxplot_data:
        print("No data to plot!")
        return plt
    
    if filtered_count > 0:
        print(f"\n  📊 Filtered {filtered_count} hardware/gate groups with < {min_samples} samples for cleaner visualization")
    
    # Create boxplot
    bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True,
                    widths=0.6, showfliers=True, whis=(0, 100),
                    showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Style other boxplot elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1)
    
    plt.setp(bp['means'], color='black', linewidth=1.5, linestyle='--')
    
    # Set labels and formatting
    ax.set_xlabel('Number of Gates by Hardware Platform', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels
    ax.set_xticks(gate_group_positions)
    ax.set_xticklabels(gate_group_labels, fontsize=tick_size, ha='center', rotation=45)
    
    # Set y-axis formatting
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.set_ylim(-0.15, 1.15)  # Error rate with padding
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create legend
    legend_elements = [Patch(facecolor=COLORBREWER_PALETTE.get(hw, '#gray'), alpha=0.7, label=hw) 
                      for hw in hardware_platforms]
    ax.legend(handles=legend_elements, fontsize=legend_size, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def run_error_rate_gates_analysis():
    """Main function to run the error rate vs gates analysis."""
    print("=" * 60)
    print("ERROR RATE vs NUMBER OF GATES ANALYSIS")
    print("=" * 60)
    
    # Load combined data
    combined_df = load_combined_hardware_data()
    
    if combined_df.empty:
        print("No data available for analysis.")
        return
    
    # Calculate error rate
    combined_df['error_rate'] = (100 - combined_df['success_rate']) / 100
    
    # Group data for statistics
    combined_df_grouped = group_gates_by_range(combined_df, range_size=10)
    
    # Print dataset statistics
    print("\n" + "=" * 40)
    print("DATASET STATISTICS (GROUPED BY GATE RANGES)")
    print("=" * 40)
    
    for hardware in combined_df_grouped['hardware'].unique():
        hw_data = combined_df_grouped[combined_df_grouped['hardware'] == hardware]
        print(f"\n{hardware} Statistics:")
        print(f"  Total experiments: {len(hw_data)}")
        print(f"  Gate range: {hw_data['num_gates'].min()} - {hw_data['num_gates'].max()}")
        print(f"  Error rate range: {hw_data['error_rate'].min():.2f}% - {hw_data['error_rate'].max():.2f}%")
        print(f"  Mean error rate: {hw_data['error_rate'].mean():.2f}%")
    
    # Generate the boxplot
    print("\n" + "=" * 40)
    print("GENERATING BOXPLOT")
    print("=" * 40)
    
    plt_obj = plot_error_rate_vs_gates_boxplot(combined_df, group_spacing=0.06, min_samples=5)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '2b_error_rate_vs_gates_grouped_boxplot_filtered.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Plot saved to: {output_file}")
    print("Analysis complete!")
    
    return combined_df


if __name__ == "__main__":
    run_error_rate_gates_analysis()

