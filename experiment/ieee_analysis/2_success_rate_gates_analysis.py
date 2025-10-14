#!/usr/bin/env python3
"""
Success Rate vs Number of Gates Analysis
========================================

This module creates boxplot visualizations showing success rate distributions
by number of gates for different hardware platforms (IBM and Rigetti).

🎛️ SPACING CONTROL:
To adjust X-axis spacing between gate groups, modify the 'group_spacing' parameter
in line ~360: plot_success_rate_vs_gates_boxplot(combined_df, group_spacing=X)

Recommended values:
- 0.2 = Very tight spacing
- 0.3 = Tight spacing (current)
- 0.5 = Default spacing  
- 0.8 = Loose spacing
- 1.0 = Very loose spacing

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Patch
from load_data import load_combined_hardware_data
from styles import COLORBREWER_PALETTE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, FIG_SIZE

# Output directory
OUTPUT_DIR = 'img'




def format_gate_count_scientific(gate_count):
    """
    Format gate count in scientific notation for better readability.
    
    Parameters:
    -----------
    gate_count : int
        Number of gates
    
    Returns:
    --------
    str
        Formatted string (e.g., 200 -> "2E2", 5000 -> "5E3")
    """
    if gate_count >= 10000:
        return f"{gate_count//1000}E3"  # e.g., 20000 -> "20E3"
    elif gate_count >= 1000:
        # Handle specific cases for 1000, 1500, 3000, 5000, 7000
        if gate_count == 1010:  # 1000-1020 range
            return "1E3"
        elif gate_count == 1510:  # 1500-1520 range
            return "1.5E3"
        elif gate_count == 3010:  # 3000-3020 range
            return "3E3"
        elif gate_count == 5010:  # 5000-5020 range
            return "5E3"
        elif gate_count == 7010:  # 7000-7020 range
            return "7E3"
        else:
            return f"{gate_count//1000}E3"
    elif gate_count >= 100:
        if gate_count == 210:   # 200-220 range
            return "2E2"
        elif gate_count == 510: # 500-520 range
            return "5E2"
        else:
            return f"{gate_count//100}E2"
    else:
        return str(gate_count)


def group_gates_by_range(df, range_size=10):
    """
    Group gates into ranges (e.g., 200±10 becomes 190-210).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'num_gates' column
    range_size : int
        Half-width of the range (default: 10)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'gate_group' and 'gate_group_label' columns
    """
    df = df.copy()
    
    # Create gate groups by rounding to nearest range
    df['gate_group'] = ((df['num_gates'] // (range_size * 2)) * (range_size * 2)) + range_size
    
    # Create readable labels with scientific notation
    df['gate_group_label'] = df['gate_group'].apply(format_gate_count_scientific)
    
    return df


def plot_success_rate_vs_gates_boxplot(df, group_spacing=0.5, min_samples=5):
    """
    Create boxplots showing success rate distributions by number of gates and hardware platform.
    Gates are grouped into ranges (±10 gates) for better readability.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results with 'hardware' column
    group_spacing : float
        Spacing between gate groups on X-axis (default: 0.5)
        - Smaller values = tighter spacing
        - Larger values = more spacing
        - Try values like: 0.2 (tight), 0.5 (default), 1.0 (loose)
    min_samples : int
        Minimum number of samples required to include a hardware/gate group (default: 5)
        Groups with fewer samples will be filtered out to avoid noise
    """
    # Use centralized IEEE format settings
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    
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
        gate_label = gate_data['gate_group_label'].iloc[0]  # Get the label for this group
        
        if len(gate_data) == 0:
            continue
        
        # Track the center position for this gate group
        group_start = pos_counter
        group_has_data = False
        
        for i, hardware in enumerate(hardware_platforms):
            hw_data = gate_data[gate_data['hardware'] == hardware]
            
            # Filter: only include if we have at least min_samples
            if len(hw_data) >= min_samples:
                boxplot_data.append(hw_data['success_rate'].values / 100)  # Convert to 0-1 scale
                positions.append(pos_counter)
                colors.append(COLORBREWER_PALETTE.get(hardware, '#gray'))
                pos_counter += 1
                group_has_data = True
            elif len(hw_data) > 0:
                # Log filtered groups
                filtered_count += 1
                print(f"  ⚠️  Filtered out {hardware} at {gate_label} gates (only {len(hw_data)} samples, need {min_samples})")
        
        # Only add the gate group label if it has data to show
        if group_has_data:
            # Calculate center position for gate group label
            group_end = pos_counter - 1
            gate_group_center = (group_start + group_end) / 2
            gate_group_positions.append(gate_group_center)
            gate_group_labels.append(gate_label)
            
            # Add spacing between different gate groups
            pos_counter += group_spacing
    
    if not boxplot_data:
        print("No data to plot!")
        return plt
    
    if filtered_count > 0:
        print(f"\n  📊 Filtered {filtered_count} hardware/gate groups with < {min_samples} samples for cleaner visualization")
    
    # Create boxplot with proper handling of edge cases (many 0% values)
    # Use whis parameter to control whisker length and handle collapsed boxes
    bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True,
                    widths=0.6, showfliers=True, whis=(0, 100),  # Use min-max range for whiskers
                    showmeans=True, meanline=True)  # Show mean line for better visibility
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Style other boxplot elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1)
    
    # Style mean lines to be more visible (black dashed)
    plt.setp(bp['means'], color='black', linewidth=1.5, linestyle='--')
    
    # Set labels and formatting
    ax.set_xlabel('Number of Gates by Hardware Platform', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=label_size, fontweight='bold')
    
    # Set x-axis ticks and labels (grouped by gate count) with rotation to prevent overlap
    ax.set_xticks(gate_group_positions)
    ax.set_xticklabels(gate_group_labels, fontsize=tick_size, ha='center', rotation=45)
    
    # Set y-axis formatting with proper bounds for success rates
    ax.tick_params(axis='y', labelsize=tick_size)
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.15, 1.15)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create simple custom legend (matching 1_ analysis style)
    legend_elements = [Patch(facecolor=COLORBREWER_PALETTE.get(hw, '#gray'), alpha=0.7, label=hw) 
                      for hw in hardware_platforms]
    ax.legend(handles=legend_elements, fontsize=legend_size, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def run_success_rate_gates_analysis():
    """
    Main function to run the success rate vs gates analysis.
    """
    print("=" * 60)
    print("SUCCESS RATE vs NUMBER OF GATES ANALYSIS")
    print("=" * 60)
    
    # Load combined data
    combined_df = load_combined_hardware_data()
    
    if combined_df.empty:
        print("No data available for analysis.")
        return
    
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
        print(f"  Success rate range: {hw_data['success_rate'].min():.2f}% - {hw_data['success_rate'].max():.2f}%")
        print(f"  Mean success rate: {hw_data['success_rate'].mean():.2f}%")
        
        # Show data distribution by gate groups
        print(f"  Distribution by gate groups:")
        for gate_group in sorted(hw_data['gate_group'].unique()):
            gate_data = hw_data[hw_data['gate_group'] == gate_group]
            gate_label = gate_data['gate_group_label'].iloc[0]
            print(f"    {gate_label} gates: {len(gate_data)} experiments, "
                  f"mean: {gate_data['success_rate'].mean():.2f}%, "
                  f"std: {gate_data['success_rate'].std():.2f}%")
    
    # Generate the boxplot
    print("\n" + "=" * 40)
    print("GENERATING BOXPLOT")
    print("=" * 40)
    
    # Generate the boxplot (adjust group_spacing to control X-axis spacing)
    plt_obj = plot_success_rate_vs_gates_boxplot(combined_df, group_spacing=0.06)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, '2_success_rate_vs_gates_grouped_boxplot_filtered.png')
    plt_obj.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Plot saved to: {output_file}")
    print("Analysis complete!")
    
    return combined_df


if __name__ == "__main__":
    # Run the analysis
    run_success_rate_gates_analysis()
