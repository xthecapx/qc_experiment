#!/usr/bin/env python3
"""
Error Rate vs Gates Stratified by Payload Size
==============================================

This analysis separates gate count effects by payload size to avoid
confounding between these two factors.

Critical insight: Previous analysis conflated gate counts across different
payload sizes, potentially masking the true relationship between gate count
and error rate.

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


def plot_error_rate_vs_gates_by_payload(df_ibm, df_rigetti, min_samples=3):
    """
    Create separate line plots for each payload size showing error rate vs gate count.
    Returns two separate figures for IBM and Rigetti (to be arranged in LaTeX).
    """
    # Set font sizes for IEEE format
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE  # Use standard figure size from styles
    
    # Color palette for payload sizes - selected for maximum contrast
    # Using ColorBrewer palette colors that are easily distinguishable
    payload_colors = {
        1: '#1b9e77',  # Teal (dark green-blue)
        2: '#d95f02',  # Orange (bright orange)
        3: '#7570b3',  # Purple (medium purple)
        4: '#e7298a',  # Magenta (bright pink)
        5: '#e6ab02',  # Yellow-gold (bright yellow-orange)
    }
    
    # ============ IBM FIGURE ============
    fig_ibm, ax_ibm = plt.subplots(figsize=fig_size)
    
    df_ibm['error_rate'] = (100 - df_ibm['success_rate']) / 100
    df_ibm_grouped = group_gates_by_range(df_ibm, range_size=10)
    
    payload_sizes = sorted(df_ibm['payload_size'].unique())
    
    print("\n" + "=" * 80)
    print("IBM: Error Rate vs Gate Count by Payload Size")
    print("=" * 80)
    
    for payload in payload_sizes:
        payload_df = df_ibm_grouped[df_ibm_grouped['payload_size'] == payload]
        gate_groups = sorted(payload_df['gate_group'].unique())
        
        gate_means = []
        gate_positions = []
        gate_stds = []
        
        for gate_group in gate_groups:
            gate_subset = payload_df[payload_df['gate_group'] == gate_group]
            if len(gate_subset) >= min_samples:
                gate_means.append(gate_subset['error_rate'].mean())
                gate_stds.append(gate_subset['error_rate'].std())
                gate_positions.append(gate_group)
        
        if len(gate_positions) > 0:
            # Convert to numpy arrays
            gate_positions = np.array(gate_positions)
            gate_means = np.array(gate_means)
            gate_stds = np.array(gate_stds)
            
            # Plot line with error bars
            ax_ibm.plot(gate_positions, gate_means, 
                    color=payload_colors.get(payload, 'gray'),
                    linewidth=2.5, marker='o', markersize=8,
                    label=f'Payload {payload}', zorder=5)
            
            # Add shaded error region (std)
            ax_ibm.fill_between(gate_positions, 
                            gate_means - gate_stds, 
                            gate_means + gate_stds,
                            color=payload_colors.get(payload, 'gray'),
                            alpha=0.2, zorder=1)
            
            print(f"\nPayload {payload}:")
            print(f"  Gate range: {gate_positions.min():.0f} - {gate_positions.max():.0f}")
            print(f"  Error rate range: {gate_means.min():.3f} - {gate_means.max():.3f}")
            print(f"  Mean error rate: {gate_means.mean():.3f}")
            print(f"  Number of gate groups: {len(gate_positions)}")
    
    # Style IBM figure (no title for LaTeX)
    ax_ibm.set_xlabel('Number of Gates', fontsize=label_size, fontweight='bold')
    ax_ibm.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    ax_ibm.set_xscale('log')
    ax_ibm.set_ylim(-0.15, 1.15)
    ax_ibm.tick_params(axis='both', which='major', labelsize=tick_size)
    ax_ibm.grid(True, linestyle='--', alpha=0.3, axis='both')
    ax_ibm.legend(fontsize=legend_size, loc='lower right', framealpha=0.95)
    
    # ============ RIGETTI FIGURE ============
    fig_rigetti, ax_rigetti = plt.subplots(figsize=fig_size)
    df_rigetti['error_rate'] = (100 - df_rigetti['success_rate']) / 100
    df_rigetti_grouped = group_gates_by_range(df_rigetti, range_size=10)
    
    payload_sizes = sorted(df_rigetti['payload_size'].unique())
    
    print("\n" + "=" * 80)
    print("Rigetti: Error Rate vs Gate Count by Payload Size")
    print("=" * 80)
    
    for payload in payload_sizes:
        payload_df = df_rigetti_grouped[df_rigetti_grouped['payload_size'] == payload]
        gate_groups = sorted(payload_df['gate_group'].unique())
        
        gate_means = []
        gate_positions = []
        gate_stds = []
        
        for gate_group in gate_groups:
            gate_subset = payload_df[payload_df['gate_group'] == gate_group]
            if len(gate_subset) >= min_samples:
                gate_means.append(gate_subset['error_rate'].mean())
                gate_stds.append(gate_subset['error_rate'].std())
                gate_positions.append(gate_group)
        
        if len(gate_positions) > 0:
            # Convert to numpy arrays
            gate_positions = np.array(gate_positions)
            gate_means = np.array(gate_means)
            gate_stds = np.array(gate_stds)
            
            # Plot line with error bars
            ax_rigetti.plot(gate_positions, gate_means, 
                    color=payload_colors.get(payload, 'gray'),
                    linewidth=2.5, marker='s', markersize=8,
                    label=f'Payload {payload}', zorder=5)
            
            # Add shaded error region (std)
            ax_rigetti.fill_between(gate_positions, 
                            gate_means - gate_stds, 
                            gate_means + gate_stds,
                            color=payload_colors.get(payload, 'gray'),
                            alpha=0.2, zorder=1)
            
            print(f"\nPayload {payload}:")
            print(f"  Gate range: {gate_positions.min():.0f} - {gate_positions.max():.0f}")
            print(f"  Error rate range: {gate_means.min():.3f} - {gate_means.max():.3f}")
            print(f"  Mean error rate: {gate_means.mean():.3f}")
            print(f"  Number of gate groups: {len(gate_positions)}")
    
    # Style Rigetti figure (no title for LaTeX)
    ax_rigetti.set_xlabel('Number of Gates', fontsize=label_size, fontweight='bold')
    ax_rigetti.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    ax_rigetti.set_xscale('log')
    ax_rigetti.set_ylim(-0.15, 1.15)
    ax_rigetti.tick_params(axis='both', which='major', labelsize=tick_size)
    ax_rigetti.grid(True, linestyle='--', alpha=0.3, axis='both')
    ax_rigetti.legend(fontsize=legend_size, loc='lower right', framealpha=0.95)
    
    fig_ibm.tight_layout()
    fig_rigetti.tight_layout()
    
    return fig_ibm, fig_rigetti


def run_stratified_gates_analysis():
    """Main function to run the stratified gates analysis."""
    print("=" * 80)
    print("ERROR RATE vs GATES STRATIFIED BY PAYLOAD SIZE")
    print("=" * 80)
    
    # Load data
    combined_df = load_combined_hardware_data()
    
    if combined_df.empty:
        print("No data available for analysis.")
        return
    
    # Split by hardware
    df_ibm = combined_df[combined_df['hardware'] == 'IBM'].copy()
    df_rigetti = combined_df[combined_df['hardware'] == 'Rigetti'].copy()
    
    # Generate the stratified plots
    fig_ibm, fig_rigetti = plot_error_rate_vs_gates_by_payload(df_ibm, df_rigetti, min_samples=3)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save both figures separately
    output_file_ibm = os.path.join(output_dir, '2d_error_rate_vs_gates_by_payload_ibm.png')
    output_file_rigetti = os.path.join(output_dir, '2d_error_rate_vs_gates_by_payload_rigetti.png')
    
    fig_ibm.savefig(output_file_ibm, dpi=300, bbox_inches='tight', facecolor='white')
    fig_rigetti.savefig(output_file_rigetti, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✓ Saved IBM plot: {output_file_ibm}")
    print(f"✓ Saved Rigetti plot: {output_file_rigetti}")
    print("\nAnalysis complete!")
    
    return combined_df


if __name__ == "__main__":
    run_stratified_gates_analysis()

