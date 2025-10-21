#!/usr/bin/env python3
"""
Success Rate vs Circuit Depth Stratified by Payload Size
========================================================

This analysis separates circuit depth effects by payload size to show
how depth impacts success rate within each payload category.

Similar to 2d but for circuit depth instead of gate count.

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


def bin_circuit_depth(df, bin_size=5):
    """
    Bin circuit depths into intervals (e.g., 5-9, 10-14, 15-19).
    """
    df = df.copy()
    # Create bins: 5-9, 10-14, 15-19, etc.
    df['depth_bin'] = ((df['circuit_depth'] // bin_size) * bin_size)
    df['depth_bin_label'] = df['depth_bin'].apply(lambda x: f"{x}-{x+bin_size-1}")
    return df


def plot_success_rate_vs_depth_by_payload(df_ibm, df_rigetti, max_depth=48, bin_size=5, min_samples=3):
    """
    Create separate line plots for each payload size showing success rate vs circuit depth.
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
    
    # Filter to max depth
    df_ibm = df_ibm[df_ibm['circuit_depth'] <= max_depth].copy()
    df_ibm['success_rate_scaled'] = df_ibm['success_rate'] / 100
    df_ibm_binned = bin_circuit_depth(df_ibm, bin_size=bin_size)
    
    payload_sizes = sorted(df_ibm['payload_size'].unique())
    
    print("\n" + "=" * 80)
    print("IBM: Success Rate vs Circuit Depth by Payload Size")
    print("=" * 80)
    
    for payload in payload_sizes:
        payload_df = df_ibm_binned[df_ibm_binned['payload_size'] == payload]
        depth_bins = sorted(payload_df['depth_bin'].unique())
        
        depth_means = []
        depth_positions = []
        depth_stds = []
        
        for depth_bin in depth_bins:
            depth_subset = payload_df[payload_df['depth_bin'] == depth_bin]
            if len(depth_subset) >= min_samples:
                depth_means.append(depth_subset['success_rate_scaled'].mean())
                depth_stds.append(depth_subset['success_rate_scaled'].std())
                depth_positions.append(depth_bin + bin_size // 2)  # Center of bin
        
        if len(depth_positions) > 0:
            # Convert to numpy arrays
            depth_positions = np.array(depth_positions)
            depth_means = np.array(depth_means)
            depth_stds = np.array(depth_stds)
            
            # Plot line with error bars
            ax_ibm.plot(depth_positions, depth_means, 
                    color=payload_colors.get(payload, 'gray'),
                    linewidth=2.5, marker='o', markersize=8,
                    label=f'Payload {payload}', zorder=5)
            
            # Add shaded error region (std)
            ax_ibm.fill_between(depth_positions, 
                            depth_means - depth_stds, 
                            depth_means + depth_stds,
                            color=payload_colors.get(payload, 'gray'),
                            alpha=0.2, zorder=1)
            
            print(f"\nPayload {payload}:")
            print(f"  Depth range: {depth_positions.min():.0f} - {depth_positions.max():.0f}")
            print(f"  Success rate range: {depth_means.min():.3f} - {depth_means.max():.3f}")
            print(f"  Mean success rate: {depth_means.mean():.3f}")
            print(f"  Number of depth bins: {len(depth_positions)}")
    
    # Style IBM figure (no title for LaTeX)
    ax_ibm.set_xlabel('Circuit Depth', fontsize=label_size, fontweight='bold')
    ax_ibm.set_ylabel('Success Rate', fontsize=label_size, fontweight='bold')
    ax_ibm.set_xlim(0, max_depth)
    ax_ibm.set_ylim(-0.15, 1.15)
    ax_ibm.tick_params(axis='both', which='major', labelsize=tick_size)
    ax_ibm.grid(True, linestyle='--', alpha=0.3, axis='both')
    ax_ibm.legend(fontsize=legend_size, loc='upper left', framealpha=0.95)
    
    # ============ RIGETTI FIGURE ============
    fig_rigetti, ax_rigetti = plt.subplots(figsize=fig_size)
    
    # Filter to max depth
    df_rigetti = df_rigetti[df_rigetti['circuit_depth'] <= max_depth].copy()
    df_rigetti['success_rate_scaled'] = df_rigetti['success_rate'] / 100
    df_rigetti_binned = bin_circuit_depth(df_rigetti, bin_size=bin_size)
    
    payload_sizes = sorted(df_rigetti['payload_size'].unique())
    
    print("\n" + "=" * 80)
    print("Rigetti: Success Rate vs Circuit Depth by Payload Size")
    print("=" * 80)
    
    for payload in payload_sizes:
        payload_df = df_rigetti_binned[df_rigetti_binned['payload_size'] == payload]
        depth_bins = sorted(payload_df['depth_bin'].unique())
        
        depth_means = []
        depth_positions = []
        depth_stds = []
        
        for depth_bin in depth_bins:
            depth_subset = payload_df[payload_df['depth_bin'] == depth_bin]
            if len(depth_subset) >= min_samples:
                depth_means.append(depth_subset['success_rate_scaled'].mean())
                depth_stds.append(depth_subset['success_rate_scaled'].std())
                depth_positions.append(depth_bin + bin_size // 2)  # Center of bin
        
        if len(depth_positions) > 0:
            # Convert to numpy arrays
            depth_positions = np.array(depth_positions)
            depth_means = np.array(depth_means)
            depth_stds = np.array(depth_stds)
            
            # Plot line with error bars
            ax_rigetti.plot(depth_positions, depth_means, 
                    color=payload_colors.get(payload, 'gray'),
                    linewidth=2.5, marker='s', markersize=8,
                    label=f'Payload {payload}', zorder=5)
            
            # Add shaded error region (std)
            ax_rigetti.fill_between(depth_positions, 
                            depth_means - depth_stds, 
                            depth_means + depth_stds,
                            color=payload_colors.get(payload, 'gray'),
                            alpha=0.2, zorder=1)
            
            print(f"\nPayload {payload}:")
            print(f"  Depth range: {depth_positions.min():.0f} - {depth_positions.max():.0f}")
            print(f"  Success rate range: {depth_means.min():.3f} - {depth_means.max():.3f}")
            print(f"  Mean success rate: {depth_means.mean():.3f}")
            print(f"  Number of depth bins: {len(depth_positions)}")
    
    # Style Rigetti figure (no title for LaTeX)
    ax_rigetti.set_xlabel('Circuit Depth', fontsize=label_size, fontweight='bold')
    ax_rigetti.set_ylabel('Success Rate', fontsize=label_size, fontweight='bold')
    ax_rigetti.set_xlim(0, max_depth)
    ax_rigetti.set_ylim(-0.15, 1.15)
    ax_rigetti.tick_params(axis='both', which='major', labelsize=tick_size)
    ax_rigetti.grid(True, linestyle='--', alpha=0.3, axis='both')
    ax_rigetti.legend(fontsize=legend_size, loc='upper left', framealpha=0.95)
    
    fig_ibm.tight_layout()
    fig_rigetti.tight_layout()
    
    return fig_ibm, fig_rigetti


def run_stratified_depth_analysis():
    """Main function to run the stratified depth analysis."""
    print("=" * 80)
    print("SUCCESS RATE vs CIRCUIT DEPTH STRATIFIED BY PAYLOAD SIZE")
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
    fig_ibm, fig_rigetti = plot_success_rate_vs_depth_by_payload(
        df_ibm, df_rigetti, max_depth=50, bin_size=5, min_samples=3
    )
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save both figures separately
    output_file_ibm = os.path.join(output_dir, '3d_success_rate_vs_depth_by_payload_ibm.png')
    output_file_rigetti = os.path.join(output_dir, '3d_success_rate_vs_depth_by_payload_rigetti.png')
    
    fig_ibm.savefig(output_file_ibm, dpi=300, bbox_inches='tight', facecolor='white')
    fig_rigetti.savefig(output_file_rigetti, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✓ Saved IBM plot: {output_file_ibm}")
    print(f"✓ Saved Rigetti plot: {output_file_rigetti}")
    print("\nAnalysis complete!")
    
    return combined_df


if __name__ == "__main__":
    run_stratified_depth_analysis()



