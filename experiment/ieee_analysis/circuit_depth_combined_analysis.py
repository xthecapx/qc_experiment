#!/usr/bin/env python3
"""
Circuit Depth Combined Analysis
===============================

This module creates bar plot visualizations showing error distribution by circuit depth
for combined datasets (IBM and AWS/Rigetti) with both linear and non-linear model fitting.

Data Sources (via load_data module):
- IBM: All CSV files from ibm/ directory
- AWS/Rigetti: All CSV files from aws/ directory, plus root-level Rigetti files

This module was extracted from target_depth_analysis.py to improve maintainability and
follow the same structure as success_rate_gates_analysis.py and hardware_success_rates_analysis.py.

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import os
from circuit_depth_models import CircuitDepthModels
from load_data import load_circuit_depth_datasets
from styles import COLORBREWER_PALETTE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, FIG_SIZE



# Output directory
OUTPUT_DIR = 'img'


def calculate_regression_stats(x, y):
    """Calculate regression statistics for the given data"""
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model



def plot_error_distribution_by_circuit_depth_combined(df1, df2, use_best_models=True):
    """
    Create a bar plot showing error distribution by circuit depth for both datasets
    Combines two datasets with different hardware platforms with IEEE formatting
    Now uses the best non-linear models instead of linear regression
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First dataset (IBM Sherbrooke)
    df2 : pandas.DataFrame
        Second dataset (Rigetti Ankaa-3)
    use_best_models : bool
        Whether to use best non-linear models (True) or linear regression (False)
    """
    # IEEE format settings
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    line_width = 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate error rate for both datasets
    df1 = df1.copy()
    df2 = df2.copy()
    df1['error_rate'] = 1 - df1['success_rate']
    df2['error_rate'] = 1 - df2['success_rate']
    
    # Group by circuit_depth and calculate statistics for Dataset 1 (IBM)
    depth_groups1 = df1.groupby('circuit_depth')
    depths1 = []
    error_means1 = []
    error_stds1 = []
    
    for depth, group in depth_groups1:
        depths1.append(depth)
        error_means1.append(group['error_rate'].mean())
        error_stds1.append(group['error_rate'].std())
    
    depths1 = np.array(depths1)
    error_means1 = np.array(error_means1)
    error_stds1 = np.array(error_stds1)
    
    # Group by circuit_depth and calculate statistics for Dataset 2 (Rigetti)
    depth_groups2 = df2.groupby('circuit_depth')
    depths2 = []
    error_means2 = []
    error_stds2 = []
    
    for depth, group in depth_groups2:
        depths2.append(depth)
        error_means2.append(group['error_rate'].mean())
        error_stds2.append(group['error_rate'].std())
    
    depths2 = np.array(depths2)
    error_means2 = np.array(error_means2)
    error_stds2 = np.array(error_stds2)
    
    print(f"IBM depth range: {depths1.min()} - {depths1.max()}, error rate range: {error_means1.min():.3f} - {error_means1.max():.3f}")
    print(f"Rigetti depth range: {depths2.min()} - {depths2.max()}, error rate range: {error_means2.min():.3f} - {error_means2.max():.3f}")
    
    # Visualization uses scatter with error bars (to match other graphs)
    marker_size = 55
    
    # Add small horizontal offset to separate IBM and Rigetti data points
    depth_offset = 0.3
    
    if use_best_models:
        # Use the best non-linear models instead of linear regression
        model_analyzer_ibm = CircuitDepthModels()
        model_analyzer_rigetti = CircuitDepthModels()
        
        print("Fitting non-linear models to IBM data...")
        model_analyzer_ibm.fit_all_models(depths1, error_means1)
        print("Fitting non-linear models to Rigetti data...")
        model_analyzer_rigetti.fit_all_models(depths2, error_means2)
        
        # Use the composite-selected best models
        try:
            ibm_best_name, _, _ = model_analyzer_ibm.get_best_model('composite')
            ibm_results = model_analyzer_ibm.fitted_params[ibm_best_name]
            print(f"Best IBM model: {ibm_best_name}")
        except:
            # Fallback to a reliable model
            ibm_best_name = 'logistic'
            ibm_results = model_analyzer_ibm.fitted_params[ibm_best_name]
            print(f"IBM model fallback to: {ibm_best_name}")
            
        try:
            rigetti_best_name, _, _ = model_analyzer_rigetti.get_best_model('composite')
            rigetti_results = model_analyzer_rigetti.fitted_params[rigetti_best_name]
            print(f"Best Rigetti model: {rigetti_best_name}")
        except:
            # Fallback to a reliable model
            rigetti_best_name = 'logistic'
            rigetti_results = model_analyzer_rigetti.fitted_params[rigetti_best_name]
            print(f"Rigetti model fallback to: {rigetti_best_name}")
    else:
        # Fallback to linear regression
        print("Using linear regression models...")
        model1 = calculate_regression_stats(depths1, error_means1)
        model2 = calculate_regression_stats(depths2, error_means2)
    
    # Scatter with error bars to match style of other analyses - with horizontal offset
    ax.errorbar(
        depths1 - depth_offset, error_means1, yerr=error_stds1,
        fmt='o', markersize=marker_size/10.0, color=COLORBREWER_PALETTE['IBM'], ecolor=COLORBREWER_PALETTE['IBM'],
        elinewidth=1.2, capsize=3, alpha=0.9, label='IBM Sherbrooke'
    )
    ax.errorbar(
        depths2 + depth_offset, error_means2, yerr=error_stds2,
        fmt='s', markersize=marker_size/10.0, color=COLORBREWER_PALETTE['Rigetti'], ecolor=COLORBREWER_PALETTE['Rigetti'],
        elinewidth=1.2, capsize=3, alpha=0.9, label='Rigetti Ankaa-3'
    )
    
    # Add trend lines for both datasets
    if use_best_models and ibm_results['success'] and rigetti_results['success']:
        # IBM trend line (best model) - no offset for smooth curve
        x_pred1 = np.linspace(depths1.min(), depths1.max(), 100)
        y_pred1 = model_analyzer_ibm.predict(ibm_best_name, x_pred1)
        ax.plot(x_pred1, y_pred1, color=COLORBREWER_PALETTE['IBM'], linestyle='--', linewidth=line_width+1, zorder=5, alpha=0.8)
        
        # Rigetti trend line (best model) - no offset for smooth curve
        x_pred2 = np.linspace(depths2.min(), depths2.max(), 100)
        y_pred2 = model_analyzer_rigetti.predict(rigetti_best_name, x_pred2)
        ax.plot(x_pred2, y_pred2, color=COLORBREWER_PALETTE['Rigetti'], linestyle='-.', linewidth=line_width+1, zorder=5, alpha=0.8)
    else:
        # Fallback to linear regression lines
        if not use_best_models:
            model1 = calculate_regression_stats(depths1, error_means1)
            model2 = calculate_regression_stats(depths2, error_means2)
        else:
            # If models failed, fallback to linear
            print("Non-linear models failed, falling back to linear regression...")
            model1 = calculate_regression_stats(depths1, error_means1)
            model2 = calculate_regression_stats(depths2, error_means2)
        
        # IBM trend line
        x_pred1 = np.linspace(depths1.min(), depths1.max(), 100)
        X_pred1 = sm.add_constant(x_pred1)
        y_pred1 = model1.predict(X_pred1)
        ax.plot(x_pred1, y_pred1, color='blue', linestyle='--', linewidth=line_width+1, zorder=5)
        
        # Rigetti trend line
        x_pred2 = np.linspace(depths2.min(), depths2.max(), 100)
        X_pred2 = sm.add_constant(x_pred2)
        y_pred2 = model2.predict(X_pred2)
        ax.plot(x_pred2, y_pred2, color='#8B0000', linestyle='-.', linewidth=line_width+1, zorder=5)
    
    # Set axis labels with updated font sizes
    ax.set_xlabel('Circuit Depth', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Improve x-axis spacing with more separation
    ax.set_xlim(min(depths1.min(), depths2.min()) - 2, max(depths1.max(), depths2.max()) + 2)
    
    # Add extra Y-axis space for legend without interfering with plot
    ax.set_ylim(0, 1.2)
    
    # Hide the 1.2 Y-axis label since it's just spacing for legend
    yticks = ax.get_yticks()
    yticks = yticks[yticks <= 1.0]  # Only show ticks up to 1.0
    ax.set_yticks(yticks)
    
    # Add grid and legend on the top left
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.legend(fontsize=legend_size-1, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
              framealpha=0.9, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def run_circuit_depth_combined_analysis():
    """
    Main function to run the circuit depth combined analysis.
    """
    print("=" * 60)
    print("CIRCUIT DEPTH COMBINED ANALYSIS")
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
    print(f"  Circuit depth range: {df1['circuit_depth'].min()} - {df1['circuit_depth'].max()}")
    print(f"  Success rate range: {df1['success_rate'].min():.3f} - {df1['success_rate'].max():.3f}")
    print(f"  Mean success rate: {df1['success_rate'].mean():.3f}")
    
    print(f"\nRigetti Dataset:")
    print(f"  Total experiments: {len(df2)}")
    print(f"  Circuit depth range: {df2['circuit_depth'].min()} - {df2['circuit_depth'].max()}")
    print(f"  Success rate range: {df2['success_rate'].min():.3f} - {df2['success_rate'].max():.3f}")
    print(f"  Mean success rate: {df2['success_rate'].mean():.3f}")
    
    # Generate the plots
    print("\n" + "=" * 40)
    print("GENERATING PLOTS")
    print("=" * 40)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate combined plot with best non-linear models (single consolidated plot)
    print("\n1. Generating combined plot with best non-linear models...")
    plt_nonlinear = plot_error_distribution_by_circuit_depth_combined(df1, df2, use_best_models=True)
    output_file_nonlinear = os.path.join(output_dir, '2_error_distribution_by_circuit_depth_combined_nonlinear.png')
    plt_nonlinear.savefig(output_file_nonlinear, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file_nonlinear}")
    plt_nonlinear.close()
    
    print("\n" + "=" * 40)
    print("ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"Generated plot:")
    print(f"  - {output_file_nonlinear}")
    
    return df1, df2


if __name__ == "__main__":
    # Run the analysis
    run_circuit_depth_combined_analysis()
