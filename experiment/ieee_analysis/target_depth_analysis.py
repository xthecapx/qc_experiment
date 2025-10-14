import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json
import os
from circuit_depth_models import CircuitDepthModels
from circuit_depth_combined_analysis import plot_error_distribution_by_circuit_depth_combined
from payload_size_boxplot_analysis import plot_error_rate_vs_payload_size_combined_boxplot
from styles import COLORBREWER_PALETTE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, FIG_SIZE

# Create output directory if it doesn't exist
OUTPUT_DIR = "img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom color palette (matching experiment_analysis.py)
COLORBREWER_PALETTE = {
    1: '#1b9e77',    # Teal
    2: '#d95f02',    # Orange
    3: '#7570b3',    # Purple
    4: '#e7298a',    # Pink
    5: '#66a61e',    # Green
    6: '#e6ab02',    # Yellow
    7: '#e377c2',    # Pink
    8: '#7f7f7f',    # Gray
    9: '#bcbd22',    # Olive
    10: '#17becf'    # Cyan
}

# Define marker styles for different payload sizes
MARKER_STYLES = {
    1: 'o',  # Circle
    2: 's',  # Square
    3: '^',  # Triangle up
    4: 'D',  # Diamond
    5: 'v',  # Triangle down
    6: 'p',  # Pentagon
    7: '*',  # Star
    8: 'h',  # Hexagon
    9: 'X',  # X
    10: 'P'  # Plus (filled)
}

def create_experiment_dataframe(csv_file_path):
    """
    Load and preprocess the experiment data from a CSV file
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Extract counts data
    df['counts_dict'] = df['counts'].apply(lambda x: json.loads(x.replace("'", '"')))
    
    # Extract counts of zeros and ones for each experiment
    def extract_counts(counts_dict, bit_length):
        zeros_count = 0
        ones_count = 0
        
        for bitstring, count in counts_dict.items():
            # Count the number of 1s in each bitstring
            ones_in_bitstring = bitstring.count('1')
            zeros_in_bitstring = len(bitstring) - ones_in_bitstring
            
            # Weight by the count
            ones_count += ones_in_bitstring * count
            zeros_count += zeros_in_bitstring * count
            
        # Normalize by the total possible bits
        total_bits = bit_length * sum(counts_dict.values())
        return zeros_count / total_bits, ones_count / total_bits
    
    # Apply the extraction function
    df['counts_zeros'] = df.apply(lambda row: extract_counts(row['counts_dict'], row['payload_size'])[0], axis=1)
    df['counts_ones'] = df.apply(lambda row: extract_counts(row['counts_dict'], row['payload_size'])[1], axis=1)
    
    return df

def calculate_regression_stats(x, y):
    """Calculate regression statistics for the given data"""
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model


def analyze_circuit_depth_models(df1, df2):
    """
    Analyze circuit depth vs error rate using various non-linear models
    
    Args:
        df1: First dataset (IBM)
        df2: Second dataset (Rigetti)
        
    Returns:
        tuple: (model_analyzer_ibm, model_analyzer_rigetti, comparison_tables)
    """
    print("\n" + "="*60)
    print("CIRCUIT DEPTH MODEL ANALYSIS")
    print("="*60)
    
    # Calculate error rates
    df1['error_rate'] = 1 - df1['success_rate']
    df2['error_rate'] = 1 - df2['success_rate']
    
    # Group by circuit depth and get mean error rates
    ibm_grouped = df1.groupby('circuit_depth')['error_rate'].mean().reset_index()
    rigetti_grouped = df2.groupby('circuit_depth')['error_rate'].mean().reset_index()
    
    # Extract data for modeling
    ibm_x = ibm_grouped['circuit_depth'].values
    ibm_y = ibm_grouped['error_rate'].values
    
    rigetti_x = rigetti_grouped['circuit_depth'].values
    rigetti_y = rigetti_grouped['error_rate'].values
    
    print(f"\nIBM Dataset: {len(ibm_x)} depth points, error rate range: {ibm_y.min():.3f} - {ibm_y.max():.3f}")
    print(f"Rigetti Dataset: {len(rigetti_x)} depth points, error rate range: {rigetti_y.min():.3f} - {rigetti_y.max():.3f}")
    
    # Create model analyzers
    print("\n" + "-"*40)
    print("FITTING MODELS TO IBM DATA")
    print("-"*40)
    model_analyzer_ibm = CircuitDepthModels()
    ibm_results = model_analyzer_ibm.fit_all_models(ibm_x, ibm_y)
    
    print("\n" + "-"*40)
    print("FITTING MODELS TO RIGETTI DATA")
    print("-"*40)
    model_analyzer_rigetti = CircuitDepthModels()
    rigetti_results = model_analyzer_rigetti.fit_all_models(rigetti_x, rigetti_y)
    
    # Get comparison tables
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    ibm_comparison = model_analyzer_ibm.get_comparison_table()
    rigetti_comparison = model_analyzer_rigetti.get_comparison_table()
    
    print("\nIBM SHERBROOKE - MODEL COMPARISON:")
    print("-" * 50)
    print(ibm_comparison.to_string(index=False))
    
    print("\n\nRIGETTI ANKAA-3 - MODEL COMPARISON:")
    print("-" * 50)
    print(rigetti_comparison.to_string(index=False))
    
    # Generate comprehensive model selection reports
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL SELECTION")
    print("="*60)
    
    try:
        # IBM Model Selection
        print("\nIBM SHERBROOKE - MODEL SELECTION REPORT:")
        ibm_report = model_analyzer_ibm.generate_model_selection_report()
        print(ibm_report)
        
        # Get best model using composite scoring
        ibm_best_name, ibm_best_results, ibm_rationale = model_analyzer_ibm.get_best_model('composite')
        print(f"\n🏆 RECOMMENDED IBM MODEL: {ibm_best_name}")
        print(f"   {ibm_rationale}")
        
        # Show detailed diagnostics
        ibm_diagnostics = model_analyzer_ibm.get_detailed_diagnostics(ibm_best_name)
        print(f"   📊 DETAILED DIAGNOSTICS:")
        print(f"      Equation: {model_analyzer_ibm.get_model_equation(ibm_best_name)}")
        if 'Durbin-Watson Statistic' in ibm_diagnostics:
            print(f"      Durbin-Watson = {ibm_diagnostics['Durbin-Watson Statistic']:.3f}")
        if 'Heteroscedasticity (BP)' in ibm_diagnostics:
            print(f"      Heteroscedasticity = {ibm_diagnostics['Heteroscedasticity (BP)']}")
            
    except ValueError as e:
        print(f"\n❌ IBM Model Selection Failed: {e}")
        ibm_best_name = None
    
    try:
        # Rigetti Model Selection  
        print(f"\n{'-'*60}")
        print("RIGETTI ANKAA-3 - MODEL SELECTION REPORT:")
        rigetti_report = model_analyzer_rigetti.generate_model_selection_report()
        print(rigetti_report)
        
        # Get best model using composite scoring
        rigetti_best_name, rigetti_best_results, rigetti_rationale = model_analyzer_rigetti.get_best_model('composite')
        print(f"\n🏆 RECOMMENDED RIGETTI MODEL: {rigetti_best_name}")
        print(f"   {rigetti_rationale}")
        
        # Show detailed diagnostics
        rigetti_diagnostics = model_analyzer_rigetti.get_detailed_diagnostics(rigetti_best_name)
        print(f"   📊 DETAILED DIAGNOSTICS:")
        print(f"      Equation: {model_analyzer_rigetti.get_model_equation(rigetti_best_name)}")
        if 'Durbin-Watson Statistic' in rigetti_diagnostics:
            print(f"      Durbin-Watson = {rigetti_diagnostics['Durbin-Watson Statistic']:.3f}")
        if 'Heteroscedasticity (BP)' in rigetti_diagnostics:
            print(f"      Heteroscedasticity = {rigetti_diagnostics['Heteroscedasticity (BP)']}")
            
    except ValueError as e:
        print(f"\n❌ Rigetti Model Selection Failed: {e}")
        rigetti_best_name = None
    
    # Generate model comparison plots
    print(f"\n{'-'*60}")
    print("GENERATING MODEL COMPARISON PLOTS")
    print(f"{'-'*60}")
    
    try:
        # IBM model comparison plot
        print("Generating IBM model comparison plot...")
        ibm_fig = model_analyzer_ibm.plot_model_comparison(
            ibm_x, ibm_y, top_n=5, 
            save_path=os.path.join(OUTPUT_DIR, 'ibm_model_comparison_plot.png')
        )
        print("✓ Saved: img/ibm_model_comparison_plot.png")
        
        # Rigetti model comparison plot
        print("Generating Rigetti model comparison plot...")
        rigetti_fig = model_analyzer_rigetti.plot_model_comparison(
            rigetti_x, rigetti_y, top_n=5,
            save_path=os.path.join(OUTPUT_DIR, 'rigetti_model_comparison_plot.png')
        )
        print("✓ Saved: img/rigetti_model_comparison_plot.png")
        
    except Exception as e:
        print(f"Warning: Could not generate model comparison plots: {e}")
    
    # Save comparison tables to CSV
    ibm_comparison.to_csv(os.path.join(OUTPUT_DIR, 'ibm_model_comparison.csv'), index=False)
    rigetti_comparison.to_csv(os.path.join(OUTPUT_DIR, 'rigetti_model_comparison.csv'), index=False)
    
    print(f"\n📊 Comparison tables saved:")
    print(f"   - {OUTPUT_DIR}/ibm_model_comparison.csv")
    print(f"   - {OUTPUT_DIR}/rigetti_model_comparison.csv")
    
    return model_analyzer_ibm, model_analyzer_rigetti, (ibm_comparison, rigetti_comparison)


def plot_error_distribution_by_circuit_depth(df):
    """
    Create a plot showing error distribution based on circuit depth
    """
    # Use centralized IEEE format settings
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate error rate (1 - success_rate)
    df['error_rate'] = 1 - df['success_rate']
    
    # Group by circuit_depth and calculate statistics
    depth_groups = df.groupby('circuit_depth')
    
    depths = []
    error_means = []
    error_stds = []
    
    for depth, group in depth_groups:
        depths.append(depth)
        error_means.append(group['error_rate'].mean())
        error_stds.append(group['error_rate'].std())
    
    # Convert to numpy arrays
    depths = np.array(depths)
    error_means = np.array(error_means)
    error_stds = np.array(error_stds)
    
    # Create bar plot with error bars using updated colors
    ax.bar(depths, error_means, yerr=error_stds, 
           color=COLORBREWER_PALETTE[1], edgecolor='black', alpha=0.7,
           capsize=5, width=0.7)
    
    # Add a trend line
    model = calculate_regression_stats(depths, error_means)
    x_pred = np.linspace(depths.min(), depths.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred)
    
    ax.plot(x_pred, y_pred, color=COLORBREWER_PALETTE[2], linestyle='--', linewidth=2,
            label=f'Trend (R²={model.rsquared:.3f})')
    
    # Set axis labels with updated font sizes (no title for IEEE format)
    ax.set_xlabel('Circuit Depth', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.legend(fontsize=legend_size)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution_by_circuit_depth.png'), dpi=300, bbox_inches='tight')






def plot_error_rate_vs_payload_size_combined(df1, df2):
    """
    Create a scatter plot showing error rate vs payload size with circuit depth as color
    Combines two datasets with different hardware platforms
    """
    # Use centralized IEEE format settings
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    marker_size = 80
    line_width = 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate error rate for both datasets
    df1['error_rate'] = 1 - df1['success_rate']
    df2['error_rate'] = 1 - df2['success_rate']
    
    # Plot Dataset 1 (IBM) - circles
    scatter1 = ax.scatter(df1['payload_size'], df1['error_rate'],
                         c=df1['circuit_depth'], cmap='viridis',
                         alpha=0.7, s=marker_size, edgecolor='black', linewidth=0.5,
                         marker='o', label='IBM Sherbrooke')
    
    # Plot Dataset 2 (Rigetti) - squares  
    scatter2 = ax.scatter(df2['payload_size'], df2['error_rate'],
                         c=df2['circuit_depth'], cmap='plasma',
                         alpha=0.7, s=marker_size, edgecolor='black', linewidth=0.5,
                         marker='s', label='Rigetti Ankaa-3')
    
    # Add colorbar for circuit depth (using the first dataset's range)
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('Circuit Depth', fontsize=label_size, fontweight='bold')
    cbar.ax.tick_params(labelsize=tick_size)
    
    # Calculate and plot error bars for Dataset 1 (IBM)
    payload_groups1 = df1.groupby('payload_size')
    payload_sizes1 = []
    error_means1 = []
    error_stds1 = []
    
    for payload_size, group in payload_groups1:
        payload_sizes1.append(payload_size)
        error_means1.append(group['error_rate'].mean())
        error_stds1.append(group['error_rate'].std())
    
    payload_sizes1 = np.array(payload_sizes1)
    error_means1 = np.array(error_means1)
    error_stds1 = np.array(error_stds1)
    
    # Calculate and plot error bars for Dataset 2 (Rigetti)
    payload_groups2 = df2.groupby('payload_size')
    payload_sizes2 = []
    error_means2 = []
    error_stds2 = []
    
    for payload_size, group in payload_groups2:
        payload_sizes2.append(payload_size)
        error_means2.append(group['error_rate'].mean())
        error_stds2.append(group['error_rate'].std())
    
    payload_sizes2 = np.array(payload_sizes2)
    error_means2 = np.array(error_means2)
    error_stds2 = np.array(error_stds2)
    
    # Add error bars for IBM (offset right)
    ax.errorbar(payload_sizes1 + 0.1, error_means1, yerr=error_stds1, 
               fmt='o', color='blue', markersize=8, 
               linewidth=2, capsize=5, capthick=2,
               label='IBM Mean ± Std', zorder=10)
    
    # Add error bars for Rigetti (offset left)
    ax.errorbar(payload_sizes2 - 0.1, error_means2, yerr=error_stds2, 
               fmt='s', color='red', markersize=8, 
               linewidth=2, capsize=5, capthick=2,
               label='Rigetti Mean ± Std', zorder=10)
    
    # Add regression lines for both datasets
    # IBM regression
    x1 = df1['payload_size']
    y1 = df1['error_rate']
    model1 = calculate_regression_stats(x1, y1)
    
    x_pred1 = np.linspace(x1.min(), x1.max(), 100)
    X_pred1 = sm.add_constant(x_pred1)
    y_pred1 = model1.predict(X_pred1)
    
    ax.plot(x_pred1, y_pred1, color='blue', linestyle='--', linewidth=line_width,
            label=f'IBM Trend (R²={model1.rsquared:.3f})', zorder=5)
    
    # Rigetti regression
    x2 = df2['payload_size']
    y2 = df2['error_rate']
    model2 = calculate_regression_stats(x2, y2)
    
    x_pred2 = np.linspace(x2.min(), x2.max(), 100)
    X_pred2 = sm.add_constant(x_pred2)
    y_pred2 = model2.predict(X_pred2)
    
    ax.plot(x_pred2, y_pred2, color='red', linestyle=':', linewidth=line_width,
            label=f'Rigetti Trend (R²={model2.rsquared:.3f})', zorder=5)
    
    # Set axis labels with updated font sizes
    ax.set_xlabel('Payload Size', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    
    # Extend x-axis limits to give more space for legend on the left
    ax.set_xlim(0.1, 5.5)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=legend_size-1, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
              framealpha=0.9, fancybox=True, shadow=True, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_rate_vs_payload_size_combined.png'), dpi=300, bbox_inches='tight')

def plot_error_rate_vs_payload_size(df):
    """
    Create a scatter plot showing error rate vs payload size with circuit depth as color
    """
    # Use centralized IEEE format settings
    title_size = TITLE_SIZE
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE
    marker_size = 80
    line_width = 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate error rate (1 - success_rate)
    df['error_rate'] = 1 - df['success_rate']
    
    # Group by payload_size and calculate statistics for error distribution
    payload_groups = df.groupby('payload_size')
    
    payload_sizes = []
    error_means = []
    error_stds = []
    
    for payload_size, group in payload_groups:
        payload_sizes.append(payload_size)
        error_means.append(group['error_rate'].mean())
        error_stds.append(group['error_rate'].std())
    
    # Convert to numpy arrays
    payload_sizes = np.array(payload_sizes)
    error_means = np.array(error_means)
    error_stds = np.array(error_stds)
    
    # Create scatter plot of individual data points with circuit depth as color (background)
    scatter = ax.scatter(df['payload_size'], df['error_rate'],
                        c=df['circuit_depth'], cmap='viridis',
                        alpha=0.7, s=marker_size, edgecolor='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Circuit Depth', fontsize=label_size, fontweight='bold')
    cbar.ax.tick_params(labelsize=tick_size)
    
    # Add mean values with error bars (more prominent) - offset to show colored points
    ax.errorbar(payload_sizes + 0.15, error_means, yerr=error_stds, 
               fmt='o', color='black', markersize=8, 
               linewidth=2, capsize=5, capthick=2,
               label='Mean ± Std Dev', zorder=10)
    
    # Add regression line
    x = df['payload_size']
    y = df['error_rate']
    model = calculate_regression_stats(x, y)
    
    # Generate points for regression line
    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred)
    
    # Plot regression line with updated styling
    ax.plot(x_pred, y_pred, color=COLORBREWER_PALETTE[2], linestyle='--', linewidth=line_width,
            label=f'Trend (R²={model.rsquared:.3f})', zorder=5)
    
    # Set axis labels with updated font sizes (no title for IEEE format)
    ax.set_xlabel('Payload Size', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=label_size, fontweight='bold')
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=legend_size)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_rate_vs_payload_size.png'), dpi=300, bbox_inches='tight')

def analyze_target_depth_experiment(csv_file_path):
    """
    Main function to analyze the target depth experiment data
    """
    print(f"Loading data from {csv_file_path}...")
    df = create_experiment_dataframe(csv_file_path)
    
    print(f"Loaded {len(df)} experiment results.")
    print(f"Circuit depths: {sorted(df['circuit_depth'].unique())}")
    print(f"Payload sizes: {sorted(df['payload_size'].unique())}")
    
    print("\nGenerating error distribution by circuit depth plot...")
    plot_error_distribution_by_circuit_depth(df)
    
    print("\nGenerating error rate vs payload size plot...")
    plot_error_rate_vs_payload_size(df)
    
    print("\nAnalysis complete. All plots have been saved.")
    
    return df


if __name__ == "__main__":
    """Generate the plots with both single and combined datasets"""
    
    csv_file1 = "2_experiment_results_target_depth.csv"
    csv_file2 = "experiment_results_target_depth_20250903_221822_updated.csv"
    
    # Check if files exist
    if not os.path.exists(csv_file1):
        print(f"Error: CSV file '{csv_file1}' not found!")
        print("Make sure you're running this script from the ieee_analysis directory.")
        exit(1)
    
    print(f"Loading data from {csv_file1}...")
    
    try:
        # Load first dataset (IBM)
        df1 = create_experiment_dataframe(csv_file1)
        
        print(f"Successfully loaded {len(df1)} IBM records")
        print(f"Circuit depths: {sorted(df1['circuit_depth'].unique())}")
        print(f"Payload sizes: {sorted(df1['payload_size'].unique())}")
        
        # Generate single dataset plots
        print("\n1. Generating error_distribution_by_circuit_depth...")
        plot_error_distribution_by_circuit_depth(df1)
        print("✓ Saved: img/error_distribution_by_circuit_depth.png")
        
        print("\n2. Generating error_rate_vs_payload_size (IBM only)...")
        plot_error_rate_vs_payload_size(df1)
        print("✓ Saved: img/error_rate_vs_payload_size.png")
        
        # Check if second dataset exists and load it
        if os.path.exists(csv_file2):
            print(f"\nLoading second dataset from {csv_file2}...")
            df2 = create_experiment_dataframe(csv_file2)
            
            print(f"Successfully loaded {len(df2)} Rigetti records")
            print(f"Circuit depths: {sorted(df2['circuit_depth'].unique())}")
            print(f"Payload sizes: {sorted(df2['payload_size'].unique())}")
            
            print("\n3. Generating combined error_distribution_by_circuit_depth (IBM + Rigetti)...")
            print("   3a. With linear regression (original)...")
            plot_error_distribution_by_circuit_depth_combined(df1, df2, use_best_models=False)
            print("   ✓ Saved: img/error_distribution_by_circuit_depth_combined.png")
            
            print("   3b. With best non-linear models...")
            plot_error_distribution_by_circuit_depth_combined(df1, df2, use_best_models=True)
            print("   ✓ Saved: img/error_distribution_by_circuit_depth_combined_nonlinear.png")
            
            print("\n4. Generating combined error_rate_vs_payload_size (IBM + Rigetti)...")
            plot_error_rate_vs_payload_size_combined(df1, df2)
            print("✓ Saved: img/error_rate_vs_payload_size_combined.png")
            
            print("\n4b. Generating boxplot error_rate_vs_payload_size (IBM + Rigetti)...")
            plot_error_rate_vs_payload_size_combined_boxplot(df1, df2)
            print("✓ Saved: img/error_rate_vs_payload_size_combined_boxplot.png")
            
            print("\n5. Analyzing circuit depth models (Non-linear alternatives to linear regression)...")
            model_analyzer_ibm, model_analyzer_rigetti, comparison_tables = analyze_circuit_depth_models(df1, df2)
            print("✓ Model analysis complete")
            
            print(f"\n🎉 Successfully generated all plots and analysis!")
            print("Images saved in: img/")
            print("- img/error_distribution_by_circuit_depth.png (IBM only)")
            print("- img/error_rate_vs_payload_size.png (IBM only)")
            print("- img/error_distribution_by_circuit_depth_combined.png (IBM + Rigetti, Linear)")
            print("- img/error_distribution_by_circuit_depth_combined_nonlinear.png (IBM + Rigetti, Best Models)")
            print("- img/error_rate_vs_payload_size_combined.png (IBM + Rigetti)")
            print("- img/error_rate_vs_payload_size_combined_boxplot.png (IBM + Rigetti, Distribution Focus)")
            print("- img/ibm_model_comparison_plot.png (IBM model fits & residuals)")
            print("- img/rigetti_model_comparison_plot.png (Rigetti model fits & residuals)")
            print("\nModel comparison tables:")
            print("- img/ibm_model_comparison.csv")
            print("- img/rigetti_model_comparison.csv")
        else:
            print(f"\nSecond dataset '{csv_file2}' not found. Generating single dataset plots only.")
            print(f"\n🎉 Successfully generated single dataset plots!")
            print("Images saved in: img/")
            print("- img/error_distribution_by_circuit_depth.png")
            print("- img/error_rate_vs_payload_size.png")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 