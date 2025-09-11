import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from matplotlib import patches
import os
import ast
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.gofplots import ProbPlot

# Custom color palette
COLORBREWER_PALETTE = {
    1: '#1b9e77',    # Teal
    2: '#d95f02',    # Orange
    3: '#7570b3',    # Purple
    4: '#e7298a',    # Pink
    5: '#66a61e',    # Green
    6: '#e6ab02'     # Yellow
}

# Distinct marker styles with different shapes and fills
MARKER_STYLES = {
    1: 'o',     # Circle
    2: 's',     # Square
    3: '^',     # Triangle up
    4: 'D',     # Diamond
    5: 'v',     # Triangle down
    6: 'p',     # Pentagon
    7: 'h',     # Hexagon
    8: '8',     # Octagon
    9: '*',     # Star
    10: 'P',    # Plus (filled)
}

# Define the gate ranges for the experiments
# Format: (start, end) where end is exclusive
GATE_RANGES = [
    (200, 205),
    (500, 505),
    (1000, 1005),
    (1500, 1505),
    (2000, 2005),
    (3000, 3005),
    (5000, 5005),
    (10000, 10005)
    # (20000, 20005),
]

# Maximum gate count for cap score calculation
MAX_GATE_COUNT = 3005

# Define the circuit depth ranges for analysis based on observed depths
# Format: (start, end) where end is inclusive
DEPTH_RANGES = [
    (100, 200),    # Covers 111-113, 141-145
    (200, 300),    # Covers 207-211, 261-263
    (300, 500),    # Covers 341-345, 405-415
    (500, 700),    # Covers 507-511, 511-513, 675-679
    (700, 1000),   # Covers 761-763
    (1000, 1200),  # Covers 1005-1015, 1007-1011, 1009-1011, 1011-1013
    (1300, 1500),  # Covers 1341-1345
    (1500, 2000),  # Covers 1507-1511, 1511-1513
    (2000, 3000),  # Covers 2005-2015, 2007-2011
    (3000, 4000),  # Covers 3005-3015, 3007-3011
    (4000, 5000),  # Covers 4005-4015
    (5000, 7000),  # Covers 6005-6015
    (8000, 10100)  # Covers 10005-10015
    # (20000, 20100)  # Covers 20000-0015
]

# Maximum circuit depth for cap score calculation
MAX_CIRCUIT_DEPTH = 7000

def create_experiment_dataframe(csv_file_paths=None):
    """
    Create a DataFrame from experiment results CSV files.
    
    Args:
        csv_file_paths (str or list): Path(s) to the CSV file(s) containing experiment results.
                                     If None, will look for CSV files in the experiment directory.
    
    Returns:
        pd.DataFrame: DataFrame containing experiment results from all CSV files.
    """
    # Handle different input types
    if csv_file_paths is None:
        # Look in the experiment directory for CSV files
        experiment_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '')
        csv_files = [os.path.join(experiment_dir, f) for f in os.listdir(experiment_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the experiment directory.")
        
        csv_file_paths = csv_files
    elif isinstance(csv_file_paths, str):
        # Convert single string path to list
        csv_file_paths = [csv_file_paths]
    
    # List to store DataFrames from each CSV
    dfs = []
    
    # Process each CSV file
    for csv_path in csv_file_paths:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert string representations of dictionaries to actual dictionaries
        for col in ['counts', 'circuit_count_ops']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Add counts_zeros and counts_ones columns for analysis
        df['counts_zeros'] = df.apply(lambda row: row['counts'].get('0' * row['payload_size'], 0) 
                                     if isinstance(row['counts'], dict) else 0, axis=1)
        
        df['counts_ones'] = df.apply(lambda row: sum(count for key, count in row['counts'].items() 
                                    if key != '0' * row['payload_size']) 
                                    if isinstance(row['counts'], dict) else 0, axis=1)
        
        # Add to list of DataFrames
        dfs.append(df)
    
    # Concatenate all DataFrames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Set job_id as index
        if 'ibm_job_id' in combined_df.columns:
            combined_df.set_index('ibm_job_id', inplace=True)
        
        return combined_df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no files were processed

def calculate_regression_stats(x, y):
    """
    Calculate regression statistics using statsmodels with error handling.
    
    Returns:
        model: Fitted OLS model, or None if regression fails
    """
    try:
        # Check if we have enough data points
        if len(x) < 3:
            print(f"Warning: Not enough data points for reliable regression (n={len(x)}, minimum=3)")
            return None
            
        # Check for NaN values
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            print(f"Warning: NaN values found in data")
            return None
            
        # Check if there's variation in x
        if np.var(x) == 0:
            print(f"Warning: No variation in x values for regression")
            return None
            
        # Check if there's variation in y
        if np.var(y) == 0:
            print(f"Warning: No variation in y values for regression")
            return None
            
        # Check if x has reasonable range (for meaningful regression)
        x_range = np.max(x) - np.min(x)
        if x_range < np.min(x) * 0.1:  # Range should be at least 10% of minimum value
            print(f"Warning: Insufficient range in x values for reliable regression (range={x_range}, min_x={np.min(x)})")
            return None
            
        X = sm.add_constant(x)  # Add constant term
        model = sm.OLS(y, X).fit()
        
        # Check if the model fit was successful
        if np.isnan(model.rsquared):
            print(f"Warning: Regression resulted in NaN R-squared")
            return None
            
        return model
        
    except Exception as e:
        print(f"Warning: Regression failed with error: {str(e)}")
        return None

def plot_success_rates(df, use_log_x=False, use_log_y=False, ieee_format=True, show_title=False):
    """
    Plot success rates vs number of gates for different payload sizes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results
    use_log_x : bool
        Whether to use logarithmic scale for x-axis
    use_log_y : bool
        Whether to use logarithmic scale for y-axis
    ieee_format : bool
        Whether to use larger fonts suitable for IEEE two-column format (default: True)
    show_title : bool
        Whether to show the plot title (default: False for IEEE format)
    
    Returns:
    --------
    list : Regression statistics for each payload size
    """
    # Set font sizes based on format
    if ieee_format:
        title_size = 18
        label_size = 16
        tick_size = 14
        legend_size = 12
        marker_size = 80  # Reduced marker size
        line_width = 2
        fig_size = (8, 7)  # Slightly taller to accommodate legend below
    else:
        title_size = 14
        label_size = 12
        tick_size = 10
        legend_size = 10
        marker_size = 60  # Reduced marker size
        line_width = 1
        fig_size = (10, 8)
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=fig_size)
    ax = plt.gca()
    
    # Store regression results for printing
    regression_stats = []
    
    # Plot for each payload size (skip payload size 5 due to insufficient data)
    payload_sizes_to_plot = [ps for ps in sorted(df['payload_size'].unique()) if ps != 5]
    
    # Print information about skipped payload sizes
    all_payload_sizes = sorted(df['payload_size'].unique())
    skipped_sizes = [ps for ps in all_payload_sizes if ps not in payload_sizes_to_plot]
    if skipped_sizes:
        print(f"\nSkipping payload size(s) {skipped_sizes} due to insufficient data for reliable analysis")
        for ps in skipped_sizes:
            ps_data = df[df['payload_size'] == ps]
            print(f"  Payload Size {ps}: {len(ps_data)} data points, gate range: {ps_data['num_gates'].min()}-{ps_data['num_gates'].max()}")
    
    for payload_size in payload_sizes_to_plot:
        payload_data = df[df['payload_size'] == payload_size]
        
        # Group by num_gates and calculate mean success rate
        grouped_data = payload_data.groupby('num_gates')['success_rate'].mean()
        
        # Remove any NaN values before regression
        grouped_data_clean = grouped_data.dropna()
        
        x = np.array(grouped_data_clean.index)
        y = np.array(grouped_data_clean.values * 100)
        
        # Debug information for problematic payload sizes
        if payload_size in [3, 5]:
            print(f"\nDebug info for Payload Size {payload_size}:")
            print(f"  Number of raw data points: {len(payload_data)}")
            print(f"  Unique gate counts: {sorted(payload_data['num_gates'].unique())}")
            print(f"  Grouped data points (before cleaning): {len(grouped_data)}")
            print(f"  Grouped data points (after cleaning): {len(grouped_data_clean)}")
            print(f"  NaN values removed: {len(grouped_data) - len(grouped_data_clean)}")
            print(f"  X values (gates): {x}")
            print(f"  Y values (success rates): {y}")
            print(f"  X range: {np.max(x) - np.min(x) if len(x) > 0 else 'N/A'}")
            print(f"  X variance: {np.var(x) if len(x) > 0 else 'N/A'}")
            print(f"  Y variance: {np.var(y) if len(y) > 0 else 'N/A'}")
        
        if use_log_x:
            x = np.log10(x)
        if use_log_y:
            y = np.log10(y)
        
        # Scatter plot with ColorBrewer colors and distinct markers
        ax.scatter(x, y, 
                  color=COLORBREWER_PALETTE[payload_size],
                  marker=MARKER_STYLES[payload_size],
                  s=marker_size,
                  label=f'Payload Size {payload_size}')
        
        # Regression analysis
        model = calculate_regression_stats(x, y)
        
        # Only plot regression line and store stats if model is valid
        if model is not None:
            # Generate points for regression line
            x_pred = np.linspace(x.min(), x.max(), 100)
            X_pred = sm.add_constant(x_pred)
            y_pred = model.predict(X_pred)
            
            # Plot regression line with matching color
            ax.plot(x_pred, y_pred,
                    color=COLORBREWER_PALETTE[payload_size],
                    linestyle='--', alpha=0.7, linewidth=line_width,
                    label=f'Trend P{payload_size} (R²={model.rsquared:.3f})')
            
            # Store regression statistics
            stats = {
                'payload_size': payload_size,
                'slope': model.params[1],
                'intercept': model.params[0],
                'r_squared': model.rsquared,
                'p_value': model.pvalues[1],
                'std_err': model.bse[1]
            }
        else:
            # Store NaN statistics for failed regression
            print(f"Skipping regression line for Payload Size {payload_size} due to insufficient or invalid data")
            stats = {
                'payload_size': payload_size,
                'slope': np.nan,
                'intercept': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_err': np.nan
            }
        
        regression_stats.append(stats)
    
    # Filter to include only ranges that have data in the DataFrame
    gate_ranges = []
    for gate_range in GATE_RANGES:
        if df[df['num_gates'].between(gate_range[0], gate_range[1])].shape[0] > 0:
            gate_ranges.append(gate_range)
    
    # Set x-axis ticks
    if use_log_x:
        tick_positions = [np.log10(r[0]) for r in gate_ranges]
        tick_labels = [f"{r[0]}-{r[1]}" for r in gate_ranges]
        plt.xlabel('Number of Gates (Range) - Log Scale', fontsize=label_size, fontweight='bold')
    else:
        tick_positions = [r[0] for r in gate_ranges]
        tick_labels = [f"{r[0]}-{r[1]}" for r in gate_ranges]
        plt.xlabel('Number of Gates (Range)', fontsize=label_size, fontweight='bold')
    
    plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=tick_size)
    
    # Set y-axis label
    if use_log_y:
        plt.ylabel('Success Rate (%) - Log Scale', fontsize=label_size, fontweight='bold')
    else:
        plt.ylabel('Success Rate (%)', fontsize=label_size, fontweight='bold')
    
    # Set y-axis tick labels
    plt.yticks(fontsize=tick_size)
    
    # Set title with appropriate scale indication (only if show_title is True)
    if show_title:
        scale_text = ""
        if use_log_x and use_log_y:
            scale_text = " (Log-Log Scale)"
        elif use_log_x:
            scale_text = " (Log X Scale)"
        elif use_log_y:
            scale_text = " (Log Y Scale)"
        
        plt.title('Success Rates by Number of Gates and Payload Size' + scale_text, fontsize=title_size, fontweight='bold', pad=20)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout without legend
    plt.tight_layout()
    
    # Save figure
    scale_suffix = ""
    if use_log_x and use_log_y:
        scale_suffix = "_logxy"
    elif use_log_x:
        scale_suffix = "_logx"
    elif use_log_y:
        scale_suffix = "_logy"
    
    plt.savefig(f'success_rate_experiment{scale_suffix}.png', 
                dpi=300, bbox_inches='tight')
    
    return regression_stats

def plot_circuit_complexity(df):
    """
    Create comprehensive circuit complexity analysis plots with colorblind-friendly colors
    """
    # Create 2D plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Success vs Gates
    for payload_size in sorted(df['payload_size'].unique()):
        payload_data = df[df['payload_size'] == payload_size]
        ax1.scatter(payload_data['num_gates'], payload_data['success_rate'] * 100,
                   color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                   marker=MARKER_STYLES.get(payload_size, 'o'),
                   s=100,
                   label=f'Payload Size {payload_size}')
    
    ax1.set_xlabel('Number of Gates')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate vs Number of Gates')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Success vs Depth
    for payload_size in sorted(df['payload_size'].unique()):
        payload_data = df[df['payload_size'] == payload_size]
        ax2.scatter(payload_data['circuit_depth'], payload_data['success_rate'] * 100,
                   color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                   marker=MARKER_STYLES.get(payload_size, 'o'),
                   s=100,
                   label=f'Payload Size {payload_size}')
    
    ax2.set_xlabel('Circuit Depth')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate vs Circuit Depth')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Success vs Width
    for payload_size in sorted(df['payload_size'].unique()):
        payload_data = df[df['payload_size'] == payload_size]
        ax3.scatter(payload_data['circuit_width'], payload_data['success_rate'] * 100,
                   color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                   marker=MARKER_STYLES.get(payload_size, 'o'),
                   s=100,
                   label=f'Payload Size {payload_size}')
    
    ax3.set_xlabel('Circuit Width')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate vs Circuit Width')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('circuit_complexity_2d.png', dpi=300, bbox_inches='tight')
    
    # Create multiple 3D plots with different viewing angles
    view_angles = [
        (20, 45),   # Default view
        (20, 135),  # Rotated 90 degrees
        (45, 90),   # Top-down view
        (60, 30)    # Higher elevation
    ]
    
    for i, (elev, azim) in enumerate(view_angles, 1):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a scatter plot for each payload size
        for payload_size in sorted(df['payload_size'].unique()):
            payload_data = df[df['payload_size'] == payload_size]
            
            # 3D scatter plot
            ax.scatter(
                payload_data['num_gates'],
                payload_data['circuit_depth'],
                payload_data['success_rate'] * 100,
                color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                marker=MARKER_STYLES.get(payload_size, 'o'),
                s=100,
                label=f'Payload Size {payload_size}'
            )
        
        # Set labels and title
        ax.set_xlabel('Number of Gates', fontsize=12)
        ax.set_ylabel('Circuit Depth', fontsize=12)
        ax.set_zlabel('Success Rate (%)', fontsize=12)
        ax.set_title('3D Visualization of Circuit Complexity vs Success Rate', fontsize=14)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Save figure with view angle indicator
        plt.tight_layout()
        plt.savefig(f'circuit_complexity_3d_view{i}.png', dpi=300, bbox_inches='tight')
    
    return None

def plot_error_analysis(df):
    """
    Create comprehensive error analysis plots with colorblind-friendly colors
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(18, 12))
    gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # 1. Bar plot comparing counts_zeros vs counts_ones
    bar_width = 0.35
    payload_sizes = sorted(df['payload_size'].unique())
    x = np.arange(len(payload_sizes))
    
    # Calculate means for each payload size
    zeros_means = [df[df['payload_size'] == size]['counts_zeros'].mean() for size in payload_sizes]
    ones_means = [df[df['payload_size'] == size]['counts_ones'].mean() for size in payload_sizes]
    
    # Create bars for each payload size with ColorBrewer colors
    for i, size in enumerate(payload_sizes):
        # Create grouped bars with matching colors
        ax1.bar(x[i] - bar_width/2, zeros_means[i], bar_width,
                color=COLORBREWER_PALETTE[size], alpha=0.7,
                edgecolor='black', linewidth=1,
                label=f'All Zeros (P{size})')
        ax1.bar(x[i] + bar_width/2, ones_means[i], bar_width,
                color=COLORBREWER_PALETTE[size], alpha=0.3,
                edgecolor='black', linewidth=1, hatch='///',
                label=f'Other States (P{size})')
    
    ax1.set_xlabel('Payload Size', fontsize=12)
    ax1.set_ylabel('Average Counts', fontsize=12)
    ax1.set_title('Distribution of Measurement Outcomes', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(payload_sizes)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. Stacked bar chart showing error distribution
    bottom = np.zeros(len(payload_sizes))
    
    # Filter to include only ranges that have data in the DataFrame
    gate_ranges = []
    for gate_range in GATE_RANGES:
        if df[df['num_gates'].between(gate_range[0], gate_range[1])].shape[0] > 0:
            gate_ranges.append(gate_range)
    
    # Create a color map for gate ranges that cycles through the COLORBREWER_PALETTE
    gate_colors = {}
    for i, gate_range in enumerate(gate_ranges):
        # Cycle through the available colors in COLORBREWER_PALETTE
        color_idx = (i % len(COLORBREWER_PALETTE)) + 1
        gate_colors[gate_range] = COLORBREWER_PALETTE[color_idx]
    
    # Calculate alpha values that scale properly with the number of gate ranges
    num_ranges = len(gate_ranges)
    
    for i, gate_range in enumerate(gate_ranges):
        errors = []
        for size in payload_sizes:
            mask = (df['payload_size'] == size) & (df['num_gates'].between(gate_range[0], gate_range[1]))
            error_rate = 1 - df[mask]['success_rate'].mean() if not df[mask].empty else 0
            errors.append(error_rate)
        
        # Use the assigned color from COLORBREWER_PALETTE with appropriate alpha
        alpha_value = 0.5 + (0.5 * i / num_ranges) if num_ranges > 1 else 0.7
        alpha_value = min(0.9, alpha_value)  # Cap at 0.9 to be safe
        
        ax2.bar(x, errors, bottom=bottom,
                label=f'{gate_range[0]}-{gate_range[1]} gates',
                color=gate_colors[gate_range],
                alpha=alpha_value,
                edgecolor='black', linewidth=0.5)
        bottom += errors
    
    ax2.set_xlabel('Payload Size', fontsize=12)
    ax2.set_ylabel('Error Rate', fontsize=12)
    ax2.set_title('Cumulative Error Rate by Payload Size and Gate Count', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(payload_sizes)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. Heatmap of success rate by payload size and gate count
    heatmap_data = np.zeros((len(payload_sizes), len(gate_ranges)))
    
    for i, size in enumerate(payload_sizes):
        for j, gate_range in enumerate(gate_ranges):
            mask = (df['payload_size'] == size) & (df['num_gates'].between(gate_range[0], gate_range[1]))
            if not df[mask].empty:
                heatmap_data[i, j] = df[mask]['success_rate'].mean()
    
    im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Success Rate', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax3.set_xticks(np.arange(len(gate_ranges)))
    ax3.set_yticks(np.arange(len(payload_sizes)))
    ax3.set_xticklabels([f'{r[0]}-{r[1]}' for r in gate_ranges])
    ax3.set_yticklabels(payload_sizes)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in the heatmap cells
    for i in range(len(payload_sizes)):
        for j in range(len(gate_ranges)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="w" if heatmap_data[i, j] < 0.5 else "black")
    
    ax3.set_title('Success Rate Heatmap', fontsize=14)
    ax3.set_xlabel('Gate Count Range', fontsize=12)
    ax3.set_ylabel('Payload Size', fontsize=12)

    # 4. Add a text box with summary statistics
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')  # Hide axes
    
    # Calculate summary statistics
    overall_success = df['success_rate'].mean()
    best_config_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    best_payload = payload_sizes[best_config_idx[0]]
    best_gates = gate_ranges[best_config_idx[1]]
    best_success = heatmap_data[best_config_idx]
    
    worst_config_idx = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
    worst_payload = payload_sizes[worst_config_idx[0]]
    worst_gates = gate_ranges[worst_config_idx[1]]
    worst_success = heatmap_data[worst_config_idx]
    
    # Create text for the summary box
    summary_text = (
        "ERROR ANALYSIS SUMMARY\n"
        "=====================\n\n"
        f"Overall Success Rate: {overall_success:.2%}\n\n"
        f"Best Configuration:\n"
        f"  - Payload Size: {best_payload}\n"
        f"  - Gate Range: {best_gates[0]}-{best_gates[1]}\n"
        f"  - Success Rate: {best_success:.2%}\n\n"
        f"Worst Configuration:\n"
        f"  - Payload Size: {worst_payload}\n"
        f"  - Gate Range: {worst_gates[0]}-{worst_gates[1]}\n"
        f"  - Success Rate: {worst_success:.2%}\n\n"
        f"Success Rate by Payload Size:\n"
    )
    
    # Add success rates by payload size
    for size in payload_sizes:
        rate = df[df['payload_size'] == size]['success_rate'].mean()
        summary_text += f"  - Payload {size}: {rate:.2%}\n"
    
    # Create a text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    
    return None

def calculate_cap_score(success_rate, payload_size, complexity_value, ps_max=4, complexity_max=None, complexity_type='gates', 
                   payload_weight=2.0, complexity_weight=1.0):
    """
    Calculate the Cap-score (Capability score) for a quantum hardware configuration.
    
    The Cap-score is a metric that combines success rate with circuit complexity,
    providing a single value that represents how well the hardware handles complex circuits.
    
    Formula: Cap-score = success_rate * ((payload_size/ps_max)^payload_weight * (complexity_value/complexity_max)^complexity_weight) * 100
    
    This rewards high success rates while accounting for the different impacts of payload size and circuit complexity.
    
    Parameters:
    -----------
    success_rate : float
        Success rate (0 to 1)
    payload_size : int
        Number of qubits in payload
    complexity_value : int
        Complexity value (gate count or circuit depth)
    ps_max : int
        Maximum payload size (default: 4)
    complexity_max : int
        Maximum complexity value (default: uses global MAX_GATE_COUNT or MAX_CIRCUIT_DEPTH)
    complexity_type : str
        Type of complexity measure ('gates' or 'depth')
    payload_weight : float
        Weight factor for payload size impact (default: 2.0)
    complexity_weight : float
        Weight factor for complexity impact (default: 1.0)
    
    Returns:
    --------
    float : Cap-score (0 to 100)
    """
    if complexity_max is None:
        if complexity_type == 'gates':
            complexity_max = MAX_GATE_COUNT
        elif complexity_type == 'depth':
            complexity_max = MAX_CIRCUIT_DEPTH
        else:
            raise ValueError("complexity_type must be 'gates' or 'depth'")
    
    # Normalize the factors
    normalized_payload = min(payload_size / ps_max, 1.0)
    normalized_complexity = min(complexity_value / complexity_max, 1.0)
    
    # Apply weights to reflect different impacts
    weighted_factor = (normalized_payload ** payload_weight) * (normalized_complexity ** complexity_weight)
    
    return success_rate * weighted_factor * 100

def plot_cap_scores(df, payload_weight=2.0, complexity_weight=1.0):
    """
    Create visualization of Cap-scores with grouped bars for each payload size and gate range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results
    payload_weight : float
        Weight factor for payload size impact (default: 2.0)
    complexity_weight : float
        Weight factor for complexity impact (default: 1.0)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Filter to include only ranges that have data in the DataFrame
    gate_ranges = []
    for gate_range in GATE_RANGES:
        if df[df['num_gates'].between(gate_range[0], gate_range[1])].shape[0] > 0:
            gate_ranges.append(gate_range)
    
    payload_sizes = sorted(df['payload_size'].unique())
    
    # Calculate positions for bars
    num_ranges = len(gate_ranges)
    bar_width = 0.8 / (num_ranges + 1)  # Adjust bar width based on number of ranges
    index = np.arange(len(payload_sizes))
    
    # Create a color map for gate ranges that cycles through the COLORBREWER_PALETTE
    gate_colors = {}
    for i, gate_range in enumerate(gate_ranges):
        # Cycle through the available colors in COLORBREWER_PALETTE
        color_idx = (i % len(COLORBREWER_PALETTE)) + 1
        gate_colors[gate_range] = COLORBREWER_PALETTE[color_idx]
    
    # Calculate and plot Cap-scores for each gate range
    for i, gate_range in enumerate(gate_ranges):
        cap_scores = []
        
        for size in payload_sizes:
            mask = (df['payload_size'] == size) & (df['num_gates'].between(gate_range[0], gate_range[1]))
            if not df[mask].empty:
                success_rate = df[mask]['success_rate'].mean()
                cap_score = calculate_cap_score(
                    success_rate, 
                    size, 
                    gate_range[0], 
                    complexity_type='gates',
                    payload_weight=payload_weight,
                    complexity_weight=complexity_weight
                )
                cap_scores.append(cap_score)
            else:
                cap_scores.append(0)
        
        # Create bars with position offset
        position = index + (i - num_ranges/2) * bar_width
        bars = ax.bar(position, cap_scores, bar_width,
                     color=gate_colors[gate_range],
                     edgecolor='black',
                     linewidth=1,
                     alpha=0.8)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}',
                   ha='center', va='bottom',
                   fontsize=8, rotation=0)
    
    # Add legend with custom patches
    legend_patches = []
    for i, gate_range in enumerate(gate_ranges):
        patch = patches.Patch(
            color=gate_colors[gate_range],
            label=f'{gate_range[0]}-{gate_range[1]} gates'
        )
        legend_patches.append(patch)
    
    # If we have many gate ranges, create a more compact legend
    if num_ranges > 5:
        ax.legend(handles=legend_patches, 
                 title='Gate Ranges',
                 loc='upper right',
                 bbox_to_anchor=(1, 1),
                 fontsize=8,
                 ncol=2)  # Use 2 columns for the legend
    else:
        ax.legend(handles=legend_patches, 
                 title='Gate Ranges',
                 loc='upper right',
                 bbox_to_anchor=(1, 1),
                 fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Payload Size', fontsize=14)
    ax.set_ylabel('Cap-Score (0-100)', fontsize=14)
    ax.set_title('Cap-Score Analysis by Payload Size and Gate Range', fontsize=16)
    
    # Set x-axis ticks
    ax.set_xticks(index)
    ax.set_xticklabels(payload_sizes)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add a horizontal line at the average Cap-score
    all_cap_scores = []
    for size in payload_sizes:
        for gate_range in gate_ranges:
            mask = (df['payload_size'] == size) & (df['num_gates'].between(gate_range[0], gate_range[1]))
            if not df[mask].empty:
                success_rate = df[mask]['success_rate'].mean()
                cap_score = calculate_cap_score(success_rate, size, gate_range[0], complexity_type='gates')
                all_cap_scores.append(cap_score)
    
    avg_cap_score = np.mean(all_cap_scores) if all_cap_scores else 0
    ax.axhline(y=avg_cap_score, color='r', linestyle='--', alpha=0.5)
    ax.text(len(payload_sizes)-1, avg_cap_score + 1, f'Avg: {avg_cap_score:.1f}', 
           color='r', ha='right', va='bottom')
    
    # Add annotations for best and worst configurations
    if all_cap_scores:
        max_cap_score = max(all_cap_scores)
        min_cap_score = min([score for score in all_cap_scores if score > 0])
        
        ax.text(0.02, 0.98, 
               f'Best Cap-Score: {max_cap_score:.1f}\nWorst Cap-Score: {min_cap_score:.1f}',
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('cap_scores_analysis.png', dpi=300, bbox_inches='tight')
    
    return None

def plot_depth_analysis(df, payload_weight=2.9, complexity_weight=0.1):
    """
    Create comprehensive analysis plots based on circuit depth instead of gate count.
    This provides an alternative perspective on how circuit complexity affects performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results
    payload_weight : float
        Weight factor for payload size impact (default: 2.0)
    complexity_weight : float
        Weight factor for complexity impact (default: 1.0)
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(18, 12))
    gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Get payload sizes and prepare data
    payload_sizes = sorted(df['payload_size'].unique())
    x = np.arange(len(payload_sizes))
    
    # 1. Scatter plot of success rate vs circuit depth
    for i, size in enumerate(payload_sizes):
        payload_data = df[df['payload_size'] == size]
        ax1.scatter(payload_data['circuit_depth'], payload_data['success_rate'] * 100,
                   color=COLORBREWER_PALETTE[size], 
                   marker=MARKER_STYLES[size],
                   s=100,
                   label=f'Payload Size {size}')
        
        # Add trend line
        if len(payload_data) > 1:
            z = np.polyfit(payload_data['circuit_depth'], payload_data['success_rate'] * 100, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(payload_data['circuit_depth'].min(), payload_data['circuit_depth'].max(), 100)
            y_trend = p(x_trend)
            ax1.plot(x_trend, y_trend, 
                    color=COLORBREWER_PALETTE[size],
                    linestyle='--', alpha=0.7,
                    label=f'Trend P{size} (slope: {z[0]:.2e})')
    
    ax1.set_xlabel('Circuit Depth', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate vs Circuit Depth', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Stacked bar chart showing error distribution by depth range
    bottom = np.zeros(len(payload_sizes))
    
    # Filter to include only depth ranges that have data in the DataFrame
    depth_ranges = []
    for depth_range in DEPTH_RANGES:
        if df[df['circuit_depth'].between(depth_range[0], depth_range[1], inclusive='both')].shape[0] > 0:
            depth_ranges.append(depth_range)
    
    # Create a color map for depth ranges
    depth_colors = {}
    for i, depth_range in enumerate(depth_ranges):
        color_idx = (i % len(COLORBREWER_PALETTE)) + 1
        depth_colors[depth_range] = COLORBREWER_PALETTE[color_idx]
    
    # Calculate error rates for each payload size and depth range
    for i, depth_range in enumerate(depth_ranges):
        error_rates = []
        
        for size in payload_sizes:
            mask = (df['payload_size'] == size) & (df['circuit_depth'].between(depth_range[0], depth_range[1], inclusive='both'))
            if not df[mask].empty:
                error_rate = 1 - df[mask]['success_rate'].mean()
                error_rates.append(error_rate * 100)  # Convert to percentage
            else:
                error_rates.append(0)
        
        ax2.bar(x, error_rates, bottom=bottom, label=f'{depth_range[0]}-{depth_range[1]}',
               color=depth_colors[depth_range], alpha=0.7, edgecolor='black', linewidth=0.5)
        bottom += error_rates
    
    ax2.set_xlabel('Payload Size', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Error Distribution by Circuit Depth', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(payload_sizes)
    ax2.legend(title='Depth Range', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. Heatmap of success rates
    heatmap_data = np.zeros((len(payload_sizes), len(depth_ranges)))
    
    for i, size in enumerate(payload_sizes):
        for j, depth_range in enumerate(depth_ranges):
            mask = (df['payload_size'] == size) & (df['circuit_depth'].between(depth_range[0], depth_range[1], inclusive='both'))
            if not df[mask].empty:
                heatmap_data[i, j] = df[mask]['success_rate'].mean()
    
    im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Success Rate', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax3.set_xticks(np.arange(len(depth_ranges)))
    ax3.set_yticks(np.arange(len(payload_sizes)))
    ax3.set_xticklabels([f'{r[0]}-{r[1]}' for r in depth_ranges])
    ax3.set_yticklabels(payload_sizes)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in the heatmap cells
    for i in range(len(payload_sizes)):
        for j in range(len(depth_ranges)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="w" if heatmap_data[i, j] < 0.5 else "black")
    
    ax3.set_title('Success Rate Heatmap by Circuit Depth', fontsize=14)
    ax3.set_xlabel('Circuit Depth Range', fontsize=12)
    ax3.set_ylabel('Payload Size', fontsize=12)

    # 4. Add a text box with summary statistics
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')  # Hide axes
    
    # Calculate summary statistics
    overall_success = df['success_rate'].mean()
    best_config_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    best_payload = payload_sizes[best_config_idx[0]]
    best_depth_range = depth_ranges[best_config_idx[1]]
    best_success = heatmap_data[best_config_idx]
    
    worst_config_idx = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
    worst_payload = payload_sizes[worst_config_idx[0]]
    worst_depth_range = depth_ranges[worst_config_idx[1]]
    worst_success = heatmap_data[worst_config_idx]
    
    # Create text for the summary box
    summary_text = (
        "CIRCUIT DEPTH ANALYSIS SUMMARY\n"
        "=============================\n\n"
        f"Overall Success Rate: {overall_success:.2%}\n\n"
        f"Best Depth Configuration:\n"
        f"  - Payload Size: {best_payload}\n"
        f"  - Depth Range: {best_depth_range[0]}-{best_depth_range[1]}\n"
        f"  - Success Rate: {best_success:.2%}\n"
        f"  - Depth-Cap Score: {best_success:.2f}\n\n"
        f"Worst Depth Configuration:\n"
        f"  - Payload Size: {worst_payload}\n"
        f"  - Depth Range: {worst_depth_range[0]}-{worst_depth_range[1]}\n"
        f"  - Success Rate: {worst_success:.2%}\n"
        f"  - Depth-Cap Score: {worst_success:.2f}\n\n"
        f"Average Depth-Cap Score: {best_success:.2f}\n\n"
        f"Weights Used:\n"
        f"  - Payload Weight: {payload_weight:.2f}\n"
        f"  - Complexity Weight: {complexity_weight:.2f}\n\n"
        f"Success Rate by Payload Size:\n"
    )
    
    # Add success rates by payload size
    for size in payload_sizes:
        rate = df[df['payload_size'] == size]['success_rate'].mean()
        summary_text += f"  - Payload {size}: {rate:.2%}\n"
    
    # Create a text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('circuit_depth_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create a separate figure for depth-based Cap-scores
    if depth_cap_scores:
        # Create a new figure for the cap scores
        plt.figure(figsize=(15, 8))
        
        # Create a dictionary to store cap scores by payload size and depth range
        cap_score_dict = {}
        for payload_size in payload_sizes:
            cap_score_dict[payload_size] = {}
            
        # Fill the dictionary with cap scores
        for score in depth_cap_scores:
            payload_size = score['payload_size']
            depth_range = score['depth_range']
            depth_range_str = f"{depth_range[0]}-{depth_range[1]}"
            cap_score_dict[payload_size][depth_range_str] = score['depth_cap_score']
        
        # Set up the bar positions
        bar_width = 0.8 / len(depth_ranges) if len(depth_ranges) > 0 else 0.8
        index = np.arange(len(payload_sizes))
        
        # Plot bars for each depth range
        for i, depth_range in enumerate(depth_ranges):
            depth_range_str = f"{depth_range[0]}-{depth_range[1]}"
            scores = []
            
            for payload_size in payload_sizes:
                scores.append(cap_score_dict[payload_size].get(depth_range_str, 0))
            
            # Calculate the position for this group of bars
            positions = index + (i - len(depth_ranges)/2 + 0.5) * bar_width
            
            # Plot the bars with the appropriate color from the depth_colors dictionary
            plt.bar(positions, scores, bar_width * 0.9,
                   label=depth_range_str,
                   color=depth_colors[depth_range],
                   edgecolor='black',
                   linewidth=0.5)
        
        # Set the x-axis labels and ticks
        plt.xlabel('Payload Size', fontsize=14)
        plt.ylabel('Depth-Cap Score (0-100)', fontsize=14)
        plt.title(f'Depth-Based Cap-Score Analysis (PW={payload_weight:.1f}, CW={complexity_weight:.1f})', fontsize=16)
        plt.xticks(index, payload_sizes)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add a horizontal line at the average Cap-score
        plt.axhline(y=avg_depth_cap, color='r', linestyle='--', alpha=0.5)
        plt.text(len(payload_sizes)-1, avg_depth_cap + 1, f'Avg: {avg_depth_cap:.1f}', 
                color='r', ha='right', va='bottom')
        
        # Add annotations for best and worst configurations
        plt.text(0.02, 0.98, 
                f'Best Depth-Cap Score: {best_config["depth_cap_score"]:.1f}\nWorst Depth-Cap Score: {worst_config["depth_cap_score"]:.1f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add a legend
        plt.legend(title='Depth Range', loc='upper right')
        
        plt.tight_layout()
        plt.savefig('depth_cap_scores_analysis.png', dpi=300, bbox_inches='tight')
    
    return None

def analyze_error_impact_weights(df):
    """
    Analyze the experimental data to determine the relative impact of payload size,
    gate count, and circuit depth on error rates.
    
    This function helps determine appropriate weights for the Cap-score calculation
    by analyzing how much each factor contributes to decreased success rates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results
        
    Returns:
    --------
    tuple : (payload_weight, complexity_weight)
        Suggested weights for payload size and complexity
    """
    # Get unique payload sizes, gate counts, and depth values
    payload_sizes = sorted(df['payload_size'].unique())
    
    # Calculate baseline success rate (smallest payload, fewest gates)
    min_payload = min(payload_sizes)
    min_gates_range = min(df['num_gates'])
    min_depth = min(df['circuit_depth'])
    
    # Baseline for gate analysis
    baseline_mask_gates = (df['payload_size'] == min_payload) & (df['num_gates'] == min_gates_range)
    if df[baseline_mask_gates].empty:
        baseline_success_gates = 1.0  # Default if no baseline data
    else:
        baseline_success_gates = df[baseline_mask_gates]['success_rate'].mean()
    
    # Baseline for depth analysis
    baseline_mask_depth = (df['payload_size'] == min_payload) & (df['circuit_depth'] == min_depth)
    if df[baseline_mask_depth].empty:
        baseline_success_depth = 1.0  # Default if no baseline data
    else:
        baseline_success_depth = df[baseline_mask_depth]['success_rate'].mean()
    
    # Calculate impact of increasing payload size (keeping gates constant)
    payload_impacts_gates = []
    for size in payload_sizes:
        if size == min_payload:
            continue
        mask = (df['payload_size'] == size) & (df['num_gates'] == min_gates_range)
        if not df[mask].empty:
            success_rate = df[mask]['success_rate'].mean()
            impact = (baseline_success_gates - success_rate) / baseline_success_gates
            payload_impacts_gates.append((size, impact))
    
    # Calculate impact of increasing gates (keeping payload constant)
    gate_impacts = []
    gate_ranges = sorted(df['num_gates'].unique())
    for gates in gate_ranges:
        if gates == min_gates_range:
            continue
        mask = (df['payload_size'] == min_payload) & (df['num_gates'] == gates)
        if not df[mask].empty:
            success_rate = df[mask]['success_rate'].mean()
            impact = (baseline_success_gates - success_rate) / baseline_success_gates
            gate_impacts.append((gates, impact))
    
    # Calculate impact of increasing payload size (keeping depth constant)
    payload_impacts_depth = []
    for size in payload_sizes:
        if size == min_payload:
            continue
        mask = (df['payload_size'] == size) & (df['circuit_depth'] == min_depth)
        if not df[mask].empty:
            success_rate = df[mask]['success_rate'].mean()
            impact = (baseline_success_depth - success_rate) / baseline_success_depth
            payload_impacts_depth.append((size, impact))
    
    # Calculate impact of increasing depth (keeping payload constant)
    depth_impacts = []
    depth_values = sorted(df['circuit_depth'].unique())
    for depth in depth_values:
        if depth == min_depth:
            continue
        mask = (df['payload_size'] == min_payload) & (df['circuit_depth'] == depth)
        if not df[mask].empty:
            success_rate = df[mask]['success_rate'].mean()
            impact = (baseline_success_depth - success_rate) / baseline_success_depth
            depth_impacts.append((depth, impact))
    
    # Calculate average impact per unit increase for gates analysis
    if payload_impacts_gates:
        avg_payload_impact_gates = np.mean([impact for _, impact in payload_impacts_gates]) / (len(payload_impacts_gates))
    else:
        avg_payload_impact_gates = 0.5  # Default if no data
        
    if gate_impacts:
        avg_gate_impact = np.mean([impact for _, impact in gate_impacts]) / (len(gate_impacts))
    else:
        avg_gate_impact = 0.25  # Default if no data
    
    # Calculate average impact per unit increase for depth analysis
    if payload_impacts_depth:
        avg_payload_impact_depth = np.mean([impact for _, impact in payload_impacts_depth]) / (len(payload_impacts_depth))
    else:
        avg_payload_impact_depth = 0.5  # Default if no data
        
    if depth_impacts:
        avg_depth_impact = np.mean([impact for _, impact in depth_impacts]) / (len(depth_impacts))
    else:
        avg_depth_impact = 0.25  # Default if no data
    
    # Use the average of both payload impacts
    avg_payload_impact = (avg_payload_impact_gates + avg_payload_impact_depth) / 2
    
    # Calculate weights for gate-based analysis
    total_impact_gates = avg_payload_impact + avg_gate_impact
    if total_impact_gates > 0:
        payload_weight_gates = 3.0 * (avg_payload_impact / total_impact_gates)
        gate_weight = 3.0 * (avg_gate_impact / total_impact_gates)
    else:
        payload_weight_gates = 2.0  # Default
        gate_weight = 1.0  # Default
    
    # Calculate weights for depth-based analysis
    total_impact_depth = avg_payload_impact + avg_depth_impact
    if total_impact_depth > 0:
        payload_weight_depth = 3.0 * (avg_payload_impact / total_impact_depth)
        depth_weight = 3.0 * (avg_depth_impact / total_impact_depth)
    else:
        payload_weight_depth = 2.0  # Default
        depth_weight = 1.0  # Default
    
    # Now test different weight combinations to find the optimal weights
    # that maximize the correlation between Cap-score and success rate
    print("\nTesting different weight combinations to find optimal weights...")
    
    # Generate test weight combinations
    weight_combinations = []
    for pw in np.linspace(1.0, 3.0, 5):  # Test payload weights from 1.0 to 3.0
        for cw in np.linspace(0.5, 2.0, 4):  # Test complexity weights from 0.5 to 2.0
            weight_combinations.append((round(pw, 1), round(cw, 1)))
    
    # Test gate-based weights
    best_correlation_gates = -1
    best_weights_gates = (payload_weight_gates, gate_weight)
    
    for pw, cw in weight_combinations:
        cap_scores = []
        success_rates = []
        
        for _, row in df.iterrows():
            cap_score = calculate_cap_score(
                row['success_rate'], 
                row['payload_size'], 
                row['num_gates'], 
                complexity_type='gates',
                payload_weight=pw,
                complexity_weight=cw
            )
            cap_scores.append(cap_score)
            success_rates.append(row['success_rate'] * 100)  # Convert to percentage
        
        correlation = np.corrcoef(cap_scores, success_rates)[0, 1]
        if correlation > best_correlation_gates:
            best_correlation_gates = correlation
            best_weights_gates = (pw, cw)
    
    # Test depth-based weights
    best_correlation_depth = -1
    best_weights_depth = (payload_weight_depth, depth_weight)
    
    for pw, cw in weight_combinations:
        cap_scores = []
        success_rates = []
        
        for _, row in df.iterrows():
            cap_score = calculate_cap_score(
                row['success_rate'], 
                row['payload_size'], 
                row['circuit_depth'], 
                complexity_type='depth',
                payload_weight=pw,
                complexity_weight=cw
            )
            cap_scores.append(cap_score)
            success_rates.append(row['success_rate'] * 100)  # Convert to percentage
        
        correlation = np.corrcoef(cap_scores, success_rates)[0, 1]
        if correlation > best_correlation_depth:
            best_correlation_depth = correlation
            best_weights_depth = (pw, cw)
    
    # Print analysis results
    print(f"\nAnalysis of error impacts:")
    print(f"  - Payload size impact (gates analysis): {avg_payload_impact_gates:.4f} per unit")
    print(f"  - Gate count impact: {avg_gate_impact:.4f} per unit")
    print(f"  - Payload size impact (depth analysis): {avg_payload_impact_depth:.4f} per unit")
    print(f"  - Circuit depth impact: {avg_depth_impact:.4f} per unit")
    
    print(f"\nInitial suggested weights (based on impact analysis):")
    print(f"  - Gates analysis: Payload weight = {payload_weight_gates:.2f}, Complexity weight = {gate_weight:.2f}")
    print(f"  - Depth analysis: Payload weight = {payload_weight_depth:.2f}, Complexity weight = {depth_weight:.2f}")
    
    print(f"\nOptimal weights (based on correlation with success rate):")
    print(f"  - Gates analysis: Payload weight = {best_weights_gates[0]:.1f}, Complexity weight = {best_weights_gates[1]:.1f}")
    print(f"  - Correlation with success rate: {best_correlation_gates:.4f}")
    print(f"  - Depth analysis: Payload weight = {best_weights_depth[0]:.1f}, Complexity weight = {best_weights_depth[1]:.1f}")
    print(f"  - Correlation with success rate: {best_correlation_depth:.4f}")
    
    # Return the optimal weights for gates analysis (primary use case)
    return best_weights_gates[0], best_weights_gates[1], best_weights_depth[0], best_weights_depth[1]

def plot_circuit_size_binned_analysis(df):
    """
    Create comprehensive analysis plots based on circuit size (total number of operations)
    using custom bins for different size ranges.
    
    This function uses predefined bins based on observed patterns in the data:
    - 200 gates: 411-437
    - 500 gates: 1011-1036
    - 1k gates: 2011-2036
    - 1.5k gates: 3011-3036
    - 2k gates: 4011-4036
    - 3k gates: 6011-6036
    - 10k gates: 20005-20036
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(18, 12))
    gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')  # Hide axes for the text box

    # Get payload sizes and prepare data
    payload_sizes = sorted(df['payload_size'].unique())
    
    # Define custom circuit size bins based on the specified patterns
    circuit_size_bins = [
        (411, 437, "200 gates"),
        (1011, 1036, "500 gates"),
        (2011, 2036, "1k gates"),
        (3011, 3036, "1.5k gates"),
        (4011, 4036, "2k gates"),
        (6011, 6036, "3k gates"),
        (20005, 20036, "10k gates")
    ]
    
    # Create a new column for circuit size category
    df['circuit_size_category'] = None
    for start, end, label in circuit_size_bins:
        mask = df['circuit_size'].between(start, end, inclusive='both')
        df.loc[mask, 'circuit_size_category'] = label
    
    # Filter out rows that don't match any of our bins
    df_filtered = df.dropna(subset=['circuit_size_category'])
    
    # If no data matches our bins, print a warning and return
    if len(df_filtered) == 0:
        print("Warning: No data matches the specified circuit size bins.")
        return None
    
    # 1. Scatter plot of success rate vs circuit size category
    # We'll use the midpoint of each bin for the x-axis
    bin_midpoints = {label: (start + end) / 2 for start, end, label in circuit_size_bins}
    bin_order = [label for _, _, label in circuit_size_bins]
    
    for i, size in enumerate(payload_sizes):
        payload_data = df_filtered[df_filtered['payload_size'] == size]
        if len(payload_data) == 0:
            continue
            
        # Group by circuit size category and calculate mean success rate
        grouped = payload_data.groupby('circuit_size_category')['success_rate'].mean() * 100
        
        # Get x-values (bin midpoints) and y-values (success rates)
        x_values = []
        y_values = []
        for category in bin_order:
            if category in grouped.index:
                x_values.append(bin_midpoints[category])
                y_values.append(grouped[category])
        
        # Plot scatter points
        if x_values:
            ax1.scatter(x_values, y_values,
                       color=COLORBREWER_PALETTE[size], 
                       marker=MARKER_STYLES[size],
                       s=100,
                       label=f'Payload Size {size}')
            
            # Add trend line if we have enough points
            if len(x_values) >= 3:
                try:
                    # Use numpy's polyfit with warnings suppressed
                    with np.errstate(invalid='ignore'):
                        z = np.polyfit(x_values, y_values, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(min(x_values), max(x_values), 100)
                        y_trend = p(x_trend)
                        ax1.plot(x_trend, y_trend, 
                                color=COLORBREWER_PALETTE[size],
                                linestyle='--', alpha=0.7,
                                label=f'Trend P{size} (slope: {z[0]:.2e})')
                except np.linalg.LinAlgError:
                    print(f"Warning: Could not fit trend line for payload size {size}")
    
    # Set x-axis to use the bin labels instead of midpoints
    ax1.set_xticks([bin_midpoints[label] for label in bin_order])
    ax1.set_xticklabels(bin_order, rotation=45)
    
    ax1.set_xlabel('Circuit Size', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate vs Circuit Size', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Stacked bar chart showing error distribution by circuit size
    # Calculate error rates for each payload size and circuit size bin
    bottom = np.zeros(len(payload_sizes))
    
    for bin_idx, (_, _, label) in enumerate(circuit_size_bins):
        error_rates = []
        
        for size in payload_sizes:
            mask = (df_filtered['payload_size'] == size) & (df_filtered['circuit_size_category'] == label)
            if not df_filtered[mask].empty:
                error_rate = 1 - df_filtered[mask]['success_rate'].mean()
                error_rates.append(error_rate * 100)  # Convert to percentage
            else:
                error_rates.append(0)
        
        # Use a consistent color scheme based on bin index
        color_idx = (bin_idx % len(COLORBREWER_PALETTE)) + 1
        
        ax2.bar(np.arange(len(payload_sizes)), error_rates, bottom=bottom, 
               label=label,
               color=COLORBREWER_PALETTE[color_idx], alpha=0.7, 
               edgecolor='black', linewidth=0.5)
        bottom += error_rates
    
    ax2.set_xlabel('Payload Size', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Error Distribution by Circuit Size', fontsize=14)
    ax2.set_xticks(np.arange(len(payload_sizes)))
    ax2.set_xticklabels(payload_sizes)
    ax2.legend(title='Circuit Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. Heatmap of success rates by payload size and circuit size
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(payload_sizes), len(circuit_size_bins)))
    
    for i, payload_size in enumerate(payload_sizes):
        for j, (_, _, label) in enumerate(circuit_size_bins):
            mask = (df_filtered['payload_size'] == payload_size) & (df_filtered['circuit_size_category'] == label)
            if not df_filtered[mask].empty:
                heatmap_data[i, j] = df_filtered[mask]['success_rate'].mean()
    
    im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Success Rate', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax3.set_xticks(np.arange(len(circuit_size_bins)))
    ax3.set_yticks(np.arange(len(payload_sizes)))
    ax3.set_xticklabels([label for _, _, label in circuit_size_bins])
    ax3.set_yticklabels(payload_sizes)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in the heatmap cells
    for i in range(len(payload_sizes)):
        for j in range(len(circuit_size_bins)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="w" if heatmap_data[i, j] < 0.5 else "black")
    
    ax3.set_title('Success Rate Heatmap by Circuit Size', fontsize=14)
    ax3.set_xlabel('Circuit Size', fontsize=12)
    ax3.set_ylabel('Payload Size', fontsize=12)

    # 4. Add a text box with summary statistics
    # Calculate summary statistics
    overall_success = df_filtered['success_rate'].mean()
    
    # Find best and worst configurations from heatmap
    best_config_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    best_payload = payload_sizes[best_config_idx[0]]
    best_size_bin = circuit_size_bins[best_config_idx[1]][2]  # Get the label
    best_success = heatmap_data[best_config_idx]
    
    worst_config_idx = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
    worst_payload = payload_sizes[worst_config_idx[0]]
    worst_size_bin = circuit_size_bins[worst_config_idx[1]][2]  # Get the label
    worst_success = heatmap_data[worst_config_idx]
    
    # Create text for the summary box
    summary_text = (
        "CIRCUIT SIZE ANALYSIS SUMMARY\n"
        "============================\n\n"
        f"Overall Success Rate: {overall_success:.2%}\n\n"
        f"Best Configuration:\n"
        f"  - Payload Size: {best_payload}\n"
        f"  - Circuit Size: {best_size_bin}\n"
        f"  - Success Rate: {best_success:.2%}\n\n"
        f"Worst Configuration:\n"
        f"  - Payload Size: {worst_payload}\n"
        f"  - Circuit Size: {worst_size_bin}\n"
        f"  - Success Rate: {worst_success:.2%}\n\n"
        f"Success Rate by Payload Size:\n"
    )
    
    # Add success rates by payload size
    for size in payload_sizes:
        rate = df_filtered[df_filtered['payload_size'] == size]['success_rate'].mean()
        summary_text += f"  - Payload {size}: {rate:.2%}\n"
    
    # Add success rates by circuit size bin
    summary_text += f"\nSuccess Rate by Circuit Size:\n"
    for _, _, label in circuit_size_bins:
        mask = df_filtered['circuit_size_category'] == label
        if not df_filtered[mask].empty:
            rate = df_filtered[mask]['success_rate'].mean()
            summary_text += f"  - {label}: {rate:.2%}\n"
    
    # Create a text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('circuit_size_binned_analysis.png', dpi=300, bbox_inches='tight')
    
    return None

def plot_circuit_width_analysis(df):
    """
    Create comprehensive analysis plots based on circuit width (total number of qubits).
    This provides insights into how the total circuit size affects performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(18, 12))
    gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')  # Hide axes for the text box

    # Get payload sizes and prepare data
    payload_sizes = sorted(df['payload_size'].unique())
    
    # Use circuit_width from the data (total number of qubits)
    circuit_widths = sorted(df['circuit_width'].unique())
    
    # 1. Scatter plot of success rate vs circuit width
    for i, size in enumerate(payload_sizes):
        payload_data = df[df['payload_size'] == size]
        ax1.scatter(payload_data['circuit_width'], payload_data['success_rate'] * 100,
                   color=COLORBREWER_PALETTE[size], 
                   marker=MARKER_STYLES[size],
                   s=100,
                   label=f'Payload Size {size}')
        
        # Add trend line if we have enough data points (at least 3 unique x values)
        unique_x_values = payload_data['circuit_width'].unique()
        if len(unique_x_values) >= 3:
            try:
                # Use numpy's polyfit with warnings suppressed
                with np.errstate(invalid='ignore'):
                    z = np.polyfit(payload_data['circuit_width'], payload_data['success_rate'] * 100, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(payload_data['circuit_width'].min(), payload_data['circuit_width'].max(), 100)
                    y_trend = p(x_trend)
                    ax1.plot(x_trend, y_trend, 
                            color=COLORBREWER_PALETTE[size],
                            linestyle='--', alpha=0.7,
                            label=f'Trend P{size} (slope: {z[0]:.2e})')
            except np.linalg.LinAlgError:
                # If polyfit fails, just skip the trend line for this payload size
                print(f"Warning: Could not fit trend line for payload size {size}")
    
    ax1.set_xlabel('Circuit Width (Total Qubits)', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate vs Circuit Width', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Stacked bar chart showing error distribution by circuit width
    # Define circuit width ranges for grouping
    width_ranges = []
    min_width = min(circuit_widths)
    max_width = max(circuit_widths)
    step = max(1, (max_width - min_width) // 5)  # Create about 5 ranges
    
    for i in range(min_width, max_width + 1, step):
        width_ranges.append((i, min(i + step - 1, max_width)))
    
    # Create a color map for width ranges
    width_colors = {}
    for i, width_range in enumerate(width_ranges):
        color_idx = (i % len(COLORBREWER_PALETTE)) + 1
        width_colors[width_range] = COLORBREWER_PALETTE[color_idx]
    
    # Calculate error rates for each payload size and circuit width range
    bottom = np.zeros(len(payload_sizes))
    for i, width_range in enumerate(width_ranges):
        error_rates = []
        
        for size in payload_sizes:
            mask = (df['payload_size'] == size) & (df['circuit_width'].between(width_range[0], width_range[1], inclusive='both'))
            if not df[mask].empty:
                error_rate = 1 - df[mask]['success_rate'].mean()
                error_rates.append(error_rate * 100)  # Convert to percentage
            else:
                error_rates.append(0)
        
        ax2.bar(np.arange(len(payload_sizes)), error_rates, bottom=bottom, 
               label=f'{width_range[0]}-{width_range[1]}',
               color=width_colors[width_range], alpha=0.7, 
               edgecolor='black', linewidth=0.5)
        bottom += error_rates
    
    ax2.set_xlabel('Payload Size', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Error Distribution by Circuit Width', fontsize=14)
    ax2.set_xticks(np.arange(len(payload_sizes)))
    ax2.set_xticklabels(payload_sizes)
    ax2.legend(title='Circuit Width Range', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. Heatmap of success rates by payload size and circuit width
    heatmap_data = np.zeros((len(payload_sizes), len(width_ranges)))
    
    for i, payload_size in enumerate(payload_sizes):
        for j, width_range in enumerate(width_ranges):
            mask = (df['payload_size'] == payload_size) & (df['circuit_width'].between(width_range[0], width_range[1], inclusive='both'))
            if not df[mask].empty:
                heatmap_data[i, j] = df[mask]['success_rate'].mean()
    
    im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Success Rate', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax3.set_xticks(np.arange(len(width_ranges)))
    ax3.set_yticks(np.arange(len(payload_sizes)))
    ax3.set_xticklabels([f'{r[0]}-{r[1]}' for r in width_ranges])
    ax3.set_yticklabels(payload_sizes)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in the heatmap cells
    for i in range(len(payload_sizes)):
        for j in range(len(width_ranges)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="w" if heatmap_data[i, j] < 0.5 else "black")
    
    ax3.set_title('Success Rate Heatmap by Circuit Width', fontsize=14)
    ax3.set_xlabel('Circuit Width Range (Total Qubits)', fontsize=12)
    ax3.set_ylabel('Payload Size', fontsize=12)

    # 4. Add a text box with summary statistics
    # Calculate summary statistics
    overall_success = df['success_rate'].mean()
    
    # Find best and worst configurations from heatmap
    best_config_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    best_payload = payload_sizes[best_config_idx[0]]
    best_width_range = width_ranges[best_config_idx[1]]
    best_success = heatmap_data[best_config_idx]
    
    worst_config_idx = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
    worst_payload = payload_sizes[worst_config_idx[0]]
    worst_width_range = width_ranges[worst_config_idx[1]]
    worst_success = heatmap_data[worst_config_idx]
    
    # Create text for the summary box
    summary_text = (
        "CIRCUIT WIDTH ANALYSIS SUMMARY\n"
        "============================\n\n"
        f"Overall Success Rate: {overall_success:.2%}\n\n"
        f"Best Configuration:\n"
        f"  - Payload Size: {best_payload}\n"
        f"  - Circuit Width Range: {best_width_range[0]}-{best_width_range[1]}\n"
        f"  - Success Rate: {best_success:.2%}\n\n"
        f"Worst Configuration:\n"
        f"  - Payload Size: {worst_payload}\n"
        f"  - Circuit Width Range: {worst_width_range[0]}-{worst_width_range[1]}\n"
        f"  - Success Rate: {worst_success:.2%}\n\n"
        f"Success Rate by Payload Size:\n"
    )
    
    # Add success rates by payload size
    for size in payload_sizes:
        rate = df[df['payload_size'] == size]['success_rate'].mean()
        summary_text += f"  - Payload {size}: {rate:.2%}\n"
    
    # Add success rates by circuit width range
    summary_text += f"\nSuccess Rate by Circuit Width Range:\n"
    for width_range in width_ranges:
        mask = df['circuit_width'].between(width_range[0], width_range[1], inclusive='both')
        if not df[mask].empty:
            rate = df[mask]['success_rate'].mean()
            summary_text += f"  - {width_range[0]}-{width_range[1]} qubits: {rate:.2%}\n"
    
    # Create a text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('circuit_width_analysis.png', dpi=300, bbox_inches='tight')
    
    return None

def analyze_job_execution_times(df):
    """
    Analyze job execution times by calculating the time differences between
    job_created, job_running, and job_finished timestamps, and visualize
    these times using histograms with logarithmic scales.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing experiment results with job timestamp columns
    """
    # Convert timestamp strings to datetime objects
    for col in ['job_created', 'job_running', 'job_finished']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Calculate time differences in seconds
    df['queue_time'] = (df['job_running'] - df['job_created']).dt.total_seconds()
    df['total_time'] = (df['job_finished'] - df['job_created']).dt.total_seconds()
    df['execution_time'] = (df['job_finished'] - df['job_running']).dt.total_seconds()
    
    # Function to calculate mode (most common value)
    def calculate_mode(data):
        # Round to 2 decimal places to group similar values
        rounded_data = np.round(data, 2)
        values, counts = np.unique(rounded_data, return_counts=True)
        mode_index = np.argmax(counts)
        return values[mode_index], counts[mode_index]
    
    # Calculate modes
    queue_mode, queue_mode_count = calculate_mode(df['queue_time'])
    total_mode, total_mode_count = calculate_mode(df['total_time'])
    exec_mode, exec_mode_count = calculate_mode(df['execution_time'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Function to create logarithmic bins
    def create_log_bins(data, num_bins=20):
        min_val = max(data.min(), 0.1)  # Avoid zero or negative values for log scale
        max_val = data.max()
        return np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
    
    # 1. Queue Time (created to running)
    log_bins_queue = create_log_bins(df['queue_time'])
    axes[0].hist(df['queue_time'], bins=log_bins_queue, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Queue Time (Created to Running)', fontsize=14)
    axes[0].set_xlabel('Time (seconds) - Log Scale', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_xscale('log')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics as text
    queue_stats = (
        f"Mean: {df['queue_time'].mean():.2f}s\n"
        f"Median: {df['queue_time'].median():.2f}s\n"
        f"Mode: {queue_mode:.2f}s (count: {queue_mode_count})\n"
        f"Min: {df['queue_time'].min():.2f}s\n"
        f"Max: {df['queue_time'].max():.2f}s\n"
        f"Std Dev: {df['queue_time'].std():.2f}s"
    )
    axes[0].text(0.95, 0.95, queue_stats, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Total Time (created to finished)
    log_bins_total = create_log_bins(df['total_time'])
    axes[1].hist(df['total_time'], bins=log_bins_total, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].set_title('Total Time (Created to Finished)', fontsize=14)
    axes[1].set_xlabel('Time (seconds) - Log Scale', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_xscale('log')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics as text
    total_stats = (
        f"Mean: {df['total_time'].mean():.2f}s\n"
        f"Median: {df['total_time'].median():.2f}s\n"
        f"Mode: {total_mode:.2f}s (count: {total_mode_count})\n"
        f"Min: {df['total_time'].min():.2f}s\n"
        f"Max: {df['total_time'].max():.2f}s\n"
        f"Std Dev: {df['total_time'].std():.2f}s"
    )
    axes[1].text(0.95, 0.95, total_stats, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Execution Time (running to finished)
    log_bins_exec = create_log_bins(df['execution_time'])
    axes[2].hist(df['execution_time'], bins=log_bins_exec, color='salmon', edgecolor='black', alpha=0.7)
    axes[2].set_title('Execution Time (Running to Finished)', fontsize=14)
    axes[2].set_xlabel('Time (seconds) - Log Scale', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_xscale('log')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics as text
    exec_stats = (
        f"Mean: {df['execution_time'].mean():.2f}s\n"
        f"Median: {df['execution_time'].median():.2f}s\n"
        f"Mode: {exec_mode:.2f}s (count: {exec_mode_count})\n"
        f"Min: {df['execution_time'].min():.2f}s\n"
        f"Max: {df['execution_time'].max():.2f}s\n"
        f"Std Dev: {df['execution_time'].std():.2f}s"
    )
    axes[2].text(0.95, 0.95, exec_stats, transform=axes[2].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('job_execution_times_log.png', dpi=300, bbox_inches='tight')
    
    # Also create a version with linear scale for comparison
    fig_linear, axes_linear = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. Queue Time (linear scale)
    axes_linear[0].hist(df['queue_time'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes_linear[0].set_title('Queue Time (Created to Running)', fontsize=14)
    axes_linear[0].set_xlabel('Time (seconds) - Linear Scale', fontsize=12)
    axes_linear[0].set_ylabel('Frequency', fontsize=12)
    axes_linear[0].grid(True, linestyle='--', alpha=0.7)
    axes_linear[0].text(0.95, 0.95, queue_stats, transform=axes_linear[0].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Total Time (linear scale)
    axes_linear[1].hist(df['total_time'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes_linear[1].set_title('Total Time (Created to Finished)', fontsize=14)
    axes_linear[1].set_xlabel('Time (seconds) - Linear Scale', fontsize=12)
    axes_linear[1].set_ylabel('Frequency', fontsize=12)
    axes_linear[1].grid(True, linestyle='--', alpha=0.7)
    axes_linear[1].text(0.95, 0.95, total_stats, transform=axes_linear[1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Execution Time (linear scale)
    axes_linear[2].hist(df['execution_time'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
    axes_linear[2].set_title('Execution Time (Running to Finished)', fontsize=14)
    axes_linear[2].set_xlabel('Time (seconds) - Linear Scale', fontsize=12)
    axes_linear[2].set_ylabel('Frequency', fontsize=12)
    axes_linear[2].grid(True, linestyle='--', alpha=0.7)
    axes_linear[2].text(0.95, 0.95, exec_stats, transform=axes_linear[2].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('job_execution_times_linear.png', dpi=300, bbox_inches='tight')
    
    # Create a new figure for execution times by payload size
    if 'payload_size' in df.columns:
        # Get unique payload sizes
        payload_sizes = sorted(df['payload_size'].unique())
        
        # Create figure with subplots
        fig_payload, axes_payload = plt.subplots(3, 1, figsize=(14, 15))
        
        # 1. Queue Time by Payload Size
        for i, size in enumerate(payload_sizes):
            payload_data = df[df['payload_size'] == size]
            axes_payload[0].boxplot(payload_data['queue_time'], positions=[i], 
                                   widths=0.6, patch_artist=True,
                                   boxprops=dict(facecolor=COLORBREWER_PALETTE.get(size, f'C{i}')),
                                   medianprops=dict(color='black'))
        
        axes_payload[0].set_title('Queue Time by Payload Size', fontsize=14)
        axes_payload[0].set_xlabel('Payload Size', fontsize=12)
        axes_payload[0].set_ylabel('Time (seconds)', fontsize=12)
        axes_payload[0].set_xticks(range(len(payload_sizes)))
        axes_payload[0].set_xticklabels(payload_sizes)
        axes_payload[0].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add mean values as text
        for i, size in enumerate(payload_sizes):
            payload_data = df[df['payload_size'] == size]
            mean_time = payload_data['queue_time'].mean()
            axes_payload[0].text(i, mean_time, f'{mean_time:.2f}s', 
                               ha='center', va='bottom', fontsize=9)
        
        # 2. Total Time by Payload Size
        for i, size in enumerate(payload_sizes):
            payload_data = df[df['payload_size'] == size]
            axes_payload[1].boxplot(payload_data['total_time'], positions=[i], 
                                   widths=0.6, patch_artist=True,
                                   boxprops=dict(facecolor=COLORBREWER_PALETTE.get(size, f'C{i}')),
                                   medianprops=dict(color='black'))
        
        axes_payload[1].set_title('Total Time by Payload Size', fontsize=14)
        axes_payload[1].set_xlabel('Payload Size', fontsize=12)
        axes_payload[1].set_ylabel('Time (seconds)', fontsize=12)
        axes_payload[1].set_xticks(range(len(payload_sizes)))
        axes_payload[1].set_xticklabels(payload_sizes)
        axes_payload[1].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add mean values as text
        for i, size in enumerate(payload_sizes):
            payload_data = df[df['payload_size'] == size]
            mean_time = payload_data['total_time'].mean()
            axes_payload[1].text(i, mean_time, f'{mean_time:.2f}s', 
                               ha='center', va='bottom', fontsize=9)
        
        # 3. Execution Time by Payload Size
        for i, size in enumerate(payload_sizes):
            payload_data = df[df['payload_size'] == size]
            axes_payload[2].boxplot(payload_data['execution_time'], positions=[i], 
                                   widths=0.6, patch_artist=True,
                                   boxprops=dict(facecolor=COLORBREWER_PALETTE.get(size, f'C{i}')),
                                   medianprops=dict(color='black'))
        
        axes_payload[2].set_title('Execution Time by Payload Size', fontsize=14)
        axes_payload[2].set_xlabel('Payload Size', fontsize=12)
        axes_payload[2].set_ylabel('Time (seconds)', fontsize=12)
        axes_payload[2].set_xticks(range(len(payload_sizes)))
        axes_payload[2].set_xticklabels(payload_sizes)
        axes_payload[2].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add mean values as text
        for i, size in enumerate(payload_sizes):
            payload_data = df[df['payload_size'] == size]
            mean_time = payload_data['execution_time'].mean()
            axes_payload[2].text(i, mean_time, f'{mean_time:.2f}s', 
                               ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('job_execution_times_by_payload.png', dpi=300, bbox_inches='tight')
        
        # Create a summary table by payload size
        payload_summary = []
        for size in payload_sizes:
            payload_data = df[df['payload_size'] == size]
            queue_mode_ps, queue_mode_count_ps = calculate_mode(payload_data['queue_time'])
            total_mode_ps, total_mode_count_ps = calculate_mode(payload_data['total_time'])
            exec_mode_ps, exec_mode_count_ps = calculate_mode(payload_data['execution_time'])
            
            payload_summary.append({
                'Payload Size': size,
                'Queue Time Mean (s)': payload_data['queue_time'].mean(),
                'Queue Time Median (s)': payload_data['queue_time'].median(),
                'Queue Time Mode (s)': queue_mode_ps,
                'Total Time Mean (s)': payload_data['total_time'].mean(),
                'Total Time Median (s)': payload_data['total_time'].median(),
                'Total Time Mode (s)': total_mode_ps,
                'Execution Time Mean (s)': payload_data['execution_time'].mean(),
                'Execution Time Median (s)': payload_data['execution_time'].median(),
                'Execution Time Mode (s)': exec_mode_ps
            })
        
        payload_summary_df = pd.DataFrame(payload_summary)
        
        print("\nJob Execution Time Summary by Payload Size:")
        print(payload_summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Create a summary table
    summary_df = pd.DataFrame({
        'Metric': ['Queue Time', 'Total Time', 'Execution Time'],
        'Mean (s)': [df['queue_time'].mean(), df['total_time'].mean(), df['execution_time'].mean()],
        'Median (s)': [df['queue_time'].median(), df['total_time'].median(), df['execution_time'].median()],
        'Mode (s)': [queue_mode, total_mode, exec_mode],
        'Mode Count': [queue_mode_count, total_mode_count, exec_mode_count],
        'Min (s)': [df['queue_time'].min(), df['total_time'].min(), df['execution_time'].min()],
        'Max (s)': [df['queue_time'].max(), df['total_time'].max(), df['execution_time'].max()],
        'Std Dev (s)': [df['queue_time'].std(), df['total_time'].std(), df['execution_time'].std()]
    })
    
    print("\nJob Execution Time Summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    return df[['queue_time', 'total_time', 'execution_time']]

def analyze_regression_models(df):
    """
    Implement and evaluate multiple regression models to predict success rates
    based on circuit parameters (circuit_depth, circuit_size, circuit_width, payload_size).
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results.
        
    Returns:
        dict: Dictionary containing regression results and model comparisons.
    """
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import jarque_bera
    import matplotlib.pyplot as plt
    from statsmodels.graphics.gofplots import ProbPlot
    
    # Ensure we have a clean dataframe with the required columns
    required_cols = ['success_rate', 'circuit_depth', 'circuit_size', 'circuit_width', 'payload_size']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Create a working copy of the dataframe with only the columns we need
    model_df = df[required_cols].copy()
    
    # Check for missing values
    if model_df.isnull().sum().sum() > 0:
        print("Warning: DataFrame contains missing values. Dropping rows with missing values.")
        model_df = model_df.dropna()
    
    # Store results for comparison
    model_results = {}
    
    # Function to evaluate model and check assumptions
    def evaluate_model(model, X, y, model_name):
        # Get predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Calculate metrics
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        aic = model.aic
        bic = model.bic
        
        # Test for heteroscedasticity - with error handling
        try:
            bp_test = het_breuschpagan(residuals, X)
            bp_pvalue = bp_test[1]
        except Exception as e:
            print(f"Warning: Breusch-Pagan test failed: {str(e)}")
            bp_test = (np.nan, np.nan, np.nan, np.nan)
            bp_pvalue = np.nan
        
        try:
            white_test = het_white(residuals, X)
            white_pvalue = white_test[1]
        except Exception as e:
            print(f"Warning: White test failed: {str(e)}")
            white_test = (np.nan, np.nan, np.nan, np.nan)
            white_pvalue = np.nan
        
        # Test for normality of residuals
        try:
            jb_test = jarque_bera(residuals)
            jb_pvalue = jb_test[1]
        except Exception as e:
            print(f"Warning: Jarque-Bera test failed: {str(e)}")
            jb_test = (np.nan, np.nan)
            jb_pvalue = np.nan
        
        # Store results
        result = {
            'model': model,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'aic': aic,
            'bic': bic,
            'bp_test': bp_test,
            'white_test': white_test,
            'jb_test': jb_test,
            'residuals': residuals,
            'y_pred': y_pred
        }
        
        model_results[model_name] = result
        
        # Print summary
        print(f"\n=== {model_name} ===")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"AIC: {aic:.4f}")
        print(f"BIC: {bic:.4f}")
        print(f"Breusch-Pagan test (p-value): {bp_pvalue:.4f}" if not np.isnan(bp_pvalue) else "Breusch-Pagan test: Failed")
        print(f"White test (p-value): {white_pvalue:.4f}" if not np.isnan(white_pvalue) else "White test: Failed")
        print(f"Jarque-Bera test (p-value): {jb_pvalue:.4f}" if not np.isnan(jb_pvalue) else "Jarque-Bera test: Failed")
        
        # Interpretation of heteroscedasticity tests
        alpha = 0.05
        if not np.isnan(bp_pvalue) and not np.isnan(white_pvalue):
            if bp_pvalue < alpha or white_pvalue < alpha:
                print("Evidence of heteroscedasticity detected (p < 0.05)")
            else:
                print("No significant heteroscedasticity detected (p >= 0.05)")
        else:
            print("Heteroscedasticity tests inconclusive due to test failures")
        
        # Interpretation of normality test
        if not np.isnan(jb_pvalue):
            if jb_pvalue < alpha:
                print("Residuals are not normally distributed (p < 0.05)")
            else:
                print("Residuals appear normally distributed (p >= 0.05)")
        else:
            print("Normality test inconclusive due to test failure")
        
        return result
    
    # Function to plot residual diagnostics
    def plot_residual_diagnostics(result, model_name):
        model = result['model']
        residuals = result['residuals']
        y_pred = result['y_pred']
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Residual Diagnostics for {model_name}', fontsize=16)
            
            # Residuals vs Fitted plot
            axes[0, 0].scatter(y_pred, residuals)
            axes[0, 0].axhline(y=0, color='r', linestyle='-')
            axes[0, 0].set_xlabel('Fitted values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted')
            
            # QQ plot
            QQ = ProbPlot(residuals)
            QQ.qqplot(line='45', ax=axes[0, 1])
            axes[0, 1].set_title('Normal Q-Q')
            
            # Scale-Location plot
            axes[1, 0].scatter(y_pred, np.sqrt(np.abs(residuals)))
            axes[1, 0].set_xlabel('Fitted values')
            axes[1, 0].set_ylabel('√|Residuals|')
            axes[1, 0].set_title('Scale-Location')
            
            # Residuals vs Leverage plot
            try:
                influence = model.get_influence()
                leverage = influence.hat_matrix_diag
                axes[1, 1].scatter(leverage, residuals)
                axes[1, 1].axhline(y=0, color='r', linestyle='-')
                axes[1, 1].set_xlabel('Leverage')
                axes[1, 1].set_ylabel('Residuals')
                axes[1, 1].set_title('Residuals vs Leverage')
            except Exception as e:
                print(f"Warning: Could not calculate leverage for {model_name}: {str(e)}")
                axes[1, 1].text(0.5, 0.5, "Leverage calculation failed", 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f'{model_name.replace(":", "_")}_residual_diagnostics.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create diagnostic plots for {model_name}: {str(e)}")
    
    # Prepare target variable and features
    y = model_df['success_rate']
    
    # Model 1: Linear model with all features
    X1 = model_df[['circuit_depth', 'circuit_size', 'circuit_width', 'payload_size']]
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(y, X1).fit()
    print("\nModel 1: All features (linear)")
    print(model1.summary())
    evaluate_model(model1, X1, y, "Model 1: All features (linear)")
    
    # Model 2: Log-transformed target variable
    # This can help with heteroscedasticity
    # Adding a small constant to avoid log(0)
    y_log = np.log(y + 0.001)
    model2 = sm.OLS(y_log, X1).fit()
    print("\nModel 2: Log-transformed target")
    print(model2.summary())
    evaluate_model(model2, X1, y_log, "Model 2: Log-transformed target")
    
    # Model 3: Interaction terms
    # Create interaction terms between payload_size and other features
    X3 = X1.copy()
    X3['depth_x_payload'] = model_df['circuit_depth'] * model_df['payload_size']
    X3['size_x_payload'] = model_df['circuit_size'] * model_df['payload_size']
    X3['width_x_payload'] = model_df['circuit_width'] * model_df['payload_size']
    model3 = sm.OLS(y, X3).fit()
    print("\nModel 3: With interaction terms")
    print(model3.summary())
    evaluate_model(model3, X3, y, "Model 3: With interaction terms")
    
    # Model 4: Polynomial terms (quadratic)
    # Check for multicollinearity before creating the model
    X4 = X1.copy()
    X4['circuit_depth_sq'] = model_df['circuit_depth'] ** 2
    X4['circuit_size_sq'] = model_df['circuit_size'] ** 2
    X4['circuit_width_sq'] = model_df['circuit_width'] ** 2
    X4['payload_size_sq'] = model_df['payload_size'] ** 2
    
    # Check condition number to detect multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print("\nChecking for multicollinearity in Model 4:")
    try:
        for i, col in enumerate(X4.columns):
            if col != 'const':
                vif = variance_inflation_factor(X4.values, i)
                print(f"VIF for {col}: {vif:.2f}")
                if vif > 10:
                    print(f"  Warning: High multicollinearity detected for {col}")
    except Exception as e:
        print(f"Warning: Could not calculate VIF: {str(e)}")
    
    model4 = sm.OLS(y, X4).fit()
    print("\nModel 4: With quadratic terms")
    print(model4.summary())
    evaluate_model(model4, X4, y, "Model 4: With quadratic terms")
    
    # Model 5: Feature selection based on p-values
    # Start with all features and remove insignificant ones
    X5 = X1.copy()
    
    # Iteratively remove features with p-value > 0.05
    while True:
        model5 = sm.OLS(y, X5).fit()
        p_values = model5.pvalues
        max_p_value = p_values.max()
        
        # If all p-values are significant, break
        if max_p_value <= 0.05:
            break
            
        # Remove the feature with the highest p-value
        feature_to_remove = p_values.idxmax()
        
        # Don't remove the constant
        if feature_to_remove == 'const':
            break
            
        print(f"Removing feature {feature_to_remove} with p-value {max_p_value:.4f}")
        X5 = X5.drop(feature_to_remove, axis=1)
        
        # If only constant is left, break
        if X5.shape[1] <= 1:
            break
    
    print("\nModel 5: Feature selection based on p-values")
    print(model5.summary())
    evaluate_model(model5, X5, y, "Model 5: Feature selection")
    
    # Plot residual diagnostics for all models
    for name, result in model_results.items():
        plot_residual_diagnostics(result, name)
    
    # Compare models and select the best one
    print("\n=== Model Comparison ===")
    comparison_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'R-squared': [result['r_squared'] for result in model_results.values()],
        'Adj R-squared': [result['adj_r_squared'] for result in model_results.values()],
        'AIC': [result['aic'] for result in model_results.values()],
        'BIC': [result['bic'] for result in model_results.values()],
        'BP Test p-value': [result['bp_test'][1] if not np.isnan(result['bp_test'][1]) else np.nan 
                           for result in model_results.values()],
        'White Test p-value': [result['white_test'][1] if not np.isnan(result['white_test'][1]) else np.nan 
                              for result in model_results.values()],
        'JB Test p-value': [result['jb_test'][1] if not np.isnan(result['jb_test'][1]) else np.nan 
                           for result in model_results.values()]
    })
    
    print(comparison_df)
    
    # Determine best model based on adjusted R-squared and diagnostic tests
    best_models = comparison_df.sort_values('Adj R-squared', ascending=False)
    print("\nModels ranked by Adjusted R-squared:")
    print(best_models[['Model', 'Adj R-squared']])
    
    # Check for models with no heteroscedasticity
    homoscedastic_models = comparison_df[
        (comparison_df['BP Test p-value'] > 0.05) & 
        (comparison_df['White Test p-value'] > 0.05)
    ]
    
    if not homoscedastic_models.empty:
        print("\nModels with homoscedasticity (no heteroscedasticity detected):")
        print(homoscedastic_models[['Model', 'Adj R-squared', 'BP Test p-value', 'White Test p-value']])
        best_model = homoscedastic_models.sort_values('Adj R-squared', ascending=False).iloc[0]
        print(f"\nRecommended model: {best_model['Model']}")
    else:
        print("\nAll models show signs of heteroscedasticity or test failures.")
        print("Consider using robust standard errors or weighted least squares.")
        best_model = best_models.iloc[0]
        print(f"\nRecommended model (despite heteroscedasticity): {best_model['Model']}")
        
        # Get the best model and apply robust standard errors
        best_model_name = best_model['Model']
        best_model_obj = model_results[best_model_name]['model']
        
        # Apply robust standard errors (HC3 is a good default choice)
        try:
            robust_model = best_model_obj.get_robustcov_results(cov_type='HC3')
            print("\nBest model with robust standard errors:")
            print(robust_model.summary())
        except Exception as e:
            print(f"Warning: Could not apply robust standard errors: {str(e)}")
    
    # Model 6: Weighted Least Squares (if heteroscedasticity is detected)
    # Use the absolute residuals from the best model to estimate weights
    try:
        best_model_name = best_models.iloc[0]['Model']
        best_model_residuals = model_results[best_model_name]['residuals']
        
        # Fit a model to predict the absolute residuals
        abs_resid = np.abs(best_model_residuals)
        mod_resid = sm.OLS(abs_resid, X1).fit()
        
        # Use fitted values as weights
        weights = 1 / (mod_resid.fittedvalues ** 2)
        
        # Fit WLS model
        wls_model = sm.WLS(y, X1, weights=weights).fit()
        print("\nModel 6: Weighted Least Squares")
        print(wls_model.summary())
        
        # Add to model results
        X6 = X1.copy()
        evaluate_model(wls_model, X6, y, "Model 6: Weighted Least Squares")
    except Exception as e:
        print(f"Warning: Could not fit Weighted Least Squares model: {str(e)}")
    
    # Model 7: Robust Regression (if outliers are suspected)
    try:
        rlm_model = sm.RLM(y, X1, M=sm.robust.norms.HuberT()).fit()
        print("\nModel 7: Robust Regression (Huber's T)")
        print(rlm_model.summary())
        
        # Add to model results
        X7 = X1.copy()
        evaluate_model(rlm_model, X7, y, "Model 7: Robust Regression")
    except Exception as e:
        print(f"Warning: Could not fit Robust Regression model: {str(e)}")
    
    return model_results

if __name__ == "__main__":
    # Example of using multiple CSV files
    # You can specify a list of CSV files
    csv_files = [
        "experiment_results_dynamic_1-5_200-205_20250304_113951.csv",
        "experiment_results_dynamic_1-4_500-505_20250304_114048.csv",
        "experiment_results_dynamic_1-4_1000-1005_20250304_114148.csv",
        "experiment_results_dynamic_1-4_1500-1505_20250304_114253.csv",
        "experiment_results_dynamic_1-4_2000-2005_20250304_114351.csv",
        "experiment_results_dynamic_1-4_3000-3005_20250304_001617_updated.csv"
    ]
    
    # Create the DataFrame from multiple CSV files
    df = create_experiment_dataframe(csv_files)
    
    # Filter to include only payload sizes up to 4 qubits
    df = df[df['payload_size'] <= 4]
    
    # Display basic information about the dataset
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nPayload size distribution:")
    print(df.groupby('payload_size').size())
    
    # Show counts distribution for each payload size
    print("\nCounts distribution by payload size:")
    for size in sorted(df['payload_size'].unique()):
        subset = df[df['payload_size'] == size]
        print(f"\nPayload size {size}:")
        print(f"Average counts_zeros: {subset['counts_zeros'].mean():.2f}")
        print(f"Average counts_ones: {subset['counts_ones'].mean():.2f}")
    
    # Create all three versions of the plot
    print("\nRegression Statistics (Linear Scale):")
    print("-"*50)
    regression_stats = plot_success_rates(df, use_log_x=False, use_log_y=False)
    for stats in regression_stats:
        print(f"\nPayload Size {stats['payload_size']}:")
        print(f"Slope: {stats['slope']:.2e} ± {stats['std_err']:.2e}")
        print(f"Intercept: {stats['intercept']:.2f}")
        print(f"R²: {stats['r_squared']:.3f}")
        print(f"p-value: {stats['p_value']:.2e}")
        print("-"*50)
    
    
    print("\nRegression Statistics (Log X Scale):")
    print("-"*50)
    regression_stats_logx = plot_success_rates(df, use_log_x=True, use_log_y=False)
    for stats in regression_stats_logx:
        print(f"\nPayload Size {stats['payload_size']}:")
        print(f"Slope: {stats['slope']:.2e} ± {stats['std_err']:.2e}")
        print(f"Intercept: {stats['intercept']:.2f}")
        print(f"R²: {stats['r_squared']:.3f}")
        print(f"p-value: {stats['p_value']:.2e}")
        print("-"*50)
    
    print("\nRegression Statistics (Log-Log Scale):")
    print("-"*50)
    regression_stats_logxy = plot_success_rates(df, use_log_x=True, use_log_y=True)
    for stats in regression_stats_logxy:
        print(f"\nPayload Size {stats['payload_size']}:")
        print(f"Slope: {stats['slope']:.2e} ± {stats['std_err']:.2e}")
        print(f"Intercept: {stats['intercept']:.2f}")
        print(f"R²: {stats['r_squared']:.3f}")
        print(f"p-value: {stats['p_value']:.2e}")
        print("-"*50)
    
    # Add circuit complexity analysis
    print("\nGenerating Circuit Complexity Analysis Plots...")
    plot_circuit_complexity(df)
    print("Circuit complexity plots saved as:")
    print("- 'circuit_complexity_2d.png'")
    print("- 'circuit_complexity_3d_view1.png' (Default view)")
    print("- 'circuit_complexity_3d_view2.png' (Rotated 90 degrees)")
    print("- 'circuit_complexity_3d_view3.png' (Top-down view)")
    print("- 'circuit_complexity_3d_view4.png' (Higher elevation)")
    
    # Add error analysis
    print("\nGenerating Error Analysis Plots...")
    plot_error_analysis(df)
    print("Error analysis plots saved as 'error_analysis.png'")
    
    # Analyze error impact weights for Cap-score calculation
    print("\nAnalyzing Error Impact Weights for Cap-score...")
    payload_weight_gates, complexity_weight_gates, payload_weight_depth, complexity_weight_depth = analyze_error_impact_weights(df)
    
    # Add Cap-score analysis
    print("\nGenerating Cap-score Analysis Plot...")
    plot_cap_scores(df, payload_weight=payload_weight_gates, complexity_weight=complexity_weight_gates)
    print("Cap-score analysis plot saved as 'cap_scores_analysis.png'")
    
    # Add depth analysis
    print("\nGenerating Circuit Depth Analysis Plots...")
    plot_depth_analysis(df, payload_weight=payload_weight_depth, complexity_weight=complexity_weight_depth)
    print("Circuit depth analysis plots saved as:")
    print("- 'circuit_depth_analysis.png'")
    print("- 'depth_cap_scores_analysis.png'")
    
    # Add circuit width analysis
    print("\nGenerating Circuit Width Analysis Plots...")
    plot_circuit_width_analysis(df)
    print("Circuit width analysis plots saved as 'circuit_width_analysis.png'")
    
    # Add circuit size binned analysis
    print("\nGenerating Circuit Size Binned Analysis Plots...")
    plot_circuit_size_binned_analysis(df)
    print("Circuit size binned analysis plots saved as 'circuit_size_binned_analysis.png'")
    
    # Add job execution time analysis
    print("\nGenerating Job Execution Time Analysis...")
    analyze_job_execution_times(df)
    print("Job execution time analysis plots saved as 'job_execution_times_log.png' and 'job_execution_times_linear.png'")
    if 'payload_size' in df.columns:
        print("Job execution times by payload size saved as 'job_execution_times_by_payload.png'")
    