import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from matplotlib import patches
import os
import ast

# ColorBrewer colorblind-friendly palette
COLORBREWER_PALETTE = {
    1: '#d7191c',    # Red
    2: '#fdae61',    # Orange
    3: '#ffffbf',    # Yellow
    4: '#abdda4',    # Light Green
    5: '#2b83ba'     # Blue
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
    """Calculate regression statistics using statsmodels."""
    X = sm.add_constant(x)  # Add constant term
    model = sm.OLS(y, X).fit()
    return model

def plot_success_rates(df, use_log_x=False, use_log_y=False):
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Store regression results for printing
    regression_stats = []
    
    # Plot for each payload size
    for payload_size in sorted(df['payload_size'].unique()):
        payload_data = df[df['payload_size'] == payload_size]
        
        # Group by num_gates and calculate mean success rate
        grouped_data = payload_data.groupby('num_gates')['success_rate'].mean()
        
        x = np.array(grouped_data.index)
        y = np.array(grouped_data.values * 100)
        
        if use_log_x:
            x = np.log10(x)
        if use_log_y:
            y = np.log10(y)
        
        # Scatter plot with ColorBrewer colors and distinct markers
        ax.scatter(x, y, 
                  color=COLORBREWER_PALETTE[payload_size],
                  marker=MARKER_STYLES[payload_size],
                  s=100,
                  label=f'Payload Size {payload_size}')
        
        # Regression analysis
        model = calculate_regression_stats(x, y)
        
        # Generate points for regression line
        x_pred = np.linspace(x.min(), x.max(), 100)
        X_pred = sm.add_constant(x_pred)
        y_pred = model.predict(X_pred)
        
        # Plot regression line with matching color
        ax.plot(x_pred, y_pred,
                color=COLORBREWER_PALETTE[payload_size],
                linestyle='--', alpha=0.5,
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
        plt.xlabel('Number of Gates (Range) - Log Scale', fontsize=12)
    else:
        tick_positions = [r[0] for r in gate_ranges]
        tick_labels = [f"{r[0]}-{r[1]}" for r in gate_ranges]
        plt.xlabel('Number of Gates (Range)', fontsize=12)
    
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # Set y-axis label
    if use_log_y:
        plt.ylabel('Success Rate (%) - Log Scale', fontsize=12)
    else:
        plt.ylabel('Success Rate (%)', fontsize=12)
    
    # Set title with appropriate scale indication
    scale_text = ""
    if use_log_x and use_log_y:
        scale_text = " (Log-Log Scale)"
    elif use_log_x:
        scale_text = " (Log X Scale)"
    elif use_log_y:
        scale_text = " (Log Y Scale)"
    
    plt.title('Success Rates by Number of Gates and Payload Size' + scale_text, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    # Adjust layout
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
    # plt.close()
    
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
        # plt.close()
    
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
    # plt.close()
    
    return None

def calculate_cap_score(success_rate, payload_size, complexity_value, ps_max=4, complexity_max=None, complexity_type='gates'):
    """
    Calculate the Cap-score (Capability score) for a quantum hardware configuration.
    
    The Cap-score is a metric that combines success rate with circuit complexity,
    providing a single value that represents how well the hardware handles complex circuits.
    
    Formula: Cap-score = success_rate * sqrt((payload_size * complexity_value)/(ps_max * complexity_max)) * 100
    
    This rewards both high success rates and the ability to handle complex circuits.
    
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
        
    return success_rate * np.sqrt((payload_size * complexity_value)/(ps_max * complexity_max)) * 100

def plot_cap_scores(df):
    """
    Create visualization of Cap-scores with grouped bars for each payload size and gate range.
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
                cap_score = calculate_cap_score(success_rate, size, gate_range[0], complexity_type='gates')
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
    # plt.close()

def plot_depth_analysis(df):
    """
    Create comprehensive analysis plots based on circuit depth instead of gate count.
    This provides an alternative perspective on how circuit complexity affects performance.
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
    
    # Calculate number of ranges
    num_ranges = len(depth_ranges)
    
    # Create a color map for depth ranges that cycles through the COLORBREWER_PALETTE
    depth_colors = {}
    for i, depth_range in enumerate(depth_ranges):
        # Cycle through the available colors in COLORBREWER_PALETTE
        color_idx = (i % len(COLORBREWER_PALETTE)) + 1
        depth_colors[depth_range] = COLORBREWER_PALETTE[color_idx]
    
    for i, depth_range in enumerate(depth_ranges):
        errors = []
        for size in payload_sizes:
            mask = (df['payload_size'] == size) & (df['circuit_depth'].between(depth_range[0], depth_range[1], inclusive='both'))
            error_rate = 1 - df[mask]['success_rate'].mean() if not df[mask].empty else 0
            errors.append(error_rate)
        
        # Use the assigned color from COLORBREWER_PALETTE with appropriate alpha
        alpha_value = 0.5 + (0.5 * i / num_ranges) if num_ranges > 1 else 0.7
        alpha_value = min(0.9, alpha_value)  # Cap at 0.9 to be safe
        
        ax2.bar(x, errors, bottom=bottom,
                label=f'{depth_range[0]}-{depth_range[1]} depth',
                color=depth_colors[depth_range],
                alpha=alpha_value,
                edgecolor='black', linewidth=0.5)
        bottom += errors
    
    ax2.set_xlabel('Payload Size', fontsize=12)
    ax2.set_ylabel('Error Rate', fontsize=12)
    ax2.set_title('Cumulative Error Rate by Payload Size and Circuit Depth', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(payload_sizes)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. Heatmap of success rate by payload size and depth range
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
    
    # Calculate depth-based Cap-scores
    depth_cap_scores = []
    for size in payload_sizes:
        for depth_range in depth_ranges:
            mask = (df['payload_size'] == size) & (df['circuit_depth'].between(depth_range[0], depth_range[1], inclusive='both'))
            if not df[mask].empty:
                success_rate = df[mask]['success_rate'].mean()
                avg_depth = df[mask]['circuit_depth'].mean()
                # Calculate a depth-based Cap-score using the unified function
                depth_cap = calculate_cap_score(success_rate, size, avg_depth, complexity_type='depth')
                depth_cap_scores.append({
                    'payload_size': size,
                    'depth_range': depth_range,
                    'success_rate': success_rate,
                    'avg_depth': avg_depth,
                    'depth_cap_score': depth_cap
                })
    
    # Find best and worst configurations
    if depth_cap_scores:
        best_config = max(depth_cap_scores, key=lambda x: x['depth_cap_score'])
        worst_config = min(depth_cap_scores, key=lambda x: x['depth_cap_score'])
        avg_depth_cap = np.mean([score['depth_cap_score'] for score in depth_cap_scores])
        
        # Create text for the summary box
        summary_text = (
            "CIRCUIT DEPTH ANALYSIS SUMMARY\n"
            "=============================\n\n"
            f"Overall Success Rate: {df['success_rate'].mean():.2%}\n\n"
            f"Best Depth Configuration:\n"
            f"  - Payload Size: {best_config['payload_size']}\n"
            f"  - Depth Range: {best_config['depth_range'][0]}-{best_config['depth_range'][1]}\n"
            f"  - Success Rate: {best_config['success_rate']:.2%}\n"
            f"  - Depth-Cap Score: {best_config['depth_cap_score']:.2f}\n\n"
            f"Worst Depth Configuration:\n"
            f"  - Payload Size: {worst_config['payload_size']}\n"
            f"  - Depth Range: {worst_config['depth_range'][0]}-{worst_config['depth_range'][1]}\n"
            f"  - Success Rate: {worst_config['success_rate']:.2%}\n"
            f"  - Depth-Cap Score: {worst_config['depth_cap_score']:.2f}\n\n"
            f"Average Depth-Cap Score: {avg_depth_cap:.2f}\n\n"
            f"Success Rate by Payload Size:\n"
        )
        
        # Add success rates by payload size
        for size in payload_sizes:
            rate = df[df['payload_size'] == size]['success_rate'].mean()
            summary_text += f"  - Payload {size}: {rate:.2%}\n"
    else:
        summary_text = "No data available for depth-based analysis."
    
    # Create a text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('circuit_depth_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
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
        plt.title('Depth-Based Cap-Score Analysis by Payload Size and Circuit Depth', fontsize=16)
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
        # plt.close()
    
    return None

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
    
    # Add Cap-score analysis
    print("\nGenerating Cap-score Analysis Plot...")
    plot_cap_scores(df)
    print("Cap-score analysis plot saved as 'cap_scores_analysis.png'")
    
    # Add depth analysis
    print("\nGenerating Circuit Depth Analysis Plots...")
    plot_depth_analysis(df)
    print("Circuit depth analysis plots saved as:")
    print("- 'circuit_depth_analysis.png'")
    print("- 'depth_cap_scores_analysis.png'")