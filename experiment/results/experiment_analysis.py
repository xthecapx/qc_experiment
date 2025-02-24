import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from results_1_4_500_505 import results_1_4_500_505
from results_1_4_1000_1005 import results_1_4_1000_1005
from results_1_4_1500_1505 import results_1_4_1500_1505
from results_1_4_2000_2005 import results_1_4_2000_2005
from results_1_5_200_205 import results_1_5_200_205

# ColorBrewer colorblind-friendly palette
COLORBREWER_PALETTE = {
    1: '#e41a1c',    # Red
    2: '#4daf4a',    # Green
    3: '#377eb8',    # Blue
    4: '#984ea3',    # Purple
}

# Distinct marker styles with different shapes and fills
MARKER_STYLES = {
    1: 'o',     # Circle
    2: 's',     # Square
    3: '^',     # Triangle up
    4: 'D',     # Diamond
}

def create_experiment_dataframe():
    # List to store all experiment results
    all_experiments = []
    
    # Combine all result groups
    result_groups = [
        results_1_4_500_505,
        results_1_4_1000_1005,
        results_1_4_1500_1505,
        results_1_4_2000_2005,
        results_1_5_200_205
    ]
    
    # Extract experiments from each group
    for group in result_groups:
        for result_group in group:
            for experiment in result_group['experiments']:
                # Only include completed experiments with payload size 1-4
                if (experiment['status'] == 'completed' and 
                    1 <= experiment['experiment_params']['payload_size'] <= 4):
                    
                    payload_size = experiment['experiment_params']['payload_size']
                    counts = experiment['results_metrics']['counts']
                    
                    # Get the all-zeros string for this payload size
                    all_zeros = '0' * payload_size
                    
                    # Sum up counts for all-zeros and the rest
                    counts_zeros = counts.get(all_zeros, 0)
                    counts_ones = sum(count for key, count in counts.items() if key != all_zeros)
                    
                    experiment_data = {
                        'job_id': experiment['ibm_data'].get('job_id', None),
                        'payload_size': payload_size,
                        'num_gates': experiment['experiment_params']['num_gates'],
                        'success_rate': experiment['results_metrics']['success_rate'],
                        'circuit_depth': experiment['circuit_metrics']['depth'],
                        'circuit_width': experiment['circuit_metrics']['width'],
                        'circuit_size': experiment['circuit_metrics']['size'],
                        'backend': experiment['ibm_data'].get('backend', None),
                        'counts_zeros': counts_zeros,  # renamed to be more clear
                        'counts_ones': counts_ones     # sum of all other combinations
                    }
                    all_experiments.append(experiment_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_experiments)
    
    # Set job_id as index
    df.set_index('job_id', inplace=True)
    
    return df

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
    
    # Set x-axis ticks
    ranges = [(200, 205), (500, 505), (1000, 1005), (1500, 1505), (2000, 2005)]
    if use_log_x:
        tick_positions = [np.log10(r[0]) for r in ranges]
        tick_labels = [f"{r[0]}-{r[1]}" for r in ranges]
        plt.xlabel('Number of Gates (Range) - Log Scale', fontsize=12)
    else:
        tick_positions = [r[0] for r in ranges]
        tick_labels = [f"{r[0]}-{r[1]}" for r in ranges]
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
                   color=COLORBREWER_PALETTE[payload_size],
                   marker=MARKER_STYLES[payload_size],
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
                   color=COLORBREWER_PALETTE[payload_size],
                   marker=MARKER_STYLES[payload_size],
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
                   color=COLORBREWER_PALETTE[payload_size],
                   marker=MARKER_STYLES[payload_size],
                   s=100,
                   label=f'Payload Size {payload_size}')
    
    ax3.set_xlabel('Circuit Width')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate vs Circuit Width')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('circuit_complexity_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create multiple 3D plots with different viewing angles
    view_angles = [
        (20, 45),   # Default view
        (20, 135),  # Rotated 90 degrees
        (45, 90),   # Top-down view
        (60, 30)    # Higher elevation
    ]
    
    for i, (elev, azim) in enumerate(view_angles):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add a light gray background grid
        ax.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.5)})
        ax.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.5)})
        ax.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.5)})
        
        # Plot points for each payload size
        for payload_size in sorted(df['payload_size'].unique()):
            payload_data = df[df['payload_size'] == payload_size]
            
            # Add scatter plot with ColorBrewer colors
            scatter = ax.scatter(payload_data['num_gates'], 
                               payload_data['circuit_depth'],
                               payload_data['success_rate'] * 100,
                               color=COLORBREWER_PALETTE[payload_size],
                               marker=MARKER_STYLES[payload_size],
                               s=150,
                               alpha=0.7,
                               edgecolor='black',
                               linewidth=0.5,
                               label=f'Payload Size {payload_size}')
            
            # Add vertical lines with matching colors
            for _, row in payload_data.iterrows():
                ax.plot([row['num_gates'], row['num_gates']], 
                       [row['circuit_depth'], row['circuit_depth']],
                       [0, row['success_rate'] * 100],
                       color=COLORBREWER_PALETTE[payload_size],
                       alpha=0.2,
                       linestyle='--')
        
        # Set labels with larger font
        ax.set_xlabel('Number of Gates', fontsize=12, labelpad=10)
        ax.set_ylabel('Circuit Depth', fontsize=12, labelpad=10)
        ax.set_zlabel('Success Rate (%)', fontsize=12, labelpad=10)
        
        # Set title with viewing angle information
        ax.set_title(f'Success Rate vs Gates and Circuit Depth\nView: {elev}° elevation, {azim}° azimuth',
                    fontsize=14, pad=20)
        
        # Adjust the legend
        ax.legend(bbox_to_anchor=(1.15, 0.5), loc='center left', fontsize=10)
        
        # Set the viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Make the panes slightly transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Adjust the aspect ratio to make the plot more cubic
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        plt.savefig(f'circuit_complexity_3d_view{i+1}.png', 
                   dpi=300, bbox_inches='tight')
        # plt.close()
    
    return None

def plot_error_analysis(df):
    """
    Create comprehensive error analysis plots:
    1. Bar plot comparing counts_zeros vs counts_ones
    2. Stacked bar chart of error distribution
    3. Error rate in log scale
    """
    # Create a figure with three subplots
    fig = plt.figure(figsize=(20, 6))
    gs = plt.GridSpec(1, 3, figure=fig)
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
    
    # Create grouped bars
    ax1.bar(x - bar_width/2, zeros_means, bar_width, label='All Zeros', 
            color='lightgray', edgecolor='black')
    ax1.bar(x + bar_width/2, ones_means, bar_width, label='Other States',
            color='darkgray', edgecolor='black')
    
    ax1.set_xlabel('Payload Size')
    ax1.set_ylabel('Average Counts')
    ax1.set_title('Distribution of Measurement Outcomes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(payload_sizes)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. Stacked bar chart showing error distribution
    bottom = np.zeros(len(payload_sizes))
    
    for gate_range in [(200, 205), (500, 505), (1000, 1005), (1500, 1505), (2000, 2005)]:
        errors = []
        for size in payload_sizes:
            mask = (df['payload_size'] == size) & (df['num_gates'].between(gate_range[0], gate_range[1]))
            error_rate = 1 - df[mask]['success_rate'].mean() if not df[mask].empty else 0
            errors.append(error_rate)
        
        ax2.bar(x, errors, bottom=bottom, label=f'{gate_range[0]}-{gate_range[1]} gates',
                color=plt.cm.viridis(gate_range[0]/2000))
        bottom += errors

    ax2.set_xlabel('Payload Size')
    ax2.set_ylabel('Cumulative Error Rate')
    ax2.set_title('Error Distribution by Gates Range')
    ax2.set_xticks(x)
    ax2.set_xticklabels(payload_sizes)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. Error rate in log scale
    for payload_size in payload_sizes:
        payload_data = df[df['payload_size'] == payload_size]
        
        # Group by num_gates and calculate mean error rate
        grouped_data = payload_data.groupby('num_gates').agg({
            'success_rate': 'mean',
            'num_gates': 'first'
        })
        
        error_rates = 1 - grouped_data['success_rate']
        
        ax3.scatter(grouped_data['num_gates'], error_rates,
                   color=COLORBREWER_PALETTE[payload_size],
                   marker=MARKER_STYLES[payload_size],
                   s=100,
                   label=f'Payload Size {payload_size}')
        
        # Add trend line
        z = np.polyfit(np.log10(grouped_data['num_gates']), np.log10(error_rates), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(grouped_data['num_gates'].min(), grouped_data['num_gates'].max(), 100)
        y_trend = 10**p(np.log10(x_trend))
        
        ax3.plot(x_trend, y_trend, 
                color=COLORBREWER_PALETTE[payload_size],
                linestyle='--', alpha=0.5)

    ax3.set_xlabel('Number of Gates')
    ax3.set_ylabel('Error Rate')
    ax3.set_title('Error Rate vs Number of Gates')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close()

    return None

if __name__ == "__main__":
    # Create the DataFrame
    df = create_experiment_dataframe()
    
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