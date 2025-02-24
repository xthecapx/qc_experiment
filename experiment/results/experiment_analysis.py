import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from results_1_4_500_505 import results_1_4_500_505
from results_1_4_1000_1005 import results_1_4_1000_1005
from results_1_4_1500_1505 import results_1_4_1500_1505
from results_1_4_2000_2005 import results_1_4_2000_2005
from results_1_5_200_205 import results_1_5_200_205

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
    
    # Create a color map for different payload sizes
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    
    # Store regression results for printing
    regression_stats = []
    
    # Plot for each payload size
    for i, payload_size in enumerate(sorted(df['payload_size'].unique())):
        payload_data = df[df['payload_size'] == payload_size]
        
        # Group by num_gates and calculate mean success rate
        grouped_data = payload_data.groupby('num_gates')['success_rate'].mean()
        
        x = np.array(grouped_data.index)
        y = np.array(grouped_data.values * 100)
        
        if use_log_x:
            x = np.log10(x)  # Log transform x-axis
        if use_log_y:
            y = np.log10(y)  # Log transform y-axis
        
        # Scatter plot
        ax.scatter(x, y, 
                  color=colors[i], marker=markers[i], s=100,
                  label=f'Payload Size {payload_size}')
        
        # Regression analysis
        model = calculate_regression_stats(x, y)
        
        # Generate points for regression line
        x_pred = np.linspace(x.min(), x.max(), 100)
        X_pred = sm.add_constant(x_pred)
        y_pred = model.predict(X_pred)
        
        # Plot regression line
        ax.plot(x_pred, y_pred, color=colors[i], linestyle='--', alpha=0.5,
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
    
    # plt.close()
    
    return regression_stats

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