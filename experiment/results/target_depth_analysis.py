import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats
import json
import os
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create output directory if it doesn't exist
OUTPUT_DIR = "circuit_depth_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define colorblind-friendly palettes
COLORBREWER_PALETTE = {
    1: '#1f77b4',  # Blue
    2: '#ff7f0e',  # Orange
    3: '#2ca02c',  # Green
    4: '#d62728',  # Red
    5: '#9467bd',  # Purple
    6: '#8c564b',  # Brown
    7: '#e377c2',  # Pink
    8: '#7f7f7f',  # Gray
    9: '#bcbd22',  # Olive
    10: '#17becf'  # Cyan
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

def plot_success_rate_vs_circuit_depth(df):
    """
    Create a plot showing success rate vs circuit depth, grouped by payload size
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Store regression results for printing
    regression_stats = []
    
    # Plot for each payload size
    for payload_size in sorted(df['payload_size'].unique()):
        payload_data = df[df['payload_size'] == payload_size]
        
        # Group by circuit_depth and calculate mean success rate
        grouped_data = payload_data.groupby('circuit_depth')['success_rate'].mean()
        
        x = np.array(grouped_data.index)
        y = np.array(grouped_data.values * 100)
        
        # Scatter plot with ColorBrewer colors and distinct markers
        ax.scatter(x, y, 
                  color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                  marker=MARKER_STYLES.get(payload_size, 'o'),
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
                color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
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
    
    # Set axis labels and title
    plt.xlabel('Circuit Depth', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Success Rates by Circuit Depth and Payload Size', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'success_rate_vs_circuit_depth.png'), dpi=300, bbox_inches='tight')
    
    return regression_stats

def plot_error_distribution_by_circuit_depth(df):
    """
    Create a plot showing error distribution based on circuit depth
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
    
    # Create bar plot with error bars
    ax.bar(depths, error_means, yerr=error_stds, 
           color='skyblue', edgecolor='black', alpha=0.7,
           capsize=5, width=0.7)
    
    # Add a trend line
    model = calculate_regression_stats(depths, error_means)
    x_pred = np.linspace(depths.min(), depths.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred)
    
    ax.plot(x_pred, y_pred, 'r--', 
            label=f'Trend (R²={model.rsquared:.3f})')
    
    # Set axis labels and title
    ax.set_xlabel('Circuit Depth', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Error Distribution by Circuit Depth', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution_by_circuit_depth.png'), dpi=300, bbox_inches='tight')

def plot_success_rate_heatmap(df):
    """
    Create a heatmap showing success rate by circuit depth and payload size
    """
    # Create pivot table for the heatmap
    pivot_data = df.pivot_table(
        values='success_rate',
        index='circuit_depth',
        columns='payload_size',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap from red to green
    cmap = LinearSegmentedColormap.from_list(
        'success_cmap', ['#d62728', '#ffff99', '#2ca02c']
    )
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.3f',
                cmap=cmap,
                linewidths=0.5,
                ax=ax,
                vmin=0, 
                vmax=1,
                cbar_kws={'label': 'Success Rate'})
    
    # Set axis labels and title
    ax.set_xlabel('Payload Size', fontsize=12)
    ax.set_ylabel('Circuit Depth', fontsize=12)
    ax.set_title('Success Rate Heatmap by Circuit Depth and Payload Size', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'success_rate_heatmap.png'), dpi=300, bbox_inches='tight')

def plot_error_distribution_by_payload_size(df):
    """
    Create a plot showing error distribution based on payload size
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
                color=COLORBREWER_PALETTE.get(size, '#333333'), alpha=0.7,
                edgecolor='black', linewidth=1,
                label=f'Zeros (P{size})' if i == 0 else "")
        
        ax1.bar(x[i] + bar_width/2, ones_means[i], bar_width,
                color=COLORBREWER_PALETTE.get(size, '#333333'), alpha=0.3,
                edgecolor='black', linewidth=1, hatch='///',
                label=f'Ones (P{size})' if i == 0 else "")
    
    # Set labels and title for first subplot
    ax1.set_xlabel('Payload Size', fontsize=12)
    ax1.set_ylabel('Average Bit Count Proportion', fontsize=12)
    ax1.set_title('Bit Distribution by Payload Size', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(payload_sizes)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Error rate by payload size (boxplot)
    df['error_rate'] = 1 - df['success_rate']
    
    # Create boxplot with ColorBrewer colors
    boxprops = {'alpha': 0.8, 'linewidth': 1.5}
    whiskerprops = {'linewidth': 1.5}
    medianprops = {'color': 'black', 'linewidth': 2}
    
    # Create a list of colors for the boxes
    box_colors = [COLORBREWER_PALETTE.get(size, '#333333') for size in payload_sizes]
    
    # Create the boxplot
    bp = ax2.boxplot([df[df['payload_size'] == size]['error_rate'] for size in payload_sizes],
                     patch_artist=True,
                     boxprops=boxprops,
                     whiskerprops=whiskerprops,
                     medianprops=medianprops)
    
    # Color the boxes
    for box, color in zip(bp['boxes'], box_colors):
        box.set_facecolor(color)
    
    # Set labels and title for second subplot
    ax2.set_xlabel('Payload Size', fontsize=12)
    ax2.set_ylabel('Error Rate', fontsize=12)
    ax2.set_title('Error Rate Distribution by Payload Size', fontsize=14)
    ax2.set_xticklabels(payload_sizes)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Scatter plot of error rate vs payload size with circuit depth as color
    scatter = ax3.scatter(df['payload_size'], df['error_rate'],
                         c=df['circuit_depth'], cmap='viridis',
                         alpha=0.7, s=80, edgecolor='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Circuit Depth', fontsize=10)
    
    # Add regression line
    x = df['payload_size']
    y = df['error_rate']
    model = calculate_regression_stats(x, y)
    
    # Generate points for regression line
    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred)
    
    # Plot regression line
    ax3.plot(x_pred, y_pred, 'r--', 
             label=f'Trend (R²={model.rsquared:.3f})')
    
    # Set labels and title for third subplot
    ax3.set_xlabel('Payload Size', fontsize=12)
    ax3.set_ylabel('Error Rate', fontsize=12)
    ax3.set_title('Error Rate vs Payload Size (colored by Circuit Depth)', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution_by_payload_size.png'), dpi=300, bbox_inches='tight')

def analyze_target_depth_experiment(csv_file_path):
    """
    Main function to analyze the target depth experiment data
    """
    print(f"Loading data from {csv_file_path}...")
    df = create_experiment_dataframe(csv_file_path)
    
    print(f"Loaded {len(df)} experiment results.")
    print(f"Circuit depths: {sorted(df['circuit_depth'].unique())}")
    print(f"Payload sizes: {sorted(df['payload_size'].unique())}")
    
    print("\nGenerating success rate vs circuit depth plot...")
    regression_stats = plot_success_rate_vs_circuit_depth(df)
    
    print("\nRegression statistics for success rate vs circuit depth:")
    for stat in regression_stats:
        print(f"Payload Size {stat['payload_size']}: slope={stat['slope']:.4f}, "
              f"intercept={stat['intercept']:.4f}, R²={stat['r_squared']:.4f}")
    
    print("\nGenerating error distribution by circuit depth plot...")
    plot_error_distribution_by_circuit_depth(df)
    
    print("\nGenerating success rate heatmap...")
    plot_success_rate_heatmap(df)
    
    print("\nGenerating error distribution by payload size plot...")
    plot_error_distribution_by_payload_size(df)
    
    print("\nPerforming regression analysis...")
    model_results = analyze_regression_models(df)
    
    print("\nAnalysis complete. All plots have been saved.")
    
    return df, model_results

def analyze_regression_models(df):
    """
    Implement and evaluate multiple regression models to predict success rates
    based on circuit parameters (circuit_depth, circuit_size, circuit_width, payload_size).
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results.
        
    Returns:
        dict: Dictionary containing regression results and model comparisons.
    """
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
            
            # Create safe filename by replacing special characters
            safe_filename = model_name.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(os.path.join(OUTPUT_DIR, f'{safe_filename}_residual_diagnostics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create actual vs predicted plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # If we're using log-transformed target, convert back to original scale
            if "Log-transformed" in model_name:
                actual = np.exp(y) - 0.001  # Reverse the log transformation
                predicted = np.exp(y_pred) - 0.001
            else:
                actual = y
                predicted = y_pred
            
            # Plot actual vs predicted
            ax.scatter(actual, predicted, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add labels and title
            ax.set_xlabel('Actual Success Rate')
            ax.set_ylabel('Predicted Success Rate')
            ax.set_title(f'Actual vs Predicted Values for {model_name}')
            
            # Add R-squared value
            r_squared = model.rsquared
            ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{safe_filename}_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
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
    
    # Model 6: Weighted Least Squares (if heteroscedasticity is detected)
    # Use the absolute residuals from the best model to estimate weights
    try:
        # Use model 1 as the base for weights
        best_model_residuals = model_results["Model 1: All features (linear)"]['residuals']
        
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
    
    # Save model equations to a file
    with open(os.path.join(OUTPUT_DIR, 'target_depth_model_equations.txt'), 'w') as f:
        f.write("# Target Depth Experiment Regression Models\n\n")
        for name, result in model_results.items():
            model = result['model']
            f.write(f"{name}:\n")
            
            # Get the equation
            equation = "success_rate = "
            for i, var in enumerate(model.params.index):
                coef = model.params[i]
                if var == 'const':
                    equation += f"{coef:.6f}"
                else:
                    if coef >= 0:
                        equation += f" + {coef:.6f} * {var}"
                    else:
                        equation += f" - {abs(coef):.6f} * {var}"
            
            f.write(equation + "\n\n")
            
            # Add model statistics
            f.write(f"R-squared: {model.rsquared:.4f}\n")
            f.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}\n")
            f.write(f"AIC: {model.aic:.4f}\n")
            f.write(f"BIC: {model.bic:.4f}\n\n")
            f.write("-" * 80 + "\n\n")
    
    print("\nModel equations saved to 'target_depth_model_equations.txt'")
    
    # Create correlation matrix visualization
    corr_matrix = model_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_depth_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nCorrelation matrix saved to 'target_depth_correlation_matrix.png'")
    
    return model_results

if __name__ == "__main__":
    # Path to the experiment results CSV file
    csv_file_path = "../experiment_results_target_depth_20250309_170751_updated.csv"
    
    # Run the analysis
    analyze_target_depth_experiment(csv_file_path) 