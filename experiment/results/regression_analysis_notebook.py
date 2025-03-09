"""
Regression Analysis for Quantum Circuit Success Rates
----------------------------------------------------
This script provides functions to analyze and visualize the two best regression models:
1. Model 2: Log-transformed target with robust standard errors
2. Model 6: Weighted Least Squares

Copy and paste this code into a Jupyter notebook to run the analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.graphics.gofplots import ProbPlot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# ColorBrewer colorblind-friendly palette
COLORBREWER_PALETTE = {
    1: '#d7191c',    # Red
    2: '#fdae61',    # Orange
    3: '#ffffbf',    # Yellow
    4: '#abdda4',    # Light Green
    5: '#2b83ba'     # Blue
}

# Simple function to load CSV data - no dependencies on other modules
def load_data(csv_file='consolidated_experiment_results.csv'):
    """
    Load experiment data from CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing experiment results
    """
    print(f"Loading data from: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check for missing values
    if df['success_rate'].isnull().sum() > 0:
        print("Warning: DataFrame contains missing success_rate values. Dropping rows with missing values.")
        df = df.dropna(subset=['success_rate'])
    
    return df

def fit_log_transformed_model(df):
    """
    Fit Model 2: Log-transformed target with robust standard errors.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        
    Returns:
        tuple: (model, X, y_log, y_pred_original) where:
            - model: Fitted statsmodels OLS model with robust standard errors
            - X: Design matrix used for fitting
            - y_log: Log-transformed target values
            - y_pred_original: Predicted values in original scale
    """
    # Prepare data
    model_df = df[['success_rate', 'circuit_depth', 'circuit_size', 'circuit_width', 'payload_size']].copy()
    
    # Log-transform target variable (adding small constant to avoid log(0))
    y = model_df['success_rate']
    y_log = np.log(y + 0.001)
    
    # Create design matrix
    X = model_df[['circuit_depth', 'circuit_size', 'circuit_width', 'payload_size']]
    X = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y_log, X).fit()
    
    # Apply robust standard errors
    robust_model = model.get_robustcov_results(cov_type='HC3')
    
    # Get predictions in original scale
    y_pred_log = robust_model.predict(X)
    y_pred_original = np.exp(y_pred_log) - 0.001
    
    # Clip predictions to valid range [0, 1]
    y_pred_original = np.clip(y_pred_original, 0, 1)
    
    return robust_model, X, y_log, y_pred_original

def fit_weighted_least_squares_model(df):
    """
    Fit Model 6: Weighted Least Squares.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        
    Returns:
        tuple: (model, X, y, weights) where:
            - model: Fitted statsmodels WLS model
            - X: Design matrix used for fitting
            - y: Target values
            - weights: Weights used in the model
    """
    # Prepare data
    model_df = df[['success_rate', 'circuit_depth', 'circuit_size', 'circuit_width', 'payload_size']].copy()
    
    # Target variable
    y = model_df['success_rate']
    
    # Create design matrix
    X = model_df[['circuit_depth', 'circuit_size', 'circuit_width', 'payload_size']]
    X = sm.add_constant(X)
    
    # First fit OLS model to get residuals
    ols_model = sm.OLS(y, X).fit()
    residuals = ols_model.resid
    
    # Fit a model to predict the absolute residuals
    abs_resid = np.abs(residuals)
    mod_resid = sm.OLS(abs_resid, X).fit()
    
    # Use fitted values as weights
    weights = 1 / (mod_resid.fittedvalues ** 2)
    
    # Fit WLS model
    wls_model = sm.WLS(y, X, weights=weights).fit()
    
    return wls_model, X, y, weights

def plot_residual_diagnostics(model, X, y, y_pred, model_name, output_dir='regression_visualizations'):
    """
    Create residual diagnostic plots for a regression model.
    
    Args:
        model: Fitted statsmodels model
        X: Design matrix
        y: Target values
        y_pred: Predicted values
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Create figure
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
        print(f"Warning: Could not calculate leverage: {str(e)}")
        axes[1, 1].text(0.5, 0.5, "Leverage calculation failed", 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_residual_diagnostics.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

def plot_actual_vs_predicted(y, y_pred, model_name, output_dir='regression_visualizations'):
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        y: Actual values
        y_pred: Predicted values
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Success Rate')
    plt.ylabel('Predicted Success Rate')
    plt.title(f'Actual vs Predicted Success Rate - {model_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate R-squared for the plot
    correlation_matrix = np.corrcoef(y, y_pred)
    correlation = correlation_matrix[0, 1]
    r_squared = correlation ** 2
    
    plt.annotate(f'R² = {r_squared:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_actual_vs_predicted.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

def plot_success_rate_by_payload(df, model, X, model_name, is_log_model=False, output_dir='regression_visualizations'):
    """
    Plot success rate vs circuit parameters by payload size.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        model: Fitted statsmodels model
        X: Design matrix
        model_name (str): Name of the model for plot titles
        is_log_model (bool): Whether the model uses log-transformed target
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots for each payload size
    for param in ['circuit_depth', 'circuit_size']:
        plt.figure(figsize=(12, 8))
        
        for payload_size in sorted(df['payload_size'].unique()):
            # Get data for this payload size
            payload_data = df[df['payload_size'] == payload_size]
            
            # Skip if not enough data
            if len(payload_data) < 5:
                continue
            
            # Plot actual data points
            plt.scatter(payload_data[param], payload_data['success_rate'], 
                       color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                       alpha=0.7, s=50, label=f'Actual (Payload {payload_size})')
            
            # Generate prediction line
            x_range = np.linspace(payload_data[param].min(), payload_data[param].max(), 100)
            
            # Create prediction data
            pred_data = pd.DataFrame({
                'circuit_depth': df['circuit_depth'].median() * np.ones(100),
                'circuit_size': df['circuit_size'].median() * np.ones(100),
                'circuit_width': payload_data['circuit_width'].iloc[0] * np.ones(100),
                'payload_size': payload_size * np.ones(100)
            })
            
            # Set the parameter we're varying
            pred_data[param] = x_range
            
            # Add constant
            pred_X = sm.add_constant(pred_data)
            
            # Generate predictions
            if is_log_model:
                y_pred_log = model.predict(pred_X)
                y_pred = np.exp(y_pred_log) - 0.001
                y_pred = np.clip(y_pred, 0, 1)
            else:
                y_pred = model.predict(pred_X)
            
            # Plot prediction line
            plt.plot(x_range, y_pred, color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                    linestyle='-', linewidth=2, label=f'Predicted (Payload {payload_size})')
        
        plt.xlabel(f'{param.replace("_", " ").title()}')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate vs {param.replace("_", " ").title()} by Payload Size - {model_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_{param}_by_payload.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()

def plot_3d_surface(df, model, model_name, is_log_model=False, output_dir='regression_visualizations'):
    """
    Create 3D surface plots of the model predictions.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        model: Fitted statsmodels model
        model_name (str): Name of the model for plot titles
        is_log_model (bool): Whether the model uses log-transformed target
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 3D plots for different combinations of parameters
    param_pairs = [
        ('circuit_depth', 'payload_size'),
        ('circuit_size', 'payload_size'),
        ('circuit_depth', 'circuit_size')
    ]
    
    for x_param, y_param in param_pairs:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid for surface
        x_min, x_max = df[x_param].min(), df[x_param].max()
        y_min, y_max = df[y_param].min(), df[y_param].max()
        
        x_range = np.linspace(x_min, x_max, 30)
        y_range = np.linspace(y_min, y_max, 30)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        
        # Create prediction data
        n_points = X_grid.size
        pred_data = pd.DataFrame({
            'circuit_depth': df['circuit_depth'].median() * np.ones(n_points),
            'circuit_size': df['circuit_size'].median() * np.ones(n_points),
            'circuit_width': df['circuit_width'].median() * np.ones(n_points),
            'payload_size': df['payload_size'].median() * np.ones(n_points)
        })
        
        # Set the parameters we're varying
        pred_data[x_param] = X_grid.flatten()
        pred_data[y_param] = Y_grid.flatten()
        
        # If y_param is payload_size, update circuit_width accordingly
        if y_param == 'payload_size':
            # In the dataset, circuit_width and payload_size are perfectly correlated
            # We need to determine the relationship
            width_by_payload = df.groupby('payload_size')['circuit_width'].first()
            
            # Map payload size to circuit width
            pred_data['circuit_width'] = pred_data['payload_size'].map(width_by_payload)
            
            # Handle any missing values (for payload sizes not in the original data)
            if pred_data['circuit_width'].isnull().any():
                # Fit a simple linear model
                from scipy.stats import linregress
                slope, intercept, _, _, _ = linregress(df['payload_size'], df['circuit_width'])
                
                # Fill missing values
                mask = pred_data['circuit_width'].isnull()
                pred_data.loc[mask, 'circuit_width'] = intercept + slope * pred_data.loc[mask, 'payload_size']
        
        # Add constant
        pred_X = sm.add_constant(pred_data)
        
        # Generate predictions
        if is_log_model:
            Z_pred_log = model.predict(pred_X)
            Z_pred = np.exp(Z_pred_log) - 0.001
            Z_pred = np.clip(Z_pred, 0, 1)
        else:
            Z_pred = model.predict(pred_X)
        
        # Reshape for plotting
        Z_grid = Z_pred.reshape(X_grid.shape)
        
        # Plot surface
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.viridis, alpha=0.8,
                              linewidth=0, antialiased=True)
        
        # Plot actual data points
        ax.scatter(df[x_param], df[y_param], df['success_rate'], 
                  c='red', marker='o', s=30, alpha=0.7)
        
        # Labels and title
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        ax.set_zlabel('Success Rate')
        ax.set_title(f'Success Rate vs {x_param.replace("_", " ").title()} and {y_param.replace("_", " ").title()} - {model_name}')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_{x_param}_{y_param}_3d.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()

def analyze_and_visualize_models(df, output_dir='regression_visualizations'):
    """
    Analyze and visualize the two best regression models.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results.
        output_dir (str): Directory to save plots.
        
    Returns:
        tuple: (log_model, wls_model) - The fitted models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fitting Model 2: Log-transformed target with robust standard errors...")
    log_model, log_X, y_log, y_pred_log = fit_log_transformed_model(df)
    
    print("Fitting Model 6: Weighted Least Squares...")
    wls_model, wls_X, y, weights = fit_weighted_least_squares_model(df)
    
    # Print model summaries
    print("\nModel 2: Log-transformed target with robust standard errors")
    print("=" * 80)
    print(log_model.summary())
    
    print("\nModel 6: Weighted Least Squares")
    print("=" * 80)
    print(wls_model.summary())
    
    # Create model equation strings
    # Get parameter names and values
    log_params = log_model.params
    wls_params = wls_model.params
    
    # Handle both pandas Series and NumPy arrays
    if hasattr(log_params, 'index'):  # It's a pandas Series
        log_param_names = log_params.index
        log_const_idx = 0 if 'const' not in log_param_names else log_param_names.get_loc('const')
    else:  # It's a NumPy array
        # Assume the first parameter is the constant
        log_param_names = ['const'] + [f'x{i}' for i in range(1, len(log_params))]
        log_const_idx = 0
    
    if hasattr(wls_params, 'index'):  # It's a pandas Series
        wls_param_names = wls_params.index
        wls_const_idx = 0 if 'const' not in wls_param_names else wls_param_names.get_loc('const')
    else:  # It's a NumPy array
        # Assume the first parameter is the constant
        wls_param_names = ['const'] + [f'x{i}' for i in range(1, len(wls_params))]
        wls_const_idx = 0
    
    # Get the parameter names from the design matrix if available
    if hasattr(log_X, 'columns'):
        log_param_names = log_X.columns
    
    if hasattr(wls_X, 'columns'):
        wls_param_names = wls_X.columns
    
    # Create equation strings
    log_model_eq = "log(success_rate + 0.001) = "
    log_model_eq += f"{log_params[log_const_idx]:.4f} "
    
    # Add the other parameters
    for i, param_name in enumerate(log_param_names):
        if i != log_const_idx:  # Skip the constant term
            coef = log_params[i]
            if coef >= 0:
                log_model_eq += f"+ {coef:.4e} × {param_name} "
            else:
                log_model_eq += f"- {abs(coef):.4e} × {param_name} "
    
    wls_model_eq = "success_rate = "
    wls_model_eq += f"{wls_params[wls_const_idx]:.4f} "
    
    # Add the other parameters
    for i, param_name in enumerate(wls_param_names):
        if i != wls_const_idx:  # Skip the constant term
            coef = wls_params[i]
            if coef >= 0:
                wls_model_eq += f"+ {coef:.4e} × {param_name} "
            else:
                wls_model_eq += f"- {abs(coef):.4e} × {param_name} "
    
    print("\nModel Equations:")
    print(f"Model 2: {log_model_eq}")
    print(f"Model 6: {wls_model_eq}")
    
    # Save model equations to file
    with open(os.path.join(output_dir, 'model_equations.txt'), 'w') as f:
        f.write("Model Equations:\n")
        f.write(f"Model 2: {log_model_eq}\n")
        f.write(f"Model 6: {wls_model_eq}\n")
    
    # Generate predictions for WLS model
    y_pred_wls = wls_model.predict(wls_X)
    
    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    
    # Residual diagnostics
    plot_residual_diagnostics(log_model, log_X, y_log, log_model.predict(log_X), 
                             "Model 2: Log-transformed target", output_dir)
    plot_residual_diagnostics(wls_model, wls_X, y, y_pred_wls, 
                             "Model 6: Weighted Least Squares", output_dir)
    
    # Actual vs Predicted plots
    plot_actual_vs_predicted(df['success_rate'], y_pred_log, 
                            "Model 2: Log-transformed target", output_dir)
    plot_actual_vs_predicted(df['success_rate'], y_pred_wls, 
                            "Model 6: Weighted Least Squares", output_dir)
    
    # Success rate by payload size plots
    plot_success_rate_by_payload(df, log_model, log_X, 
                                "Model 2: Log-transformed target", True, output_dir)
    plot_success_rate_by_payload(df, wls_model, wls_X, 
                                "Model 6: Weighted Least Squares", False, output_dir)
    
    # 3D surface plots
    plot_3d_surface(df, log_model, "Model 2: Log-transformed target", True, output_dir)
    plot_3d_surface(df, wls_model, "Model 6: Weighted Least Squares", False, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return log_model, wls_model

# Example usage:
# df = load_data('consolidated_experiment_results.csv')
# log_model, wls_model = analyze_and_visualize_models(df) 