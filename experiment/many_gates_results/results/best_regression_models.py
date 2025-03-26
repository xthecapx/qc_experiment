#!/usr/bin/env python3
"""
Best regression models for predicting quantum circuit success rates.
This module provides functions to analyze and visualize the two best regression models:
1. Model 2: Log-transformed target with robust standard errors
2. Model 6: Weighted Least Squares
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
from .experiment_analysis import create_experiment_dataframe

# ColorBrewer colorblind-friendly palette
COLORBREWER_PALETTE = {
    1: '#d7191c',    # Red
    2: '#fdae61',    # Orange
    3: '#ffffbf',    # Yellow
    4: '#abdda4',    # Light Green
    5: '#2b83ba'     # Blue
}

def load_data(csv_file=None):
    """
    Load experiment data from CSV file.
    
    Args:
        csv_file (str): Path to the CSV file. If None, uses consolidated_experiment_results.csv
        
    Returns:
        pd.DataFrame: DataFrame containing experiment results
    """
    if csv_file is None:
        csv_file = 'consolidated_experiment_results.csv'
    
    if not os.path.exists(csv_file):
        # Try to find the file in the parent directory
        parent_dir = os.path.dirname(os.getcwd())
        alternative_path = os.path.join(parent_dir, csv_file)
        if os.path.exists(alternative_path):
            csv_file = alternative_path
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    print(f"Loading data from: {csv_file}")
    df = create_experiment_dataframe(csv_file)
    
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

def plot_residual_diagnostics(model, X, y, y_pred, model_name, output_dir=None):
    """
    Create residual diagnostic plots for a regression model.
    
    Args:
        model: Fitted statsmodels model
        X: Design matrix
        y: Target values
        y_pred: Predicted values
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save plots. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
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
    
    # Residuals vs Leverage plot - skip if not available
    try:
        # Check if the model has the get_influence method
        if hasattr(model, 'get_influence'):
            influence = model.get_influence()
            leverage = influence.hat_matrix_diag
            axes[1, 1].scatter(leverage, residuals)
            axes[1, 1].axhline(y=0, color='r', linestyle='-')
            axes[1, 1].set_xlabel('Leverage')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals vs Leverage')
        else:
            # For models without get_influence, create a simpler plot
            axes[1, 1].text(0.5, 0.5, "Leverage calculation not available for this model type", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Residuals vs Leverage (Not Available)')
    except Exception as e:
        print(f"Warning: Could not calculate leverage: {str(e)}")
        axes[1, 1].text(0.5, 0.5, "Leverage calculation failed", 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_residual_diagnostics.png'), 
                dpi=300, bbox_inches='tight')

def plot_actual_vs_predicted(y, y_pred, model_name, output_dir=None):
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        y: Actual values
        y_pred: Predicted values
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save plots. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
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
    plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_actual_vs_predicted.png'), 
                dpi=300, bbox_inches='tight')

def plot_success_rate_by_payload(df, model, X, model_name, is_log_model=False, output_dir=None):
    """
    Plot success rate vs circuit parameters by payload size.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        model: Fitted statsmodels model
        X: Design matrix used for fitting
        model_name (str): Name of the model for plot titles
        is_log_model (bool): Whether the model uses log-transformed target
        output_dir (str): Directory to save plots. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
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
            
            # Create prediction data with the same columns as the original design matrix
            if hasattr(X, 'columns'):
                # It's a pandas DataFrame
                pred_data = pd.DataFrame(index=range(100), columns=X.columns)
                
                # Fill with median values from the original data
                for col in X.columns:
                    if col == 'const':
                        pred_data[col] = 1.0
                    else:
                        pred_data[col] = df[col].median() if col in df.columns else X[col].median()
                
                # Set the parameter we're varying and payload size
                if param in pred_data.columns:
                    pred_data[param] = x_range
                
                if 'payload_size' in pred_data.columns:
                    pred_data['payload_size'] = payload_size
                
                # Set circuit_width based on payload_size if needed
                if 'circuit_width' in pred_data.columns:
                    # In the dataset, circuit_width and payload_size are perfectly correlated
                    # Use the circuit_width from the payload_data
                    pred_data['circuit_width'] = payload_data['circuit_width'].iloc[0]
            else:
                # It's a numpy array - create a new array with the same shape
                n_features = X.shape[1]
                pred_data = np.ones((100, n_features))
                
                # Fill with median values from the original data
                for i in range(1, n_features):  # Skip the constant
                    pred_data[:, i] = df.iloc[:, i].median()
                
                # Determine which column corresponds to our parameter and payload_size
                param_cols = {
                    'circuit_depth': 1,
                    'circuit_size': 2,
                    'circuit_width': 3,
                    'payload_size': 4
                }
                
                # Set the parameter we're varying
                if param in param_cols:
                    pred_data[:, param_cols[param]] = x_range
                
                # Set payload_size
                if 'payload_size' in param_cols:
                    pred_data[:, param_cols['payload_size']] = payload_size
                
                # Set circuit_width based on payload_size
                if 'circuit_width' in param_cols:
                    pred_data[:, param_cols['circuit_width']] = payload_data['circuit_width'].iloc[0]
            
            # Generate predictions
            try:
                if is_log_model:
                    y_pred_log = model.predict(pred_data)
                    y_pred = np.exp(y_pred_log) - 0.001
                    y_pred = np.clip(y_pred, 0, 1)
                else:
                    y_pred = model.predict(pred_data)
                
                # Plot prediction line
                plt.plot(x_range, y_pred, color=COLORBREWER_PALETTE.get(payload_size, '#333333'),
                        linestyle='-', linewidth=2, label=f'Predicted (Payload {payload_size})')
            except Exception as e:
                print(f"Warning: Could not generate predictions for payload_size={payload_size}: {str(e)}")
                print(f"X shape: {X.shape}, pred_data shape: {pred_data.shape}")
        
        plt.xlabel(f'{param.replace("_", " ").title()}')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate vs {param.replace("_", " ").title()} by Payload Size - {model_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_{param}_by_payload.png'), 
                    dpi=300, bbox_inches='tight')

def plot_3d_surface(df, model, model_name, is_log_model=False, output_dir=None):
    """
    Create 3D surface plots of the model predictions.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        model: Fitted statsmodels model
        model_name (str): Name of the model for plot titles
        is_log_model (bool): Whether the model uses log-transformed target
        output_dir (str): Directory to save plots. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create 3D plots for different combinations of parameters
    param_pairs = [
        ('circuit_depth', 'payload_size'),
        ('circuit_size', 'payload_size'),
        ('circuit_depth', 'circuit_size')
    ]
    
    # Get the design matrix structure
    if hasattr(model, 'model') and hasattr(model.model, 'exog'):
        X_structure = model.model.exog
    else:
        # Skip 3D plots if we can't determine the model structure
        print("Warning: Could not determine model structure for 3D plots. Skipping.")
        return
    
    # Determine if X is a pandas DataFrame or numpy array
    is_pandas = hasattr(X_structure, 'columns')
    
    # Get column names or indices
    if is_pandas:
        column_names = X_structure.columns
    else:
        # Assume standard order: const, circuit_depth, circuit_size, circuit_width, payload_size
        column_names = ['const', 'circuit_depth', 'circuit_size', 'circuit_width', 'payload_size']
    
    # Create a mapping of parameter names to column indices
    param_indices = {}
    for i, name in enumerate(column_names):
        param_indices[name] = i
    
    for x_param, y_param in param_pairs:
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create grid for surface
            x_min, x_max = df[x_param].min(), df[x_param].max()
            y_min, y_max = df[y_param].min(), df[y_param].max()
            
            x_range = np.linspace(x_min, x_max, 30)
            y_range = np.linspace(y_min, y_max, 30)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            
            # Create prediction data with the same structure as the original design matrix
            n_points = X_grid.size
            
            if is_pandas:
                # Create a DataFrame with the same columns
                pred_data = pd.DataFrame(index=range(n_points), columns=column_names)
                
                # Fill with median values
                for col in column_names:
                    if col == 'const':
                        pred_data[col] = 1.0
                    else:
                        pred_data[col] = df[col].median() if col in df.columns else 0.0
                
                # Set the parameters we're varying
                pred_data[x_param] = X_grid.flatten()
                pred_data[y_param] = Y_grid.flatten()
                
                # If y_param is payload_size, update circuit_width accordingly
                if y_param == 'payload_size' and 'circuit_width' in pred_data.columns:
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
            else:
                # Create a numpy array with the same shape
                n_features = len(column_names)
                pred_data = np.ones((n_points, n_features))
                
                # Fill with median values
                for i, col in enumerate(column_names):
                    if col != 'const' and col in df.columns:
                        pred_data[:, i] = df[col].median()
                
                # Set the parameters we're varying
                if x_param in param_indices:
                    pred_data[:, param_indices[x_param]] = X_grid.flatten()
                
                if y_param in param_indices:
                    pred_data[:, param_indices[y_param]] = Y_grid.flatten()
                
                # If y_param is payload_size, update circuit_width accordingly
                if y_param == 'payload_size' and 'circuit_width' in param_indices:
                    # Get unique payload sizes and corresponding circuit widths
                    unique_payloads = df['payload_size'].unique()
                    width_by_payload = {ps: df[df['payload_size'] == ps]['circuit_width'].iloc[0] for ps in unique_payloads}
                    
                    # Update circuit_width based on payload_size
                    for i in range(n_points):
                        ps = pred_data[i, param_indices['payload_size']]
                        if ps in width_by_payload:
                            pred_data[i, param_indices['circuit_width']] = width_by_payload[ps]
                        else:
                            # Use a linear approximation for payload sizes not in the data
                            # This is a simple approach - could be improved
                            pred_data[i, param_indices['circuit_width']] = ps * 2 + 3  # Assuming width = 2*payload + 3
            
            # Generate predictions
            try:
                if is_log_model:
                    Z_pred_log = model.predict(pred_data)
                    Z_pred = np.exp(Z_pred_log) - 0.001
                    Z_pred = np.clip(Z_pred, 0, 1)
                else:
                    Z_pred = model.predict(pred_data)
                
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
                plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_{x_param}_{y_param}_3d.png'), 
                            dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Warning: Could not create 3D plot for {x_param} vs {y_param}: {str(e)}")
        except Exception as e:
            print(f"Error creating 3D plot for {x_param} vs {y_param}: {str(e)}")

def analyze_homoscedasticity(model, X, y, y_pred, model_name, output_dir=None):
    """
    Analyze homoscedasticity of residuals using Breusch-Pagan test and visualization.
    
    Args:
        model: Fitted statsmodels model
        X: Design matrix
        y: Target values
        y_pred: Predicted values
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save plots. If None, uses current directory.
        
    Returns:
        dict: Results of homoscedasticity tests
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Perform Breusch-Pagan test for homoscedasticity
    try:
        bp_test = het_breuschpagan(residuals, X)
        bp_lm_stat, bp_lm_pvalue, bp_fstat, bp_f_pvalue = bp_test
        
        # Perform White's test for homoscedasticity
        try:
            white_test = het_white(residuals, X)
            white_lm_stat, white_lm_pvalue, white_fstat, white_f_pvalue = white_test
        except Exception as e:
            print(f"Warning: White's test failed: {str(e)}")
            white_lm_stat, white_lm_pvalue, white_fstat, white_f_pvalue = np.nan, np.nan, np.nan, np.nan
        
        # Create visualization of residuals vs fitted values
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of residuals vs fitted values
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-')
        
        # Add a smoothed line to help visualize patterns
        try:
            from scipy.stats import binned_statistic
            bins = 20
            bin_means, bin_edges, _ = binned_statistic(y_pred, residuals, statistic='mean', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Mean residual')
        except Exception as e:
            print(f"Warning: Could not create smoothed line: {str(e)}")
        
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title(f'Homoscedasticity Analysis - {model_name}')
        
        # Add test results to the plot
        if not np.isnan(bp_lm_pvalue):
            plt.annotate(f'Breusch-Pagan test p-value: {bp_lm_pvalue:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        if not np.isnan(white_lm_pvalue):
            plt.annotate(f'White test p-value: {white_lm_pvalue:.4f}', 
                        xy=(0.05, 0.90), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add interpretation
        alpha = 0.05
        if not np.isnan(bp_lm_pvalue) and not np.isnan(white_lm_pvalue):
            if bp_lm_pvalue < alpha or white_lm_pvalue < alpha:
                interpretation = "Evidence of heteroscedasticity (p < 0.05)"
            else:
                interpretation = "No significant heteroscedasticity detected (p >= 0.05)"
        elif not np.isnan(bp_lm_pvalue):
            if bp_lm_pvalue < alpha:
                interpretation = "Evidence of heteroscedasticity (BP test, p < 0.05)"
            else:
                interpretation = "No significant heteroscedasticity detected (BP test, p >= 0.05)"
        elif not np.isnan(white_lm_pvalue):
            if white_lm_pvalue < alpha:
                interpretation = "Evidence of heteroscedasticity (White test, p < 0.05)"
            else:
                interpretation = "No significant heteroscedasticity detected (White test, p >= 0.05)"
        else:
            interpretation = "Heteroscedasticity tests inconclusive"
        
        plt.annotate(interpretation, 
                    xy=(0.05, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_homoscedasticity.png'), 
                    dpi=300, bbox_inches='tight')
        # Display in notebook
        
        # Print results
        print(f"\nHomoscedasticity Analysis for {model_name}:")
        print(f"Breusch-Pagan test:")
        print(f"  LM statistic: {bp_lm_stat:.4f}")
        print(f"  LM p-value: {bp_lm_pvalue:.4f}")
        print(f"  F statistic: {bp_fstat:.4f}")
        print(f"  F p-value: {bp_f_pvalue:.4f}")
        
        if not np.isnan(white_lm_pvalue):
            print(f"White's test:")
            print(f"  LM statistic: {white_lm_stat:.4f}")
            print(f"  LM p-value: {white_lm_pvalue:.4f}")
            print(f"  F statistic: {white_fstat:.4f}")
            print(f"  F p-value: {white_f_pvalue:.4f}")
        
        print(f"Interpretation: {interpretation}")
        
        # Create a more detailed visualization - residuals vs each predictor
        if hasattr(X, 'columns'):
            # It's a pandas DataFrame
            predictors = [col for col in X.columns if col != 'const']
            
            # Create a multi-panel figure
            n_predictors = len(predictors)
            fig, axes = plt.subplots(1, n_predictors, figsize=(5*n_predictors, 6))
            
            # If there's only one predictor, axes won't be an array
            if n_predictors == 1:
                axes = [axes]
            
            for i, predictor in enumerate(predictors):
                axes[i].scatter(X[predictor], residuals, alpha=0.7)
                axes[i].axhline(y=0, color='r', linestyle='-')
                axes[i].set_xlabel(predictor)
                axes[i].set_ylabel('Residuals')
                axes[i].set_title(f'Residuals vs {predictor}')
                axes[i].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_").replace(" ", "_")}_residuals_vs_predictors.png'), 
                        dpi=300, bbox_inches='tight')
            # Display in notebook
        
        return {
            'bp_test': bp_test,
            'white_test': (white_lm_stat, white_lm_pvalue, white_fstat, white_f_pvalue) if not np.isnan(white_lm_pvalue) else None,
            'interpretation': interpretation
        }
    
    except Exception as e:
        print(f"Warning: Homoscedasticity analysis failed: {str(e)}")
        return {
            'bp_test': None,
            'white_test': None,
            'interpretation': "Analysis failed"
        }

def analyze_and_visualize_models(df=None, output_dir=None):
    """
    Analyze and visualize the two best regression models.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results. If None, loads from consolidated CSV.
        output_dir (str): Directory to save plots. If None, uses current directory.
        
    Returns:
        tuple: (log_model, wls_model) - The fitted models
    """
    if df is None:
        df = load_data()
    
    if output_dir is None:
        output_dir = os.getcwd()
        
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
    
    # Use iloc for positional indexing to avoid FutureWarning
    if hasattr(log_params, 'iloc'):
        log_model_eq += f"{log_params.iloc[log_const_idx]:.4f} "
    else:
        log_model_eq += f"{log_params[log_const_idx]:.4f} "
    
    # Add the other parameters
    for i, param_name in enumerate(log_param_names):
        if i != log_const_idx:  # Skip the constant term
            # Use iloc for positional indexing if available
            if hasattr(log_params, 'iloc'):
                coef = log_params.iloc[i]
            else:
                coef = log_params[i]
                
            if coef >= 0:
                log_model_eq += f"+ {coef:.4e} × {param_name} "
            else:
                log_model_eq += f"- {abs(coef):.4e} × {param_name} "
    
    wls_model_eq = "success_rate = "
    
    # Use iloc for positional indexing to avoid FutureWarning
    if hasattr(wls_params, 'iloc'):
        wls_model_eq += f"{wls_params.iloc[wls_const_idx]:.4f} "
    else:
        wls_model_eq += f"{wls_params[wls_const_idx]:.4f} "
    
    # Add the other parameters
    for i, param_name in enumerate(wls_param_names):
        if i != wls_const_idx:  # Skip the constant term
            # Use iloc for positional indexing if available
            if hasattr(wls_params, 'iloc'):
                coef = wls_params.iloc[i]
            else:
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
    
    # Homoscedasticity analysis
    print("\nAnalyzing homoscedasticity...")
    homoscedasticity_log = analyze_homoscedasticity(log_model, log_X, y_log, log_model.predict(log_X),
                                                  "Model 2: Log-transformed target", output_dir)
    homoscedasticity_wls = analyze_homoscedasticity(wls_model, wls_X, y, y_pred_wls,
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
    
    # Print homoscedasticity comparison
    print("\nHomoscedasticity Comparison:")
    print(f"Model 2 (Log-transformed): {homoscedasticity_log['interpretation']}")
    print(f"Model 6 (Weighted LS): {homoscedasticity_wls['interpretation']}")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return log_model, wls_model

def main():
    """Main function to run the analysis."""
    # Set up output directory
    output_dir = 'regression_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    csv_file = 'consolidated_experiment_results.csv'
    df = load_data(csv_file)
    
    # Run analysis and create visualizations
    log_model, wls_model = analyze_and_visualize_models(df, output_dir)
    
    print("\nAnalysis completed.")

if __name__ == "__main__":
    main() 