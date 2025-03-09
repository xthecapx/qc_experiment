"""
Example usage of regression analysis functions in a Jupyter notebook.

Copy and paste this code into a new cell after importing the regression analysis functions.
"""

# Load the data
df = load_data('consolidated_experiment_results.csv')

# Print basic statistics
print("\nDataset Overview:")
print(f"Number of experiments: {len(df)}")
print("\nFeature statistics:")
print(df[['circuit_depth', 'circuit_size', 'circuit_width', 'payload_size', 'success_rate']].describe())

# Check correlations
print("\nCorrelation matrix:")
corr_matrix = df[['circuit_depth', 'circuit_size', 'circuit_width', 'payload_size', 'success_rate']].corr()
print(corr_matrix)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(corr_matrix, fignum=1)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(i, j, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Run the analysis and create visualizations
log_model, wls_model = analyze_and_visualize_models(df)

# Print the model equations again for easy reference
print("\nModel Equations:")

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
if hasattr(log_model, 'model') and hasattr(log_model.model, 'exog_names'):
    log_param_names = log_model.model.exog_names

if hasattr(wls_model, 'model') and hasattr(wls_model.model, 'exog_names'):
    wls_param_names = wls_model.model.exog_names

# Print log model equation
print(f"Model 2 (Log-transformed): log(success_rate + 0.001) = {log_params[log_const_idx]:.4f}", end="")
for i, param_name in enumerate(log_param_names):
    if i != log_const_idx:  # Skip the constant term
        coef = log_params[i]
        if coef >= 0:
            print(f" + {coef:.4e} × {param_name}", end="")
        else:
            print(f" - {abs(coef):.4e} × {param_name}", end="")
print()

# Print WLS model equation
print(f"Model 6 (Weighted LS): success_rate = {wls_params[wls_const_idx]:.4f}", end="")
for i, param_name in enumerate(wls_param_names):
    if i != wls_const_idx:  # Skip the constant term
        coef = wls_params[i]
        if coef >= 0:
            print(f" + {coef:.4e} × {param_name}", end="")
        else:
            print(f" - {abs(coef):.4e} × {param_name}", end="")
print()

# Analyze effect of payload size
print("\nAnalyzing effect of payload size...")
for payload_size in sorted(df['payload_size'].unique()):
    subset = df[df['payload_size'] == payload_size]
    print(f"\nPayload size {payload_size} (n={len(subset)}):")
    
    # Calculate average success rate
    avg_success = subset['success_rate'].mean()
    print(f"Average success rate: {avg_success:.4f}")
    
    # Calculate correlation with circuit parameters for this payload size
    corr = subset[['circuit_depth', 'circuit_size', 'circuit_width', 'success_rate']].corr()
    print("Correlations with success_rate:")
    for col in ['circuit_depth', 'circuit_size', 'circuit_width']:
        print(f"  {col}: {corr.loc[col, 'success_rate']:.4f}")

print("\nRegression analysis completed.") 