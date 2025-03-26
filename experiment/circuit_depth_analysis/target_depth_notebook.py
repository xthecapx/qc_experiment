# Target Depth Analysis Notebook
# Copy this code into a Jupyter notebook to run the analysis

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Output directory for all generated files
OUTPUT_DIR = "circuit_depth_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add the results directory to the path if needed
sys.path.append('results')

# Import the analysis functions
from results.target_depth_analysis import (
    create_experiment_dataframe,
    plot_success_rate_vs_circuit_depth,
    plot_error_distribution_by_circuit_depth,
    plot_success_rate_heatmap,
    plot_error_distribution_by_payload_size,
    analyze_regression_models
)

# Path to the CSV file
csv_file_path = "experiment_results_target_depth_20250309_170751_updated.csv"

# Load and preprocess the data
print(f"Loading data from {csv_file_path}...")
df = create_experiment_dataframe(csv_file_path)

print(f"Loaded {len(df)} experiment results.")
print(f"Circuit depths: {sorted(df['circuit_depth'].unique())}")
print(f"Payload sizes: {sorted(df['payload_size'].unique())}")

# Generate the plots
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

# Run the regression analysis
# Note: This may take some time to complete
print("\nPerforming regression analysis...")
model_results = analyze_regression_models(df)

print("\nAnalysis complete. All plots have been saved to the circuit_depth_analysis folder.")

# Display the plots in the notebook
from IPython.display import Image, display

print("\nDisplaying generated plots:")

# Function to safely display images
def display_image(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        display(Image(filepath))
    else:
        print(f"Warning: Image file not found: {filepath}")

print("\nSuccess Rate vs Circuit Depth:")
display_image("success_rate_vs_circuit_depth.png")

print("\nError Distribution by Circuit Depth:")
display_image("error_distribution_by_circuit_depth.png")

print("\nSuccess Rate Heatmap:")
display_image("success_rate_heatmap.png")

print("\nError Distribution by Payload Size:")
display_image("error_distribution_by_payload_size.png")

print("\nCorrelation Matrix:")
display_image("target_depth_correlation_matrix.png")

# Display regression model plots - find the actual filenames
print("\nRegression Model Diagnostics:")

# Find all diagnostic plot files
diagnostic_files = glob.glob(os.path.join(OUTPUT_DIR, "*_residual_diagnostics.png"))
actual_vs_pred_files = glob.glob(os.path.join(OUTPUT_DIR, "*_actual_vs_predicted.png"))

# Display them
for file in sorted(diagnostic_files):
    print(f"\nDiagnostic plot: {os.path.basename(file)}")
    display(Image(file))

for file in sorted(actual_vs_pred_files):
    print(f"\nActual vs Predicted plot: {os.path.basename(file)}")
    display(Image(file))

# You can also run individual parts of the analysis if needed:

# For just the success rate vs circuit depth plot
# regression_stats = plot_success_rate_vs_circuit_depth(df)

# For just the error distribution by circuit depth
# plot_error_distribution_by_circuit_depth(df)

# For just the success rate heatmap
# plot_success_rate_heatmap(df)

# For just the error distribution by payload size
# plot_error_distribution_by_payload_size(df)

# For just the regression analysis
# model_results = analyze_regression_models(df) 