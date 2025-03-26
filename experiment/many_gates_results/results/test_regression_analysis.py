#!/usr/bin/env python3
"""
Test script for regression analysis of quantum circuit experiment data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from experiment_analysis import create_experiment_dataframe, analyze_regression_models

def main():
    """Run regression analysis on experiment data."""
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.dirname(current_dir)
    
    # Use the consolidated CSV file
    csv_file = os.path.join(experiment_dir, 'consolidated_experiment_results.csv')
    
    if not os.path.exists(csv_file):
        print(f"Consolidated CSV file not found: {csv_file}")
        print("Please run join_csv_files.py first to create the consolidated file.")
        return
    
    print(f"Loading data from: {csv_file}")
    
    # Create dataframe from CSV file
    df = create_experiment_dataframe(csv_file)
    
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
    
    # Run regression analysis
    print("\nRunning regression analysis...")
    model_results = analyze_regression_models(df)
    
    # Additional analysis: Examine the effect of payload size
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

if __name__ == "__main__":
    main() 