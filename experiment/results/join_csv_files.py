#!/usr/bin/env python3
"""
Script to join multiple experiment CSV files into a single consolidated file.
"""

import os
import pandas as pd
import glob

def join_csv_files(csv_files, output_file):
    """
    Join multiple CSV files into a single consolidated file.
    
    Args:
        csv_files (list): List of CSV file paths to join
        output_file (str): Path to save the consolidated CSV file
    
    Returns:
        pd.DataFrame: The consolidated DataFrame
    """
    # List to store individual dataframes
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        if os.path.exists(file):
            print(f"Reading {os.path.basename(file)}...")
            df = pd.read_csv(file)
            print(f"  - Found {len(df)} rows")
            dfs.append(df)
        else:
            print(f"Warning: File not found: {file}")
    
    if not dfs:
        print("No valid CSV files found.")
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined DataFrame has {len(combined_df)} rows")
    
    # Save to output file
    combined_df.to_csv(output_file, index=False)
    print(f"Saved consolidated data to {output_file}")
    
    return combined_df

def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.dirname(current_dir)
    
    # List of CSV files to join
    csv_files = [
        os.path.join(experiment_dir, "experiment_results_dynamic_1-5_200-205_20250304_113951.csv"),
        os.path.join(experiment_dir, "experiment_results_dynamic_1-4_500-505_20250304_114048.csv"),
        os.path.join(experiment_dir, "experiment_results_dynamic_1-4_1000-1005_20250304_114148.csv"),
        os.path.join(experiment_dir, "experiment_results_dynamic_1-4_1500-1505_20250304_114253.csv"),
        os.path.join(experiment_dir, "experiment_results_dynamic_1-4_2000-2005_20250304_114351.csv"),
        os.path.join(experiment_dir, "experiment_results_dynamic_1-4_3000-3005_20250304_001617_updated.csv"),
        os.path.join(experiment_dir, "experiment_results_dynamic_1-4_5000-5005_20250306_131017_updated.csv"),
        os.path.join(experiment_dir, "experiment_results_dynamic_1-2_10000-10005_20250306_234959_updated.csv")
    ]
    
    # Output file path
    output_file = os.path.join(experiment_dir, "consolidated_experiment_results.csv")
    
    # Join the CSV files
    combined_df = join_csv_files(csv_files, output_file)
    
    if combined_df is not None:
        # Print some statistics about the combined data
        print("\nDataset Overview:")
        print(f"Number of experiments: {len(combined_df)}")
        
        # Print distribution by payload size
        print("\nDistribution by payload size:")
        payload_counts = combined_df['payload_size'].value_counts().sort_index()
        for size, count in payload_counts.items():
            print(f"Payload size {size}: {count} experiments")
        
        # Print distribution by gate count range
        print("\nDistribution by gate count range:")
        gate_ranges = [(200, 205), (500, 505), (1000, 1005), (1500, 1505), 
                      (2000, 2005), (3000, 3005), (5000, 5005), (10000, 10005)]
        
        for start, end in gate_ranges:
            count = len(combined_df[(combined_df['num_gates'] >= start) & (combined_df['num_gates'] < end)])
            print(f"Gates {start}-{end-1}: {count} experiments")

if __name__ == "__main__":
    main() 