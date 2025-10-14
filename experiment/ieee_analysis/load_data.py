#!/usr/bin/env python3
"""
Data Loading Module
==================

Centralized data loading functionality for IEEE analysis modules.
Handles loading, deduplication, and preprocessing of experimental data
from both IBM and Rigetti quantum computing platforms.

Data Sources:
- IBM: 2_experiment_results_target_depth.csv, 3_experiment_results.csv
- Rigetti: experiment_results_target_depth_20250903_221822_updated.csv

Features:
- Automatic deduplication based on job IDs
- Hardware platform identification
- Data validation and preprocessing
- Consistent data format across all analysis modules

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import os
import json


def load_combined_hardware_data():
    """
    Load and combine data from ALL available IBM and AWS/Rigetti files.
    
    Loads complete dataset from:
    - IBM: 2_experiment_results_target_depth.csv, 3_experiment_results.csv, and all files from ibm/ directory
    - AWS/Rigetti: experiment_results_target_depth_20250903_221822_updated.csv and all files from aws/ directory
    
    Automatically removes duplicate entries based on job IDs:
    - IBM: removes duplicates based on 'ibm_job_id'
    - Rigetti: removes duplicates based on 'qbraid_job_id' or 'job_id'
    
    Returns:
    --------
    pandas.DataFrame
        Combined dataset with hardware identifiers and no duplicates from ALL available data sources
    """
    all_data = []
    base_dir = os.path.dirname(__file__)
    
    # Load ALL IBM files (root-level + ibm/ directory)
    import glob
    
    ibm_files = []
    
    # Root-level IBM files
    ibm_root_files = [
        '2_experiment_results_target_depth.csv',
        '3_experiment_results.csv'
    ]
    
    for filename in ibm_root_files:
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            ibm_files.append(file_path)
    
    # IBM directory files
    ibm_dir = os.path.join(base_dir, "ibm")
    if os.path.exists(ibm_dir):
        ibm_dir_files = glob.glob(os.path.join(ibm_dir, "*.csv"))
        ibm_files.extend(ibm_dir_files)
    
    print(f"Found {len(ibm_files)} IBM dataset files")
    
    # Process ALL IBM files
    ibm_dataframes = []
    for csv_file in ibm_files:
        try:
            df = pd.read_csv(csv_file)
            if 'status' in df.columns:
                df = df[df['status'] == 'completed']
            
            # Add hardware identifier
            df['hardware'] = 'IBM'
            ibm_dataframes.append(df)
            print(f"  Loaded {len(df)} records from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"  Error loading IBM file {os.path.basename(csv_file)}: {e}")
    
    # Combine all IBM data and deduplicate
    if ibm_dataframes:
        ibm_combined = pd.concat(ibm_dataframes, ignore_index=True)
        initial_count = len(ibm_combined)
        if 'ibm_job_id' in ibm_combined.columns:
            ibm_combined = ibm_combined.drop_duplicates(subset=['ibm_job_id'], keep='first')
            dedup_count = len(ibm_combined)
            if initial_count != dedup_count:
                print(f"  Removed {initial_count - dedup_count} duplicate IBM job IDs")
        all_data.append(ibm_combined)
        print(f"Total IBM records after deduplication: {len(ibm_combined)}")
    
    # Load ALL AWS/Rigetti files (root-level + aws/ directory)
    aws_files = []
    
    # Root-level AWS/Rigetti file
    aws_root_file = os.path.join(base_dir, 'experiment_results_target_depth_20250903_221822_updated.csv')
    if os.path.exists(aws_root_file):
        aws_files.append(aws_root_file)
    
    # AWS directory files
    aws_dir = os.path.join(base_dir, "aws")
    if os.path.exists(aws_dir):
        aws_dir_files = glob.glob(os.path.join(aws_dir, "*.csv"))
        aws_files.extend(aws_dir_files)
    
    print(f"Found {len(aws_files)} AWS/Rigetti dataset files")
    
    # Process ALL AWS/Rigetti files
    aws_dataframes = []
    for csv_file in aws_files:
        try:
            df = pd.read_csv(csv_file)
            if 'status' in df.columns:
                df = df[df['status'] == 'completed']
            
            # Add hardware identifier (use existing vendor column if available)
            if 'vendor' in df.columns:
                df['hardware'] = df['vendor'].str.title()  # rigetti -> Rigetti
            else:
                df['hardware'] = 'Rigetti'
            
            aws_dataframes.append(df)
            print(f"  Loaded {len(df)} records from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"  Error loading AWS file {os.path.basename(csv_file)}: {e}")
    
    # Combine all AWS/Rigetti data and deduplicate
    if aws_dataframes:
        aws_combined = pd.concat(aws_dataframes, ignore_index=True)
        initial_count = len(aws_combined)
        
        # Find appropriate job ID column for deduplication
        job_id_col = None
        if 'qbraid_job_id' in aws_combined.columns:
            job_id_col = 'qbraid_job_id'
        elif 'job_id' in aws_combined.columns:
            job_id_col = 'job_id'
        
        if job_id_col:
            aws_combined = aws_combined.drop_duplicates(subset=[job_id_col], keep='first')
            dedup_count = len(aws_combined)
            if initial_count != dedup_count:
                print(f"  Removed {initial_count - dedup_count} duplicate {job_id_col}s")
        
        all_data.append(aws_combined)
        print(f"Total AWS/Rigetti records after deduplication: {len(aws_combined)}")
    
    if not all_data:
        print("No data files found!")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Final deduplication check across all data
    initial_total = len(combined_df)
    print(f"Total records before final validation: {initial_total}")
    
    # Validate required columns
    required_cols = ['payload_size', 'success_rate', 'hardware']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        return pd.DataFrame()
    
    # Convert success_rate to percentage if it's in decimal format
    if combined_df['success_rate'].max() <= 1.0:
        combined_df['success_rate'] = combined_df['success_rate'] * 100
    
    # Filter out erroneous 100% success rates (likely data errors)
    original_count = len(combined_df)
    filtered_100_count = len(combined_df[combined_df['success_rate'] == 100.0])
    combined_df = combined_df[combined_df['success_rate'] < 100.0]
    
    if filtered_100_count > 0:
        print(f"⚠️  Filtered out {filtered_100_count} records with 100% success rate (likely errors)")
    
    print(f"\nCombined dataset: {len(combined_df)} total records (after filtering)")
    print(f"Original dataset: {original_count} records")
    print(f"Hardware platforms: {combined_df['hardware'].unique()}")
    print(f"Payload size range: {combined_df['payload_size'].min()} - {combined_df['payload_size'].max()}")
    print(f"Success rate range: {combined_df['success_rate'].min():.2f}% - {combined_df['success_rate'].max():.2f}%")
    
    return combined_df


def load_ibm_data_only():
    """
    Load only IBM data from the specified files.
    
    Returns:
    --------
    pandas.DataFrame
        IBM dataset with deduplication applied
    """
    all_data = []
    base_dir = os.path.dirname(__file__)
    
    # Specific IBM files to load
    ibm_files = [
        '2_experiment_results_target_depth.csv',
        '3_experiment_results.csv'
    ]
    
    # Process IBM files
    for filename in ibm_files:
        csv_file = os.path.join(base_dir, filename)
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if 'status' in df.columns:
                    df = df[df['status'] == 'completed']
                
                # Remove duplicates based on IBM job ID
                initial_count = len(df)
                if 'ibm_job_id' in df.columns:
                    df = df.drop_duplicates(subset=['ibm_job_id'], keep='first')
                    dedup_count = len(df)
                    if initial_count != dedup_count:
                        print(f"Removed {initial_count - dedup_count} duplicate IBM job IDs from {filename}")
                
                all_data.append(df)
                print(f"Loaded {len(df)} IBM records from {filename}")
            except Exception as e:
                print(f"Error loading IBM file {filename}: {e}")
        else:
            print(f"Warning: IBM file {filename} not found")
    
    if not all_data:
        print("No IBM data files found!")
        return pd.DataFrame()
    
    # Combine IBM data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined IBM dataset: {len(combined_df)} total records")
    
    return combined_df


def load_rigetti_data_only():
    """
    Load only Rigetti data from the specified file.
    
    Returns:
    --------
    pandas.DataFrame
        Rigetti dataset with deduplication applied
    """
    base_dir = os.path.dirname(__file__)
    aws_file = os.path.join(base_dir, 'experiment_results_target_depth_20250903_221822_updated.csv')
    
    if os.path.exists(aws_file):
        try:
            df = pd.read_csv(aws_file)
            if 'status' in df.columns:
                df = df[df['status'] == 'completed']
            
            # Remove duplicates based on QBraid job ID or job_id
            initial_count = len(df)
            job_id_col = None
            if 'qbraid_job_id' in df.columns:
                job_id_col = 'qbraid_job_id'
            elif 'job_id' in df.columns:
                job_id_col = 'job_id'
            
            if job_id_col:
                df = df.drop_duplicates(subset=[job_id_col], keep='first')
                dedup_count = len(df)
                if initial_count != dedup_count:
                    print(f"Removed {initial_count - dedup_count} duplicate {job_id_col}s from AWS file")
            
            print(f"Loaded {len(df)} Rigetti records from experiment_results_target_depth_20250903_221822_updated.csv")
            return df
        except Exception as e:
            print(f"Error loading AWS file: {e}")
    else:
        print("Warning: AWS file experiment_results_target_depth_20250903_221822_updated.csv not found")
    
    return pd.DataFrame()


def get_data_summary(df):
    """
    Get a summary of the loaded data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to summarize
        
    Returns:
    --------
    dict
        Summary statistics
    """
    if df.empty:
        return {"error": "No data available"}
    
    summary = {
        "total_records": len(df),
        "hardware_platforms": df['hardware'].unique().tolist() if 'hardware' in df.columns else [],
        "payload_size_range": {
            "min": df['payload_size'].min() if 'payload_size' in df.columns else None,
            "max": df['payload_size'].max() if 'payload_size' in df.columns else None
        },
        "success_rate_range": {
            "min": df['success_rate'].min() if 'success_rate' in df.columns else None,
            "max": df['success_rate'].max() if 'success_rate' in df.columns else None
        }
    }
    
    return summary


def create_experiment_dataframe(csv_file_path):
    """
    Load and preprocess the experiment data from a CSV file with counts processing.
    
    This function is specifically designed for circuit depth analysis where we need
    to extract detailed counts information from the quantum measurement results.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing experiment data
        
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with counts_zeros and counts_ones columns added
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Filter for completed experiments only
    if 'status' in df.columns:
        df = df[df['status'] == 'completed']
    
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


def load_circuit_depth_datasets():
    """
    Load and combine ALL available datasets from IBM and AWS directories for circuit depth analysis.
    
    This function loads all CSV files from the ibm/ and aws/ subdirectories plus root-level files
    and processes them with detailed counts extraction. Data is filtered to circuit depths <= 50 for 
    optimal visualization and focus on practical circuit depth ranges.
    
    Data Sources:
    - IBM: 2_experiment_results_target_depth.csv, 3_experiment_results.csv, and all files from ibm/ directory
    - AWS/Rigetti: experiment_results_target_depth_20250903_221822_updated.csv and all files from aws/ directory
    
    All data is filtered to circuit_depth <= 50 for optimal visualization.
    
    Returns:
    --------
    tuple: (df_ibm_combined, df_rigetti_combined)
        Two DataFrames containing the combined IBM and Rigetti datasets with counts processing,
        filtered to circuit depths <= 50 for optimal visualization
    """
    import glob
    
    base_dir = os.path.dirname(__file__)
    
    # Load all IBM datasets from ibm/ directory
    ibm_dir = os.path.join(base_dir, "ibm")
    ibm_files = glob.glob(os.path.join(ibm_dir, "*.csv"))
    
    # Load all AWS datasets from aws/ directory
    aws_dir = os.path.join(base_dir, "aws")
    aws_files = glob.glob(os.path.join(aws_dir, "*.csv"))
    
    # Also include the root-level Rigetti file for backward compatibility
    rigetti_root_file = os.path.join(base_dir, "experiment_results_target_depth_20250903_221822_updated.csv")
    if os.path.exists(rigetti_root_file):
        aws_files.append(rigetti_root_file)
    
    # Also include the root-level IBM files for backward compatibility
    ibm_root_files = [
        "2_experiment_results_target_depth.csv",
        "3_experiment_results.csv"
    ]
    for filename in ibm_root_files:
        ibm_root_file = os.path.join(base_dir, filename)
        if os.path.exists(ibm_root_file):
            ibm_files.append(ibm_root_file)
    
    print(f"Found {len(ibm_files)} IBM dataset files")
    print(f"Found {len(aws_files)} AWS/Rigetti dataset files")
    
    # Load and combine IBM datasets
    ibm_dataframes = []
    total_ibm_records = 0
    
    for file_path in ibm_files:
        if os.path.exists(file_path):
            try:
                df = create_experiment_dataframe(file_path)
                if len(df) > 0:
                    ibm_dataframes.append(df)
                    total_ibm_records += len(df)
                    print(f"  Loaded {len(df)} records from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Warning: Could not load {os.path.basename(file_path)}: {e}")
    
    # Load and combine AWS/Rigetti datasets
    rigetti_dataframes = []
    total_rigetti_records = 0
    
    for file_path in aws_files:
        if os.path.exists(file_path):
            try:
                df = create_experiment_dataframe(file_path)
                if len(df) > 0:
                    rigetti_dataframes.append(df)
                    total_rigetti_records += len(df)
                    print(f"  Loaded {len(df)} records from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Warning: Could not load {os.path.basename(file_path)}: {e}")
    
    if not ibm_dataframes:
        raise FileNotFoundError("No IBM datasets could be loaded!")
    
    if not rigetti_dataframes:
        raise FileNotFoundError("No AWS/Rigetti datasets could be loaded!")
    
    # Combine all IBM datasets
    df_ibm_combined = pd.concat(ibm_dataframes, ignore_index=True)
    print(f"\nCombined IBM datasets: {len(df_ibm_combined)} total records")
    
    # Remove duplicates based on IBM job ID
    initial_count = len(df_ibm_combined)
    if 'ibm_job_id' in df_ibm_combined.columns:
        df_ibm_combined = df_ibm_combined.drop_duplicates(subset=['ibm_job_id'], keep='first')
        dedup_count = len(df_ibm_combined)
        if initial_count != dedup_count:
            print(f"  Removed {initial_count - dedup_count} duplicate IBM job IDs")
    
    # Filter IBM data to circuit depth <= 50 for meaningful visualization
    initial_count = len(df_ibm_combined)
    df_ibm_combined = df_ibm_combined[df_ibm_combined['circuit_depth'] <= 50]
    filtered_count = len(df_ibm_combined)
    if initial_count != filtered_count:
        print(f"  Filtered to circuit depth <= 50: {filtered_count} records (removed {initial_count - filtered_count})")
    
    # Combine all AWS/Rigetti datasets
    df_rigetti_combined = pd.concat(rigetti_dataframes, ignore_index=True)
    print(f"Combined AWS/Rigetti datasets: {len(df_rigetti_combined)} total records")
    
    # Remove duplicates based on job ID
    initial_count = len(df_rigetti_combined)
    job_id_col = None
    if 'qbraid_job_id' in df_rigetti_combined.columns:
        job_id_col = 'qbraid_job_id'
    elif 'job_id' in df_rigetti_combined.columns:
        job_id_col = 'job_id'
    
    if job_id_col:
        df_rigetti_combined = df_rigetti_combined.drop_duplicates(subset=[job_id_col], keep='first')
        dedup_count = len(df_rigetti_combined)
        if initial_count != dedup_count:
            print(f"  Removed {initial_count - dedup_count} duplicate {job_id_col}s")
    
    # Filter Rigetti data to circuit depth <= 50 for meaningful visualization
    initial_count = len(df_rigetti_combined)
    df_rigetti_combined = df_rigetti_combined[df_rigetti_combined['circuit_depth'] <= 50]
    filtered_count = len(df_rigetti_combined)
    if initial_count != filtered_count:
        print(f"  Filtered to circuit depth <= 50: {filtered_count} records (removed {initial_count - filtered_count})")
    
    # Print summary statistics
    print(f"\nFinal IBM dataset:")
    print(f"  Total records: {len(df_ibm_combined)}")
    if len(df_ibm_combined) > 0:
        print(f"  Circuit depths: {sorted(df_ibm_combined['circuit_depth'].unique())}")
        print(f"  Payload sizes: {sorted(df_ibm_combined['payload_size'].unique())}")
    
    print(f"\nFinal AWS/Rigetti dataset:")
    print(f"  Total records: {len(df_rigetti_combined)}")
    if len(df_rigetti_combined) > 0:
        print(f"  Circuit depths: {sorted(df_rigetti_combined['circuit_depth'].unique())}")
        print(f"  Payload sizes: {sorted(df_rigetti_combined['payload_size'].unique())}")
    
    return df_ibm_combined, df_rigetti_combined


def load_optimization_experiment_data(filename: str = None):
    """
    Load optimization experiment data from CSV file.
    
    This function loads data from the circuit optimization experiment that measures
    the impact of ISA transpilation on circuit depth, width, and size.
    
    Parameters:
    -----------
    filename : str, optional
        Specific filename to load. If None, loads the most recent file matching
        the pattern '5_optimization_experiment_results_*.csv'
        
    Returns:
    --------
    pandas.DataFrame
        Optimization experiment data with columns:
        - timestamp, payload_size, num_gates, iteration
        - original_depth, original_width, original_size
        - isa_depth, isa_width, isa_size
        - depth_reduction, depth_reduction_percent
        - width_change, width_change_percent
        - size_change, size_change_percent
        - backend_name
    """
    base_dir = os.path.dirname(__file__)
    
    if filename is None:
        # Find the most recent optimization experiment file
        pattern = os.path.join(base_dir, "5_optimization_experiment_results_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(
                f"No optimization experiment results found matching pattern: "
                f"5_optimization_experiment_results_*.csv"
            )
        
        # Sort by modification time and get the most recent
        filename = max(files, key=os.path.getmtime)
        print(f"Loading most recent optimization experiment file: {os.path.basename(filename)}")
    else:
        filename = os.path.join(base_dir, filename)
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Optimization experiment file not found: {filename}")
    
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} optimization experiment records")
    
    # Print summary
    if len(df) > 0:
        print(f"  Payload sizes: {sorted(df['payload_size'].unique())}")
        print(f"  Gate ranges: {sorted(df['num_gates'].unique())}")
        print(f"  Mean depth reduction: {df['depth_reduction_percent'].mean():.2f}%")
        print(f"  Mean width change: {df['width_change_percent'].mean():.2f}%")
    
    return df


if __name__ == "__main__":
    # Test the data loading functions
    print("Testing data loading functions...")
    print("=" * 50)
    
    # Test combined data loading
    combined_data = load_combined_hardware_data()
    print(f"\nCombined data summary: {get_data_summary(combined_data)}")
    
    # Test IBM only
    print("\n" + "=" * 50)
    ibm_data = load_ibm_data_only()
    print(f"IBM data summary: {get_data_summary(ibm_data)}")
    
    # Test Rigetti only
    print("\n" + "=" * 50)
    rigetti_data = load_rigetti_data_only()
    print(f"Rigetti data summary: {get_data_summary(rigetti_data)}")
    
    # Test circuit depth specific loading
    print("\n" + "=" * 50)
    print("Testing circuit depth data loading...")
    try:
        ibm_circuit, rigetti_circuit = load_circuit_depth_datasets()
        print(f"Circuit depth IBM data: {len(ibm_circuit)} records")
        print(f"Circuit depth Rigetti data: {len(rigetti_circuit)} records")
    except Exception as e:
        print(f"Circuit depth loading failed: {e}")
    
    # Test optimization experiment loading
    print("\n" + "=" * 50)
    print("Testing optimization experiment data loading...")
    try:
        opt_data = load_optimization_experiment_data()
        print(f"Optimization experiment data: {len(opt_data)} records")
    except FileNotFoundError as e:
        print(f"Optimization experiment loading skipped: {e}")
