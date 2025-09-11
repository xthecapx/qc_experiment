# Quantum Circuit Execution Results on IBM Quantum Hardware - IEEE DataPort Dataset

## Dataset Summary

### Overview
This dataset contains comprehensive experimental results from quantum circuit executions performed on IBM Quantum hardware, specifically targeting the evaluation of quantum teleportation protocol performance under varying circuit complexities and payload sizes.

### Dataset Details

**File:** `3_experiment_results.csv`  
**Total Records:** 292 experimental runs  
**Data Collection Period:** May 1, 2025  
**Quantum Backend:** IBM Sherbrooke quantum processor  

### Experimental Design

The dataset encompasses two distinct experimental protocols:

1. **Fixed Payload Experiments** (161 records)
   - Systematic evaluation of quantum circuits with payload sizes ranging from 1-10 qubits
   - Circuit depths varying from 11-43 layers
   - Focus on understanding payload size impact on teleportation fidelity

2. **Target Depth Experiments** (131 records)
   - Controlled studies targeting specific circuit depths
   - Evaluation of depth-dependent quantum coherence effects
   - Circuit complexity analysis with varying gate compositions

### Key Measured Parameters

**Circuit Characteristics:**
- `circuit_depth`: Number of quantum gate layers (range: 8-43)
- `circuit_width`: Number of qubits used (range: 4-13)
- `circuit_size`: Total number of operations (range: 13-181)
- `payload_size`: Number of qubits in teleported state (range: 1-5)
- `circuit_count_ops`: Detailed gate composition (H, CNOT, X, Y, Z, S, S†, CZ gates)

**Performance Metrics:**
- `success_rate`: Quantum teleportation fidelity (range: 0.024-0.874)
- `job_execution_duration`: Hardware execution time
- `counts`: Complete measurement outcome histograms

**Hardware Metadata:**
- IBM job IDs for result reproducibility
- Execution timestamps and durations
- Backend-specific calibration flags

### Analysis Methodology

The dataset supports comprehensive analysis approaches including:

- **Data Processing:** CSV loading with JSON parsing for gate compositions and measurement outcomes
- **Visualization Capabilities:** 
  - Success rate vs payload size regression analysis
  - Error distribution analysis and heatmaps
  - Multi-dimensional performance mapping
- **Statistical Analysis Options:** 
  - Multiple regression model formulations
  - Residual diagnostics and model validation
  - Feature selection and multicollinearity analysis
- **Research Applications:** Quantum error scaling, circuit optimization, and hardware characterization

### Statistical Coverage

**Regression Models Implemented:**
1. Linear baseline model (all features)
2. Log-transformed target variable model
3. Interaction terms model (payload×depth)
4. Quadratic feature model
5. Feature selection model (stepwise)
6. Weighted least squares model

**Diagnostic Tests:**
- Breusch-Pagan and White tests for heteroscedasticity
- Jarque-Bera normality tests
- Variance Inflation Factor analysis
- Residual autocorrelation assessment

### Research Applications

This dataset enables investigation of:
- **Quantum Error Scaling:** How decoherence affects multi-qubit teleportation
- **Circuit Optimization:** Identifying optimal depth-payload trade-offs
- **Hardware Characterization:** IBM quantum processor performance profiling
- **Protocol Validation:** Quantum communication protocol benchmarking

### Data Quality and Reproducibility

- **Complete Provenance:** All experiments include IBM job IDs for independent verification
- **Hardware Timestamps:** Precise execution timing for temporal analysis
- **Error Handling:** All experiments completed successfully with full measurement data
- **Statistical Rigor:** Multiple validation approaches and diagnostic testing

### Technical Specifications

**Data Format:** CSV with JSON-encoded measurement dictionaries  
**Analysis Environment:** Python 3.x with pandas, numpy, matplotlib, statsmodels, seaborn  
**Hardware Platform:** IBM Quantum Network, Sherbrooke backend  
**Measurement Protocol:** Full computational basis measurement with 4096 shots per circuit  

---

## Detailed Instructions for Dataset Utilization

### Prerequisites

**Required Software:**
- Python 3.7 or higher
- Required Python packages:
  ```bash
  pip install pandas numpy matplotlib seaborn statsmodels scipy
  ```

**Package Versions:**
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
statsmodels>=0.12.0
scipy>=1.7.0
```

### Dataset Structure

The dataset is provided as `3_experiment_results.csv` with the following key columns:

**Primary Metrics:**
- `success_rate`: Quantum teleportation fidelity (0.0-1.0)
- `payload_size`: Number of qubits teleported (1-5)
- `circuit_depth`: Number of gate layers (8-43)
- `circuit_width`: Total qubits used (4-13)
- `circuit_size`: Total operations count (13-181)

**Circuit Composition:**
- `circuit_count_ops`: JSON dictionary of gate counts (H, CNOT, X, Y, Z, S, S†, CZ)
- `counts`: JSON dictionary of measurement outcomes

**Experimental Context:**
- `experiment_type`: "fixed_payload_experiments" or "target_depth_experiment"
- `execution_type`: "ibm" (hardware execution)
- `ibm_backend`: Quantum processor used ("ibm_sherbrooke")

### Quick Start Guide

#### 1. Basic Data Loading and Exploration

```python
import pandas as pd
import json

# Load the dataset
df = pd.read_csv('3_experiment_results.csv')

# Basic exploration
print(f"Dataset contains {len(df)} experiments")
print(f"Payload sizes: {sorted(df['payload_size'].unique())}")
print(f"Circuit depths: {df['circuit_depth'].min()}-{df['circuit_depth'].max()}")
print(f"Success rate range: {df['success_rate'].min():.3f}-{df['success_rate'].max():.3f}")

# Parse gate composition
df['gates_dict'] = df['circuit_count_ops'].apply(lambda x: json.loads(x.replace("'", '"')))
```

#### 2. Data Preprocessing

```python
import json

# Parse JSON columns for analysis
df['gates_dict'] = df['circuit_count_ops'].apply(lambda x: json.loads(x.replace("'", '"')))
df['counts_dict'] = df['counts'].apply(lambda x: json.loads(x.replace("'", '"')))

# Extract gate counts for analysis
gate_types = ['h', 'cx', 'x', 'y', 'z', 's', 'sdg', 'cz']
for gate in gate_types:
    df[f'{gate}_count'] = df['gates_dict'].apply(lambda x: x.get(gate, 0))

# Calculate derived metrics
df['error_rate'] = 1 - df['success_rate']
df['complexity_score'] = df['circuit_depth'] * df['payload_size']
```

#### 3. Custom Analysis Examples

**Success Rate Analysis by Payload Size:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Group by payload size
payload_analysis = df.groupby('payload_size').agg({
    'success_rate': ['mean', 'std', 'count'],
    'circuit_depth': ['mean', 'std']
}).round(4)

print(payload_analysis)

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='payload_size', y='success_rate')
plt.title('Success Rate Distribution by Payload Size')
plt.ylabel('Teleportation Fidelity')
plt.xlabel('Payload Size (qubits)')
plt.show()
```

**Circuit Complexity vs Performance:**
```python
# Create complexity metric
df['complexity_score'] = df['circuit_depth'] * df['payload_size']

# Correlation analysis
correlation_matrix = df[['success_rate', 'payload_size', 'circuit_depth', 
                        'circuit_width', 'complexity_score']].corr()
print(correlation_matrix)

# Scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['complexity_score'], df['success_rate'], 
                     c=df['payload_size'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Payload Size')
plt.xlabel('Circuit Complexity Score')
plt.ylabel('Success Rate')
plt.title('Performance vs Circuit Complexity')
plt.show()
```

**Gate Composition Analysis:**
```python
# Extract gate counts
gate_types = ['h', 'cx', 'x', 'y', 'z', 's', 'sdg', 'cz']
for gate in gate_types:
    df[f'{gate}_count'] = df['gates_dict'].apply(lambda x: x.get(gate, 0))

# Gate usage correlation with success rate
gate_correlations = df[[f'{gate}_count' for gate in gate_types] + ['success_rate']].corr()['success_rate'].sort_values()
print("Gate type correlations with success rate:")
print(gate_correlations)
```

### Advanced Usage

#### 1. Statistical Modeling

```python
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Prepare features
features = ['payload_size', 'circuit_depth', 'circuit_width']
X = df[features]
y = df['success_rate']

# Add interaction terms
X['payload_depth_interaction'] = X['payload_size'] * X['circuit_depth']

# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Fit regression model
X_with_const = sm.add_constant(X_scaled)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
```

#### 2. Time Series Analysis

```python
# Convert timestamps
df['job_created'] = pd.to_datetime(df['job_created'])
df['execution_hour'] = df['job_created'].dt.hour

# Temporal performance analysis
temporal_stats = df.groupby('execution_hour')['success_rate'].agg(['mean', 'count'])
print("Performance by execution time:")
print(temporal_stats)
```

#### 3. Filtering and Subsetting

```python
# Filter by experiment type
fixed_payload_data = df[df['experiment_type'] == 'fixed_payload_experiments']
target_depth_data = df[df['experiment_type'] == 'target_depth_experiment']

# Filter by performance threshold
high_performance = df[df['success_rate'] > 0.7]
print(f"High-performance circuits: {len(high_performance)} out of {len(df)}")

# Filter by payload size
single_qubit = df[df['payload_size'] == 1]
multi_qubit = df[df['payload_size'] > 1]
```

### Recommended Analysis Outputs

The dataset structure enables generation of various analytical outputs:

**Visualizations:**
- Success rate vs payload size relationships
- 2D performance heatmaps by circuit parameters
- Error distribution analysis by experimental conditions
- Hardware execution timing patterns

**Statistical Results:**
- Regression model formulations and coefficients
- Feature correlation matrices
- Model diagnostic plots and residual analysis

### Research Applications

#### 1. Quantum Error Analysis
```python
# Calculate error rates
df['error_rate'] = 1 - df['success_rate']

# Analyze error scaling
error_scaling = df.groupby('payload_size')['error_rate'].mean()
print("Error scaling with payload size:")
print(error_scaling)
```

#### 2. Hardware Characterization
```python
# Job execution analysis
execution_stats = df.groupby('payload_size').agg({
    'job_execution_duration': ['mean', 'std'],
    'job_quantum_duration': ['mean', 'std']
})
print("Hardware execution statistics:")
print(execution_stats)
```

#### 3. Protocol Optimization
```python
# Find optimal operating points
optimal_circuits = df.loc[df.groupby('payload_size')['success_rate'].idxmax()]
print("Optimal circuits by payload size:")
print(optimal_circuits[['payload_size', 'circuit_depth', 'success_rate']])
```

### Troubleshooting

**Common Issues:**

1. **JSON Parsing Errors**: The `counts` and `circuit_count_ops` columns contain JSON strings that may need quote replacement:
   ```python
   df['counts_dict'] = df['counts'].apply(lambda x: json.loads(x.replace("'", '"')))
   ```

2. **Memory Issues**: For large-scale analysis, process data in chunks:
   ```python
   chunk_size = 50
   for i in range(0, len(df), chunk_size):
       chunk = df.iloc[i:i+chunk_size]
       # Process chunk
   ```

3. **Plot Display**: If plots don't display, ensure proper backend:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For non-interactive environments
   ```

### Citation and Reproducibility

Each experiment includes an `ibm_job_id` for independent verification on IBM Quantum systems. The exact hardware configuration and execution timestamps are preserved for full reproducibility.

**Example verification:**
```python
# Extract job IDs for specific experiments
job_ids = df[df['success_rate'] > 0.8]['ibm_job_id'].tolist()
print(f"High-performance job IDs: {job_ids[:5]}")
```

### Analysis Implementation Examples

**Complete analysis workflow:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load and process data
df = pd.read_csv('3_experiment_results.csv')
df['gates_dict'] = df['circuit_count_ops'].apply(lambda x: json.loads(x.replace("'", '"')))
df['error_rate'] = 1 - df['success_rate']

# Generate key visualizations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='payload_size', y='success_rate', hue='circuit_depth')
plt.title('Success Rate vs Payload Size by Circuit Depth')
plt.show()

# Statistical modeling
X = df[['payload_size', 'circuit_depth', 'circuit_width']]
y = df['success_rate']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
```

### Expected Analysis Results

When analyzing this dataset, researchers should observe:

- **Performance Trends:** Success rates generally decrease with increasing payload size and circuit depth
- **Optimal Regions:** Single-qubit teleportation achieves highest fidelities (up to 87.4%)
- **Error Scaling:** Exponential degradation patterns consistent with quantum decoherence theory
- **Hardware Characteristics:** Consistent execution timing patterns indicating stable quantum processor operation

This comprehensive dataset enables research into quantum communication protocols, error characterization, hardware performance optimization, and quantum algorithm benchmarking.

---

## Dataset Files Included

1. **3_experiment_results.csv** - Primary dataset (292 experiments)
2. **IEEE_DataPort_Dataset_Documentation.md** - This documentation file

## Contact and Support

This dataset supports reproducible quantum computing research. All experimental data includes full provenance information for independent verification on IBM Quantum systems.