# Plot Generation Guide

Quick reference for generating the key plots used in the IEEE report.

## Required Plots

### 1. Success Rate by Payload Size
**File**: `1_combined_hardware_success_rates_boxplot.png`
```bash
python 1_hardware_success_rates_analysis.py
```
- Shows success rate distributions by payload size for IBM vs Rigetti
- Boxplot format with side-by-side comparison
- Compares hardware performance across different payload sizes

### 1b. Error Rate vs Payload Size (Complementary View)
**File**: `1b_error_rate_vs_payload_size_combined_boxplot.png`
```bash
python 1b_payload_size_boxplot_analysis.py
```
- Error rate distributions by payload size for both platforms (complement to plot 1)
- Same data as plot 1, but showing error rate (1 - success rate)
- Side-by-side comparison of IBM and Rigetti
- Filtered to circuit depth ≤ 50

### 1c. Error Rate Envelope vs Payload Size
**File**: `1c_error_rate_envelope_vs_payload_size.png`
```bash
python 1c_error_rate_envelope_analysis.py
```
- Scatter plot with mean, min, and max error rates by payload size
- Shows envelope (range of variation) for error rates
- Shaded regions indicate the full range between min and max
- Useful for understanding error rate variability and bounds
- Filtered to circuit depth ≤ 50

### 2. Success Rate vs Number of Gates
**File**: `2_success_rate_vs_gates_grouped_boxplot_filtered.png`
```bash
python 2_success_rate_gates_analysis.py
```
- Success rates grouped by gate count ranges (2E2, 5E2, 1E3, etc.)
- Includes datasets up to 20k gates
- Boxplot format focusing on data distribution
- Shows how success rate changes with circuit complexity
- Filters out gate ranges with < 5 samples for cleaner visualization

### 2b. Error Rate vs Number of Gates (Complementary View)
**File**: `2b_error_rate_vs_gates_grouped_boxplot_filtered.png`
```bash
python 2b_error_rate_gates_analysis.py
```
- Error rate distributions by gate count (complement to plot 2)
- Same data as plot 2, but showing error rate (1 - success rate)
- Grouped by gate count ranges with filtering for data quality
- Shows how error rate changes with increasing gate count

### 2c. Error Rate Envelope vs Number of Gates
**File**: `2c_error_rate_envelope_vs_gates.png`
```bash
python 2c_error_rate_envelope_gates_analysis.py
```
- Scatter plot with mean, min, and max error rates by gate count
- Shows envelope (range of variation) for error rates across gate ranges
- Shaded regions indicate the full range between min and max
- Useful for understanding error rate variability and bounds with gate complexity
- Log-scale x-axis for better visualization across wide gate range
- Filters out gate ranges with < 5 samples
- **Note**: This plot conflates gate counts across different payload sizes (see plot 2d for stratified analysis)

### 2d. Error Rate vs Gates Stratified by Payload Size ⚠️ CRITICAL
**File**: `2d_error_rate_vs_gates_by_payload.png`
```bash
python 2d_error_rate_gates_by_payload_analysis.py
```
- **Critical analysis**: Separates gate count effects by payload size
- Side-by-side comparison of IBM and Rigetti
- Each payload size shown as a separate line with different color
- Reveals confounding between gate count and payload size in plot 2c
- Shows that high gate counts were only tested on small payloads
- Essential for proper interpretation of gate count effects
- Log-scale x-axis with shaded standard deviation regions

### 3. Success Rate vs Circuit Depth
**File**: `3_success_rate_vs_circuit_depth_boxplot.png`
```bash
python 3_success_rate_circuit_depth_analysis.py
```
- Success rate distributions by circuit depth (binned in 5-depth intervals)
- Side-by-side comparison of IBM and Rigetti
- Filtered to circuit depth ≤ 50
- Shows how success rate changes with circuit depth

### 3b. Error Rate vs Circuit Depth (Complementary View)
**File**: `3b_error_rate_vs_circuit_depth_boxplot.png`
```bash
python 3b_error_rate_circuit_depth_analysis.py
```
- Error rate distributions by circuit depth (complement to plot 3)
- Same data as plot 3, but showing error rate (1 - success rate)
- Side-by-side comparison of IBM and Rigetti
- Filtered to circuit depth ≤ 50

### 3c. Error Rate Envelope vs Circuit Depth
**File**: `3c_error_rate_envelope_vs_circuit_depth.png`
```bash
python 3c_error_rate_envelope_circuit_depth_analysis.py
```
- Scatter plot with mean, min, and max error rates by circuit depth
- Shows envelope (range of variation) for error rates across depth bins
- Shaded regions indicate the full range between min and max
- Useful for understanding error rate variability and bounds with circuit complexity
- Filtered to circuit depth ≤ 50
- Circuit depths binned in 5-unit intervals (5-9, 10-14, etc.)

### 3d. Error Rate vs Circuit Depth Stratified by Payload Size ⚠️ CRITICAL
**File**: `3d_error_rate_vs_depth_by_payload_ibm.png` and `3d_error_rate_vs_depth_by_payload_rigetti.png`
```bash
python 3d_error_rate_depth_by_payload_analysis.py
```
- **Critical analysis**: Separates circuit depth effects by payload size
- Two separate plots (IBM and Rigetti) for LaTeX side-by-side arrangement
- Each payload size shown as a separate line with different color
- Shows how circuit depth affects error rate within each payload category
- Filtered to circuit depth ≤ 50
- 5-depth binning with shaded standard deviation regions
- No titles (for LaTeX control)

## Quick Regeneration

To regenerate all plots:
```bash
cd experiment/ieee_analysis
python 1_hardware_success_rates_analysis.py
python 1b_payload_size_boxplot_analysis.py
python 1c_error_rate_envelope_analysis.py
python 2_success_rate_gates_analysis.py
python 2b_error_rate_gates_analysis.py
python 2c_error_rate_envelope_gates_analysis.py
python 2d_error_rate_gates_by_payload_analysis.py  # CRITICAL - stratified analysis
python 3_success_rate_circuit_depth_analysis.py
python 3b_error_rate_circuit_depth_analysis.py
python 3c_error_rate_envelope_circuit_depth_analysis.py
python 3d_error_rate_depth_by_payload_analysis.py  # CRITICAL - stratified analysis
```

All plots are saved in the `img/` directory.

## Notes

- All scripts use centralized styling from `styles.py` for consistency
- Data loading is handled by `load_data.py` module
- Scripts have numbered prefixes (1_, 1b_, 1c_, 2_, 2b_, 2c_, 2d_, 3_, 3b_, 3c_, 3d_) matching their output files
- Complementary plot sets show the same data from different perspectives:
  - Plot 1, 1b, 1c: Success rate, error rate (boxplot), and error rate (envelope) by payload size
  - Plot 2, 2b, 2c, 2d: Success rate, error rate (boxplot), error rate (envelope), and **stratified analysis** by gate count
  - Plot 3, 3b, 3c, 3d: Success rate, error rate (boxplot), error rate (envelope), and **stratified analysis** by circuit depth
- Plots 1c, 2c, and 3c use envelope visualization (signal, low, high) to show error rate bounds
- **Plots 2d and 3d are CRITICAL**: They show how error rate varies with gate count and circuit depth, respectively, when controlling for payload size
- **Plot 2d reveals confounding**: The original gate count analysis conflated high gate counts on small payloads with low gate counts on large payloads
- **Plot 3d shows depth effects**: Within each payload size, circuit depth shows clear monotonic increase in error rate
- Plots 2d and 3d generate separate files for IBM and Rigetti (no titles) for LaTeX arrangement
- Output plots are optimized for IEEE paper format

## Important Methodological Note

**Stratified analyses (2d and 3d) are essential**: They reveal how factors affect error rate within each payload size category, avoiding confounding. Plot 2d shows that gate count has modest effects (10-20% increase) within payloads. Plot 3d confirms that circuit depth is a strong predictor within all payload sizes, with clear monotonic trends. Always stratify by potential confounding variables!
