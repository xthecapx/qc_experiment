#!/usr/bin/env python3
"""
Filtered Success Rate Correlation vs Number of Gates Analysis
=============================================================

This module analyses the correlation between the number of gates and
success rate, focusing on lower gate counts (≤ 70 gates) where a
negative correlation is visually apparent. It generates a focused plot
with correlation statistics (Pearson, Spearman, R², slope, p-value) for
both IBM and Rigetti datasets.

Author: Analysis Script
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from load_data import load_combined_hardware_data
from styles import (
    COLORBREWER_PALETTE,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    FIG_SIZE,
)

# Output directory
OUTPUT_DIR = "img"


def format_gate_count_scientific(gate_count: int) -> str:
    """Format gate count in scientific notation-like labels."""
    if gate_count >= 10000:
        return f"{gate_count // 1000}E3"
    if gate_count >= 1000:
        mapping = {
            1010: "1E3",
            1510: "1.5E3",
            3010: "3E3",
            5010: "5E3",
            7010: "7E3",
        }
        return mapping.get(gate_count, f"{gate_count // 1000}E3")
    if gate_count >= 100:
        mapping = {
            210: "2E2",
            510: "5E2",
        }
        return mapping.get(gate_count, f"{gate_count // 100}E2")
    return str(gate_count)


def group_gates_by_range(df: pd.DataFrame, range_size: int = 10) -> pd.DataFrame:
    """Group gates into ranges for plotting/aggregation."""
    df = df.copy()
    df["gate_group"] = ((df["num_gates"] // (range_size * 2)) * (range_size * 2)) + range_size
    df["gate_group_label"] = df["gate_group"].apply(format_gate_count_scientific)
    return df


def calculate_correlation_stats(depths, success_rates):
    """Calculate correlation statistics between gates and success rate."""
    pearson_r, pearson_p = stats.pearsonr(depths, success_rates)
    spearman_r, spearman_p = stats.spearmanr(depths, success_rates)
    slope, intercept, r_value, p_value, _ = stats.linregress(depths, success_rates)
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "regression_p": p_value,
    }


def format_correlation_stats(stats_dict, platform_name: str) -> str:
    """Format correlation statistics for printing."""
    lines = [
        f"\n{platform_name} Correlation Statistics:",
        f"  Pearson r = {stats_dict['pearson_r']:.4f} (p = {stats_dict['pearson_p']:.4e})",
        f"  Spearman ρ = {stats_dict['spearman_r']:.4f} (p = {stats_dict['spearman_p']:.4e})",
        "  Linear Regression:",
        f"    R² = {stats_dict['r_squared']:.4f}",
        f"    Slope = {stats_dict['slope']:.4f}",
        f"    p-value = {stats_dict['regression_p']:.4e}",
    ]
    if stats_dict["pearson_p"] < 0.001:
        lines.append("  → Highly significant correlation (p < 0.001)")
    elif stats_dict["pearson_p"] < 0.01:
        lines.append("  → Very significant correlation (p < 0.01)")
    elif stats_dict["pearson_p"] < 0.05:
        lines.append("  → Significant correlation (p < 0.05)")
    else:
        lines.append("  → Not statistically significant (p ≥ 0.05)")
    return "\n".join(lines)


def plot_filtered_success_rate_correlation(df, bin_size: int = 10):
    """Plot filtered success rate correlation for lower gate counts."""
    # Set styles
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE

    # Filter lower gate counts (≤ 70 gates)
    df_filtered = df[df["num_gates"] <= 70].copy()
    if df_filtered.empty:
        raise ValueError("Filtered dataset has no rows (num_gates <= 70).")

    # Convert success_rate (%) to 0-1 scale for correlation
    df_filtered["success_rate_fraction"] = df_filtered["success_rate"] / 100.0

    print("\nFiltered datasets (≤ 70 gates):")
    for platform in sorted(df_filtered["hardware"].unique()):
        count = len(df_filtered[df_filtered["hardware"] == platform])
        print(f"  {platform}: {count} experiments")

    # Split by platform
    df_ibm = df_filtered[df_filtered["hardware"] == "IBM"].copy()
    df_rigetti = df_filtered[df_filtered["hardware"] == "Rigetti"].copy()

    # Correlation statistics
    ibm_stats = calculate_correlation_stats(df_ibm["num_gates"], df_ibm["success_rate_fraction"])
    rigetti_stats = calculate_correlation_stats(df_rigetti["num_gates"], df_rigetti["success_rate_fraction"])

    print(format_correlation_stats(ibm_stats, "IBM"))
    print(format_correlation_stats(rigetti_stats, "Rigetti"))

    # Group for plotting
    df_grouped = group_gates_by_range(df_filtered, range_size=bin_size)

    # Prepare plot
    fig, ax = plt.subplots(figsize=fig_size)

    for platform, color, marker in [
        ("IBM", COLORBREWER_PALETTE["IBM"], "o"),
        ("Rigetti", COLORBREWER_PALETTE["Rigetti"], "s"),
    ]:
        df_platform = df_grouped[df_grouped["hardware"] == platform]
        gate_groups = sorted(df_platform["gate_group"].unique())
        means = []
        mins = []
        maxs = []
        valid_groups = []

        for gate_group in gate_groups:
            subset = df_platform[df_platform["gate_group"] == gate_group]
            if subset.empty:
                continue
            success_frac = subset["success_rate"] / 100.0
            means.append(success_frac.mean())
            mins.append(success_frac.min())
            maxs.append(success_frac.max())
            valid_groups.append(gate_group)

        if not valid_groups:
            continue

        valid_groups = np.array(valid_groups)
        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)

        ax.plot(valid_groups, means, color=color, linewidth=2.5, marker=marker, markersize=10, label=platform)
        ax.plot(valid_groups, mins, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.plot(valid_groups, maxs, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.fill_between(valid_groups, mins, maxs, color=color, alpha=0.2)

    # Labels and styling
    ax.set_xlabel("Number of Gates", fontsize=label_size, fontweight="bold")
    ax.set_ylabel("Mean Success Rate", fontsize=label_size, fontweight="bold")

    tick_labels = [format_gate_count_scientific(g) for g in sorted(df_grouped["gate_group"].unique())]
    ax.set_xticks(sorted(df_grouped["gate_group"].unique()))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=tick_size)

    ax.set_ylim(-0.1, 1.05)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.grid(True, linestyle="--", alpha=0.3, axis="both")

    # Add correlation statistics box
    rigetti_p_text = "< 0.001" if rigetti_stats["pearson_p"] < 0.001 else f"= {rigetti_stats['pearson_p']:.3f}"
    stats_text = (
        f"IBM: r = {ibm_stats['pearson_r']:.3f}, R² = {ibm_stats['r_squared']:.3f}, p < 0.001\n"
        f"Rigetti: r = {rigetti_stats['pearson_r']:.3f}, R² = {rigetti_stats['r_squared']:.3f}, p {rigetti_p_text}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=legend_size - 2,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        family="monospace",
    )

    ax.legend(fontsize=legend_size, loc="upper right", framealpha=0.95)
    plt.tight_layout()

    return plt, ibm_stats, rigetti_stats


def run_filtered_gates_correlation_analysis():
    """Run the filtered success rate correlation analysis for number of gates."""
    print("=" * 60)
    print("FILTERED SUCCESS RATE CORRELATION vs NUMBER OF GATES")
    print("=" * 60)

    combined_df = load_combined_hardware_data()
    if combined_df.empty:
        print("No data available for analysis.")
        return

    print("\n" + "=" * 40)
    print("FULL DATASET STATISTICS")
    print("=" * 40)
    for platform in sorted(combined_df["hardware"].unique()):
        df_platform = combined_df[combined_df["hardware"] == platform]
        print(f"\n{platform} Dataset:")
        print(f"  Total experiments: {len(df_platform)}")
        print(f"  Gate range: {df_platform['num_gates'].min()} - {df_platform['num_gates'].max()}")

    print("\n" + "=" * 40)
    print("GENERATING FILTERED CORRELATION PLOT")
    print("=" * 40)

    plt_obj, ibm_stats, rigetti_stats = plot_filtered_success_rate_correlation(combined_df, bin_size=10)

    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "2b_filtered_success_rate_correlation_gates.png")
    plt_obj.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"\n✓ Saved: {output_path}")
    print("Analysis complete!")
    return {
        "IBM": ibm_stats,
        "Rigetti": rigetti_stats,
        "output": output_path,
    }


if __name__ == "__main__":
    run_filtered_gates_correlation_analysis()
