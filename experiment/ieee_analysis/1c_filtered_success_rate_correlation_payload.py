#!/usr/bin/env python3
"""
Filtered Success Rate Correlation vs Payload Size Analysis
==========================================================

This module quantifies and visualises the correlation between payload size
(number of payload qubits) and success rate for IBM and Rigetti platforms.
It mirrors the filtered correlation analyses created for gate count and
circuit depth by reporting Pearson and Spearman coefficients, linear
regression statistics, and a focused plot with min/mean/max envelopes.

Author: Analysis Script
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def calculate_correlation_stats(x_values, success_rates):
    """Compute correlation metrics between payload size and success rate."""
    pearson_r, pearson_p = stats.pearsonr(x_values, success_rates)
    spearman_r, spearman_p = stats.spearmanr(x_values, success_rates)
    slope, intercept, r_value, p_value, _ = stats.linregress(x_values, success_rates)
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
    """Format correlation statistics for console output."""
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


def plot_payload_success_rate_correlation(df):
    """Produce correlation plot for payload size vs success rate."""
    # Style constants
    label_size = LABEL_SIZE
    tick_size = TICK_SIZE
    legend_size = LEGEND_SIZE
    fig_size = FIG_SIZE

    # Ensure payload sizes are within expected range (1-5)
    df_filtered = df[(df["payload_size"] >= 1) & (df["payload_size"] <= 5)].copy()
    if df_filtered.empty:
        raise ValueError("Filtered dataset has no rows for payload sizes 1-5.")

    # Convert success_rate from percentage to fraction for correlation calculations
    df_filtered["success_rate_fraction"] = df_filtered["success_rate"] / 100.0

    print("\nFiltered datasets (payload sizes 1-5):")
    for platform in sorted(df_filtered["hardware"].unique()):
        count = len(df_filtered[df_filtered["hardware"] == platform])
        print(f"  {platform}: {count} experiments")

    # Separate by hardware and compute correlation statistics
    stats_per_platform = {}
    for platform in sorted(df_filtered["hardware"].unique()):
        subset = df_filtered[df_filtered["hardware"] == platform]
        stats_per_platform[platform] = calculate_correlation_stats(
            subset["payload_size"].values,
            subset["success_rate_fraction"].values,
        )
        print(format_correlation_stats(stats_per_platform[platform], platform))

    # Prepare data for plotting: mean/min/max by payload size per hardware
    fig, ax = plt.subplots(figsize=fig_size)
    for platform, color, marker in [
        ("IBM", COLORBREWER_PALETTE["IBM"], "o"),
        ("Rigetti", COLORBREWER_PALETTE["Rigetti"], "s"),
    ]:
        subset = df_filtered[df_filtered["hardware"] == platform]
        payload_groups = sorted(subset["payload_size"].unique())
        means, mins, maxs = [], [], []
        for payload in payload_groups:
            group = subset[subset["payload_size"] == payload]["success_rate_fraction"]
            if group.empty:
                continue
            means.append(group.mean())
            mins.append(group.min())
            maxs.append(group.max())
        payload_groups = np.array(payload_groups)
        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)

        ax.plot(payload_groups, means, color=color, linewidth=2.5, marker=marker, markersize=10, label=platform)
        ax.plot(payload_groups, mins, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.plot(payload_groups, maxs, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.fill_between(payload_groups, mins, maxs, color=color, alpha=0.2)

    # Axis labels and ticks
    ax.set_xlabel("Payload Size", fontsize=label_size, fontweight="bold")
    ax.set_ylabel("Mean Success Rate", fontsize=label_size, fontweight="bold")
    ax.set_xticks(sorted(df_filtered["payload_size"].unique()))
    ax.set_xticklabels(sorted(df_filtered["payload_size"].unique()), fontsize=tick_size)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.grid(True, linestyle="--", alpha=0.3, axis="both")

    # Annotate statistics
    ibm_stats = stats_per_platform.get("IBM")
    rigetti_stats = stats_per_platform.get("Rigetti")
    rigetti_p_text = "< 0.001" if rigetti_stats and rigetti_stats["pearson_p"] < 0.001 else f"= {rigetti_stats['pearson_p']:.3f}" if rigetti_stats else "N/A"
    legend_handles = []
    legend_labels = []
    if ibm_stats:
        legend_handles.append(Line2D([0], [0], color=COLORBREWER_PALETTE["IBM"], linewidth=2.5))
        legend_labels.append(
            f"IBM — r = {ibm_stats['pearson_r']:.3f}, R² = {ibm_stats['r_squared']:.3f}, p < 0.001"
        )
    if rigetti_stats:
        legend_handles.append(Line2D([0], [0], color=COLORBREWER_PALETTE["Rigetti"], linewidth=2.5))
        legend_labels.append(
            f"Rigetti — r = {rigetti_stats['pearson_r']:.3f}, R² = {rigetti_stats['r_squared']:.3f}, p {rigetti_p_text}"
        )

    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=legend_size - 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
        framealpha=0.95,
        title="Correlation Statistics",
        title_fontsize=legend_size - 1,
        edgecolor="gray",
    )

    plt.tight_layout()

    return plt, stats_per_platform


def run_filtered_payload_correlation_analysis():
    """Entry point for the payload size vs success rate correlation analysis."""
    print("=" * 60)
    print("FILTERED SUCCESS RATE CORRELATION vs PAYLOAD SIZE")
    print("=" * 60)

    combined_df = load_combined_hardware_data()
    if combined_df.empty:
        print("No data available for analysis.")
        return

    print("\n" + "=" * 40)
    print("FULL DATASET STATISTICS")
    print("=" * 40)
    for platform in sorted(combined_df["hardware"].unique()):
        subset = combined_df[combined_df["hardware"] == platform]
        print(f"\n{platform} Dataset:")
        print(f"  Total experiments: {len(subset)}")
        print(f"  Payload size range: {subset['payload_size'].min()} - {subset['payload_size'].max()}")

    print("\n" + "=" * 40)
    print("GENERATING FILTERED CORRELATION PLOT")
    print("=" * 40)

    plt_obj, stats_per_platform = plot_payload_success_rate_correlation(combined_df)

    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "1c_filtered_success_rate_correlation_payload.png")
    plt_obj.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"\n✓ Saved: {output_path}")
    print("Analysis complete!")
    return stats_per_platform


if __name__ == "__main__":
    run_filtered_payload_correlation_analysis()
