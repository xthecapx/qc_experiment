#!/usr/bin/env python3
"""
Shared plotting styles for IEEE analysis scripts.

Centralizes color palette and default typography/figure sizes so all plots
across this folder remain consistent.

You can extend this module with more properties later (fonts, line styles, etc.).
"""

# ColorBrewer-like palette used consistently across analyses
COLORBREWER_PALETTE = {
    'IBM': '#1b9e77',      # Teal
    'Rigetti': '#d95f02',  # Orange
    1: '#1b9e77',          # Also keep numeric keys used in some modules
    2: '#d95f02',
    3: '#7570b3',
    4: '#e7298a',
    5: '#66a61e',
    6: '#e6ab02',
    7: '#e377c2',
    8: '#7f7f7f',
    9: '#bcbd22',
    10: '#17becf'
}

# Default typography and figure size - increased for paper readability
TITLE_SIZE = 32
LABEL_SIZE = 28
TICK_SIZE = 24
LEGEND_SIZE = 22
FIG_SIZE = (15, 12)  # Updated default as requested
MARKER_SIZE = 80     # Default marker size for scatter plots

# Marker styles for different payload sizes or data series
MARKER_STYLES = {
    1: 'o',  # Circle
    2: 's',  # Square
    3: '^',  # Triangle up
    4: 'D',  # Diamond
    5: 'v',  # Triangle down
    6: 'p',  # Pentagon
    7: '*',  # Star
    8: 'h',  # Hexagon
    9: 'X',  # X
    10: 'P'  # Plus (filled)
}

def apply_axes_defaults(ax):
    """Apply common axes tick params and grid style."""
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax


