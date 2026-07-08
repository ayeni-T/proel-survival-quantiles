"""
fig_config.py
=============
Publication figure configuration for:
  "Profile Empirical Likelihood Confidence Bands for the Difference of
   Survival Quantiles under Stratified Cox Models"
   Author Name, Institution Name

Publication standards
---------------------
  Target journals : Statistical Methods in Medical Research,
                    Statistics in Medicine, Lifetime Data Analysis
  Column width    : 7.09 in double-column (180 mm)
  Resolution      : 300 dpi PNG only
  Font            : Times New Roman serif (matches LaTeX default)
  Font sizes      : 9 pt labels, 8 pt ticks, 8 pt legend
  Colour palette  : Wong (2011) colour-blind-safe 4-colour palette
  No figure titles (captions go in the LaTeX source)
  Legend always inside axes, never outside
  Tick marks inward on all four sides

Usage
-----
  from fig_config import apply_style, COLORS, FIGSIZE, save_fig
  apply_style()
  fig, ax = plt.subplots(figsize=FIGSIZE['double'])
  ...
  save_fig(fig, 'figure1_coverage', out_dir='./figures')
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# ── Output directory (overridden at runtime by qdiff_plots.py CLI) ────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'figures')

# ── Figure sizes (inches) ─────────────────────────────────────────────
FIGSIZE = {
    'single'      : (3.46, 2.8),
    'double'      : (7.09, 4.5),
    'double_tall' : (7.09, 6.0),
    'double_short': (7.09, 3.2),
    'square'      : (3.46, 3.46),
}

# ── Colour palette — Wong (2011) colour-blind-safe ────────────────────
COLORS = {
    'proel'    : '#0072B2',
    'na'       : '#E69F00',
    'plel'     : '#009E73',
    'bootstrap': '#D55E00',
    'nominal'  : '#999999',
    'band'     : '#0072B2',
    'group1'   : '#0072B2',
    'group2'   : '#D55E00',
    'zero'     : '#BBBBBB',
}

ALPHA = {
    'ptwise_band': 0.25,
    'sim_band'   : 0.12,
    'km_ci'      : 0.15,
}

LINESTYLE = {
    'proel'    : '-',
    'na'       : '--',
    'plel'     : '-.',
    'bootstrap': ':',
    'nominal'  : (0, (3, 1, 1, 1)),
    'sim_band' : '--',
    'zero'     : '-',
}

LINEWIDTH = {
    'data'     : 1.4,
    'reference': 0.9,
    'axis'     : 0.8,
}

MARKER = {
    'proel'    : 'o',
    'na'       : 's',
    'plel'     : '^',
    'bootstrap': 'D',
}
MARKERSIZE = 4.5

METHOD_LABELS = {
    'proel'    : 'PROEL (proposed)',
    'na'       : 'Normal approx.',
    'plel'     : 'Plug-in EL',
    'bootstrap': 'Bootstrap Wald',
}

XLABEL_N        = 'Sample size $n_i$'
XLABEL_QUANTILE = r'Quantile level $p$'
XLABEL_DELTA    = r'Hypothesised $\delta$'
YLABEL_COVERAGE = 'Coverage probability'
YLABEL_WIDTH    = 'Average CI width (days)'
YLABEL_EL_RATIO = r'$-2\log\,\mathcal{R}(\delta,\,p)$'

RC = {
    'font.family'          : 'serif',
    'font.serif'           : ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size'            : 9,
    'axes.titlesize'       : 9,
    'axes.labelsize'       : 9,
    'xtick.labelsize'      : 8,
    'ytick.labelsize'      : 8,
    'legend.fontsize'      : 8,
    'legend.title_fontsize': 8,
    'axes.linewidth'       : 0.8,
    'axes.spines.top'      : True,
    'axes.spines.right'    : True,
    'axes.grid'            : False,
    'axes.axisbelow'       : True,
    'xtick.direction'      : 'in',
    'ytick.direction'      : 'in',
    'xtick.major.size'     : 3.5,
    'ytick.major.size'     : 3.5,
    'xtick.minor.size'     : 2.0,
    'ytick.minor.size'     : 2.0,
    'xtick.major.width'    : 0.8,
    'ytick.major.width'    : 0.8,
    'xtick.top'            : True,
    'ytick.right'          : True,
    'lines.linewidth'      : 1.4,
    'lines.markersize'     : 4.5,
    'legend.frameon'       : True,
    'legend.framealpha'    : 0.92,
    'legend.edgecolor'     : '0.7',
    'legend.fancybox'      : False,
    'legend.borderpad'     : 0.4,
    'legend.labelspacing'  : 0.3,
    'legend.handlelength'  : 1.8,
    'legend.handletextpad' : 0.5,
    'figure.dpi'           : 150,
    'savefig.dpi'          : 300,
    'savefig.bbox'         : 'tight',
    'savefig.pad_inches'   : 0.03,
    'mathtext.fontset'     : 'stix',
    'figure.constrained_layout.use': False,
}


def apply_style():
    """Apply all publication rcParams. Call once before any plt.subplots()."""
    mpl.rcParams.update(RC)


def save_fig(fig, name, out_dir=None, tight=True):
    """
    Save figure as PNG at 300 dpi.

    Parameters
    ----------
    fig     : matplotlib Figure
    name    : str   filename without extension e.g. 'figure1_coverage'
    out_dir : str   output directory (default: OUTPUT_DIR)
    tight   : bool  call tight_layout before saving

    Returns
    -------
    str  full path to saved file
    """
    if tight:
        fig.tight_layout()
    d    = out_dir or OUTPUT_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'{name}.png')
    fig.savefig(path, format='png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'Saved: {path}', flush=True)
    return path


def add_panel_label(ax, label, x=-0.12, y=1.04, fontsize=9, bold=True):
    """Add bold panel label (a)(b)(c) outside top-left corner of axes."""
    ax.text(x, y, label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight='bold' if bold else 'normal',
            va='bottom', ha='right')


def set_axis_limits_with_margin(ax, ymin, ymax, margin=0.05):
    """Set y limits with fractional margin so data never touches spines."""
    span = ymax - ymin
    ax.set_ylim(ymin - margin * span, ymax + margin * span)


def nominal_line(ax, level, **kwargs):
    """Draw horizontal nominal coverage reference line."""
    kw = dict(color=COLORS['nominal'], linewidth=LINEWIDTH['reference'],
              linestyle=LINESTYLE['nominal'], zorder=1)
    kw.update(kwargs)
    ax.axhline(level, **kw)


def zero_line(ax, **kwargs):
    """Draw horizontal zero reference line."""
    kw = dict(color=COLORS['zero'], linewidth=LINEWIDTH['reference'],
              linestyle=LINESTYLE['zero'], zorder=1)
    kw.update(kwargs)
    ax.axhline(0, **kw)


def safe_legend(ax, loc='best', ncol=1, **kwargs):
    """Place legend inside axes, never outside."""
    return ax.legend(loc=loc, ncol=ncol, **kwargs)
