#!/usr/bin/env python3
"""
qdiff_plots_v2.py
=================
Publication figures for:
  "Profile Empirical Likelihood Confidence Bands for the
   Difference of Survival Quantiles under Stratified Cox Models"
   Author Name, Institution Name

Reads output from qdiff_combined.py (merged_results.json +
realdata_results.json).

Figures
-------
  Figure 1 — Coverage probability vs n  (PROEL, PLEL, NA)
  Figure 2 — Average CI width vs n      (PROEL, PLEL, NA)
  Figure 3 — Kaplan-Meier survival curves (3 datasets)
  Figure 4 — Quantile difference with PROEL CI (3 datasets)
  Figure 5 — Profile EL ratio curves (one per dataset)

Usage
-----
  # After merge:
  python qdiff_combined.py merge --results-dir ./results

  # Generate all figures:
  python qdiff_plots_v2.py \\
      --sim-results ./results/merged_results.json \\
      --rd-results  ./results/realdata_results.json \\
      --out-dir     ./figures

  # Demo mode (no results needed):
  python qdiff_plots_v2.py --demo --out-dir ./figures
"""

import argparse, json, os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_config import (
    apply_style, COLORS, FIGSIZE, LINEWIDTH, LINESTYLE, ALPHA,
    MARKER, MARKERSIZE, METHOD_LABELS,
    save_fig, set_axis_limits_with_margin,
    nominal_line, zero_line,
    XLABEL_N, XLABEL_QUANTILE, XLABEL_DELTA,
    YLABEL_COVERAGE, YLABEL_WIDTH, YLABEL_EL_RATIO,
)
warnings.filterwarnings('ignore')

SAMPLE_SIZES        = [50, 100, 200, 500]
CENS_RATES          = [0.10, 0.20, 0.40]
CENS_LABELS         = ['10%', '20%', '40%']
QUANT_LEVELS        = [0.25, 0.50, 0.75]
NOMINAL_ALPHAS      = [0.10, 0.05]
NOMINAL_LEVELS_PLOT = [0.90, 0.95]
SIM_METHODS         = ['proel', 'plel', 'na']
P_SELECTED          = [0.25, 0.50, 0.75]


# =====================================================================
# DATA HELPERS
# =====================================================================

def load_sim(path):
    """Load merged_results.json → nested dict [n][cens_rate]."""
    with open(path) as f:
        blocks = json.load(f)
    data = {}
    for b in blocks:
        n = b['n']; c = b['cens_rate']
        data.setdefault(n, {})[c] = {
            'coverage': b.get('coverage', {}),
            'width'   : b.get('width',    {}),
        }
    return data


def load_rd(path):
    with open(path) as f:
        return json.load(f)


def _extract(data, metric, method, alpha, p, n, c):
    key = f'a{int((1-alpha)*100)}_p{int(p*100)}'
    col = f'{method}_mean'
    try:
        return data[n][c][metric][key][col]
    except KeyError:
        return np.nan


def _avg_over_p(data, metric, method, alpha, n, c):
    """Average metric over all three quantile levels."""
    return np.nanmean([_extract(data, metric, method, alpha, p, n, c)
                       for p in QUANT_LEVELS])


def _make_demo_sim():
    """Synthetic simulation data for demo mode."""
    rng  = np.random.default_rng(2025)
    data = {}
    for n in SAMPLE_SIZES:
        data[n] = {}
        for ic, c in enumerate(CENS_RATES):
            data[n][c] = {'coverage': {}, 'width': {}}
            for alpha in NOMINAL_ALPHAS:
                nom = 1 - alpha
                for p in QUANT_LEVELS:
                    key = f'a{int(nom*100)}_p{int(p*100)}'
                    excess  = 0.012*(30/n)**0.45*(1+0.4*ic)
                    cov_pr  = nom + excess + rng.normal(0, 0.003)
                    cov_pl  = nom + excess*0.6 + rng.normal(0, 0.004)
                    cov_na  = nom - 0.022*(30/n)**0.55*(1+0.5*ic) \
                              + rng.normal(0, 0.003)
                    import math
                    cp   = -math.log(1-p)
                    base = (cp - math.sqrt(cp)) * 2*(1+1.5*c)/np.sqrt(n/30)
                    base = max(abs(base), 0.1)
                    data[n][c]['coverage'][key] = {
                        'proel_mean': float(np.clip(cov_pr, 0.4, 1.0)),
                        'plel_mean' : float(np.clip(cov_pl, 0.4, 1.0)),
                        'na_mean'   : float(np.clip(cov_na, 0.4, 1.0)),
                        'estimability_rate': 1.0 if (p < 0.75 or c < 0.4) else 0.85,
                    }
                    data[n][c]['width'][key] = {
                        'proel_mean': float(base * 1.00),
                        'plel_mean' : float(base * 1.40),
                        'na_mean'   : float(base * 0.85),
                    }
    return data


def _make_demo_rd():
    """Synthetic real data for demo mode."""
    import math
    rng = np.random.default_rng(2025)
    pg  = list(np.linspace(0.15, 0.75, 25))

    def _ds(name, g1, g2, n1, n2, unit, scale):
        dhat = [scale*(-math.log(1-p) - math.sqrt(-math.log(1-p)))
                + rng.normal(0, 0.05) for p in pg]
        hw   = [0.3*abs(d) + 0.1 for d in dhat]
        T1   = rng.exponential(300, n1); E1 = (T1 < 400).astype(float)
        T2   = rng.exponential(200, n2); E2 = (T2 < 400).astype(float)
        obs1 = np.minimum(T1, 400); obs2 = np.minimum(T2, 400)
        def km(T, E):
            times = np.concatenate([[0], np.sort(np.unique(T[E==1]))])
            s = [1.0]; n = len(T)
            for t in times[1:]:
                ar = (T>=t).sum(); ev = ((T==t)&(E==1)).sum()
                s.append(s[-1]*(1-ev/ar) if ar>0 else s[-1])
            return times.tolist(), s
        t1,s1 = km(obs1,E1); t2,s2 = km(obs2,E2)
        el = {}
        for p in P_SELECTED:
            dp = scale*(-math.log(1-p)-math.sqrt(-math.log(1-p)))
            dg = np.linspace(dp-1, dp+1, 60).tolist()
            ev = [float(max((d-dp)**2/0.1, 0)) for d in dg]
            el[str(p)] = {'delta_grid':dg,'el_vals':ev,'dhat':float(dp)}
        return {
            'meta': {'name':name,'g1':g1,'g2':g2,'unit':unit,'n1':n1,'n2':n2},
            'p_grid': pg,
            'dhat':  dhat,
            'pt_lo': [d-h for d,h in zip(dhat,hw)],
            'pt_hi': [d+h for d,h in zip(dhat,hw)],
            'pl_lo': [d-h*1.5 for d,h in zip(dhat,hw)],
            'pl_hi': [d+h*1.5 for d,h in zip(dhat,hw)],
            'na_lo': [d-h*0.7 for d,h in zip(dhat,hw)],
            'na_hi': [d+h*0.7 for d,h in zip(dhat,hw)],
            'km': {'1':{'times':t1,'surv':s1,'se':[0]*len(t1)},
                   '2':{'times':t2,'surv':s2,'se':[0]*len(t2)}},
            'el_ratio': el,
        }
    return [
        _ds('German Breast Cancer (GBSG2)',
            'Hormone therapy','No hormone therapy', 299,387,'days', 300),
        _ds('NCCTG Lung Cancer','Female','Male',90,137,'days', 150),
        _ds('VA Lung Cancer','Standard','Test chemo.',69,68,'days', 30),
    ]


# =====================================================================
# FIGURE 1 — Coverage probability
# =====================================================================

def plot_figure1(sim_data):
    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE['double_tall'],
                              sharey=False, sharex=True)
    panel_ids = [['(a)','(b)'],['(c)','(d)'],['(e)','(f)']]
    handles   = []

    for row, (c, clabel) in enumerate(zip(CENS_RATES, CENS_LABELS)):
        for col, (alpha, nom) in enumerate(
                zip(NOMINAL_ALPHAS, NOMINAL_LEVELS_PLOT)):
            ax = axes[row, col]
            nominal_line(ax, nom)
            x  = np.array(SAMPLE_SIZES)

            for meth in SIM_METHODS:
                y = np.array([_avg_over_p(sim_data,'coverage',meth,alpha,n,c)
                              for n in SAMPLE_SIZES])
                line, = ax.plot(x, y,
                    color=COLORS[meth], lw=LINEWIDTH['data'],
                    ls=LINESTYLE[meth], marker=MARKER[meth],
                    ms=MARKERSIZE, label=METHOD_LABELS[meth], zorder=3)
                if row == 0 and col == 0:
                    handles.append(line)

            all_y = np.array([_avg_over_p(sim_data,'coverage',m,alpha,n,c)
                              for m in SIM_METHODS for n in SAMPLE_SIZES])
            all_y = all_y[np.isfinite(all_y)]
            if len(all_y):
                ylo = min(np.nanmin(all_y)-0.015, nom-0.04)
                yhi = max(np.nanmax(all_y)+0.010, nom+0.015)
                ax.set_ylim(ylo, yhi)

            ax.set_xticks(SAMPLE_SIZES)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y,_: f'{y:.2f}'))
            if col == 0:
                ax.set_ylabel(YLABEL_COVERAGE, fontsize=8)
            if row == 0:
                ax.set_title(f'Nominal = {nom:.0%}',
                             fontsize=9, fontweight='bold', pad=6)
            if row == 2:
                ax.set_xlabel(XLABEL_N, fontsize=9)
            if col == 1:
                ax.annotate(f'Censoring {clabel}',
                    xy=(1.03,0.5), xycoords='axes fraction',
                    fontsize=8, rotation=90, ha='left', va='center',
                    annotation_clip=False)
            ax.text(-0.14, 1.06, panel_ids[row][col],
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='bottom', ha='left')

    fig.legend(handles, [METHOD_LABELS[m] for m in SIM_METHODS],
               loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.0),
               frameon=True, framealpha=0.92, edgecolor='0.7',
               fontsize=8, handlelength=2.0)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fig.subplots_adjust(right=0.93, bottom=0.10)
    return save_fig(fig, 'figure1_coverage', tight=False)


# =====================================================================
# FIGURE 2 — Average CI width
# =====================================================================

def plot_figure2(sim_data):
    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE['double_tall'],
                              sharey=False, sharex=True)
    panel_ids = [['(a)','(b)'],['(c)','(d)'],['(e)','(f)']]
    handles   = []

    for row, (c, clabel) in enumerate(zip(CENS_RATES, CENS_LABELS)):
        for col, (alpha, nom) in enumerate(
                zip(NOMINAL_ALPHAS, NOMINAL_LEVELS_PLOT)):
            ax = axes[row, col]
            x  = np.array(SAMPLE_SIZES)

            for meth in SIM_METHODS:
                y = np.array([_avg_over_p(sim_data,'width',meth,alpha,n,c)
                              for n in SAMPLE_SIZES])
                line, = ax.plot(x, y,
                    color=COLORS[meth], lw=LINEWIDTH['data'],
                    ls=LINESTYLE[meth], marker=MARKER[meth],
                    ms=MARKERSIZE, label=METHOD_LABELS[meth], zorder=3)
                if row == 0 and col == 0:
                    handles.append(line)

            all_y = np.array([_avg_over_p(sim_data,'width',m,alpha,n,c)
                              for m in SIM_METHODS for n in SAMPLE_SIZES])
            all_y = all_y[np.isfinite(all_y)]
            if len(all_y):
                set_axis_limits_with_margin(
                    ax, np.nanmin(all_y), np.nanmax(all_y), margin=0.08)

            ax.set_xticks(SAMPLE_SIZES)
            if col == 0:
                ax.set_ylabel(YLABEL_WIDTH, fontsize=8)
            if row == 0:
                ax.set_title(f'Nominal = {nom:.0%}',
                             fontsize=9, fontweight='bold', pad=6)
            if row == 2:
                ax.set_xlabel(XLABEL_N, fontsize=9)
            if col == 1:
                ax.annotate(f'Censoring {clabel}',
                    xy=(1.03,0.5), xycoords='axes fraction',
                    fontsize=8, rotation=90, ha='left', va='center',
                    annotation_clip=False)
            ax.text(-0.14, 1.06, panel_ids[row][col],
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='bottom', ha='left')

    fig.legend(handles, [METHOD_LABELS[m] for m in SIM_METHODS],
               loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.0),
               frameon=True, framealpha=0.92, edgecolor='0.7',
               fontsize=8, handlelength=2.0)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fig.subplots_adjust(right=0.93, bottom=0.10)
    return save_fig(fig, 'figure2_width', tight=False)


# =====================================================================
# FIGURE 3 — Kaplan-Meier curves
# =====================================================================

def plot_figure3(rd_results):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.6))
    panel_ids = ['(a)','(b)','(c)']

    for idx, (res, ax) in enumerate(zip(rd_results, axes)):
        meta = res['meta']; km = res['km']
        t_max = 0
        for gk, color, lk in [('1',COLORS['group1'],'g1'),
                                ('2',COLORS['group2'],'g2')]:
            g   = km[gk]
            t   = np.array(g['times']); s = np.array(g['surv'])
            se  = np.array(g['se'])
            lbl = f"{meta[lk]}  (n={meta[f'n{gk}']})"
            ax.step(t, s, where='post', color=color,
                    lw=LINEWIDTH['data'], label=lbl, zorder=3)
            ax.fill_between(t,
                np.clip(s-1.96*se, 0, 1), np.clip(s+1.96*se, 0, 1),
                step='post', alpha=ALPHA['km_ci'],
                color=color, linewidth=0)
            t_max = max(t_max, t[-1])

        ax.set_xlabel(f"Time ({meta['unit']})", fontsize=9)
        ax.set_ylim(-0.03, 1.07)
        ax.set_xlim(left=0, right=t_max*1.02)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y,_: f'{y:.1f}'))
        if idx == 0:
            ax.set_ylabel('Survival probability', fontsize=9)
        ax.set_title(meta['name'], fontsize=9, fontweight='bold', pad=5)
        ax.text(-0.13, 1.07, panel_ids[idx], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='left')
        t_arr = np.array(km['1']['times'])
        s_arr = np.array(km['1']['surv'])
        mid   = min(np.searchsorted(t_arr, t_max*0.6), len(s_arr)-1)
        leg_loc = 'upper right' if s_arr[mid] < 0.5 else 'lower left'
        ax.legend(loc=leg_loc, fontsize=7.2, framealpha=0.92,
                  edgecolor='0.7', fancybox=False, borderpad=0.5,
                  labelspacing=0.3, handlelength=1.5)

    fig.text(0.5, -0.03,
             'Kaplan\u2013Meier estimates with 95% pointwise confidence bands',
             ha='center', va='top', fontsize=8, style='italic')
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.14)
    return save_fig(fig, 'figure3_km_curves', tight=False)


# =====================================================================
# FIGURE 4 — Quantile difference bands (PROEL CI)
# =====================================================================

def plot_figure4(rd_results):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.8))
    panel_ids = ['(a)','(b)','(c)']

    for idx, (res, ax) in enumerate(zip(rd_results, axes)):
        meta  = res['meta']
        p     = np.array(res['p_grid'])
        dhat  = np.array([v if v is not None else np.nan
                          for v in res['dhat']])
        pt_lo = np.array([v if v is not None else np.nan
                          for v in res['pt_lo']])
        pt_hi = np.array([v if v is not None else np.nan
                          for v in res['pt_hi']])
        na_lo = np.array([v if v is not None else np.nan
                          for v in res.get('na_lo', [np.nan]*len(p))])
        na_hi = np.array([v if v is not None else np.nan
                          for v in res.get('na_hi', [np.nan]*len(p))])

        valid = np.isfinite(dhat); pv = p[valid]
        if len(pv) == 0:
            ax.text(0.5, 0.5, 'Not estimable', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='gray')
        else:
            zero_line(ax)
            # PROEL CI shaded band
            ax.fill_between(pv, pt_lo[valid], pt_hi[valid],
                            alpha=0.30, color=COLORS['proel'],
                            linewidth=0, zorder=2, label='PROEL 95% CI')
            # NA CI dashed outline
            na_v = np.isfinite(na_lo[valid]) & np.isfinite(na_hi[valid])
            if na_v.sum() > 1:
                ax.plot(pv[na_v], na_lo[valid][na_v],
                        color=COLORS['na'], lw=0.9,
                        ls='--', zorder=2, label='NA 95% CI')
                ax.plot(pv[na_v], na_hi[valid][na_v],
                        color=COLORS['na'], lw=0.9, ls='--', zorder=2)
            # Point estimate
            ax.plot(pv, dhat[valid], color=COLORS['proel'],
                    lw=LINEWIDTH['data']+0.2, ls='-', zorder=4,
                    label=r'$\hat{\Delta}(p)$')

            ax.set_xticks([0.25, 0.50, 0.75])
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x,_: f'{x:.2f}'))
            ax.set_xlim(pv[0]-0.01, pv[-1]+0.01)
            all_y = np.concatenate([dhat[valid],
                pt_lo[valid][np.isfinite(pt_lo[valid])],
                pt_hi[valid][np.isfinite(pt_hi[valid])]])
            if len(all_y):
                span = np.nanmax(all_y)-np.nanmin(all_y)
                margin = max(span*0.10, 2.0)
                ax.set_ylim(min(np.nanmin(all_y)-margin, -margin*0.5),
                            np.nanmax(all_y)+margin)
            ax.legend(loc='best', fontsize=7, framealpha=0.92,
                      edgecolor='0.7', fancybox=False,
                      borderpad=0.5, labelspacing=0.3)

        ax.set_xlabel(XLABEL_QUANTILE, fontsize=9)
        if idx == 0:
            ax.set_ylabel(r'$\hat{\Delta}(p)$' +
                          f'  ({meta["unit"]})', fontsize=9)
        ax.set_title(meta['name'], fontsize=9, fontweight='bold', pad=5)
        ax.text(-0.13, 1.07, panel_ids[idx], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='left')

    fig.tight_layout(w_pad=1.0)
    return save_fig(fig, 'figure4_qdiff_bands', tight=False)


# =====================================================================
# FIGURE 5 — Profile EL ratio (one figure per dataset)
# =====================================================================

def plot_figure5(rd_results):
    apply_style()
    chi2_95   = chi2.ppf(0.95, df=1)
    y_cap     = chi2_95 * 2.2
    ds_colors = [COLORS['proel'], COLORS['plel'], COLORS['na']]
    fig_tags  = ['figure5a', 'figure5b', 'figure5c']
    saved     = []

    for ds_idx, res in enumerate(rd_results):
        meta    = res['meta']
        el_data = res.get('el_ratio', {})
        color   = ds_colors[ds_idx % 3]
        tag     = fig_tags[ds_idx] if ds_idx < 3 else f'figure5_{ds_idx}'
        p_avail = [p for p in P_SELECTED if str(p) in el_data]
        if not p_avail: continue

        n_cols = len(p_avail)
        fig_w  = 2.5*n_cols + 0.3
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 3.4))
        if n_cols == 1: axes = [axes]
        panel_ids = ['(a)','(b)','(c)']

        for col, p in enumerate(p_avail):
            ax  = axes[col]; key = str(p)
            ed  = el_data[key]
            dgrid  = np.array(ed['delta_grid'])
            el_v   = np.array([v if v is not None else np.nan
                               for v in ed['el_vals']], dtype=float)
            dhat_p = ed['dhat']
            visible = np.isfinite(el_v)&(el_v>=0)&(el_v<=y_cap*1.05)
            inside  = np.isfinite(el_v)&(el_v>=0)&(el_v<=chi2_95)

            if inside.sum() >= 2:
                ci_lo = dgrid[inside][0]; ci_hi = dgrid[inside][-1]
                ax.fill_between(dgrid[inside], 0, el_v[inside],
                                alpha=0.25, color=color,
                                linewidth=0, zorder=1)
                for b in [ci_lo, ci_hi]:
                    ax.axvline(b, color=color, lw=1.0,
                               ls='--', alpha=0.75, zorder=3)
                ax.annotate(f'95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]',
                            xy=(0.5,0.97), xycoords='axes fraction',
                            fontsize=6.5, ha='center', va='top',
                            color=color, zorder=5)
            if visible.sum() >= 2:
                ax.plot(dgrid[visible], el_v[visible],
                        color=color, lw=LINEWIDTH['data'], zorder=4)
                xm = (dgrid[visible][-1]-dgrid[visible][0])*0.03
                ax.set_xlim(dgrid[visible][0]-xm, dgrid[visible][-1]+xm)
            ax.axhline(chi2_95, color=COLORS['nominal'],
                       lw=LINEWIDTH['reference'],
                       ls=LINESTYLE['nominal'], zorder=2)
            ax.axvline(dhat_p, color='black', lw=1.6, ls='-', zorder=5)
            ax.set_ylim(-0.04*y_cap, y_cap)
            ax.set_yticks([0, round(chi2_95,1), round(y_cap,1)])
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y,_: f'{y:.1f}'))
            ax.set_title(f'$p={p:.2f}$', fontsize=9,
                         fontweight='bold', pad=5)
            ax.set_xlabel(f'{XLABEL_DELTA}  ({meta["unit"]})', fontsize=9)
            if col == 0:
                ax.set_ylabel(YLABEL_EL_RATIO, fontsize=9)
            ax.text(-0.14, 1.07, panel_ids[col], transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='bottom', ha='left')

        fig.suptitle(f'{meta["name"]}  —  '
                     f'{meta["g1"]} vs {meta["g2"]}',
                     fontsize=9, fontweight='bold', y=1.03)
        h_ci   = mpatches.Patch(facecolor=color, alpha=0.25,
                                 label='95% CI region')
        h_chi2 = Line2D([0],[0], color=COLORS['nominal'],
                         lw=LINEWIDTH['reference'],
                         ls=LINESTYLE['nominal'],
                         label=r'$\chi^2_{1,\,0.05}=3.84$')
        h_dhat = Line2D([0],[0], color='black', lw=1.6, ls='-',
                         label=r'$\hat{\Delta}(p)$')
        fig.legend(handles=[h_ci, h_chi2, h_dhat],
                   loc='lower center', bbox_to_anchor=(0.5,-0.04),
                   ncol=3, fontsize=8, frameon=True,
                   framealpha=0.92, edgecolor='0.7', handlelength=1.8)
        fig.tight_layout(w_pad=1.0)
        fig.subplots_adjust(bottom=0.24)
        path = save_fig(fig, tag, tight=False)
        saved.append(path)

    return saved


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate publication figures from qdiff_combined.py output.')
    parser.add_argument('--sim-results', type=str, default=None)
    parser.add_argument('--rd-results',  type=str, default=None)
    parser.add_argument('--demo',        action='store_true')
    parser.add_argument('--out-dir',     type=str, default='./figures')
    parser.add_argument('--fig5-only',   action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    import fig_config as _fc
    _fc.OUTPUT_DIR = args.out_dir

    # Simulation figures
    if args.demo or args.sim_results is None:
        print('Demo simulation data...')
        sim_data = _make_demo_sim()
    else:
        print(f'Loading: {args.sim_results}')
        sim_data = load_sim(args.sim_results)

    if not args.fig5_only:
        print('Figure 1 — Coverage...')
        print(f'  Saved: {plot_figure1(sim_data)}')
        print('Figure 2 — Width...')
        print(f'  Saved: {plot_figure2(sim_data)}')

    # Real data figures
    if args.demo or args.rd_results is None:
        print('Demo real data...')
        rd_results = _make_demo_rd()
    else:
        print(f'Loading: {args.rd_results}')
        rd_results = load_rd(args.rd_results)

    if not args.fig5_only:
        print('Figure 3 — KM curves...')
        print(f'  Saved: {plot_figure3(rd_results)}')
        print('Figure 4 — Quantile difference bands...')
        print(f'  Saved: {plot_figure4(rd_results)}')

    print('Figures 5a/5b/5c — EL ratio...')
    for path in plot_figure5(rd_results):
        print(f'  Saved: {path}')

    print('\nAll figures complete.')
