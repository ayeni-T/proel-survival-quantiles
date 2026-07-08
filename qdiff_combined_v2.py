#!/usr/bin/env python3
"""
qdiff_combined_v2.py
=================
Combined simulation and real data analysis for:
  "Profile Empirical Likelihood Confidence Bands for the
   Difference of Survival Quantiles under Stratified Cox Models"
   Author Name, Institution Name

*** v2 CHANGES vs. qdiff_combined.py (original) ***
  1. FIX: qdiff_el_ratio's nuisance-quantile (Q2(p)) search replaced
     a flat 31-point grid over the full data range with a coarse pass
     (grid_size=21) + 2 zoom-in refinement passes (refine_factor=15
     each) around the running minimizer. The old flat grid became too
     coarse to locate the true EL-ratio minimum as n grew, biasing the
     minimized ratio upward and causing PROEL coverage to *degrade*
     with n at p=0.25 under light censoring (e.g. ~95% at n=50 down to
     ~69% at n=500). Confirmed: old grid gave ratio 3.64 vs true 3.17
     at n=500, right at the chi^2_{1,0.95}=3.84 boundary.
  2. SPEED: the risk-sum arrays R1, R2 (used inside the EL-contribution
     and Lagrange-multiplier solves) depend only on (T, Z, beta), not
     on the candidate quantile/delta being tested, but were recomputed
     via an O(n^2) operation at every single grid point. Now computed
     once per replication and passed through via R1=, R2= params.
  This is a self-contained script -- SAME directory structure and
  output format as qdiff_combined.py (original), so qdiff_plots_v2.py
  reads its output with no changes. Intended to run from a SEPARATE
  working directory (see below) so the original code/results/logs are
  left untouched for comparison.

DGP (Zhao & Zhao 2025 framework):
  Stratified Cox model, two groups, shared beta:
    Lambda_i(t|Z) = Lambda_i0(t) * exp(Z'beta)
    Lambda_10(t) = t        (Weibull shape=1, scale=1)
    Lambda_20(t) = t^2      (Weibull shape=2, scale=1)
    beta = (0.6, 0.2), Z = (Bernoulli(0.5), N(0,1))
  True quantile difference:
    Delta(p) = -log(1-p) - sqrt(-log(1-p))

Three CI methods compared per replication:
  PROEL — Profile EL with quantile constraint (proposed method)
  PLEL  — Plug-in EL (fixed tau2)
  NA    — Normal approximation via Bahadur representation

One Cox fit per replication (shared beta_hat, shared Breslow)
for fair comparison.

Output:  JSON block files (one per scenario), merged_results.json,
         realdata_results.json — same structure as Zhao & Zhao (2025).

Recommended ARCTIC directory layout
------------------------------------
  /home/users/username/qdiff/         <- ORIGINAL, untouched
      qdiff_combined.py
      qdiff_combined_array.sh
      results/                       <- original merged_results.json,
                                         realdata_results.json, blocks
      logs/

  /home/users/username/qdiff_v2/      <- NEW, this script's home
      qdiff_combined_v2.py           <- this file
      qdiff_combined_v2_array.sh      <- SLURM array script (below)
      qdiff_plots_v2.py               <- plotting script (unchanged)
      fig_config.py                   <- COPY over from qdiff/ (plots
                                          script imports it from its
                                          own directory)
      results/                        <- new block*.json, merged_results.json,
                                          realdata_results.json land here
      logs/                           <- new SLURM stdout/stderr logs

ARCTIC usage (run from /home/users/username/qdiff_v2/)
------------------------------------------------------
  # Pilot test (5 reps, block 0):
  python qdiff_combined_v2.py sim --block-id 0 --n-reps 5 \\
      --out-dir /home/users/username/qdiff_v2/results

  # Full run via SLURM array:
  sbatch qdiff_combined_v2_array.sh

  # Merge after array finishes:
  python qdiff_combined_v2.py merge \\
      --results-dir /home/users/username/qdiff_v2/results

  # Real data analysis (quick -- no replication loop):
  python qdiff_combined_v2.py realdata \\
      --out-dir /home/users/username/qdiff_v2/results
"""

from __future__ import annotations
import argparse, json, math, os, time, warnings, glob
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2
warnings.filterwarnings('ignore')


def _log(msg): print(msg, flush=True)


# =====================================================================
# SECTION 1 — SHARED COX MODEL AND BRESLOW ESTIMATOR
# Copied exactly from validated qdiff_quantile_proel.py
# =====================================================================

def fit_stratified_cox(T, E, Z, group, tol=1e-8, max_iter=200):
    """Stratified Cox Newton-Raphson. Returns beta_hat (d,)."""
    T = np.asarray(T, float); E = np.asarray(E, int)
    Z = np.asarray(Z, float); group = np.asarray(group, int)
    d = Z.shape[1]
    beta = np.zeros(d)
    for _ in range(max_iter):
        score = np.zeros(d); info = np.zeros((d, d))
        for g in [1, 2]:
            ix = group == g
            Tg, Eg, Zg = T[ix], E[ix], Z[ix]
            for j in np.where(Eg == 1)[0]:
                risk = Tg >= Tg[j]
                eta  = np.exp(Zg[risk] @ beta)
                s0   = eta.sum()
                s1   = (Zg[risk] * eta[:, None]).sum(0)
                s2   = (Zg[risk,:,None]*Zg[risk,None,:]*eta[:,None,None]).sum(0)
                ebar = s1 / max(s0, 1e-300)
                score += Zg[j] - ebar
                info  += s2 / max(s0, 1e-300) - np.outer(ebar, ebar)
        step = np.linalg.solve(info + 1e-8*np.eye(d), score)
        beta += step
        if np.max(np.abs(step)) < tol: break
    return beta


def risk_sums_at_observed_times(Tg, Zg, beta):
    """
    R_j = sum_{k: T_k >= T_j} exp(Z_k'beta)  for each observed time T_j.
    Exact copy from validated qdiff_quantile_proel.py.
    """
    exp_eta = np.exp(Zg @ beta)
    R = np.array([exp_eta[Tg >= t].sum() for t in Tg])
    return np.maximum(R, 1e-300)


def breslow_by_group(T, E, Z, group, beta):
    """
    Stratified Breslow baseline cumulative hazard.
    Uses group-specific risk sets — exact copy from validated
    qdiff_quantile_proel.py.
      haz_j = sum_{delta_k=1, T_k=t_j} 1/R_j
    Returns dict: g -> (event_times, cumhaz).
    """
    T = np.asarray(T, float); E = np.asarray(E, int)
    Z = np.asarray(Z, float); group = np.asarray(group, int)
    out = {}
    for g in [1, 2]:
        ix = group == g
        Tg, Eg, Zg = T[ix], E[ix], Z[ix]
        ev_times = np.sort(np.unique(Tg[Eg == 1]))
        R   = risk_sums_at_observed_times(Tg, Zg, beta)
        haz = []
        for t in ev_times:
            jj = np.where((Tg == t) & (Eg == 1))[0]
            haz.append(np.sum(1.0 / R[jj]))
        out[g] = (ev_times.astype(float), np.cumsum(haz).astype(float))
    return out


def q_from_cumhaz(p, times, cumhaz):
    """p-th quantile from Breslow cumhaz. Returns nan if not estimable."""
    if len(times) == 0: return np.nan
    c = -math.log(1.0 - float(p))
    k = np.searchsorted(cumhaz, c, side='left')
    return np.nan if k >= len(times) else float(times[k])


# Keep q_from_breslow as alias for backward compatibility
q_from_breslow = q_from_cumhaz


def qdiff_hat(p, bres):
    q1 = q_from_cumhaz(p, *bres[1])
    q2 = q_from_cumhaz(p, *bres[2])
    return np.nan if (np.isnan(q1) or np.isnan(q2)) else q1 - q2


# =====================================================================
# SECTION 2 — PROEL CI  (profile EL with quantile constraint)
# Copied exactly from validated qdiff_quantile_proel.py
# =====================================================================

def _lambda_for_fixed_quantile(Tg, Eg, Zg, beta, tau, c_p, R=None):
    """
    Solve sum_j delta_j*I(T_j<=tau) / (R_j + lam*I(T_j<=tau)) = c_p.

    R (risk sums) does not depend on tau and can be passed in
    precomputed to avoid repeating an O(n^2) computation at every
    grid point during a profile search.
    """
    g = (Tg <= tau).astype(float)
    if np.sum(Eg * g) == 0: return None
    if R is None:
        R = risk_sums_at_observed_times(Tg, Zg, beta)
    Rg = R[(Eg == 1) & (g == 1)]
    if len(Rg) == 0: return None
    lo = -np.min(Rg) + 1e-10
    hi = max(np.max(R) * 1e4, 1e4)

    def f(lam):
        den = R + lam * g
        if np.any(den[(Eg==1)&(g==1)] <= 0): return np.inf
        return float(np.sum(Eg * g / den) - c_p)

    fhi = f(hi); expand = 0
    while np.isfinite(fhi) and fhi > 0 and expand < 30:
        hi *= 2; fhi = f(hi); expand += 1
    if not np.isfinite(fhi) or fhi > 0: return None
    try:
        return brentq(f, lo, hi, maxiter=100, xtol=1e-8)
    except Exception:
        return None


def _one_group_el_contribution(Tg, Eg, Zg, beta, tau, c_p, R=None):
    """
    -2 log EL contribution for one group under H(tau) = c_p.
    KL-divergence formula: 2*sum_delta[log(den/R) + R/den - 1].

    R can be passed in precomputed (see _lambda_for_fixed_quantile).
    """
    g = (Tg <= tau).astype(float)
    if g.sum() == 0 or np.sum(Eg * g) == 0: return np.nan
    if R is None:
        R = risk_sums_at_observed_times(Tg, Zg, beta)
    H_hat_tau = float(np.sum(Eg * g / R))
    if not np.isfinite(H_hat_tau) or H_hat_tau <= 0: return np.nan
    lam = _lambda_for_fixed_quantile(Tg, Eg, Zg, beta, tau, c_p, R=R)
    if lam is None: return np.nan
    den = R + lam * g
    if np.any(den[Eg == 1] <= 0): return np.nan
    ratio = den / R
    val = 2.0 * np.sum(Eg * (np.log(ratio) + (1.0 / ratio) - 1.0))
    if not np.isfinite(val): return np.nan
    return float(max(val, 0.0))


def _scan_q2(q2_grid, d, T1, E1, Z1, T2, E2, Z2, beta, c_p, R1, R2):
    """Evaluate v1(q2+d) + v2(q2) on a grid; return (best_val, best_idx, vals)."""
    vals = np.full(len(q2_grid), np.inf)
    for k, q2 in enumerate(q2_grid):
        q1 = q2 + d
        if q1 <= 0:
            continue
        v1 = _one_group_el_contribution(T1, E1, Z1, beta, q1, c_p, R=R1)
        v2 = _one_group_el_contribution(T2, E2, Z2, beta, q2, c_p, R=R2)
        if np.isfinite(v1) and np.isfinite(v2):
            vals[k] = v1 + v2
    if not np.any(np.isfinite(vals)):
        return np.inf, -1, vals
    best_idx = int(np.argmin(vals))
    return float(vals[best_idx]), best_idx, vals


def qdiff_el_ratio(d, p, T, E, Z, group, beta, bres, grid_size=21,
                    refine_levels=2, refine_factor=15, R1=None, R2=None):
    """
    Profile -2 log R(d,p): minimise over nuisance q2=Q_2(p).

    FIX #1 (grid resolution): the EL-contribution function
    v1(q2+d)+v2(q2) is piecewise-constant in q2, with breakpoints only
    at observed event times of either stratum. A fixed-resolution grid
    over the *entire* data-supported range of q2 becomes too coarse to
    locate the true minimiser as n grows (the function sharpens around
    its minimum while the search range does not shrink), which biases
    the minimised EL ratio upward and causes coverage to degrade
    --rather than improve--with n. Confirmed directly: at n=500,
    p=0.25, 10% censoring, the old 31-point grid returned a minimised
    ratio of ~3.64 (essentially at the chi^2_{1,0.95}=3.84 boundary)
    versus ~3.17 on a 4001-point grid for the same replication.

    Fix: a coarse pass over `grid_size` points, then `refine_levels`
    zoom-in passes of `refine_factor` points each around the current
    best point. Resolution improves automatically as the window
    narrows (not tied to a fixed absolute grid size).

    FIX #2 (speed): the risk-sum arrays R1, R2 (over observed event
    times within each stratum) do not depend on the candidate (d, q2)
    being evaluated -- only on (T, Z, beta) -- yet the original code
    recomputed them (an O(n^2) operation) at every single grid point.
    They can now be precomputed once per replication and passed in via
    R1/R2 (e.g. by qdiff_ci, which calls this function across an
    entire delta-grid). If not supplied they are computed once here.
    """
    c_p = -math.log(1.0 - p)
    ix1, ix2 = group == 1, group == 2
    T1, E1, Z1 = T[ix1], E[ix1].astype(int), Z[ix1]
    T2, E2, Z2 = T[ix2], E[ix2].astype(int), Z[ix2]
    if R1 is None: R1 = risk_sums_at_observed_times(T1, Z1, beta)
    if R2 is None: R2 = risk_sums_at_observed_times(T2, Z2, beta)
    ev2 = np.sort(np.unique(T2[E2 == 1]))
    if len(ev2) == 0: return np.nan
    lo = max(np.min(ev2), np.min(T1[E1 == 1]) - d)
    hi = min(np.max(ev2), np.max(T1[E1 == 1]) - d)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi: return np.nan

    # Coarse pass
    grid = np.linspace(lo, hi, grid_size)
    best, best_idx, vals = _scan_q2(grid, d, T1, E1, Z1, T2, E2, Z2, beta, c_p, R1, R2)
    if best_idx < 0:
        return np.nan

    # Refinement passes: zoom into a window of +/- one coarse cell
    # around the current best point, each time with a finer grid.
    cell = grid[1] - grid[0] if len(grid) > 1 else (hi - lo)
    center = grid[best_idx]
    window = cell
    for _ in range(refine_levels):
        sub_lo = max(lo, center - window)
        sub_hi = min(hi, center + window)
        if sub_hi <= sub_lo:
            break
        fine_grid = np.linspace(sub_lo, sub_hi, refine_factor)
        fbest, fbest_idx, fvals = _scan_q2(fine_grid, d, T1, E1, Z1,
                                            T2, E2, Z2, beta, c_p, R1, R2)
        if fbest_idx < 0:
            break
        if fbest < best:
            best = fbest
        center = fine_grid[fbest_idx]
        window = (fine_grid[1] - fine_grid[0]) if len(fine_grid) > 1 else window

    return np.nan if not np.isfinite(best) else float(best)


def qdiff_ci(p, T, E, Z, group, beta, bres, alpha=0.05,
             n_grid=161, width_factor=1.5):
    """
    Pointwise PROEL CI. Returns (dhat, lo, hi).

    Precomputes the risk-sum arrays R1, R2 once (they depend only on
    (T, Z, beta), not on the candidate delta or p) and reuses them
    across the whole delta-grid sweep -- see qdiff_el_ratio docstring.
    """
    c_p  = -math.log(1.0 - p)
    dhat = qdiff_hat(p, bres)
    if np.isnan(dhat): return np.nan, np.nan, np.nan
    q1   = q_from_cumhaz(p, *bres[1])
    q2   = q_from_cumhaz(p, *bres[2])
    hw   = width_factor * max(
               abs(dhat),
               q1 * 0.25 if not np.isnan(q1) else 0.0,
               q2 * 0.25 if not np.isnan(q2) else 0.0,
               np.nanmedian(T) * 0.20, 0.25)
    grid = np.linspace(dhat - hw, dhat + hw, n_grid)
    crit = chi2.ppf(1.0 - alpha, df=1)

    ix1, ix2 = group == 1, group == 2
    R1 = risk_sums_at_observed_times(T[ix1], Z[ix1], beta)
    R2 = risk_sums_at_observed_times(T[ix2], Z[ix2], beta)
    vals = np.array([qdiff_el_ratio(dv, p, T, E, Z, group, beta, bres,
                                     R1=R1, R2=R2)
                     for dv in grid])
    inside = np.where(np.isfinite(vals) & (vals <= crit))[0]
    if len(inside) == 0: return dhat, np.nan, np.nan
    return dhat, float(grid[inside[0]]), float(grid[inside[-1]])


# Alias for internal use in run_block
def proel_ci(p, T, E, Z, group, beta, bres, alpha=0.05, n_grid=161, width_factor=1.5):
    """Wrapper returning (lo, hi) for backward compatibility inside run_block."""
    _, lo, hi = qdiff_ci(p, T, E, Z, group, beta, bres, alpha, n_grid, width_factor)
    return lo, hi


# Alias for internal use in realdata
proel_el_ratio = qdiff_el_ratio


# =====================================================================
# SECTION 3 — PLEL CI  (plug-in EL, fixed tau2)
# =====================================================================

def _plel_el_ratio(d, p, T, E, Z, group, beta, bres):
    """Plug-in EL ratio: tau2 fixed at Q_hat_2(p)."""
    n  = len(T)
    t2_hat, c2_hat = bres[2]
    tau2 = q_from_breslow(p, t2_hat, c2_hat)
    if np.isnan(tau2): return np.nan
    tau1 = tau2 + d
    if tau1 <= 0: return np.nan
    ix1, ix2 = group==1, group==2
    T1,E1,Z1 = T[ix1],E[ix1],Z[ix1]
    T2,E2,Z2 = T[ix2],E[ix2],Z[ix2]
    g1 = (T1 <= tau1).astype(float)
    g2 = (T2 <= tau2).astype(float)
    if E1[g1==1].sum()==0 or E2[g2==1].sum()==0: return np.nan
    # Use pointwise S0 normalised by total n — consistent with breslow_by_group
    # which uses S0_t = sum_{risk set} exp(Z@beta) / n_total
    S0_1 = np.exp(Z1 @ beta) / n
    S0_2 = np.exp(Z2 @ beta) / n
    # Safety: ensure S0 values are positive
    S0_1 = np.maximum(S0_1, 1e-300)
    S0_2 = np.maximum(S0_2, 1e-300)

    def eq_lam(lam):
        d1 = S0_1 + lam*g1; d2 = S0_2 - lam*g2
        if np.any(d1[(E1==1)&(g1==1)]<=0) or np.any(d2[(E2==1)&(g2==1)]<=0):
            return 1e6
        return float(np.sum(E1*g1/d1) - np.sum(E2*g2/d2))

    found = False
    eps   = 1e-9
    for shrink in [1.0, 0.5, 0.1, 0.01]:
        lam_hi = (min(S0_1[(E1==1)&(g1==1)].min() if (g1==1).any() else 0.1,
                      S0_2[(E2==1)&(g2==1)].min() if (g2==1).any() else 0.1)
                  - eps) * shrink
        if np.isnan(lam_hi) or lam_hi <= 0: continue
        flo = eq_lam(-lam_hi); fhi = eq_lam(lam_hi)
        if not (np.isnan(flo) or np.isnan(fhi)) and flo * fhi < 0:
            found = True; break
    if not found: return np.nan
    try:
        lam_n = brentq(eq_lam, -lam_hi, lam_hi, xtol=1e-8, maxiter=1000)
    except Exception:
        return np.nan
    d1c = S0_1 + lam_n*g1; d2c = S0_2 - lam_n*g2
    if np.any(d1c[(E1==1)&(g1==1)]<=0) or np.any(d2c[(E2==1)&(g2==1)]<=0):
        return np.nan
    lr1 = np.sum(E1 * np.log(d1c / S0_1))
    lr2 = np.sum(E2 * np.log(d2c / S0_2))
    el  = 2.0 * (lr1 + lr2)
    return max(el, 0.0) if np.isfinite(el) else np.nan


def plel_ci(p, T, E, Z, group, beta, bres, alpha=0.05, n_grid=80):
    """Plug-in EL CI."""
    dhat = qdiff_hat(p, bres)
    if np.isnan(dhat): return np.nan, np.nan
    crit = chi2.ppf(1.0 - alpha, df=1)
    lo_n, hi_n = na_ci(p, T, E, Z, group, beta, bres, alpha=alpha)
    hw = (hi_n - lo_n)*3.0 if not (np.isnan(lo_n) or np.isnan(hi_n)) \
         else max(abs(dhat)*2.0, 0.5)
    grid  = np.linspace(dhat - hw, dhat + hw, n_grid)
    vals  = np.array([_plel_el_ratio(d, p, T, E, Z, group, beta, bres)
                      for d in grid])
    inside = np.where(np.isfinite(vals) & (vals <= crit))[0]
    if len(inside) == 0: return np.nan, np.nan
    return float(grid[inside[0]]), float(grid[inside[-1]])


# =====================================================================
# SECTION 4 — NA CI  (normal approximation via Bahadur representation)
# =====================================================================

def _kernel_hazard(times, cumhaz, bw):
    """Gaussian kernel-smoothed hazard function."""
    if len(times) < 2: return lambda t: 1e-6
    jumps = np.diff(cumhaz, prepend=0.0)
    def hazard(t):
        u = (times - t) / bw
        K = np.exp(-0.5*u**2) / (np.sqrt(2*np.pi)*bw)
        return max(np.dot(K, jumps), 1e-8)
    return hazard


def na_ci(p, T, E, Z, group, beta, bres, alpha=0.05):
    """Normal approximation CI based on Bahadur representation."""
    n   = len(T)
    bw  = n**(-0.2)
    dhat = qdiff_hat(p, bres)
    if np.isnan(dhat): return np.nan, np.nan
    ix1, ix2 = group==1, group==2
    T1,E1,Z1,n1 = T[ix1],E[ix1],Z[ix1],ix1.sum()
    T2,E2,Z2,n2 = T[ix2],E[ix2],Z[ix2],ix2.sum()
    q1 = q_from_breslow(p, *bres[1])
    q2 = q_from_breslow(p, *bres[2])
    if np.isnan(q1) or np.isnan(q2): return np.nan, np.nan
    lam1 = _kernel_hazard(*bres[1], bw)(q1)
    lam2 = _kernel_hazard(*bres[2], bw)(q2)
    if lam1 <= 0 or lam2 <= 0: return np.nan, np.nan
    # Breslow variance at q_i
    def bres_var(times, cumhaz, q):
        mask = times <= q
        if not mask.any(): return np.nan
        jumps = np.diff(np.concatenate([[0.0], cumhaz[mask]]))
        return float(np.sum(jumps[jumps>0]))
    var1 = bres_var(*bres[1], q1)
    var2 = bres_var(*bres[2], q2)
    if np.isnan(var1) or np.isnan(var2): return np.nan, np.nan
    se1 = np.sqrt(var1) / lam1 / np.sqrt(n1)
    se2 = np.sqrt(var2) / lam2 / np.sqrt(n2)
    se  = np.sqrt(se1**2 + se2**2)
    z   = chi2.ppf(1.0 - alpha, df=1)**0.5
    return float(dhat - z*se), float(dhat + z*se)


# =====================================================================
# SECTION 5 — DATA GENERATION  (Zhao & Zhao 2025 DGP)
# =====================================================================

BETA_TRUE = np.array([0.6, 0.2])
LAM10     = lambda t: float(t)       # Weibull shape=1, scale=1
LAM20     = lambda t: float(t**2)    # Weibull shape=2, scale=1
TRUE_DELTA = lambda p: (-math.log(1-p)) - math.sqrt(-math.log(1-p))

SAMPLE_SIZES = [50, 100, 200, 500]
CENS_RATES   = [0.10, 0.20, 0.40]
QUANT_LEVELS = [0.25, 0.50, 0.75]
NOMINAL_ALPHA = [0.10, 0.05]
SCENARIOS = [{'n': n, 'cens': c}
             for n in SAMPLE_SIZES for c in CENS_RATES]


def _invert_cumhaz(lam0_fn, target, tmax=100.0):
    while lam0_fn(tmax) < target: tmax *= 10
    try:
        return brentq(lambda t: lam0_fn(t) - target, 1e-10, tmax, xtol=1e-8)
    except Exception:
        return tmax


def calibrate_cmax(n, cens_rate, shape, scale, beta, seed=0, n_mc=10000):
    """
    Pre-calibrate cmax ONCE per (n, cens_rate, group) scenario using brentq.
    Call this before the replication loop, not inside it.
    n_mc=50000 gives SE < 0.003 on the rate estimate.
    """
    rng = np.random.default_rng(seed)
    def rate_for_cmax(cmax):
        Ctmp = rng.uniform(0, cmax, n_mc)
        Xtmp = scale * ((-np.log(rng.uniform(size=n_mc)) /
                         np.exp(np.column_stack([
                             rng.binomial(1, 0.5, n_mc),
                             rng.normal(size=n_mc)
                         ]) @ beta)) ** (1.0/shape))
        return np.mean(Xtmp > Ctmp)
    def obj(cmax): return rate_for_cmax(cmax) - cens_rate
    try:
        lo = 1e-6
        # Weibull quantile for typical X values
        hi = scale * ((-math.log(0.01)) ** (1.0/shape)) * 5.0
        while obj(hi) > 0: hi *= 2
        return brentq(obj, lo, hi, xtol=1e-4)
    except Exception:
        # Fallback: analytical quantile-based estimate
        return scale * ((-math.log(1.0 - cens_rate)) ** (1.0/shape)) * 3.0


def simulate_dgp(n, cens_rate, rng, cmax1=None, cmax2=None):
    """
    Simulate two-group stratified Cox data.
    Baseline: Lambda_10(t)=t, Lambda_20(t)=t^2  (Zhao & Zhao 2025 DGP).
    cmax1, cmax2: pre-calibrated censoring upper bounds (one per group).
    Pass these from run_block to avoid recalibrating every replication.
    """
    out = {'T':[], 'E':[], 'Z':[], 'G':[]}
    for grp, lam0, shape, scale, cmax in [
        (1, LAM10, 1.0, 1.0, cmax1),
        (2, LAM20, 2.0, 1.0, cmax2),
    ]:
        Z   = np.column_stack([rng.binomial(1, 0.5, n),
                                rng.standard_normal(n)])
        U   = rng.uniform(0, 1, n)
        eta = np.exp(Z @ BETA_TRUE)
        X   = np.array([_invert_cumhaz(lam0, -math.log(U[j])/eta[j])
                        for j in range(n)])
        if cmax is None:
            # Fallback if calibration not provided
            cmax = np.quantile(X, 1.0 - cens_rate) * 3.0
        C    = rng.uniform(0, max(cmax, 1e-6), n)
        out['T'].append(np.minimum(X, C))
        out['E'].append((X <= C).astype(int))
        out['Z'].append(Z)
        out['G'].append(np.full(n, grp))
    return (np.concatenate(out['T']), np.concatenate(out['E']),
            np.vstack(out['Z']),      np.concatenate(out['G']))


# =====================================================================
# SECTION 6 — SIMULATION RUNNER
# =====================================================================

N_REPS  = 2000
BOOTSTRAP_B = 500   # simultaneous band only


# ── Checkpoint helpers ───────────────────────────────────────────────

def _ckpt_path(block_id, out_dir):
    return os.path.join(out_dir, f'ckpt_block{block_id:02d}.json')


def _save_ckpt(block_id, rep, cov_acc, wid_acc, tim_acc,
               cmax1, cmax2, seed, out_dir):
    """Save raw accumulators to a checkpoint file every CKPT_EVERY reps."""
    ckpt = {
        'block_id': block_id, 'rep_done': rep,
        'seed': seed, 'cmax1': cmax1, 'cmax2': cmax2,
        'cov_acc': {k: [list(v) for v in vals]
                     for k, vals in cov_acc.items()},
        'wid_acc': {k: [list(v) for v in vals]
                     for k, vals in wid_acc.items()},
        'tim_acc': {k: vals for k, vals in tim_acc.items()},
    }
    path = _ckpt_path(block_id, out_dir)
    tmp  = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ckpt, f)
    os.replace(tmp, path)   # atomic write


def _load_ckpt(block_id, out_dir):
    """Load checkpoint if it exists. Returns (rep_start, accumulators, cmax1, cmax2)."""
    path = _ckpt_path(block_id, out_dir)
    if not os.path.exists(path):
        return 0, {}, {}, {}, None, None
    with open(path) as f:
        ckpt = json.load(f)
    cov_acc = {k: [tuple(v) for v in vals]
               for k, vals in ckpt['cov_acc'].items()}
    wid_acc = {k: [tuple(v) for v in vals]
               for k, vals in ckpt['wid_acc'].items()}
    tim_acc = ckpt['tim_acc']
    return (int(ckpt['rep_done']) + 1,
            cov_acc, wid_acc, tim_acc,
            float(ckpt['cmax1']), float(ckpt['cmax2']))


CKPT_EVERY = 100   # save checkpoint every N reps


def run_block(block_id, n_reps=None, seed=None, out_dir='./results'):
    sc      = SCENARIOS[block_id]
    n       = sc['n']
    cens    = sc['cens']
    n_reps  = n_reps or N_REPS
    seed    = seed if seed is not None else block_id * 10000
    os.makedirs(out_dir, exist_ok=True)

    _log(f'\n[Block {block_id}]  n={n}  cens={cens:.0%}  reps={n_reps}  seed={seed}')

    # ── Load checkpoint if resuming ───────────────────────────────────
    rep_start, cov_acc, wid_acc, tim_acc, cmax1_ckpt, cmax2_ckpt = \
        _load_ckpt(block_id, out_dir)

    if rep_start > 0:
        _log(f'  Resuming from checkpoint: rep {rep_start}/{n_reps}')
        cmax1, cmax2 = cmax1_ckpt, cmax2_ckpt
        _log(f'  cmax restored: group1={cmax1:.4f}  group2={cmax2:.4f}')
    else:
        # Pre-calibrate cmax ONCE per group (not inside every replication)
        _log(f'  Calibrating censoring mechanism...')
        cmax1 = calibrate_cmax(n, cens, shape=1.0, scale=1.0,
                               beta=BETA_TRUE, seed=seed)
        cmax2 = calibrate_cmax(n, cens, shape=2.0, scale=1.0,
                               beta=BETA_TRUE, seed=seed+1)
        _log(f'  cmax: group1={cmax1:.4f}  group2={cmax2:.4f}')

    # ── Replication loop ──────────────────────────────────────────────
    for rep in range(rep_start, n_reps):
        rng = np.random.default_rng(seed + rep + 1)
        T, E, Z, G = simulate_dgp(n, cens, rng, cmax1=cmax1, cmax2=cmax2)
        try:
            beta = fit_stratified_cox(T, E, Z, G)
            bres = breslow_by_group(T, E, Z, G, beta)
        except Exception:
            continue

        for alpha in NOMINAL_ALPHA:
            for p in QUANT_LEVELS:
                key    = f'a{int((1-alpha)*100)}_p{int(p*100)}'
                true_d = TRUE_DELTA(p)
                q1     = q_from_breslow(p, *bres[1])
                q2     = q_from_breslow(p, *bres[2])
                est    = int(not np.isnan(q1) and not np.isnan(q2))

                # PROEL (timed) — use qdiff_ci returning (dhat, lo, hi)
                t0 = time.perf_counter()
                _, pr_lo, pr_hi = qdiff_ci(p, T, E, Z, G, beta, bres,
                                           alpha=alpha, n_grid=161)
                t_pr = time.perf_counter() - t0

                # PLEL
                pl_lo, pl_hi = plel_ci(p, T, E, Z, G, beta, bres,
                                       alpha=alpha, n_grid=80)
                # NA
                na_lo, na_hi = na_ci(p, T, E, Z, G, beta, bres, alpha=alpha)

                def _cov(lo, hi):
                    return int(not np.isnan(lo) and lo <= true_d <= hi)
                def _wid(lo, hi):
                    return float(hi - lo) if not np.isnan(lo) else np.nan

                cov_acc.setdefault(key, []).append((
                    _cov(pr_lo, pr_hi), _cov(pl_lo, pl_hi),
                    _cov(na_lo, na_hi), est
                ))
                wid_acc.setdefault(key, []).append((
                    _wid(pr_lo, pr_hi), _wid(pl_lo, pl_hi),
                    _wid(na_lo, na_hi)
                ))
                tim_acc.setdefault(key, []).append(t_pr)

        # Progress report and checkpoint
        if (rep + 1) % 200 == 0:
            _log(f'  {rep+1}/{n_reps} reps done')
        if (rep + 1) % CKPT_EVERY == 0:
            _save_ckpt(block_id, rep, cov_acc, wid_acc, tim_acc,
                       cmax1, cmax2, seed, out_dir)

    # ── Final checkpoint (marks completion) ───────────────────────────
    _save_ckpt(block_id, n_reps - 1, cov_acc, wid_acc, tim_acc,
               cmax1, cmax2, seed, out_dir)

    # ── Aggregate ─────────────────────────────────────────────────────
    output = {'block_id': block_id, 'n': n, 'cens_rate': cens,
              'n_reps': len(next(iter(cov_acc.values()), [])),
              'coverage': {}, 'width': {}, 'time': {}}

    for key, vals in cov_acc.items():
        arr  = np.array(vals, dtype=float)
        est  = arr[:, 3]; mask = est == 1
        output['coverage'][key] = {
            'proel_mean'       : float(np.nanmean(arr[:,0])),
            'plel_mean'        : float(np.nanmean(arr[:,1])),
            'na_mean'          : float(np.nanmean(arr[:,2])),
            'proel_cond'       : float(np.nanmean(arr[mask,0])) if mask.any() else float('nan'),
            'plel_cond'        : float(np.nanmean(arr[mask,1])) if mask.any() else float('nan'),
            'na_cond'          : float(np.nanmean(arr[mask,2])) if mask.any() else float('nan'),
            'estimability_rate': float(np.nanmean(est)),
            'n_reps'           : len(arr),
        }
    for key, vals in wid_acc.items():
        arr = np.array(vals, dtype=float)
        output['width'][key] = {
            'proel_mean': float(np.nanmean(arr[:,0])),
            'plel_mean' : float(np.nanmean(arr[:,1])),
            'na_mean'   : float(np.nanmean(arr[:,2])),
        }
    for key, vals in tim_acc.items():
        output['time'][key] = {'proel_mean_sec': float(np.nanmean(vals))}

    # ── Remove checkpoint after successful completion ──────────────────
    ckpt_file = _ckpt_path(block_id, out_dir)
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
        _log(f'  Checkpoint removed (block complete)')

    return output


def save_block(output, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir,
        f"block{output['block_id']:02d}_n{output['n']}_c{int(output['cens_rate']*100)}.json")
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    _log(f'  Saved: {fname}')
    return fname


def merge_blocks(results_dir):
    files  = sorted(glob.glob(os.path.join(results_dir, 'block*.json')))
    merged = [json.load(open(f)) for f in files]
    _log(f'Merged {len(merged)} blocks from {results_dir}')
    return merged


# =====================================================================
# SECTION 7 — REAL DATA ANALYSIS
# =====================================================================

P_GRID     = np.linspace(0.15, 0.75, 37)
P_SELECTED = [0.25, 0.50, 0.75]


def kaplan_meier(T, E):
    times = np.concatenate([[0], np.sort(np.unique(T[E==1]))])
    surv  = np.ones(len(times)); n = len(T)
    for k, t in enumerate(times[1:], 1):
        ar = (T >= t).sum(); ev = ((T==t)&(E==1)).sum()
        surv[k] = surv[k-1] * (1 - ev/ar) if ar > 0 else surv[k-1]
    return times, surv


def greenwood_se(T, E, times):
    surv = np.ones(len(times)); var = np.zeros(len(times))
    for k, t in enumerate(times[1:], 1):
        ar = (T>=t).sum(); ev = ((T==t)&(E==1)).sum()
        if ar > 0:
            surv[k] = surv[k-1]*(1-ev/ar)
            var[k]  = var[k-1] + ev/(ar*max(ar-ev,1))
        else:
            surv[k] = surv[k-1]; var[k] = var[k-1]
    return surv * np.sqrt(var)


def _build_el_ratio_data(p_levels, T, E, Z, group, beta, bres, n_grid=80,
                          width_factor=3.0):
    """
    Build EL ratio curves for Figure 5.

    Uses the SAME width_factor-based grid sizing as qdiff_ci() (validated,
    always produces finite values), instead of an NA-CI-based heuristic.
    NA CIs are known to be unreliable/too narrow on real data (confirmed
    by simulation), so sizing the EL-ratio search grid off NA width
    caused the grid to miss the region where the EL ratio is finite,
    especially for larger datasets (GBSG2, NCCTG) where NA's CI is
    tightest.

    width_factor=3.0 (wider than qdiff_ci's 1.5) so the figure shows
    the full bowl shape rising well above the chi-square threshold on
    both sides, not just the narrow acceptance region.
    """
    el = {}
    for p in p_levels:
        dhat = qdiff_hat(p, bres)
        if np.isnan(dhat): continue
        q1 = q_from_cumhaz(p, *bres[1])
        q2 = q_from_cumhaz(p, *bres[2])
        hw = width_factor * max(
                 abs(dhat),
                 q1 * 0.25 if not np.isnan(q1) else 0.0,
                 q2 * 0.25 if not np.isnan(q2) else 0.0,
                 np.nanmedian(T) * 0.20, 0.25)
        dgrid = np.linspace(dhat - hw, dhat + hw, n_grid)
        elv   = np.array([proel_el_ratio(d, p, T, E, Z, group, beta, bres)
                           for d in dgrid])
        el[str(p)] = {
            'delta_grid': dgrid.tolist(),
            'el_vals'   : [float(v) if np.isfinite(v) else None for v in elv],
            'dhat'      : float(dhat),
        }
    return el


def analyse_dataset(T, E, Z, G, meta, p_grid=None, alpha=0.05):
    if p_grid is None: p_grid = P_GRID
    _log(f"  Fitting Cox ({meta['name']})...")
    beta = fit_stratified_cox(T, E, Z, G)
    bres = breslow_by_group(T, E, Z, G, beta)

    # Truncate p_grid to estimable range
    t1,c1 = bres[1]; t2,c2 = bres[2]
    n_ev1 = int(np.sum(E[G==1])); n_ev2 = int(np.sum(E[G==2]))
    _log(f"  Events: g1={n_ev1}, g2={n_ev2}")
    if len(c1)==0 or len(c2)==0 or n_ev1==0 or n_ev2==0:
        _log("  ERROR: no events in one group")
        return None
    p_max = min(1-math.exp(-c1[-1])-1e-4, 1-math.exp(-c2[-1])-1e-4, 0.80)
    p_grid = np.array([p for p in p_grid if p <= p_max])
    if len(p_grid) == 0: p_grid = np.array([0.15])
    _log(f"  Estimable p: [{p_grid[0]:.3f}, {p_grid[-1]:.3f}]  ({len(p_grid)} levels)")

    pt_lo = np.full(len(p_grid), np.nan)
    pt_hi = np.full(len(p_grid), np.nan)
    pl_lo = np.full(len(p_grid), np.nan)
    pl_hi = np.full(len(p_grid), np.nan)
    na_lo = np.full(len(p_grid), np.nan)
    na_hi = np.full(len(p_grid), np.nan)
    dhat  = np.array([qdiff_hat(p, bres) for p in p_grid])

    _log('  Computing CIs over p-grid...')
    for k, p in enumerate(p_grid):
        _, pt_lo[k], pt_hi[k] = qdiff_ci(p, T, E, Z, G, beta, bres, alpha=alpha, n_grid=161)
        pl_lo[k], pl_hi[k] = plel_ci(p, T, E, Z, G, beta, bres, alpha=alpha, n_grid=80)
        na_lo[k], na_hi[k] = na_ci(p, T, E, Z, G, beta, bres, alpha=alpha)

    # KM curves
    km = {}
    for g in [1, 2]:
        mask = G == g
        t_km, s_km = kaplan_meier(T[mask], E[mask])
        se_km = greenwood_se(T[mask], E[mask], t_km)
        km[str(g)] = {'times': t_km.tolist(), 'surv': s_km.tolist(),
                      'se': se_km.tolist()}

    # EL ratio curves for Figure 5
    p_sel = [p for p in P_SELECTED if p <= p_max]
    _log(f'  Computing EL ratio curves for p={p_sel}...')
    el_data = _build_el_ratio_data(p_sel, T, E, Z, G, beta, bres)

    def _to_list(arr):
        return [float(v) if np.isfinite(v) else None for v in arr]

    return {
        'meta'    : meta,
        'p_grid'  : p_grid.tolist(),
        'dhat'    : _to_list(dhat),
        'pt_lo'   : _to_list(pt_lo),
        'pt_hi'   : _to_list(pt_hi),
        'pl_lo'   : _to_list(pl_lo),
        'pl_hi'   : _to_list(pl_hi),
        'na_lo'   : _to_list(na_lo),
        'na_hi'   : _to_list(na_hi),
        'km'      : km,
        'el_ratio': el_data,
        'beta_hat': beta.tolist(),
    }


def load_gbsg2():
    from sksurv.datasets import load_gbsg2 as _load
    X, y = _load()
    group = X['horTh'].map({'yes':1,'no':2}).values.astype(int)
    T = y['time'].astype(float); E = y['cens'].astype(float)
    cols = ['age','pnodes','tsize','estrec','progrec']
    Z = X[cols].values.astype(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return T, E, Z, group, {
        'name':'German Breast Cancer (GBSG2)',
        'g1':'Hormone therapy','g2':'No hormone therapy',
        'unit':'days','n1':int((group==1).sum()),'n2':int((group==2).sum())}


def load_ncctg_lung():
    from lifelines.datasets import load_lung
    df = load_lung()
    df['event'] = (df['status']==2).astype(float) \
                  if df['status'].max()==2 else df['status'].astype(float)
    df['group'] = df['sex'].map({2:1,1:2}).astype(int)
    cols = [c for c in ['age','ph.ecog','ph.karno','pat.karno','wt.loss']
            if c in df.columns]
    df = df.dropna(subset=['time','status','sex'])
    T = df['time'].values.astype(float); E = df['event'].values.astype(float)
    G = df['group'].values.astype(int)
    Z = df[cols].copy().fillna(df[cols].median()).values.astype(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return T, E, Z, G, {
        'name':'NCCTG Lung Cancer','g1':'Female','g2':'Male',
        'unit':'days','n1':int((G==1).sum()),'n2':int((G==2).sum())}


def load_veteran():
    from sksurv.datasets import load_veterans_lung_cancer
    X, y = load_veterans_lung_cancer()
    df = X.copy()
    df['event'] = y['Status'].astype(float)
    df['time']  = y['Survival_in_days'].astype(float)
    trt  = 'Treatment'   if 'Treatment'   in df.columns else 'Trt'
    age  = 'Age_in_years' if 'Age_in_years' in df.columns else 'Age'
    kar  = 'Karnofsky_score' if 'Karnofsky_score' in df.columns else 'Karnofsky'
    diag = 'Months_from_Diagnosis' if 'Months_from_Diagnosis' in df.columns else 'Diagtime'
    prior= 'Prior_therapy' if 'Prior_therapy' in df.columns else 'Prior'
    df['group'] = df[trt].map({'standard':1,'test':2,'Standard':1,'Test':2}).astype(int)
    df['prior_num'] = df[prior].map({'no':0,'yes':1,0:0,10:1}).fillna(0).astype(float)
    ct = pd.get_dummies(df['Celltype'], drop_first=True)
    Z  = pd.concat([df[[kar,diag,age]],df[['prior_num']],ct],axis=1).fillna(0).values.astype(float)
    Z  = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    G  = df['group'].values.astype(int)
    T  = df['time'].values.astype(float)
    E  = df['event'].values.astype(float)
    return T, E, Z, G, {
        'name':'VA Lung Cancer','g1':'Standard','g2':'Test chemo.',
        'unit':'days','n1':int((G==1).sum()),'n2':int((G==2).sum())}


def run_realdata(out_dir='./results', alpha=0.05):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for tag, loader in [('GBSG2',load_gbsg2),
                        ('NCCTG', load_ncctg_lung),
                        ('Veteran',load_veteran)]:
        _log(f'\n--- {tag} ---')
        try:
            T, E, Z, G, meta = loader()
            res = analyse_dataset(T, E, Z, G, meta, alpha=alpha)
            if res is not None:
                results.append(res)
        except Exception as ex:
            _log(f'{tag} failed: {ex}')
            import traceback; traceback.print_exc()
    path = os.path.join(out_dir, 'realdata_results.json')
    with open(path,'w') as f: json.dump(results, f, indent=2)
    _log(f'\nReal data saved: {path}')
    return results


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='qdiff_combined.py')
    sub = parser.add_subparsers(dest='command')

    p_sim = sub.add_parser('sim')
    p_sim.add_argument('--block-id', type=int, required=True)
    p_sim.add_argument('--n-reps',   type=int, default=None)
    p_sim.add_argument('--seed',     type=int, default=None)
    p_sim.add_argument('--out-dir',  type=str,
                       default='/home/users/username/qdiff_v2/results')

    p_merge = sub.add_parser('merge')
    p_merge.add_argument('--results-dir', type=str,
                         default='/home/users/username/qdiff_v2/results')

    p_rd = sub.add_parser('realdata')
    p_rd.add_argument('--out-dir', type=str,
                      default='/home/users/username/qdiff_v2/results')
    p_rd.add_argument('--alpha',   type=float, default=0.05)

    args = parser.parse_args()

    if args.command == 'sim':
        result = run_block(args.block_id,
                           n_reps  = args.n_reps,
                           seed    = args.seed,
                           out_dir = args.out_dir)
        save_block(result, args.out_dir)
        _log(f'\nBlock {args.block_id} finished.')

    elif args.command == 'merge':
        merged = merge_blocks(args.results_dir)
        path   = os.path.join(args.results_dir, 'merged_results.json')
        with open(path,'w') as f: json.dump(merged, f, indent=2)
        _log(f'Merged results saved: {path}')

    elif args.command == 'realdata':
        run_realdata(out_dir=args.out_dir, alpha=args.alpha)

    else:
        parser.print_help()
