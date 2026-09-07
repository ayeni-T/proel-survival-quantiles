# PROEL: Profile Empirical Likelihood for Survival Quantile Differences

Code accompanying:

**Profile Empirical Likelihood Confidence Bands for the Difference of Survival Quantiles under Stratified Cox Models**
Taiwo Michael Ayeni and Yichuan Zhao
Department of Mathematics and Statistics, Georgia State University, Atlanta, GA, USA

Correspondence: your_email@your_institution.edu

> **Note:** account/path placeholders throughout the SLURM scripts and
> main script (`YOUR_ACCOUNT`, `YOUR_PARTITION`, `/path/to/your/workdir`,
> `/path/to/your/conda/env`, `your_email@your_institution.edu`) need to
> be replaced with your own cluster account, working directory, conda
> environment path, and notification email before running.

This repository contains the simulation and real-data analysis code for comparing four confidence-interval procedures for the difference in survival quantiles between two strata of a stratified Cox model:

- **PROEL** -- Profile empirical likelihood with a jointly profiled nuisance quantile (proposed method)
- **PLEL** -- Plug-in empirical likelihood (nuisance quantile fixed at its point estimate)
- **NA** -- Normal approximation via the Bahadur representation
- **Bartlett-corrected PROEL** -- A bootstrap Bartlett-type scale correction applied to PROEL, addressing finite-sample undercoverage at extreme quantiles under heavy censoring (see paper Section 2.6)

## Files

| File | Purpose |
|---|---|
| `qdiff_combined_v2.py` | Main script: Monte Carlo simulation study and real-data analysis for all four methods |
| `qdiff_combined_v2_array.sh` | SLURM array job script for running the full simulation (12 scenario blocks) on an HPC cluster |
| `realdata_v2.sh` | SLURM job script for the real-data analysis |
| `run_pilot.py` | Quick pilot check (20 replications on 3 scenario blocks) for validating a code change before a full cluster run |
| `qdiff_plots_v2.py` | Generates all figures (coverage, width, KM curves with model-based baseline overlay, quantile-difference bands, profile likelihood ratio curves, Bartlett factor vs. n) |
| `fig_config.py` | Shared publication-quality figure styling (fonts, colors, sizes) |
| `plot_realdata_only.py` | Convenience script to regenerate only the real-data figures (3, 4, 5a-c) |

## Requirements

- numpy
- pandas
- scipy
- matplotlib

## Usage

### Simulation study

```bash
# Quick local test (one block, 5 replications)
python qdiff_combined_v2.py sim --block-id 0 --n-reps 5 --out-dir ./results

# Full run via SLURM array (12 blocks x 2000 replications each)
sbatch qdiff_combined_v2_array.sh

# Merge block results after the array job finishes
python qdiff_combined_v2.py merge --results-dir ./results
```

### Real-data analysis

```bash
python qdiff_combined_v2.py realdata --out-dir ./results
```

The three datasets used (GBSG2, NCCTG lung, VA lung) are loaded via the scikit-survival and lifelines Python packages; see the paper for full source citations. The real-data output now also includes, per group, the covariate-adjusted (Z=0 reference level) baseline survival curve from the same stratified Cox fit used throughout, alongside the raw Kaplan-Meier curve, for direct comparison in Figure 3.

### Figures

```bash
python qdiff_plots_v2.py --sim-results ./results/merged_results.json \
    --rd-results ./results/realdata_results.json --out-dir ./figures
```

### Pilot / sanity check

```bash
python run_pilot.py
```

## Notes on the simulation design

The data-generating process follows a stratified Cox model with two covariates, shared regression coefficient, and Weibull baseline hazards (shape 1 and 2 respectively in the two strata), giving a closed-form true quantile difference. The design crosses four sample sizes, three censoring rates, and three quantile levels, evaluated at 90% and 95% nominal confidence with 2,000 Monte Carlo replications per scenario. See the paper for full methodological and simulation detail.

## Notes on the Bartlett correction

The bootstrap scale factor $\hat{b}$ used in the Bartlett-type correction is derived from case-resampled bootstrap replicates, seeded deterministically per dataset/quantile combination (via `zlib.crc32`, not Python's built-in `hash()`, which is randomized per-process for strings and does not give reproducible results across runs).
