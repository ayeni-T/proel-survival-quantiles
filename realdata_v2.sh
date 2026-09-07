#!/bin/bash
#SBATCH --job-name=qdiff_v2_realdata
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/your/workdir/logs/realdata_%j.out
#SBATCH --error=/path/to/your/workdir/logs/realdata_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@your_institution.edu

# ── Real-data pipeline: GBSG2, NCCTG, Veteran ─────────────────────────
# Runs analyse_dataset on all three real datasets (Table 5, Figure 5),
# with BOTH the PLEL fix and the Bartlett correction (at p=0.25/0.50/0.75,
# B=150 -- same BARTLETT_B as the simulation grid). Datasets are all
# built-in package data (sksurv.datasets.load_gbsg2 /
# load_veterans_lung_cancer, lifelines.datasets.load_lung) -- no separate
# file uploads needed, just the packages themselves (already confirmed
# available in this environment).
#
# Estimated cost: GBSG2 (n~686, the largest) is the slow one -- a
# 37-point continuous PROEL/PLEL/NA band plus the Bartlett bootstrap at
# 3 quantile levels. Rough estimate ~15-20 min for GBSG2, faster for
# NCCTG/Veteran (smaller n) -- likely 25-40 min total. --time is set
# generously (4h) since this is a first real run and true cost on this
# dataset size hasn't been directly benchmarked yet.
#
# Separate from the qdiff_v4 sim array job -- writes to
# realdata_results.json (distinct filename, no collision with the
# block*.json files the sim array job produces), safe to run alongside
# it in parallel.

mkdir -p /path/to/your/workdir/logs
mkdir -p /path/to/your/workdir/results
cd /path/to/your/workdir

module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate /path/to/your/conda/env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "============================================"
echo "Job:     qdiff_combined_v2 realdata"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "============================================"

python qdiff_combined_v2.py realdata \
    --out-dir /path/to/your/workdir/results \
    --alpha 0.05

echo "============================================"
echo "Real-data analysis completed: $(date)"
echo "============================================"
