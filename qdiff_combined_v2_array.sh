#!/bin/bash
#SBATCH --job-name=qdiff_v2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --array=0-11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=120:00:00
#SBATCH --output=/path/to/your/workdir/logs/comb_%A_%a.out
#SBATCH --error=/path/to/your/workdir/logs/comb_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@your_institution.edu
#SBATCH --exclude=NODE1,NODE2   # optional: exclude specific problematic nodes on your cluster, if any

# ── Scenario map ──────────────────────────────────────────────────────
# Block  0: n= 50, cens=10%    Block  1: n= 50, cens=20%
# Block  2: n= 50, cens=40%    Block  3: n=100, cens=10%
# Block  4: n=100, cens=20%    Block  5: n=100, cens=40%
# Block  6: n=200, cens=10%    Block  7: n=200, cens=20%
# Block  8: n=200, cens=40%    Block  9: n=500, cens=10%
# Block 10: n=500, cens=20%    Block 11: n=500, cens=40%
#
# ── FIXES INCLUDED IN THIS SCRIPT ───────────────────────────────────
# Runs qdiff_combined_v2.py, which includes the following fixes
# relative to earlier internal versions:
#
# 1. PLEL risk-set fix (same as v3): _plel_el_ratio/plel_ci now use
#    risk-set sums (R1/R2, matching breslow_by_group and PROEL's own
#    construction) instead of the old S0_i = exp(Z_i@beta)/n
#    individual-weight term, which produced PLEL intervals far too
#    wide / near-100% coverage purely as a scaling artifact.
#
# 2. Bartlett-type finite-sample correction for PROEL, applied
#    UNIFORMLY across all three quantile levels (p=0.25, 0.50, 0.75)
#    by default (BARTLETT_P_LEVELS in the script) -- a smaller oracle
#    diagnostic suggested the correction is a near no-op at p=0.25/0.50,
#    but for a publication-defensible result, that should be a direct
#    output of this same production procedure rather than inferred from
#    a smaller side-check, so it is computed everywhere. Benchmarked
#    cost @ n=500: ~38s per b_hat estimate (B=150-200) vs ~12s for a
#    full existing PROEL CI -- estimated ~100h total for the heaviest
#    (n=500) blocks with all 3 quantile levels active, vs ~32-39h
#    without the correction at all. Still inside the 120h SBATCH budget,
#    with less slack than a p=0.75-only scope would leave.
#
#    To narrow the correction back down to p=0.75 only (e.g. if a
#    specific block's wall-clock becomes a real problem -- checkpointing
#    protects against losing progress either way, but does not reduce
#    total compute cost), set BARTLETT_P_LEVELS=0.75 when submitting.
#    To disable the Bartlett correction entirely for a given run
#    (PLEL-fix-only, identical behavior to v3), set SKIP_BARTLETT=1
#    when submitting: `SKIP_BARTLETT=1 sbatch qdiff_combined_v2_array.sh`
#
# Both PROEL and NA are otherwise UNCHANGED from v2/v3.

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
echo "Job:     qdiff_combined_v2 (PLEL fix + Bartlett correction @ p=0.75)"
echo "Block:   ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "============================================"

SEED=$(( SLURM_ARRAY_TASK_ID * 10000 ))
REPS=${REPS:-2000}
# BARTLETT_P_LEVELS=${BARTLETT_P_LEVELS:-}   # default: all of 0.25,0.5,0.75 (unset here)
                                              # set e.g. to "0.75" to narrow scope
SKIP_BARTLETT=${SKIP_BARTLETT:-0}

EXTRA_ARGS=""
if [ "${SKIP_BARTLETT}" = "1" ]; then
    EXTRA_ARGS="--skip-bartlett"
elif [ -n "${BARTLETT_P_LEVELS}" ]; then
    EXTRA_ARGS="--bartlett-p-levels ${BARTLETT_P_LEVELS}"
fi

python qdiff_combined_v2.py sim \
    --block-id ${SLURM_ARRAY_TASK_ID} \
    --n-reps   ${REPS} \
    --seed     ${SEED} \
    --out-dir  /path/to/your/workdir/results \
    ${EXTRA_ARGS}

echo "============================================"
echo "Block ${SLURM_ARRAY_TASK_ID} completed: $(date)"
echo "============================================"

