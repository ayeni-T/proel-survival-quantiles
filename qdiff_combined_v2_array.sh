#!/bin/bash
#SBATCH --job-name=qdiff_v2
#SBATCH --account=PROJECT_ACCOUNT
#SBATCH --partition=compute_partition
#SBATCH --array=0-11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=120:00:00
#SBATCH --output=/home/users/username/qdiff_v2/logs/comb_%A_%a.out
#SBATCH --error=/home/users/username/qdiff_v2/logs/comb_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.edu
#SBATCH --exclude=node01,node02,node03,node04,node05

# ── Scenario map ──────────────────────────────────────────────────────
# Block  0: n= 50, cens=10%    Block  1: n= 50, cens=20%
# Block  2: n= 50, cens=40%    Block  3: n=100, cens=10%
# Block  4: n=100, cens=20%    Block  5: n=100, cens=40%
# Block  6: n=200, cens=10%    Block  7: n=200, cens=20%
# Block  8: n=200, cens=40%    Block  9: n=500, cens=10%
# Block 10: n=500, cens=20%    Block 11: n=500, cens=40%
#
# ── v2 NOTE ──────────────────────────────────────────────────────────
# This runs qdiff_combined_v2.py (fixed nuisance-grid search + cached
# risk sums), writing ONLY to qdiff_v2/results and qdiff_v2/logs --
# the original qdiff/results and qdiff/logs are untouched.
#
# Benchmarked at the n=500 scale that previously needed 2 job
# submissions (~156h total) to finish 2000 reps under the original
# code: v2 measures ~9.5s/qdiff_ci call here vs. the old code's
# effective ~47s/call (the old code recomputed risk sums twice
# redundantly per grid point; v2 caches them once per replication).
# Estimated ~32h for a full n=500 block, so --time=120:00:00 below
# should comfortably cover it in a single submission -- but the
# checkpoint/resume logic is still in place as a safety net if a node
# gets preempted or a block runs longer than expected.

mkdir -p /home/users/username/qdiff_v2/logs
mkdir -p /home/users/username/qdiff_v2/results
cd /home/users/username/qdiff_v2

module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate /home/users/username/myenv

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "============================================"
echo "Job:     qdiff_combined_v2"
echo "Block:   ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "============================================"

SEED=$(( SLURM_ARRAY_TASK_ID * 10000 ))
REPS=${REPS:-2000}

python qdiff_combined_v2.py sim \
    --block-id ${SLURM_ARRAY_TASK_ID} \
    --n-reps   ${REPS} \
    --seed     ${SEED} \
    --out-dir  /home/users/username/qdiff_v2/results

echo "============================================"
echo "Block ${SLURM_ARRAY_TASK_ID} completed: $(date)"
echo "============================================"
