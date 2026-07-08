#!/bin/bash
#SBATCH --job-name=realdata_v2
#SBATCH --account=PROJECT_ACCOUNT
#SBATCH --partition=compute_partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/username/qdiff_v2/logs/realdata_v2_%j.out
#SBATCH --error=/home/users/username/qdiff_v2/logs/realdata_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.edu

mkdir -p /home/users/username/qdiff_v2/logs
mkdir -p /home/users/username/qdiff_v2/results
cd /home/users/username/qdiff_v2

module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate /home/users/username/myenv

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Started: $(date)"
python qdiff_combined_v2.py realdata \
    --out-dir /home/users/username/qdiff_v2/results
echo "Completed: $(date)"
