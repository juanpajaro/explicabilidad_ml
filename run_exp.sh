#!/bin/bash
#SBATCH --job-name=explications
#SBATCH --output=logs/exp_out_%j.txt
#SBATCH --error=logs/exp_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Load the conda environment
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate ali_env_v2
echo "Starting computational phenotyping job..."
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

srun python3 exp.py