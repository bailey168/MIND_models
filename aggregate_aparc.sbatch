#!/bin/bash
#SBATCH --job-name=avg_matrix_68
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/matrix_logs/cartbind_avg_matrix_68_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/matrix_logs/cartbind_avg_matrix_68_%J.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

python3 aggregate_aparc.py