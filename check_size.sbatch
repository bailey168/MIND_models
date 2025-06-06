#!/bin/bash
#SBATCH --job-name=averaging_matrix
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/matrix_logs/%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/matrix_logs/%J.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

python3 check_size.py