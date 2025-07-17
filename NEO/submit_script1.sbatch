#!/bin/bash
#SBATCH --job-name=add_FC_csv_07-17
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/add_FC_csv_07-17_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/add_FC_csv_07-17_%J.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=2
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/NEO/add_FC_csv.py