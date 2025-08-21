#!/bin/bash
#SBATCH --job-name=GF_FC_matrices
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/GF_FC_matrices_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/GF_FC_matrices_%J.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin_scc/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/create_FC_matrices.py