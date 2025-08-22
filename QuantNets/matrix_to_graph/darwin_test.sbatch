#!/bin/bash
#SBATCH --job-name=convert_graph_PAL
#SBATCH --output=/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_PAL_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_PAL_%J.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=12
#SBATCH --array=0
#SBATCH --partition=low-moby

source /scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /scratch/bng/cartbind/code/MIND_models/QuantNets/matrix_to_graph/matrix_to_graph_parallelized.py