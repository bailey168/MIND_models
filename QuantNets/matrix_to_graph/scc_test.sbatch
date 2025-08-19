#!/bin/bash
#SBATCH --job-name=convert_graph
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=36
#SBATCH --array=0
#SBATCH --partition=low-moby

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin_scc/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/QuantNets/matrix_to_graph/matrix_to_graph_parallelized_scc.py