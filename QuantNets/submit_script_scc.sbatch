#!/bin/bash
#SBATCH --job-name=run_dsst_regression_test_09_03
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/run_dsst_regression_test_09_03_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/run_dsst_regression_test_09_03_%J.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin_scc/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/QuantNets/matrix_to_graph/run_dsst_regression_test.py