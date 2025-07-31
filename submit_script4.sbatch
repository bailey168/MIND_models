#!/bin/bash
#SBATCH --job-name=elasticnet_TMT_07-30
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_TMT_07-30_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_TMT_07-30_%J.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --array=0

module load PYTHON/3.6
module load PYTHON/3.6

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/models_elasticnet/elasticnet_TMT.py