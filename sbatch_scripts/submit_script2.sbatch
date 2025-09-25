#!/bin/bash
#SBATCH --job-name=elasticnet_PAL_07-30
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_PAL_07-30_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_PAL_07-30_%J.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

module load PYTHON/3.6
module load PYTHON/3.6

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/models_elasticnet/elasticnet_PAL.py