#!/bin/bash
#SBATCH --job-name=elasticnet_GF_07-30
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_GF_07-30_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_GF_07-30_%J.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --array=0

echo "Loading modules..."
module load PYTHON/3.6
module load PYTHON/3.6

echo "Activating virtual environment..."
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

echo "Running elastic net analysis script..."
python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/models_elasticnet/elasticnet_GF.py