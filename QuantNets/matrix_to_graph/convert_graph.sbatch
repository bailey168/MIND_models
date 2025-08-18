#!/bin/bash
#SBATCH --job-name=convert_graph
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

module load PYTHON/3.6
module load PYTHON/3.6

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

PATHS_FILE=/home/p/pz249/baileyng/MIND_models/3D+4D/aparc_paths.loc
OUTPUT_DIR=/scratch/p/pz249/baileyng/3D+4D_data/MIND_results/full
PARCELLATION=aparc

mapfile -s $SLURM_ARRAY_TASK_ID \
    -t -n1 SUBJECT < "$PATHS_FILE"

python3 -u /home/p/pz249/baileyng/MIND_models/MIND/run_MIND_script.py --subj_dir "${SUBJECT[0]}" --output_dir "$OUTPUT_DIR" --parcellation "$PARCELLATION"