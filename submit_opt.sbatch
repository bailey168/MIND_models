#!/bin/bash
#SBATCH --job-name=MIND_processing
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/opt_logs/MIND_processing_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/opt_logs/MIND_processing_%J.err
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=600M
#SBATCH --cpus-per-task=1
#SBATCH --array=0-208

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

PATHS_FILE=/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/paths_aparc_opt.loc
OUTPUT_DIR=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/opt_results
PARCELLATION=aparc

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1000))
mapfile -s $LINE_NUM \
    -t -n1 SUBJECT < "$PATHS_FILE"

python3 run_MIND_script.py --subj_dir "${SUBJECT[0]}" --output_dir "$OUTPUT_DIR" --parcellation "$PARCELLATION"
