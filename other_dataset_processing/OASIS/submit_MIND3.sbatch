#!/bin/bash
#SBATCH --job-name=MIND_processing4_09_04
#SBATCH --output=/scratch/p/pz249/baileyng/logs/MIND_processing4_09_04_%J.out
#SBATCH --error=/scratch/p/pz249/baileyng/logs/MIND_processing4_09_04_%J.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-752

module load python/3.8.5
source /home/p/pz249/baileyng/mind_env/bin/activate

PATHS_FILE=/home/p/pz249/baileyng/MIND_models/OASIS/unprocessed_FS2_paths.loc
OUTPUT_DIR=/scratch/p/pz249/baileyng/OASIS_data/MIND/full
PARCELLATION=aparc

mapfile -s $SLURM_ARRAY_TASK_ID \
    -t -n1 SUBJECT < "$PATHS_FILE"

python3 /home/p/pz249/baileyng/MIND_models/MIND/run_MIND_script.py --subj_dir "${SUBJECT[0]}" --output_dir "$OUTPUT_DIR" --parcellation "$PARCELLATION"
