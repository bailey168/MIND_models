#!/bin/bash
#SBATCH --job-name=MIND_batches
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/mind_%A_%a.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/mind_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-92

DATA_DIR=/external/rprshnas01/external_data/uk_biobank/imaging/brain/nifti/t1_surface/data/
OUTPUT_DIR=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/MIND_results
TOTAL_SUBJECTS=46114
BATCH_SIZE=500

# Calculate the subject range
START_INDEX=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END_INDEX=$((START_INDEX + BATCH_SIZE))

# Cap end index at total number of subjects
if [ "$END_INDEX" -gt "$TOTAL_SUBJECTS" ]; then
    END_INDEX=$TOTAL_SUBJECTS
fi

echo "Running subjects from $START_INDEX to $END_INDEX (task ID $SLURM_ARRAY_TASK_ID)"
python3 run_MIND_script.py --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" $START_INDEX $END_INDEX
