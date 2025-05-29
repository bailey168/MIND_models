#!/bin/bash

for path_file in /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/subject_paths/paths_part_*; do
    num_lines=$(wc -l < "$path_file")
    max_index=$((num_lines - 1))
    sbatch --array=0-$max_index submit_MIND_merged_loop.sbatch "$path_file"
    echo "Submitted job for $path_file with array range 0-$max_index"
    sleep 2h
done