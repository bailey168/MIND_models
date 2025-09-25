#!/bin/bash

SOURCE_DIR="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_2"
PATH_FILE="/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/pipe_paths/paths_HCP-MMP_base.loc"
OUTPUT_DIR="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_unique"

mkdir -p "$OUTPUT_DIR"

declare -A prefix_lookup

while IFS= read -r line; do
    prefix="${line:0:7}"
    prefix_lookup["$prefix"]=1
done < "$PATH_FILE"

for file in "$SOURCE_DIR"/*; do
    base=$(basename "$file")
    prefix=${base:0:7}
    if [[ ${prefix_lookup["$prefix"]+_} ]]; then
        cp "$file" "$OUTPUT_DIR"
    fi
done

echo "Files copied successfully from $SOURCE_DIR to $OUTPUT_DIR"