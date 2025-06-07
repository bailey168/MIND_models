#!/bin/bash

SOURCE_DIR="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/HCP-MMP_avg"
LOC_FILE="/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/paths_HCP-MMP_base.loc"

> "$LOC_FILE"

for file in "$SOURCE_DIR"/*; do
    filename=$(basename "$file")
    echo "$filename" >> "$LOC_FILE"
done

echo "Files piped successfully!"