#!/bin/bash

SOURCE_DIR="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg"

TARGET_DIR_2="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_2"
TARGET_DIR_3="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_3"

mkdir -p "$TARGET_DIR_2"
mkdir -p "$TARGET_DIR_3"

for file in "$SOURCE_DIR"/*; do
    filename=$(basename "$file")
    if [[ $filename == *2_0_aparc_MIND_matrix.csv ]]; then
        cp "$file" "$TARGET_DIR_2"
    elif [[ $filename == *3_0_aparc_MIND_matrix.csv ]]; then
        cp "$file" "$TARGET_DIR_3"
    fi
done

echo "Files copied successfully!"