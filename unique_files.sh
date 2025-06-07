#!/bin/bash

dir1="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_2"
dir2="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/HCP-MMP_avg"
output_dir="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_unique"

mkdir -p "$output_dir"

for file1 in "$dir1"/*; do
    base1=$(basename "$file1")
    prefix1=${base1:0:7}
    match=$(find "$dir2" -type f -name "${prefix1}*" | head -n 1)
    if [[ -n "$match" ]]; then
        cp "$file1" "$output_dir"
    fi
done

echo "Files copied successfully from $dir1 to $output_dir"