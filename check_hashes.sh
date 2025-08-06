#!/bin/bash

# Directories
target_dir="/scratch/p/pz249/baileyng"
orig_dir="/scratch/a/arisvoin/arisvoin/pz"
log_dir="./hash_logs"

# Create output directory for logs
mkdir -p "$log_dir"

# Loop over all files in the target directory
for target_path in "$target_dir"/*; do
    filename="$(basename "$target_path")"
    orig_path="$orig_dir/$filename"
    output_file="$log_dir/${filename}.hash.txt"

    echo "Checking: $filename"
    sha512sum "$target_path" "$orig_path" &> "$output_file" &
done

wait
echo "Done. All individual outputs stored in $log_dir/"