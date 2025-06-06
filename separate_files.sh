#!/bin/bash

SOURCE_DIR="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/MIND_results/aparc"

TARGET_DIR_2="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/MIND_results/aparc_2"
TARGET_DIR_3="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/MIND_results/aparc_3"

mkdir -p "$TARGET_DIR_2"
mkdir -p "$TARGET_DIR_3"

cp "$SOURCE_DIR"/*2_0_aparc_MIND_matrix.csv "$TARGET_DIR_2"
cp "$SOURCE_DIR"/*3_0_aparc_MIND_matrix.csv "$TARGET_DIR_3"

echo "Files copied successfully!"