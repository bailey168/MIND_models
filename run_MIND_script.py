#!/usr/bin/env python3

import os
import argparse
import time
from memory_profiler import profile
from MIND import compute_MIND

def main():
    parser = argparse.ArgumentParser(description="Run MIND on a subject")
    parser.add_argument("--subj_dir", type=str, help="Subject directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    args = parser.parse_args()

    # Input and output directories
    subj_dir = args.subj_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Choose features and parcellation
    features = ['CT', 'MC', 'Vol', 'SD', 'SA']
    #parcellation = 'aparc'
    parcellation = 'HCP-MMP'

    subj = os.path.basename(os.path.normpath(subj_dir))
    print(f"Processing {subj}...")
    time_start = time.time()

    # Run MIND on subject
    try:
        mind_matrix = compute_MIND(
            surf_dir=os.path.join(subj_dir),
            features=features,
            parcellation=parcellation,
            filter_vertices=True,
            resample=False,
            n_samples=4000
        )
        output_csv = os.path.join(output_dir, f"{subj}_MIND_matrix.csv")
        mind_matrix.to_csv(output_csv)

        print(f"{subj}: Done in {time.time() - time_start:.2f} seconds. Saved to {output_csv}")

    except Exception as e:
        print(f"Error processing {subj}: {e}")


if __name__ == "__main__":
    main()