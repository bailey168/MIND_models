#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm
from MIND import compute_MIND

def main():
    parser = argparse.ArgumentParser(description="Run MIND on a range of subjects")
    parser.add_argument("--data_dir", type=str, help="Input data directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("start_idx", type=int, help="Start index of subjects to process")
    parser.add_argument("end_idx", type=int, help="End index (exclusive) of subjects to process")
    args = parser.parse_args()

    # Input and output directories
    data_dir = args.data_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Choose features and parcellation
    features = ['CT', 'MC', 'Vol', 'SD', 'SA']
    parcellation = 'aparc'

    all_subjects = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    if args.start_idx < 0 or args.end_idx > len(all_subjects):
        print(f"Index range out of bounds. Available subjects: {len(all_subjects)}")
        return
    if args.start_idx >= args.end_idx:
        print("Start index must be less than end index.")
        return  

    batch_subjects = all_subjects[args.start_idx:args.end_idx]

    # Iterate through subject directories
    for subj in tqdm(batch_subjects, desc="Processing subjects"):
        try:
            mind_matrix = compute_MIND(
                surf_dir=os.path.join(data_dir, subj),
                features=features,
                parcellation=parcellation,
                filter_vertices=True,
                resample=False,
                n_samples=4000
            )
            output_csv = os.path.join(output_dir, f"{subj}_MIND_matrix.csv")
            mind_matrix.to_csv(output_csv)
        except Exception as e:
            print(f"Error processing {subj}: {e}")


if __name__ == "__main__":
    main()