#!/bin/bash

printf "%s\n" /KIMEL/tigrlab/archive/data/OPT/pipelines/in_progress/baseline_6m_24m/pipeline/freesurfer_longitudinal/subject_dir/* | while read -r dir; do
    [ -d "$dir" ] && printf "%s\n" "$dir"
done > /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/paths_aparc_opt.loc
