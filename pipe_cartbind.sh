#!/bin/bash

printf "%s\n" /KIMEL/tigrlab/projects/colin/CARTBIND_FMRIPrep/fmriprep_cartbind_cmh/freesurfer/sub*/ | while read -r dir; do
    [ -d "$dir" ] && printf "%s\n" "$dir"
done > /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/paths_aparc_camh.loc
