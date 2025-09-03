#!/bin/bash

# keep header from file1
head -n 1 FS_ID.csv > freesurfer2_unloaded2.csv

# append ONLY lines from file1 whose (first 8 chars + last 5 chars) are NOT in file2
awk '
  NR==FNR {
    if (FNR>1) {
      line=$0; sub(/\r$/,"",line)
      if (length(line)>=13) {
        p=substr(line,1,8)
        s=substr(line,length(line)-4)
        seen[p,s]=1
      }
    }
    next
  }
  FNR==1 { next }  # skip header from file1
  {
    line=$0; sub(/\r$/,"",line)
    if (length(line)>=13) {
      p=substr(line,1,8)
      s=substr(line,length(line)-4)
      if (!((p,s) in seen)) print
    }
  }
' freesurfer2_loaded2.csv FS_ID.csv >> freesurfer2_unloaded2.csv