#!/bin/bash

for f in /scratch/p/pz249/baileyng/*.tar; do
  echo "Unpacking $f"
  tar --overwrite -xvf "$f"
done

for f in /scratch/p/pz249/baileyng/*.tree.gz; do
  echo "Unpacking $f"
  tar --overwrite -xvzf "$f" 2>/dev/null || gunzip -k "$f"
done