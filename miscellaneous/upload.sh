#!/bin/bash

find /opt/notebooks/FC100_graphs/raw/GF100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC100_graphs/raw/GF100/"
find /opt/notebooks/FC100_graphs/raw/PAL100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC100_graphs/raw/PAL100/"
find /opt/notebooks/FC100_graphs/raw/DSST100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC100_graphs/raw/DSST100/"
find /opt/notebooks/FC100_graphs/raw/TMT100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC100_graphs/raw/TMT100/"

find /opt/notebooks/FC25_graphs/raw/GF100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC25_graphs/raw/GF100/"
find /opt/notebooks/FC25_graphs/raw/PAL100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC25_graphs/raw/PAL100/"
find /opt/notebooks/FC25_graphs/raw/DSST100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC25_graphs/raw/DSST100/"
find /opt/notebooks/FC25_graphs/raw/TMT100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/FC25_graphs/raw/TMT100/"

find /opt/notebooks/MIND_graphs/raw/GF100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/MIND_graphs/raw/GF100/"
find /opt/notebooks/MIND_graphs/raw/PAL100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/MIND_graphs/raw/PAL100/"
find /opt/notebooks/MIND_graphs/raw/DSST100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/MIND_graphs/raw/DSST100/"
find /opt/notebooks/MIND_graphs/raw/TMT100 -maxdepth 1 -name "*" -type f | xargs -P 72 -I {} dx upload --brief {} --path "/Data/graphs/MIND_graphs/raw/TMT100/"