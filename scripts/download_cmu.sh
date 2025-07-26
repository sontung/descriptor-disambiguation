#!/bin/bash -l
#PBS -N download
#PBS -l select=1:ncpus=2:mem=100GB
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

cd /home/n11373598/work/descriptor-disambiguation/datasets/

wget -r -np -nH --cut-dirs=3 -R "index.html*" https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/

for f in *.tar; do tar -xf "$f"; done
