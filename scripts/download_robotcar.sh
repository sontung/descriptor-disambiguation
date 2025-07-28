#!/bin/bash -l
#PBS -N download
#PBS -l select=1:ncpus=2:mem=100GB
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

cd /home/n11373598/work/descriptor-disambiguation/

export dataset=datasets/robotcar
wget -r -np -nH -R "index.html*" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/ -P $dataset
for condition in $dataset/images/*.zip; do unzip condition -d $dataset/images/; done