#!/bin/bash -l
#PBS -N aachen_dd
#PBS -l select=1:ncpus=20:ngpus=1:mem=100GB:qlist=qvpr
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
cd /home/n11373598/work/descriptor-disambiguation
cd env_pixi
/home/n11373598/.pixi/bin/pixi shell
cd ..

python main_aachen.py --local_desc d2net --local_desc_dim 512 --global_desc eigenplaces --global_desc_dim 2048
