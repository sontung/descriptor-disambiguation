#!/bin/bash -l
#PBS -N aachen_dd
#PBS -l select=1:ncpus=20:ngpus=1:mem=100GB:qlist=qvpr
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
cd /home/n11373598/work/descriptor-disambiguation

# Run the test command inside the pixi environment
/home/n11373598/.pixi/bin/pixi run python main_aachen.py --local_desc d2net --local_desc_dim 512 --global_desc megaloc --global_desc_dim 8448 || {
  echo "Python crashed!"
  exit 1
}
