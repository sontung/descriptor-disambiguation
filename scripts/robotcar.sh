#!/bin/bash -l
#PBS -N robotcar_dd
#PBS -l select=1:ncpus=20:ngpus=1:mem=100GB
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
cd /home/n11373598/work/descriptor-disambiguation

# Run the test command inside the pixi environment
/home/n11373598/.pixi/bin/pixi run python main_robotcar.py --convert 0 --dataset datasets/robotcar --global_desc megaloc || {
  echo "Python crashed!"
  exit 1
}
