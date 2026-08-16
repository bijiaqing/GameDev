#!/bin/bash -l
#SBATCH --job-name=GameDev
#SBATCH --output=/u/jibi/verbose/%x.%j.out
#SBATCH --error=/u/jibi/verbose/%x.%j.err
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiaqing.bi@gmail.com
#SBATCH --time=02:00:00

module purge
module load gcc/14
module load rocm/7.2
module load python-waterboa/2025.06

set -e

cd ~/gamedev

ROCPROFCOMPUTE_COLOR=0 python3 qav/perf/profile.py \
    --case fluid_3d_production \
    --tool compute \
    --kernel advection_ybl \
    --target gfx942
