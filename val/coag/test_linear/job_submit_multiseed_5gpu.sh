#!/bin/bash -l
#SBATCH --job-name=test_linear_multiseed_5gpu
#SBATCH --output=/u/jibi/verbose/%x.%A_%a.out
#SBATCH --error=/u/jibi/verbose/%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=p.gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --array=1-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiaqing.bi@gmail.com
#SBATCH --time=1-12:00:00

module purge
module load gcc/12
module load cuda/12.1
module load python-waterboa/2025.06

set -e
set -o pipefail

cd ~/gamedev

python3 val/coag/test_linear/run_multiseed.py --cards 5 --job "${SLURM_ARRAY_TASK_ID}"
