#!/bin/bash -l
#SBATCH --job-name=test_linear
#SBATCH --output=/u/jibi/verbose/%x.%j.out
#SBATCH --error=/u/jibi/verbose/%x.%j.err
#SBATCH --nodes=1
#SBATCH --partition=p.gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiaqing.bi@gmail.com
#SBATCH --time=1-00:00:00

module purge
module load gcc/12
module load cuda/12.1
module load python-waterboa/2025.06

set -e
set -o pipefail

cd ~/gamedev

JOBNAME="${SLURM_JOB_NAME}"
python3 val/coag/"${JOBNAME}"/run_models.py --batch "${SLURM_ARRAY_TASK_ID}"
