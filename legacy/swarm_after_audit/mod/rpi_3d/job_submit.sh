#!/bin/bash -l
#SBATCH --job-name=rpi_3d
#SBATCH --output=/u/jibi/verbose/%x.%j.out
#SBATCH --error=/u/jibi/verbose/%x.%j.err
#SBATCH --nodes=1
#SBATCH --partition="p.gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiaqing.bi@gmail.com
#SBATCH --time=1-00:00:00

JOBNAME="${SLURM_JOB_NAME}"
ROOTDIR="/u/jibi/graffiti"

cd "${ROOTDIR}"
make MODEL="${JOBNAME}"
"${ROOTDIR}/mod/${JOBNAME}/gamedev"
