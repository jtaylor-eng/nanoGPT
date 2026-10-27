#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm1012
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:30:00
#SBATCH --output=%x_%A_%a.log
# array over lines of $MANIFEST: "<row> <lr> [seed]"
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $MANIFEST)
bash $REPO/osc/run_lm.sh $line
