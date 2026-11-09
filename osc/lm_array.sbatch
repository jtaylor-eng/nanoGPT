#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm1012
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:30:00
#SBATCH --output=%x_%A_%a.log
export SCRATCH=/fs/scratch/PAS2836/$USER
source ${REPO:-$SCRATCH/nanoGPT}/osc/env.sh
# array over lines of $MANIFEST: "[VAR=value ...] <row> <lr> [seed]"; leading VAR=value words (CONFIG, RESULTS_TAG)
# are exported per line, so one chain can mix configs. DATA_SRC comes from --export (env.sh keeps it).
set -- $(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $MANIFEST)
while [[ ${1:-} == *=* ]]; do export "$1"; shift; done
bash $REPO/osc/run_lm.sh "$@"
