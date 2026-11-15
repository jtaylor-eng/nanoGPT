#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=ks_smoke
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/fs/scratch/PAS2836/jacktaylor/nanogpt_results/lm1012/kernel_speed/slurm/%x_%j.log
# Toy-size run of every command in kernel_gate.sbatch / kernel_time.sbatch (tier-2 smoke, used when the desktop GPU is
# busy). Submit with --export=ALL,SMOKE=research/experiments/stieltjes-kernel-speed/smoke_e001.sh
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$SCRATCH/wt/nanogpt/stieltjes-kernel-speed VENV=$SCRATCH/venvs/nanogpt DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B
source $REPO/osc/env.sh; source $VENV/bin/activate
[ -z "$(git -C $REPO status --porcelain)" ] || { echo "dirty tree in $REPO"; exit 2; }
echo "commit $(git -C $REPO rev-parse HEAD) job $SLURM_JOB_ID host $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $(date '+%F %T')"
PY=$VENV/bin/python bash $REPO/${SMOKE:?} $REPO $TMPDIR/smoke
echo DONE $(date '+%T')
