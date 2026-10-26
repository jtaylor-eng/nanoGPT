#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm1012_16k
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.log
# Length ladder 512..16384 for every finished exp-slope run; writes lengths_16k.json next to lengths.json.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh; source $REPO/osc/stage.sh; source $VENV/bin/activate
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; stage_data; cd $RUNDIR
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for d in $LM_RESULTS/*_xs/lr*; do
  [ -f $d/TRAIN_DONE ] || continue
  [ -f $d/lengths_16k.json ] && { echo "skip $d"; continue; }
  echo "== $d $(date '+%H:%M:%S')"
  python $REPO/eval_lengths.py --out_dir=$d --data_dir=$DATA_DIR --lengths=512,1024,2048,4096,8192,16384 \
    --batch_tokens=16384 --stats_lengths=512,4096,16384 --stats_seqs=2 --out_name=lengths_16k.json 2>&1 | grep -E "^L=|^stats|wrote|Error"
done
echo DONE $(date '+%H:%M:%S')
