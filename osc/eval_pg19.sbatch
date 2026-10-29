#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm1012_pg19
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.log
# PG-19 long-document ladder (data/pg19_long/pg19_long.bin) for every finished exp-slope run -> pg19.json
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh; source $VENV/bin/activate
cd $TMPDIR
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for d in $LM_RESULTS/*_xs/lr*; do
  [ -f $d/TRAIN_DONE ] || continue
  [ -f $d/pg19.json ] && { echo "skip $d"; continue; }
  echo "== $d $(date '+%H:%M:%S')"
  python $REPO/eval_lengths.py --out_dir=$d --data_file=$REPO/data/pg19_long/pg19_long.bin --doc_len=16385 \
    --lengths=512,1024,2048,4096,8192,16384 --batch_tokens=16384 --stats_lengths= --out_name=pg19.json 2>&1 | grep -E "^L=|wrote|Error"
done
echo DONE $(date '+%H:%M:%S')
