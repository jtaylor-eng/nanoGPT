#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm1012_arxiv
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.log
# Long-document ladder 512..16384 on 128 Pile-ArXiv test papers (data/arxiv_long/arxiv_long.bin, in the repo) for
# every finished exp-slope run; writes arxiv.json next to lengths.json. Each length scores the first L+1 tokens of
# every document, so longer context only adds earlier text of the same paper.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh; source $VENV/bin/activate
cd $TMPDIR
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for d in $LM_RESULTS/*_xs/lr*; do
  [ -f $d/TRAIN_DONE ] || continue
  [ -f $d/arxiv.json ] && { echo "skip $d"; continue; }
  echo "== $d $(date '+%H:%M:%S')"
  python $REPO/eval_lengths.py --out_dir=$d --data_file=$REPO/data/arxiv_long/arxiv_long.bin --doc_len=16385 \
    --lengths=512,1024,2048,4096,8192,16384 --batch_tokens=16384 --stats_lengths=512,16384 --stats_seqs=2 \
    --out_name=arxiv.json 2>&1 | grep -E "^L=|^stats|wrote|Error"
done
echo DONE $(date '+%H:%M:%S')
