#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lqd_smoke
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.log
# Pre-submit smoke: the run_lm.sh train command for $ROW at $CONFIG, from $TMPDIR with absolute paths, 6 iters,
# then a resume for 2 more and the arxiv eval at 2048 (wall time is mostly torch.compile). Results to $SCRATCH/nanogpt_results/smoke/<row> (wiped first).
# sbatch --export=ALL,REPO=<checkout>,ROW=<row>,CONFIG=lm_stieltjes_125m.py osc/smoke.sbatch
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $REPO/osc/env.sh; source $REPO/osc/rows.sh
DATA_DIR=$SCRATCH/nanogpt_data/fineweb25B   # read in place (memmap): a few batches do not justify a 50 GB stage
source $VENV/bin/activate
OUT=$SCRATCH/nanogpt_results/smoke/$ROW; rm -rf $OUT; mkdir -p $OUT
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; cd $RUNDIR
NI=$(cat $LM_RESULTS/NUM_ITER)
A="$REPO/config/$CONFIG --data_root=$DATA_DIR $(row_args $ROW) --stieltjes_num_iter=$NI --learning_rate=1.2e-3 --min_lr=1.2e-4
   --seed=1337 --out_dir=$OUT --batch_size=8 --gradient_accumulation_steps=2 --eval_interval=3 --eval_iters=2 --final_eval_iters=2"
python $REPO/train.py $A --init_from=scratch --max_iters=6 --lr_decay_iters=8 --warmup_iters=2
python $REPO/train.py $A --init_from=resume --max_iters=8 --lr_decay_iters=8 --warmup_iters=2
python $REPO/eval_lengths.py --out_dir=$OUT --data_file=$REPO/data/arxiv_long/arxiv_long.bin --doc_len=16385 \
  --lengths=2048 --batch_tokens=16384 --out_name=arxiv.json
echo "SMOKE OK commit=$(git -C $REPO rev-parse --short HEAD) row=$ROW"
