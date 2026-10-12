#!/usr/bin/env bash
# run_lm.sh <row> <lr>   -- train (resume if ckpt exists without TRAIN_DONE), then length eval.
# num_iter for Stieltjes rows: $LM_RESULTS/NUM_ITER (written by probe_decide.py), default 30.
set -uo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/rows.sh"; source "$(dirname "$0")/stage.sh"
source $VENV/bin/activate
row=$1; lr=$2
OUT=$LM_RESULTS/$row/lr$lr; mkdir -p $OUT; LOG=$OUT/run.log
NI=30; is_stieltjes $row && [ -f $LM_RESULTS/NUM_ITER ] && NI=$(cat $LM_RESULTS/NUM_ITER)
echo "[$(date '+%m-%d %H:%M:%S')] run_lm $row lr=$lr num_iter=$NI host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git -C $REPO rev-parse --short HEAD)" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $LOG
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; stage_data
cd $RUNDIR; ln -sfn $REPO/config config
minlr=$(python -c "print($lr/10)")
if [ ! -f $OUT/TRAIN_DONE ]; then
  INIT=scratch; [ -f $OUT/ckpt.pt ] && INIT=resume
  python $REPO/train.py config/lm_stieltjes.py $(row_args $row) --stieltjes_num_iter=$NI \
    --learning_rate=$lr --min_lr=$minlr --out_dir=$OUT --init_from=$INIT --wall_limit_s=${WALL_LIMIT_S:-48600} >> $LOG 2>&1
  rc=$?; echo "[$(date '+%m-%d %H:%M:%S')] train exit $rc" | tee -a $LOG
  [ -f $OUT/TRAIN_DONE ] || exit $rc
fi
if [ ! -f $OUT/lengths.json ]; then
  python $REPO/eval_lengths.py --out_dir=$OUT --data_dir=$DATA_DIR --batch_tokens=16384 >> $LOG 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] eval exit $?" | tee -a $LOG
fi
