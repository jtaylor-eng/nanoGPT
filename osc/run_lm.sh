#!/usr/bin/env bash
# run_lm.sh <row> <lr> [seed]  -- train (resume if ckpt exists without TRAIN_DONE), then the three evals
# (FineWeb 512..4096 -> lengths.json; FineWeb to 16k -> lengths_16k.json; ArXiv papers -> arxiv.json).
# seed defaults to 1337 (dir lr<lr>); other seeds go to lr<lr>_s<seed>.
# CONFIG env var selects the train config (default lm_stieltjes.py; lm_stieltjes_t2048.py for stage 3); results
# for a non-default config go under $LM_RESULTS/<config tag>/ (e.g. t2048/).
# num_iter for Stieltjes rows: $LM_RESULTS/NUM_ITER (written by probe_decide.py), default 30.
set -uo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/rows.sh"; source "$(dirname "$0")/stage.sh"
source $VENV/bin/activate
row=$1; lr=$2; seed=${3:-1337}
CFG=${CONFIG:-lm_stieltjes.py}
ROOT=$LM_RESULTS; [ "$CFG" != "lm_stieltjes.py" ] && ROOT=$LM_RESULTS/${CFG#lm_stieltjes_}; ROOT=${ROOT%.py}
OUT=$ROOT/$row/lr$lr; [ "$seed" != "1337" ] && OUT=${OUT}_s$seed
mkdir -p $OUT; LOG=$OUT/run.log
NI=30; is_stieltjes $row && [ -f $LM_RESULTS/NUM_ITER ] && NI=$(cat $LM_RESULTS/NUM_ITER)
echo "[$(date '+%m-%d %H:%M:%S')] run_lm $row lr=$lr seed=$seed num_iter=$NI host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git -C $REPO rev-parse --short HEAD)" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $LOG
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; stage_data
cd $RUNDIR; ln -sfn $REPO/config config
minlr=$(python -c "print($lr/10)")
if [ ! -f $OUT/TRAIN_DONE ]; then
  INIT=scratch
  # resume only from a checkpoint that is plausibly complete (a stalled GPFS write leaves 0 B / partial files)
  if [ -f $OUT/ckpt.pt ] && [ $(stat -c %s $OUT/ckpt.pt) -gt 100000000 ] && python -c "import torch,sys;torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)" 2>/dev/null; then
    INIT=resume
  elif [ -f $OUT/ckpt.pt ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] discarding unusable ckpt.pt ($(stat -c %s $OUT/ckpt.pt) B); restarting from scratch" | tee -a $LOG
    rm -f $OUT/ckpt.pt $OUT/ckpt.pt.tmp $OUT/metrics.jsonl
  fi
  python $REPO/train.py $REPO/config/$CFG --data_root=$DATA_DIR $(row_args $row) --stieltjes_num_iter=$NI \
    --learning_rate=$lr --min_lr=$minlr --seed=$seed --out_dir=$OUT --init_from=$INIT --wall_limit_s=${WALL_LIMIT_S:-9000} >> $LOG 2>&1
  rc=$?; echo "[$(date '+%m-%d %H:%M:%S')] train exit $rc" | tee -a $LOG
  [ -f $OUT/TRAIN_DONE ] || exit $rc
fi
if [ ! -f $OUT/lengths.json ]; then
  python $REPO/eval_lengths.py --out_dir=$OUT --data_dir=$DATA_DIR --batch_tokens=8192 >> $LOG 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] eval exit $?" | tee -a $LOG
fi
if [ ! -f $OUT/lengths_16k.json ]; then
  python $REPO/eval_lengths.py --out_dir=$OUT --data_dir=$DATA_DIR --lengths=512,1024,2048,4096,8192,16384 \
    --batch_tokens=16384 --stats_lengths=512,4096,16384 --stats_seqs=2 --out_name=lengths_16k.json >> $LOG 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] eval16k exit $?" | tee -a $LOG
fi
if [ ! -f $OUT/arxiv.json ]; then
  python $REPO/eval_lengths.py --out_dir=$OUT --data_file=$REPO/data/arxiv_long/arxiv_long.bin --doc_len=16385 \
    --lengths=512,1024,2048,4096,8192,16384 --batch_tokens=16384 --stats_lengths=512,16384 --stats_seqs=2 --out_name=arxiv.json >> $LOG 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] arxiv exit $?" | tee -a $LOG
fi
if [ ! -f $OUT/pg19.json ]; then
  python $REPO/eval_lengths.py --out_dir=$OUT --data_file=$REPO/data/pg19_long/pg19_long.bin --doc_len=16385 \
    --lengths=512,1024,2048,4096,8192,16384 --batch_tokens=16384 --stats_lengths= --out_name=pg19.json >> $LOG 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] pg19 exit $?" | tee -a $LOG
fi
