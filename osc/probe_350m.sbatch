#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm350m_probe
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.log
# Stage 5 (350M) throughput / memory probe: tok/s and peak GPU memory per row at candidate micro-batches
# (accum 4, 40 steps, median over steps >= 15, compile on, Stieltjes num_iter from $LM_RESULTS/NUM_ITER), so the
# 350M budget is measured rather than scaled from 125M. Writes $LM_RESULTS/350m/probe/speed.tsv and the per-run
# metrics.jsonl. Submit with --export=ALL,REPO=<direction checkout>.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source ${REPO:-$SCRATCH/nanoGPT}/osc/env.sh; source $REPO/osc/rows.sh; source $REPO/osc/stage.sh
source $VENV/bin/activate
export DATA_SRC=${DATA_SRC_OVERRIDE:-$SCRATCH/nanogpt_data/fineweb25B}
P=$LM_RESULTS/350m/probe; mkdir -p $P
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; stage_data; cd $RUNDIR; ln -sfn $REPO/config config
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
NI=30; [ -f $LM_RESULTS/NUM_ITER ] && NI=$(cat $LM_RESULTS/NUM_ITER)
T=$P/speed.tsv; printf "row\tmicro_batch\tnum_iter\trc\tms_per_step_accum4\ttok_per_s\tpeak_mem_gb\thours_4770_steps\n" > $T
speed() {  # row micro-batch
  local row=$1 bs=$2 n=${1}_bs$2
  echo "== $n $(date '+%H:%M:%S')"
  python $REPO/train.py $REPO/config/lm_stieltjes_350m.py --data_root=$DATA_DIR $(row_args $row) --stieltjes_num_iter=$NI \
     --batch_size=$bs --gradient_accumulation_steps=4 --learning_rate=6e-4 --min_lr=6e-5 --out_dir=$RUNDIR/speed_$n \
     --max_iters=40 --lr_decay_iters=4770 --eval_interval=100000 --final_eval_iters=0 --log_interval=5 \
     --always_save_checkpoint=False > $P/speed_$n.log 2>&1
  local rc=$?
  cp -f $RUNDIR/speed_$n/metrics.jsonl $P/speed_$n.jsonl 2>/dev/null; rm -rf $RUNDIR/speed_$n
  python - "$row" "$bs" "$NI" "$rc" "$P/speed_$n.jsonl" <<'EOF' | tee -a $T
import json, sys, os, statistics as s
row, bs, ni, rc, f = sys.argv[1:]
r = []
if os.path.exists(f):
    r = [json.loads(l) for l in open(f) if '"train"' in l]; r = [x for x in r if x['iter'] >= 15]
if rc != '0' or len(r) < 3:
    print(f"{row}\t{bs}\t{ni}\t{rc}\tnan\tnan\tnan\tnan"); sys.exit()
ms = s.median(x['dt_ms'] for x in r); tps = s.median(x['tok_per_s'] for x in r); mem = max(x.get('mem_gb', 0) for x in r)
print(f"{row}\t{bs}\t{ni}\t{rc}\t{ms:.0f}\t{tps:.0f}\t{mem:.1f}\t{4770 * 2**19 / tps / 3600:.2f}")
EOF
  grep -iE "out of memory|Error" $P/speed_$n.log | head -2
}
speed softmax_xs 16
speed softmax_xs 8
speed asentmax_zi_xs 8
speed asentmax_zi_xs 4
speed wstieltjes_q3_d2_xs 8
speed wstieltjes_q3_d2_xs 4
echo "== done $(date '+%H:%M:%S')"; cat $T
