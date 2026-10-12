#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm1012_probe
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.log
# (1) tokens/s at the real batch for every row (120 steps each, compile on, num_iter 30 and 15 for Stieltjes),
# (2) solver residual |S-1| at num_iter 15 vs 30 on Stieltjes checkpoints after 800 steps, at T=512 and 4096,
# (3) decision: writes $LM_RESULTS/NUM_ITER (15 iff max residual at 15 < 1e-4 for every Stieltjes row/length).
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh; source $REPO/osc/rows.sh; source $REPO/osc/stage.sh
source $VENV/bin/activate
P=$LM_RESULTS/probe; mkdir -p $P
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; stage_data; cd $RUNDIR; ln -sfn $REPO/config config
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
speed() {  # name extra-args...
  local n=$1; shift
  python $REPO/train.py config/lm_stieltjes.py "$@" --out_dir=$P/speed_$n --max_iters=120 --lr_decay_iters=7630 \
     --eval_interval=100000 --final_eval_iters=0 --log_interval=10 > $P/speed_$n.log 2>&1
  echo "$n rc=$? $(python -c "
import json;r=[json.loads(l) for l in open('$P/speed_$n/metrics.jsonl') if '\"train\"' in l]
r=[x for x in r if x['iter']>=40];import statistics as s
print('median ms/step %.0f  tok/s %.0f  -> 7630 steps = %.2f h'%(s.median(x['dt_ms'] for x in r),s.median(x['tok_per_s'] for x in r),s.median(x['dt_ms'] for x in r)*7630/3.6e6))" 2>&1)"
  rm -f $P/speed_$n/ckpt.pt
}
for row in softmax stieltjes_q4 wstieltjes_q4_d2 aswstieltjes_q2_d4_zi; do
  speed $row $(row_args $row) --learning_rate=1.5e-3
  is_stieltjes $row && speed ${row}_it15 $(row_args $row) --learning_rate=1.5e-3 --stieltjes_num_iter=15
done
for row in stieltjes_q4 wstieltjes_q4_d2 aswstieltjes_q2_d4_zi; do
  python $REPO/train.py config/lm_stieltjes.py $(row_args $row) --out_dir=$P/res_$row --max_iters=800 --lr_decay_iters=7630 \
     --learning_rate=1.5e-3 --eval_interval=400 --eval_iters=20 --final_eval_iters=0 --log_interval=100 > $P/res_$row.log 2>&1
  grep "^step" $P/res_$row.log | tail -1 | sed "s/^/$row /"
  python $REPO/residual_probe.py --out_dir=$P/res_$row --data_dir=$DATA_DIR --lengths=512,4096 --iters=10,15,30 --n_seqs=4 2>&1 | grep -E "^L=" | sed "s/^/$row /"
done
python $REPO/osc/probe_decide.py $P $LM_RESULTS/NUM_ITER
