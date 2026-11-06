#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=hs_e001
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:15:00
#SBATCH --output=%x_%j.log
# pe-ssmax-headsplit e001: (1) regression tests on GPU (old-vs-new model.py bit-identity, fused-vs-eager for ssmax and
# the 2/10 split), (2) 40-step 125M throughput / NaN probe at the production batch for softmax_xs (reference),
# ssmax_xs, softmax_xs_a2, wstieltjes_q3_d2_xs_a2. Submit with --export=ALL,REPO=<direction checkout>.
# Reads data in place from scratch (memmap) instead of staging: 40 steps touch a few hundred MB.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source ${REPO:-$SCRATCH/nanoGPT}/osc/env.sh; source $REPO/osc/rows.sh
source $VENV/bin/activate
P=$LM_RESULTS/125m/hs_probe; mkdir -p $P
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; cd $RUNDIR
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
bash $REPO/tests/test_rows.sh; echo "test_rows rc=$?"
python $REPO/tests/test_split_ssmax.py 2>&1 | grep -v "^number of\|^num \|^using"
for sl in exp exp_a2; do
  echo "test_attn_paths SLOPES=$sl"; SLOPES=$sl python $REPO/tests/test_attn_paths.py 2>&1 | grep -E "BAD|ALL OK|FAILURES|Error"
done
NI=$(cat $LM_RESULTS/NUM_ITER 2>/dev/null || echo 30)
speed() {  # row
  local row=$1 extra=""
  [[ $row == *stieltjes* || $row == *entmax* ]] && extra="--batch_size=8 --gradient_accumulation_steps=32"
  python $REPO/train.py $REPO/config/lm_stieltjes_125m.py --data_root=$SCRATCH/nanogpt_data/fineweb25B $(row_args $row) \
    --stieltjes_num_iter=$NI --learning_rate=1.2e-3 --min_lr=1.2e-4 --out_dir=$P/$row --max_iters=40 --eval_interval=100000 \
    --eval_iters=2 --final_eval_iters=0 --log_interval=5 --wall_limit_s=100000 $extra > $P/$row.log 2>&1
  echo "$row rc=$? $(python -c "
import json, math, statistics as s
r = [json.loads(l) for l in open('$P/$row/metrics.jsonl')]; r = [x for x in r if x.get('kind') == 'train']
w = [x for x in r if x['iter'] >= 15]
nan = any(not math.isfinite(x['loss']) for x in r)
print('median ms/step %.0f  tok/s %.0f  -> 4770 steps = %.2f h  loss@%d %.3f  nan %s' % (s.median(x['dt_ms'] for x in w),
      s.median(x['tok_per_s'] for x in w), s.median(x['dt_ms'] for x in w) * 4770 / 3.6e6, r[-1]['iter'], r[-1]['loss'], nan))" 2>&1)"
  rm -f $P/$row/ckpt.pt
}
for row in softmax_xs ssmax_xs softmax_xs_a2 wstieltjes_q3_d2_xs_a2; do speed $row; done
