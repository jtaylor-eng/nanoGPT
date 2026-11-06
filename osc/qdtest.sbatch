#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lqd_qdtest
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=%x_%j.log
# learnable-qd e002 gate before the training chain (all from $TMPDIR, absolute paths):
#  1. fixed-(q, d) Triton path bit-identical to the kernel at $OLD_SHA (fwd + bwd)
#  2. tests/test_attn_paths.py incl. lwstieltjes_q3_d2 (fused per-head q/d vs eager per-head, incl. theta grads), both slope types
#  3. throughput probe: 125M, row $ROW, mb8 x accum32 (the run_lm.sh flags), 12 iters; fails if the median step
#     (iters 4..11) exceeds MAX_STEP_MS (the 25 GPU-h per-run ceiling minus evals) or no qd record was logged.
# sbatch --export=ALL,REPO=<checkout>,ROW=lwstieltjes_q3_d2_xs,OLD_SHA=3074a63,MAX_STEP_MS=17500 osc/qdtest.sbatch
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $REPO/osc/env.sh; source $REPO/osc/rows.sh
source $VENV/bin/activate
mkdir -p $TMPDIR/old $TMPDIR/run; cd $TMPDIR/run
git -C $REPO show $OLD_SHA:stieltjes/triton_stieltjes.py > $TMPDIR/old/triton_stieltjes_old.py
echo "== row_args $ROW: $(row_args $ROW)"
echo "== 1. fixed path identical vs $OLD_SHA"
python $REPO/tests/test_fixed_path_identical.py $TMPDIR/old/triton_stieltjes_old.py
echo "== 2. attention paths (linear slopes, exp slopes)"
python $REPO/tests/test_attn_paths.py
SLOPES=exp python $REPO/tests/test_attn_paths.py
echo "== 3. throughput probe"
OUT=$SCRATCH/nanogpt_results/smoke/probe_$ROW; rm -rf $OUT; mkdir -p $OUT
NI=$(cat $LM_RESULTS/NUM_ITER)
python $REPO/train.py $REPO/config/lm_stieltjes_125m.py --data_root=$SCRATCH/nanogpt_data/fineweb25B $(row_args $ROW) \
  --stieltjes_num_iter=$NI --learning_rate=1.2e-3 --min_lr=1.2e-4 --seed=1337 --out_dir=$OUT \
  --batch_size=8 --gradient_accumulation_steps=32 --eval_interval=1000 --log_interval=1 --eval_iters=2 --final_eval_iters=2 \
  --init_from=scratch --max_iters=12 --lr_decay_iters=12 --warmup_iters=2
python - $OUT/metrics.jsonl ${MAX_STEP_MS:-17500} <<'EOF'
import json, sys, statistics
recs = [json.loads(l) for l in open(sys.argv[1])]
dts = [r['dt_ms'] for r in recs if r.get('kind') == 'train' and 4 <= r['iter'] <= 11]
qd = [r for r in recs if r.get('kind') == 'qd']
med = statistics.median(dts)
print(f"PROBE median step {med:.0f} ms over {len(dts)} iters; limit {sys.argv[2]} ms; qd records {len(qd)}")
if qd:
    print("PROBE qd layer0 q", qd[-1]['q'][0], "d", qd[-1]['d'][0])
sys.exit(0 if (dts and med <= float(sys.argv[2]) and qd) else 1)
EOF
echo "QDTEST OK commit=$(git -C $REPO rev-parse --short HEAD) row=$ROW"
