#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=ks_time
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/fs/scratch/PAS2836/jacktaylor/nanogpt_results/lm1012/kernel_speed/slurm/%x_%j.log
# Timings on one A100 40 GB node, candidate and frozen baseline in the same job. Direction stieltjes-kernel-speed.
# Submit with --export=ALL,TAG=e00N[,PLAN=profile|ab].
#   micro:  bench_kernel.py micro  (kernel-level, both kernels)
#   steps:  unmodified train.py at 125M / T=2048 / 524,288 tok per step through `bench_kernel.py wrap`
#     PLAN=profile (e001): wstieltjes_q3_d2_xs 30 steps + 10 with synced timers, softmax_xs 30, asentmax_zi_xs 20 + 8 timed,
#                          wstieltjes micro-batch 16 x 16 (does it fit 40 GB?) 4 steps
#     PLAN=ab:             wstieltjes_q3_d2_xs, aswstieltjes_q3_d2_zi_xs, wstieltjes_q4_d2_xs: baseline then candidate, 20 steps each
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$SCRATCH/wt/nanogpt/stieltjes-kernel-speed VENV=$SCRATCH/venvs/nanogpt DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B
source $REPO/osc/env.sh; source $REPO/osc/rows.sh; source $VENV/bin/activate
TAG=${TAG:?set TAG=e00N}; PLAN=${PLAN:-profile}; OUT=$LM_RESULTS/kernel_speed/$TAG; mkdir -p $OUT
[ -z "$(git -C $REPO status --porcelain)" ] || { echo "dirty tree in $REPO"; exit 2; }
# Triton/inductor caches: env.sh defaults on scratch (persistent, keyed by source hash: the frozen kernel compiles once)
cd $TMPDIR
echo "commit $(git -C $REPO rev-parse HEAD) job $SLURM_JOB_ID host $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) plan=$PLAN $(date '+%F %T')"

echo "== micro $(date '+%T')"
python $REPO/bench_kernel.py micro --out=$OUT/micro.json 2>&1 | tee $OUT/micro.log
echo "== micro rc=${PIPESTATUS[0]} $(date '+%T')"

# real FineWeb batches from node-local disk: the first 4 GB (2B tokens) of train.bin is enough for a timing probe
FW=$TMPDIR/fw; mkdir -p $FW; head -c 4294967296 $DATA_SRC/train.bin > $FW/train.bin; cp -L $DATA_SRC/val.bin $FW/val.bin
SPARSE="--batch_size=8 --gradient_accumulation_steps=32"     # as osc/run_lm.sh for the sparse rows at 125M
step() {  # name row steps timers kernel [extra train.py args]
  local name=$1 row=$2 steps=$3 timers=$4 kern=$5; shift 5
  echo "== step $name $(date '+%T')"
  python $REPO/bench_kernel.py wrap --kernel=$kern --timers=$timers --out=$OUT/step_$name.json -- \
    $REPO/train.py $REPO/config/lm_stieltjes_125m.py --data_root=$FW $(row_args $row) --stieltjes_num_iter=15 \
    --learning_rate=1.2e-3 --min_lr=1.2e-4 --out_dir=$TMPDIR/out_$name --max_iters=$((steps - 1)) --eval_interval=100000 \
    --eval_iters=1 --final_eval_iters=0 --log_interval=1 "$@" > $OUT/step_$name.log 2>&1
  echo "== step $name rc=$? $(grep -c '^iter ' $OUT/step_$name.log) iters $(date '+%T')"
  rm -rf $TMPDIR/out_$name
}
if [ "$PLAN" = profile ]; then
  step wst_q3d2_cand        wstieltjes_q3_d2_xs 30 0 cand $SPARSE
  step wst_q3d2_cand_timed  wstieltjes_q3_d2_xs 10 1 cand $SPARSE
  step softmax              softmax_xs          30 0 cand
  step asentmax             asentmax_zi_xs      20 0 cand $SPARSE
  step asentmax_timed       asentmax_zi_xs       8 1 cand $SPARSE
  step wst_q3d2_cand_mb16   wstieltjes_q3_d2_xs  4 0 cand
else
  for row in wstieltjes_q3_d2_xs aswstieltjes_q3_d2_zi_xs wstieltjes_q4_d2_xs; do
    step ${row}_ref  $row 20 0 ref  $SPARSE
    step ${row}_cand $row 20 0 cand $SPARSE
  done
fi
python $REPO/bench_kernel.py summary $OUT
echo DONE $(date '+%T')
