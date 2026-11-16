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
#     PLAN=ab_trained (e004): bench_kernel.py real (forward kernel on trained q/k), then the candidate without tile skipping
#                          (the e002 kernel) against the candidate, STEPS steps each: wstieltjes_q3_d2_xs and
#                          aswstieltjes_q3_d2_zi_xs RESUMED from the trained checkpoints (tile skipping depends on how
#                          sparse the heads are; at initialisation nothing is dead), then wstieltjes_q3_d2_xs from scratch.
#                          QUICK=1 STEPS=3: the pre-submit smoke of this plan (small micro, 2 layers, one resumed row).
#                          CMP=cand_nobskip (e005): the comparator is the candidate with backward tile skipping off (the
#                          e004 kernel) instead of cand_noskip (forward and backward skipping off, the e002 kernel).
#                          CMP=cand_noexit (e006): the comparator is the candidate with the Newton cycle exit off (the e005 kernel).
#                          CMP=cand_copy (e007): the comparator is the candidate with ZERO_COPY off and q/k/v copied at the
#                          call site (the e006 kernel and call site; reverted, the switch is gone). REAL=0 skips
#                          bench_kernel.py real (kernel-level timings on captured q/k).
#                          CMP=cand_nocache (e008): the comparator is the candidate with CACHE_SMEM off (the e006 wrapper).
#                          MB=1 (e008): after the pairs, softmax_xs at mb 8 x 32 and 16 x 16 (same-node reference, and
#                          what the micro-batch split costs outside the kernels) and the trained wstieltjes_q3_d2_xs
#                          step at mb 16 x 16 with the candidate.
#     PLAN=final (e009):   end-to-end confirmation, the final kernel against the frozen baseline, STEPS steps each, in two
#                          jobs (the full set is 58 min at STEPS=30; every pair stays inside one job). TAG=e009/<part>.
#                          PART=a (the chart rows on one node): wstieltjes_q3_d2_xs trained ref / cand, softmax_xs 16 x 16,
#                          asentmax_zi_xs trained, wstieltjes_q3_d2_xs from scratch ref / cand, asentmax_zi_xs from scratch.
#                          PART=b: aswstieltjes_q3_d2_zi_xs trained ref / cand, wstieltjes_q3_d1.5_xs trained ref / cand,
#                          wstieltjes_q4_d2_xs from scratch ref / cand (no trained q4 checkpoint exists), and the trained
#                          aswstieltjes_q3_d2_zi_xs step at mb 16 x 16 with the candidate (does the AS row fit 40 GB?).
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$SCRATCH/wt/nanogpt/stieltjes-kernel-speed VENV=$SCRATCH/venvs/nanogpt DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B
source $REPO/osc/env.sh; source $REPO/osc/rows.sh; source $VENV/bin/activate
TAG=${TAG:?set TAG=e00N}; PLAN=${PLAN:-profile}; STEPS=${STEPS:-20}; QUICK=${QUICK:-0}; CMP=${CMP:-cand_noskip}; CS=${CMP#cand_}; OUT=$LM_RESULTS/kernel_speed/$TAG; mkdir -p $OUT
[ -z "$(git -C $REPO status --porcelain)" ] || { echo "dirty tree in $REPO"; exit 2; }
# Triton/inductor caches: env.sh defaults on scratch (persistent, keyed by source hash: the frozen kernel compiles once)
cd $TMPDIR
echo "commit $(git -C $REPO rev-parse HEAD) job $SLURM_JOB_ID host $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) plan=$PLAN $(date '+%F %T')"

echo "== aliascheck $(date '+%T')"
python $REPO/bench_kernel.py aliascheck
echo "== aliascheck rc=$? $(date '+%T')"

echo "== micro $(date '+%T')"
python $REPO/bench_kernel.py micro $([ "$QUICK" = 1 ] && echo --small) --out=$OUT/micro.json 2>&1 | tee $OUT/micro.log
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
CK=$LM_RESULTS/125m_fw25_rng
rstep() {  # name row kernel [timers]: STEPS steps resumed from the trained checkpoint (a copy: the run dirs are read-only here)
  local name=$1 row=$2 kern=$3 tm=${4:-0} mb=${5:-$SPARSE} O=$TMPDIR/out_$1
  mkdir -p $O; cp $CK/$row/lr1.2e-3/ckpt.pt $O/ckpt.pt
  local it; it=$(python -c "import torch, sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['iter_num']))" $O/ckpt.pt)
  echo "== step $name (resume at iter $it) $(date '+%T')"
  python $REPO/bench_kernel.py wrap --kernel=$kern --timers=$tm --out=$OUT/step_$name.json -- \
    $REPO/train.py $REPO/config/lm_stieltjes_125m.py --data_root=$FW $(row_args $row) --stieltjes_num_iter=15 \
    --learning_rate=1.2e-3 --min_lr=1.2e-4 --out_dir=$O --init_from=resume --max_iters=$((it + STEPS - 1)) --eval_interval=100000 \
    --eval_iters=1 --final_eval_iters=0 --log_interval=1 $mb > $OUT/step_$name.log 2>&1
  echo "== step $name rc=$? $(grep -c '^iter ' $OUT/step_$name.log) iters $(date '+%T')"
  rm -rf $O
}
if [ "$PLAN" = ab_trained ]; then
  if [ "${REAL:-1}" = 1 ]; then
  echo "== real $(date '+%T')"
  if [ "$QUICK" = 1 ]; then RA="--ckpts=$CK/wstieltjes_q3_d2_xs/lr1.2e-3/ckpt.pt --layers=0,11"
  else RA="--ckpts=$CK/wstieltjes_q3_d2_xs/lr1.2e-3/ckpt.pt,$CK/aswstieltjes_q3_d2_zi_xs/lr1.2e-3/ckpt.pt"; fi
  python $REPO/bench_kernel.py real $RA --fineweb_val=$DATA_SRC/val.bin --arxiv=$REPO/data/arxiv_long/arxiv_long.bin --out=$OUT/real.json 2>&1 | tee $OUT/real.log
  echo "== real rc=${PIPESTATUS[0]} $(date '+%T')"
  fi
  rstep wst_q3d2_trained_$CS wstieltjes_q3_d2_xs $CMP
  rstep wst_q3d2_trained_cand   wstieltjes_q3_d2_xs cand
  if [ "$QUICK" != 1 ]; then
    rstep aswst_q3d2_trained_$CS aswstieltjes_q3_d2_zi_xs $CMP
    rstep aswst_q3d2_trained_cand   aswstieltjes_q3_d2_zi_xs cand
    step wst_q3d2_scratch_$CS wstieltjes_q3_d2_xs $STEPS 0 $CMP $SPARSE
    step wst_q3d2_scratch_cand   wstieltjes_q3_d2_xs $STEPS 0 cand        $SPARSE
    # TIMED=1 (e007), last so that it cannot take the A/B with it: the trained step again with CUDA-synchronised timers
    # around the call site and the Function's forward / backward (as e001's profile, on the current kernel)
    [ "${TIMED:-0}" = 1 ] && rstep wst_q3d2_trained_cand_timed wstieltjes_q3_d2_xs cand 1
  fi
  if [ "${MB:-0}" = 1 ]; then
    step softmax_mb8  softmax_xs $STEPS 0 cand $SPARSE
    step softmax_mb16 softmax_xs $STEPS 0 cand
    rstep wst_q3d2_trained_cand_mb16 wstieltjes_q3_d2_xs cand 0 "--batch_size=16 --gradient_accumulation_steps=16"
  fi
elif [ "$PLAN" = final ]; then
  PART=${PART:?set PART=a|b}
  pair() { rstep ${1}_trained_ref $2 ref; rstep ${1}_trained_cand $2 cand; }                 # name row
  spair() { step ${1}_scratch_ref $2 $STEPS 0 ref $SPARSE; step ${1}_scratch_cand $2 $STEPS 0 cand $SPARSE; }
  if [ "$PART" = a ]; then
    pair wst_q3d2 wstieltjes_q3_d2_xs
    step softmax_mb16 softmax_xs $STEPS 0 cand
    rstep asentmax_trained asentmax_zi_xs cand
    spair wst_q3d2 wstieltjes_q3_d2_xs
    step asentmax_scratch asentmax_zi_xs $STEPS 0 cand $SPARSE
  else
    pair aswst_q3d2 aswstieltjes_q3_d2_zi_xs
    pair wst_q3d1.5 wstieltjes_q3_d1.5_xs
    spair wst_q4d2 wstieltjes_q4_d2_xs
    rstep aswst_q3d2_trained_cand_mb16 aswstieltjes_q3_d2_zi_xs cand 0 "--batch_size=16 --gradient_accumulation_steps=16"
  fi
elif [ "$PLAN" = profile ]; then
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
