#!/usr/bin/env bash
# Pre-submit smoke for e001 on the desktop GPU (tier 1): the job commands of osc/kernel_gate.sbatch and
# osc/kernel_time.sbatch at toy size, from an empty directory with absolute paths.
#   gpu-run -- bash <this file> <worktree> <scratch dir>
# Toy model (2L x 4H x 256, T=512) trained 8 steps on the ArXiv eval tokens; timings here mean nothing (32x32 tiles).
set -u
W=$1; S=$2; PY=${PY:-$HOME/.venvs/nanogpt/bin/python}; R=$S/res
mkdir -p $S/empty $S/fw $R
ln -sfn $W/data/arxiv_long/arxiv_long.bin $S/fw/train.bin; ln -sfn $W/data/arxiv_long/arxiv_long.bin $S/fw/val.bin
export TRITON_CACHE_DIR=$S/triton TORCHINDUCTOR_CACHE_DIR=$S/inductor
cd $S/empty
source $W/osc/rows.sh
TINY="--n_layer=2 --n_head=4 --n_embd=256 --block_size=512 --batch_size=4 --gradient_accumulation_steps=2 --warmup_iters=2"
for spec in "wst wstieltjes_q3_d2_xs 1 cand" "wstref wstieltjes_q3_d2_xs 0 ref" "ent asentmax_zi_xs 1 cand"; do
  set -- $spec
  echo "== step $1 $(date +%T)"
  $PY $W/bench_kernel.py wrap --kernel=$4 --timers=$3 --out=$R/step_$1.json -- $W/train.py $W/config/lm_stieltjes_125m.py \
    --data_root=$S/fw $(row_args $2) --stieltjes_num_iter=15 --learning_rate=1.2e-3 --min_lr=1.2e-4 --out_dir=$S/out_$1 \
    --max_iters=7 --eval_interval=100000 --eval_iters=1 --final_eval_iters=0 --log_interval=1 $TINY > $R/step_$1.log 2>&1
  echo "== step $1 rc=$? $(grep -c '^iter ' $R/step_$1.log) iters $(date +%T)"
done
for kern in ref cand; do
  $PY $W/bench_kernel.py wrap --kernel=$kern -- $W/eval_lengths.py --out_dir=$S/out_wst \
    --data_file=$W/data/arxiv_long/arxiv_long.bin --doc_len=16385 --lengths=2048 --max_docs=8 --batch_tokens=16384 \
    --stats_lengths= --out_name=ax_$kern.json > $R/eval_$kern.log 2>&1
  echo "== eval $kern rc=$? $(grep -E '^L=' $R/eval_$kern.log) $(date +%T)"
done
$PY $W/bench_kernel.py summary $R; echo "== summary rc=$? $(date +%T)"
$PY $W/bench_kernel.py micro --small --out=$R/micro.json > $R/micro.log 2>&1; echo "== micro rc=$? $(date +%T)"; tail -34 $R/micro.log
$PY $W/tests/test_kernel_equiv.py --part=real --ckpts=$S/out_wst/ckpt.pt --fineweb_val=$W/data/arxiv_long/arxiv_long.bin \
  --arxiv=$W/data/arxiv_long/arxiv_long.bin --heads16k=0,3 --out=$R/equiv_real.json > $R/equiv_real.log 2>&1
echo "== equiv_real rc=$? $(date +%T)"; tail -3 $R/equiv_real.log | cut -c1-500
$PY $W/tests/test_kernel_equiv.py --part=random --quick --out=$R/equiv_quick.json > $R/equiv_quick.log 2>&1
echo "== equiv_quick rc=$? $(date +%T)"; tail -3 $R/equiv_quick.log | cut -c1-700
