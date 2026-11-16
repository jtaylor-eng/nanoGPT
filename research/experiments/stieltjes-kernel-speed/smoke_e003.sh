#!/usr/bin/env bash
# Pre-submit smoke for e003 (tier 2 through osc/kernel_smoke.sbatch: the trained checkpoints are only on OSC; the
# checkpoint-free parts, `tiles --selftest` and `sweep --small`, also ran on the desktop through gpu-run).
# Both commands of osc/kernel_measure.sbatch at reduced size, from an empty directory with absolute paths.
#   bash <this file> <worktree> <scratch dir>
set -u
W=$1; S=$2; PY=${PY:-$HOME/.venvs/nanogpt/bin/python}; R=${SMOKE_OUT:-$S/res}
CK=$LM_RESULTS/125m_fw25_rng
mkdir -p $S/empty $R
cd $S/empty
$PY $W/bench_kernel.py tiles --ckpts=$CK/wstieltjes_q3_d2_xs/lr1.2e-3/ckpt.pt --fineweb_val=$DATA_SRC/val.bin \
  --arxiv=$W/data/arxiv_long/arxiv_long.bin --layers=0,11 --n_long=16384 --heads_long=0,6 --out=$R/tiles_smoke.json > $R/tiles_smoke.log 2>&1
echo "== tiles rc=$? $(date +%T)"; cut -c1-600 $R/tiles_smoke.log | tail -12
$PY $W/bench_kernel.py sweep --small --out=$R/sweep_smoke.json > $R/sweep_smoke.log 2>&1
echo "== sweep rc=$? $(date +%T)"; cut -c1-300 $R/sweep_smoke.log | tail -40
