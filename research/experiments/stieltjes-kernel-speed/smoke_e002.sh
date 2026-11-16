#!/usr/bin/env bash
# Pre-submit smoke for e002 (tier 2, osc/kernel_smoke.sbatch; the desktop GPU was held by another process):
# the candidate kernel compiles and runs on triton 3.1.0, is correct on the quick random cases, and the micro table
# (training shape, both kernels, NUM_ITER 0/5/15/30) is produced. Run from an empty directory with absolute paths.
#   bash <this file> <worktree> <scratch dir>
set -u
W=$1; S=$2; PY=${PY:-$HOME/.venvs/nanogpt/bin/python}; R=${SMOKE_OUT:-$S/res}
mkdir -p $S/empty $R
cd $S/empty
$PY $W/bench_kernel.py micro --out=$R/micro_smoke.json > $R/micro_smoke.log 2>&1; echo "== micro rc=$? $(date +%T)"; cat $R/micro_smoke.log
$PY $W/tests/test_kernel_equiv.py --part=random --quick --out=$R/equiv_quick.json > $R/equiv_quick.log 2>&1
echo "== equiv_quick rc=$? $(date +%T)"; grep -vc '^OK ' $R/equiv_quick.log; tail -4 $R/equiv_quick.log | cut -c1-600
