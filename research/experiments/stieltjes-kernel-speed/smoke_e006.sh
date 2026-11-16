#!/usr/bin/env bash
# Second pre-submit smoke for e006 (tier 2 through osc/kernel_smoke.sbatch). The first one is the job script itself
# (osc/kernel_time.sbatch PLAN=ab_trained QUICK=1 STEPS=3 CMP=cand_noexit); this one covers what only the gate job runs
# and what failed for a harness reason in the first: `aliascheck` on the A100's triton 3.1.0 and the quick skipcheck
# (the e004 / e005 cases with gradients and the 112 cycle-exit cases at other sweep counts). From an empty directory.
#   bash <this file> <worktree> <scratch dir>
set -u
W=$1; S=$2; PY=${PY:-$HOME/.venvs/nanogpt/bin/python}
mkdir -p $S/empty
cd $S/empty
$PY $W/bench_kernel.py aliascheck
echo "== aliascheck rc=$? $(date +%T)"
$PY $W/bench_kernel.py skipcheck --quick > $S/skipcheck_quick.log 2>&1
echo "== skipcheck --quick rc=$? $(date +%T)"; grep -a "^BAD" $S/skipcheck_quick.log | cut -c1-240 | head -20; tail -3 $S/skipcheck_quick.log | cut -c1-240
