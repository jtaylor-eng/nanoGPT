#!/usr/bin/env bash
# submit_hs_resume_ssmax.sh -> pe-ssmax-headsplit e002 retry: finish ssmax_xs (125M, lr 1.2e-3, fineweb25B) from ckpt iter 4250.
# Segment 4 (8453516_0) died with CUDA OOM in the first backward after resuming exactly on an eval boundary
# (eval + compile at micro-batch 16 left 7.3 GB reserved-unallocated). Fix: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
# Segment A: same micro-batch (16 x 16). Segment B (afterany, exits at once if A finished): micro-batch 8 x 32 (same 524,288
# tok/step) as a fallback. 520 steps x 6.47 s = 0.94 h + ~0.3 h evals.
# SMOKE=1: copy the ckpt to smoke_hs/ssmax_resume and reproduce the failing path (resume at 4250, eval, 2 steps), debug-nextgen.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
EXP=ALL,REPO=$REPO,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,MIN_TRAIN_TOKENS=2490000000,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRC=$LM_RESULTS/125m_fw25/ssmax_xs/lr1.2e-3
if [ -n "${SMOKE:-}" ]; then
  D=$LM_RESULTS/smoke_hs; mkdir -p $D/ssmax_resume/ssmax_xs/lr1.2e-3; cp $SRC/ckpt.pt $D/ssmax_resume/ssmax_xs/lr1.2e-3/
  cd $D; M=$D/manifest_resume.txt; echo "CONFIG=lm_stieltjes_125m.py RESULTS_TAG=smoke_hs/ssmax_resume ssmax_xs 1.2e-3" > $M
  sbatch --parsable --job-name=hs_smoke --partition=debug-nextgen --time=0:30:00 --array=0 \
    --export="$EXP,MANIFEST=$M,SKIP_EVAL=1,TRAIN_EXTRA=--max_iters=4252 --final_eval_iters=2" $REPO/osc/lm_array.sbatch
  exit 0
fi
D=$LM_RESULTS/125m_fw25; cd $LM_RESULTS/slurm; M=$D/manifest_hs_resume_ssmax.txt
echo "CONFIG=lm_stieltjes_125m.py RESULTS_TAG=125m_fw25 ssmax_xs 1.2e-3" > $M
a=$(sbatch --parsable --job-name=hs_fw25 --time=2:00:00 --array=0 --exclude=a0339 --export=$EXP,MANIFEST=$M,WALL_LIMIT_S=6600 $REPO/osc/lm_array.sbatch)
b=$(sbatch --parsable --job-name=hs_fw25 --time=2:00:00 --array=0 --exclude=a0339 --dependency=afterany:$a \
  --export="$EXP,MANIFEST=$M,WALL_LIMIT_S=6600,TRAIN_EXTRA=--batch_size=8 --gradient_accumulation_steps=32" $REPO/osc/lm_array.sbatch)
echo "hs ssmax resume arrays: $a $b  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $D/jobs_hs_resume_ssmax.txt
