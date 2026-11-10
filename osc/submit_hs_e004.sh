#!/usr/bin/env bash
# submit_hs_e004.sh  -> pe-ssmax-headsplit e004: asentmax_zi_xs_a4, asentmax_zi_xs_a2, 125M, lr 1.2e-3, fineweb25B (1 epoch).
# One chain of 5 segment arrays (4:15, WALL_LIMIT_S=13800, afterany): ASEntmax 10.1 s/step (51.9k tok/s, lm350m-qdf e004)
# -> 13.4 h train + ~0.3 h evals = ceil(13.7 / 3.83) = 4 segments + 1 spare.
# SMOKE=1: both lines, 6 steps, no evals, debug-nextgen, results under smoke_hs/.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
EXP=ALL,REPO=$REPO,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,MIN_TRAIN_TOKENS=2490000000
ROWS="asentmax_zi_xs_a4 asentmax_zi_xs_a2"
if [ -n "${SMOKE:-}" ]; then
  D=$LM_RESULTS/smoke_hs; mkdir -p $D; cd $D; M=$D/manifest_e004.txt; : > $M
  for row in $ROWS; do echo "CONFIG=lm_stieltjes_125m.py RESULTS_TAG=smoke_hs $row 1.2e-3" >> $M; done
  sbatch --parsable --job-name=hs_smoke --partition=debug-nextgen --time=0:30:00 --array=0-1 \
    --export="$EXP,MANIFEST=$M,SKIP_EVAL=1,TRAIN_EXTRA=--max_iters=6 --lr_decay_iters=6 --eval_interval=3 --eval_iters=2 --final_eval_iters=2" $REPO/osc/lm_array.sbatch
  exit 0
fi
D=$LM_RESULTS/125m_fw25; mkdir -p $D $LM_RESULTS/slurm; cd $LM_RESULTS/slurm; M=$D/manifest_hs_e004.txt; : > $M
for row in $ROWS; do echo "CONFIG=lm_stieltjes_125m.py RESULTS_TAG=125m_fw25 $row 1.2e-3" >> $M; done
dep=""; ids=""
for s in $(seq 1 5); do
  j=$(sbatch --parsable --job-name=hs_e004 --time=4:15:00 --array=0-1 --exclude=a0339 --export=$EXP,MANIFEST=$M,WALL_LIMIT_S=13800 $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "hs e004 arrays:$ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $D/jobs_hs_e004.txt
