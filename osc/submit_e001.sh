#!/usr/bin/env bash
# submit_e001.sh  -> learnable-qd e001 (the gate): aswstieltjes_q3_d2_zi_xs at 125M, lr {3e-4, 6e-4, 1.2e-3},
# config/lm_stieltjes_125m.py, fineweb25B, 2.5B tokens, seed 1337. Chain of 8 x 3:30 segment arrays (afterany),
# WALL_LIMIT_S=11400 (~3.1 h of training per segment). Measured 125M q3/d2 step 13.7 s, +5 % for the AS scale
# (51M ratio) -> ~19 h of training = 7 segments, the 8th is a spare for the evals. Spare segments of finished
# tasks exit after the TRAIN_DONE / json checks. Run on the ascend login node from the direction checkout.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
for f in data/arxiv_long/arxiv_long.bin data/pg19_long/pg19_long.bin; do [ -s $REPO/$f ] || { echo "missing $REPO/$f"; exit 1; }; done
[ -s $LM_RESULTS/NUM_ITER ] || { echo "missing $LM_RESULTS/NUM_ITER"; exit 1; }
mkdir -p $LM_RESULTS/slurm $LM_RESULTS/125m; cd $LM_RESULTS/slurm
M=$LM_RESULTS/125m/manifest_lqd_e001.txt; : > $M
for lr in 3e-4 6e-4 1.2e-3; do echo "aswstieltjes_q3_d2_zi_xs $lr" >> $M; done
EXP=ALL,REPO=$REPO,MANIFEST=$M,CONFIG=lm_stieltjes_125m.py,DATA_SRC_OVERRIDE=$SCRATCH/nanogpt_data/fineweb25B,WALL_LIMIT_S=11400
# pre-submit smoke gates the chain: segment 1 waits afterok on it (a failed smoke leaves the chain pending)
sm=$(sbatch --parsable --export=ALL,REPO=$REPO,ROW=aswstieltjes_q3_d2_zi_xs,CONFIG=lm_stieltjes_125m.py $REPO/osc/smoke.sbatch)
dep="--dependency=afterok:$sm"; ids="smoke $sm arrays"
for s in $(seq 1 8); do
  j=$(sbatch --parsable --job-name=lqd_e001 --time=3:30:00 --array=0-2 --export=$EXP $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "e001 $ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/125m/lqd_e001_jobs.txt
