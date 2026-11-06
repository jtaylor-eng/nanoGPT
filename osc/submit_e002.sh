#!/usr/bin/env bash
# submit_e002.sh  -> learnable-qd e002 (Variant A): lwstieltjes_q3_d2_xs (learnable per-head q_h, d_h, init 3 / 2) at
# 125M, lr {3e-4, 6e-4, 1.2e-3}, config/lm_stieltjes_125m.py, fineweb25B, 2.5B tokens, seed 1337.
# Gates: qdtest.sbatch (fixed path bit-identical, fused-vs-eager incl. q/d grads, throughput <= 17.5 s/step) and
# smoke.sbatch (scratch + resume + arxiv eval). Segment 1 waits afterok on both; later segments afterany.
# 9 x 3:30 segment arrays, WALL_LIMIT_S=11400: 17.5 s/step worst case -> 23.2 h = 7.3 segments + evals; spares exit.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
ROW=lwstieltjes_q3_d2_xs
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
for f in data/arxiv_long/arxiv_long.bin data/pg19_long/pg19_long.bin; do [ -s $REPO/$f ] || { echo "missing $REPO/$f"; exit 1; }; done
[ -s $LM_RESULTS/NUM_ITER ] || { echo "missing $LM_RESULTS/NUM_ITER"; exit 1; }
mkdir -p $LM_RESULTS/slurm $LM_RESULTS/125m; cd $LM_RESULTS/slurm
M=$LM_RESULTS/125m/manifest_lqd_e002.txt; : > $M
for lr in 3e-4 6e-4 1.2e-3; do echo "$ROW $lr" >> $M; done
EXP=ALL,REPO=$REPO,MANIFEST=$M,CONFIG=lm_stieltjes_125m.py,DATA_SRC_OVERRIDE=$SCRATCH/nanogpt_data/fineweb25B,WALL_LIMIT_S=11400
qt=$(sbatch --parsable --export=ALL,REPO=$REPO,ROW=$ROW,OLD_SHA=3074a63,MAX_STEP_MS=17500 $REPO/osc/qdtest.sbatch)
sm=$(sbatch --parsable --export=ALL,REPO=$REPO,ROW=$ROW,CONFIG=lm_stieltjes_125m.py $REPO/osc/smoke.sbatch)
dep="--dependency=afterok:$qt:$sm"; ids="qdtest $qt smoke $sm arrays"
for s in $(seq 1 9); do
  j=$(sbatch --parsable --job-name=lqd_e002 --time=3:30:00 --array=0-2 --export=$EXP $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "e002 $ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/125m/lqd_e002_jobs.txt
