#!/usr/bin/env bash
# submit_350m.sh  -> stage 5 (350M, e002): 3 rows x 3 LRs, 2.5B tokens, as a chain of 4:15 segment arrays
# (afterany). Run on the ascend login node from the direction checkout. Manifest order: 0-2 softmax, 3-5 ASEntmax,
# 6-8 windowed Stieltjes (superseded by submit_fw25.sh: e002 ran on fineweb10B via the env.sh bug), so later segments shrink to the slower rows. Segment counts from the e001 probe
# (21.5 / 33.1 / 46.4 h at micro-batch 8, ~3.8 h of training per segment) plus one spare each:
# softmax 7, ASEntmax 10, Stieltjes 14. Spare segments of finished tasks exit after the TRAIN_DONE / json checks.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
mkdir -p $LM_RESULTS/slurm $LM_RESULTS/350m; cd $LM_RESULTS/slurm
M=$LM_RESULTS/350m/manifest_e002.txt; : > $M
for row in softmax_xs asentmax_zi_xs wstieltjes_q3_d2_xs; do for lr in 3e-4 6e-4 1.2e-3; do echo "$row $lr" >> $M; done; done
EXP=ALL,REPO=$REPO,MANIFEST=$M,CONFIG=lm_stieltjes_350m.py,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,WALL_LIMIT_S=13800
dep=""; ids=""
for s in $(seq 1 14); do
  arr=0-8; [ $s -gt 7 ] && arr=3-8; [ $s -gt 10 ] && arr=6-8
  j=$(sbatch --parsable --job-name=lm350m_e002 --time=4:15:00 --array=$arr --export=$EXP $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "e002 arrays:$ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/350m/e002_jobs.txt
