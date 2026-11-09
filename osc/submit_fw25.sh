#!/usr/bin/env bash
# submit_fw25.sh  -> e003 (350M, 9 runs) + e004 (125M redo, 6 runs) on fineweb25B (2.5B tok = 1 epoch for 4770 steps),
# as ONE chain of 14 segment arrays (4:15, WALL_LIMIT_S=13800, afterany) over a 15-line mixed-config manifest.
# Segments 2-14 are submitted held: release them (`scontrol release <ids>`) after segment 1's run.log shows
# DATA_SRC=.../fineweb25B. Segment count per task = ceil(train h / 3.8) + 1 spare (+1 for 350M Stieltjes, as e002):
# 125M softmax 4, ASEntmax 5, Stieltjes 6; 350M softmax 7, ASEntmax 10, Stieltjes 14.
# Results: $LM_RESULTS/350m_fw25/<row>/lr<lr>, $LM_RESULTS/125m_fw25/<row>/lr<lr>.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
mkdir -p $LM_RESULTS/slurm $LM_RESULTS/350m_fw25; cd $LM_RESULTS/slurm
M=$LM_RESULTS/350m_fw25/manifest_e003_e004.txt; : > $M
C1="CONFIG=lm_stieltjes_125m.py RESULTS_TAG=125m_fw25"; C3="CONFIG=lm_stieltjes_350m.py RESULTS_TAG=350m_fw25"
for row in softmax_xs asentmax_zi_xs wstieltjes_q3_d2_xs; do for lr in 6e-4 1.2e-3; do echo "$C1 $row $lr" >> $M; done; done
for row in softmax_xs asentmax_zi_xs wstieltjes_q3_d2_xs; do for lr in 3e-4 6e-4 1.2e-3; do echo "$C3 $row $lr" >> $M; done; done
NSEG=(4 4 5 5 6 6 7 7 7 10 10 10 14 14 14)
EXP=ALL,REPO=$REPO,MANIFEST=$M,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,MIN_TRAIN_TOKENS=2490000000,WALL_LIMIT_S=13800
dep=""; hold=""; ids=""
for s in $(seq 1 14); do
  arr=$(for i in ${!NSEG[@]}; do [ ${NSEG[$i]} -ge $s ] && echo $i; done | paste -sd,)
  j=$(sbatch --parsable $hold --job-name=lm_fw25 --time=4:15:00 --array=$arr --export=$EXP $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; hold="--hold"; ids="$ids $j"
done
echo "e003+e004 arrays:$ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/350m_fw25/jobs.txt
