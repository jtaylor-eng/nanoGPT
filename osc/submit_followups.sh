#!/usr/bin/env bash
# submit_followups.sh [smoke] -> e006 / e007 / e008: 125M follow-ups to e005 (user decision 2026-11-11), fineweb25B 1 epoch,
# RNG-restore train.py, results $LM_RESULTS/125m_fw25_rng/<row>/lr<lr>[_s<seed>] (same tag as e005).
#   e006 (a) seeds 1, 2 at lr 1.2e-3: softmax_xs, asentmax_zi_xs, wstieltjes_q3_d2_xs        manifest lines 0-5
#   e008 (c) lr 1.2e-3 seed 1337: aswstieltjes_q3_d2_zi_xs, wstieltjes_q3_d1.5_xs            manifest lines 6-7
#   e007 (b) lr 2.4e-3 seed 1337: softmax_xs, asentmax_zi_xs, wstieltjes_q3_d2_xs            manifest lines 8-10
# One chain of 6 segment arrays (4:15, WALL_LIMIT_S=13800, afterany); segments per task softmax 4, ASEntmax 5, Stieltjes 6
# (measured e005: 2.25 / 3.57 / 4.87 segments of train + evals, so each has at least one spare).
# `smoke`: debug-nextgen, 30 iters, no evals, tag smoke_e006: both new rows, the seed path and the lr 2.4e-3 path,
# plus the solver-residual check for the two new rows at num_iter 15 (osc/residual_probe.sbatch).
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
EXP=ALL,REPO=$REPO,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,MIN_TRAIN_TOKENS=2490000000
if [ "${1:-}" = smoke ]; then
  TAG=smoke_e006; C="CONFIG=lm_stieltjes_125m.py RESULTS_TAG=$TAG"
  mkdir -p $LM_RESULTS/slurm $LM_RESULTS/$TAG; cd $LM_RESULTS/slurm
  M=$LM_RESULTS/$TAG/manifest.txt
  printf '%s\n' "$C aswstieltjes_q3_d2_zi_xs 1.2e-3" "$C wstieltjes_q3_d1.5_xs 1.2e-3" "$C softmax_xs 1.2e-3 1" "$C asentmax_zi_xs 2.4e-3" > $M
  export SKIP_EVAL=1 TRAIN_EXTRA="--max_iters=30 --lr_decay_iters=30 --eval_interval=15 --eval_iters=2 --final_eval_iters=2 --log_interval=5"
  j=$(sbatch --parsable --job-name=lm_smoke6 --partition=debug-nextgen --time=0:30:00 --exclude=a0339 --array=0-3 \
      --export=$EXP,MANIFEST=$M,WALL_LIMIT_S=1500 $REPO/osc/lm_array.sbatch)
  r=$(sbatch --parsable --partition=debug-nextgen --exclude=a0339 --export=ALL,REPO=$REPO,OUT=$LM_RESULTS/$TAG/residual \
      $REPO/osc/residual_probe.sbatch \
      $LM_RESULTS/125m/aswstieltjes_q3_d2_zi_xs/lr1.2e-3:2048,16384:1 $LM_RESULTS/wstieltjes_q3_d1.5_xs/lr1.5e-3:512,4096:4)
  echo "smoke array $j residual $r  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/$TAG/jobs.txt
  exit 0
fi
TAG=125m_fw25_rng; C="CONFIG=lm_stieltjes_125m.py RESULTS_TAG=$TAG"
mkdir -p $LM_RESULTS/slurm $LM_RESULTS/$TAG; cd $LM_RESULTS/slurm
M=$LM_RESULTS/$TAG/manifest_e006_e008.txt; : > $M
for row in softmax_xs asentmax_zi_xs wstieltjes_q3_d2_xs; do for seed in 1 2; do echo "$C $row 1.2e-3 $seed" >> $M; done; done
for row in aswstieltjes_q3_d2_zi_xs wstieltjes_q3_d1.5_xs; do echo "$C $row 1.2e-3" >> $M; done
for row in softmax_xs asentmax_zi_xs wstieltjes_q3_d2_xs; do echo "$C $row 2.4e-3" >> $M; done
NSEG=(4 4 5 5 6 6 6 6 4 5 6)
[ $(wc -l < $M) -eq ${#NSEG[@]} ] || { echo "manifest / NSEG mismatch"; exit 1; }
dep=""; ids=""
for s in $(seq 1 6); do
  arr=$(for i in ${!NSEG[@]}; do [ ${NSEG[$i]} -ge $s ] && echo $i; done | paste -sd,)
  j=$(sbatch --parsable --job-name=lm_e006 --time=4:15:00 --exclude=a0339 --array=$arr \
      --export=$EXP,MANIFEST=$M,WALL_LIMIT_S=13800 $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "e006-e008 arrays:$ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/$TAG/jobs_e006_e008.txt
