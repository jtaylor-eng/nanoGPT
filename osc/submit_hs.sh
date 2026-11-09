#!/usr/bin/env bash
# submit_hs.sh  -> pe-ssmax-headsplit e002 (ssmax_xs) + e003 (4 ALiBi/NoPE split cells), 125M, lr 1.2e-3, fineweb25B (1 epoch),
# one chain of 6 segment arrays (4:15, WALL_LIMIT_S=13800, afterany) over a 5-line manifest.
# Segments per task = ceil(train h / 3.83) + 1 spare: softmax-speed rows (8.4-8.5 h, probe 8368600) 4, Stieltjes (18.1 h) 6.
# Results: $LM_RESULTS/125m_fw25/<row>/lr1.2e-3, next to the 6/6 baselines of lm350m-qdf e004.
# SMOKE=1: 2 lines (ssmax_xs, wstieltjes_q3_d2_xs_a2), 6 steps, no evals, debug-nextgen, results under smoke_hs/.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
EXP=ALL,REPO=$REPO,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,MIN_TRAIN_TOKENS=2490000000
if [ -n "${SMOKE:-}" ]; then
  D=$LM_RESULTS/smoke_hs; mkdir -p $D; cd $D; M=$D/manifest.txt
  printf '%s\n' "CONFIG=lm_stieltjes_125m.py RESULTS_TAG=smoke_hs ssmax_xs 1.2e-3" "CONFIG=lm_stieltjes_125m.py RESULTS_TAG=smoke_hs wstieltjes_q3_d2_xs_a2 1.2e-3" > $M
  sbatch --parsable --job-name=hs_smoke --partition=debug-nextgen --time=0:30:00 --array=0-1 \
    --export="$EXP,MANIFEST=$M,SKIP_EVAL=1,TRAIN_EXTRA=--max_iters=6 --lr_decay_iters=6 --eval_interval=3 --eval_iters=2 --final_eval_iters=2" $REPO/osc/lm_array.sbatch
  exit 0
fi
D=$LM_RESULTS/125m_fw25; mkdir -p $D $LM_RESULTS/slurm; cd $LM_RESULTS/slurm; M=$D/manifest_hs_e002_e003.txt; : > $M
C="CONFIG=lm_stieltjes_125m.py RESULTS_TAG=125m_fw25"
for row in ssmax_xs softmax_xs_a4 softmax_xs_a2 wstieltjes_q3_d2_xs_a4 wstieltjes_q3_d2_xs_a2; do echo "$C $row 1.2e-3" >> $M; done
NSEG=(4 4 4 6 6)
dep=""; ids=""
for s in $(seq 1 6); do
  arr=$(for i in ${!NSEG[@]}; do [ ${NSEG[$i]} -ge $s ] && echo $i; done | paste -sd,)
  j=$(sbatch --parsable --job-name=hs_fw25 --time=4:15:00 --array=$arr --export=$EXP,MANIFEST=$M,WALL_LIMIT_S=13800 $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "hs e002+e003 arrays:$ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $D/jobs_hs.txt
