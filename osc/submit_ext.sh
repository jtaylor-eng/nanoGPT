#!/usr/bin/env bash
# submit_ext.sh [probe|smoke] -> e009 / e010: two softmax-family rows for the bottom row of figures/e006_e008_lengths.png
# (user extension 2026-11-15). 125M, fineweb25B 1 epoch, RNG-restore train.py, seed 1337, results under
# $LM_RESULTS/125m_fw25_rng/<row>/lr<lr> (same tag and protocol as e005-e008).
#   e009 ssmax_xs   lr 1.2e-3, 2.4e-3                                   manifest lines 0-1
#   e010 adatemp_xs lr 1.2e-3 (+ 2.4e-3 with ADATEMP_LR24=1)            manifest lines 2-3
# Set from the e010 probe (osc/probe_adatemp.sbatch): ADATEMP_MB (micro-batch; accumulation = 256 / micro-batch, so every
# row keeps 524,288 tok/step), ADATEMP_NSEG (4:15 segments for one adatemp run), ADATEMP_LR24.
# ssmax: 3 segments (8.95 GPU-h measured on direction pe-ssmax-headsplit = 2.3 segments of 3.83 h + evals).
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: the pe-ssmax-headsplit ssmax_xs run hit a fragmentation OOM when it
# resumed on an eval boundary at micro-batch 16 (8453516_0); allocator setting only, no numerical effect.
# `probe`: debug-nextgen throughput / memory probe.  `smoke`: debug-nextgen, 30 iters, no evals, tag smoke_ext.
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
MB=${ADATEMP_MB:-16}
EXP=ALL,REPO=$REPO,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,MIN_TRAIN_TOKENS=2490000000,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $LM_RESULTS/slurm; cd $LM_RESULTS/slurm
if [ "${1:-}" = probe ]; then
  j=$(sbatch --parsable --exclude=a0339 --export=ALL,REPO=$REPO $REPO/osc/probe_adatemp.sbatch)
  mkdir -p $LM_RESULTS/probe_adatemp; echo "probe $j  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/probe_adatemp/jobs.txt
  exit 0
fi
if [ "${1:-}" = smoke ]; then
  TAG=smoke_ext; C="CONFIG=lm_stieltjes_125m.py RESULTS_TAG=$TAG"; mkdir -p $LM_RESULTS/$TAG
  M=$LM_RESULTS/$TAG/manifest.txt
  printf '%s\n' "$C ssmax_xs 2.4e-3" "MICRO_BATCH=$MB $C adatemp_xs 1.2e-3" > $M
  export SKIP_EVAL=1 TRAIN_EXTRA="--max_iters=30 --lr_decay_iters=30 --eval_interval=15 --eval_iters=2 --final_eval_iters=2 --log_interval=5"
  j=$(sbatch --parsable --job-name=lm_smoke9 --partition=debug-nextgen --time=0:30:00 --exclude=a0339 --array=0-1 \
      --export=$EXP,MANIFEST=$M,WALL_LIMIT_S=1500 $REPO/osc/lm_array.sbatch)
  echo "smoke array $j  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/$TAG/jobs.txt
  exit 0
fi
TAG=125m_fw25_rng; C="CONFIG=lm_stieltjes_125m.py RESULTS_TAG=$TAG"; mkdir -p $LM_RESULTS/$TAG
M=$LM_RESULTS/$TAG/manifest_e009_e010.txt; : > $M
for lr in 1.2e-3 2.4e-3; do echo "$C ssmax_xs $lr" >> $M; done
NSEG=(3 3)
: ${ADATEMP_NSEG:?set ADATEMP_NSEG from the probe}
for lr in 1.2e-3 $([ -n "${ADATEMP_LR24:-}" ] && echo 2.4e-3); do echo "MICRO_BATCH=$MB $C adatemp_xs $lr" >> $M; NSEG+=($ADATEMP_NSEG); done
max=0; for n in ${NSEG[@]}; do [ $n -gt $max ] && max=$n; done
dep=""; ids=""
for s in $(seq 1 $max); do
  arr=$(for i in ${!NSEG[@]}; do [ ${NSEG[$i]} -ge $s ] && echo $i; done | paste -sd,)
  j=$(sbatch --parsable --job-name=lm_e009 --time=4:15:00 --exclude=a0339 --array=$arr \
      --export=$EXP,MANIFEST=$M,WALL_LIMIT_S=13800 $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "e009-e010 arrays:$ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/$TAG/jobs_e009_e010.txt
