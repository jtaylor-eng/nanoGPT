#!/usr/bin/env bash
# submit_e005.sh -> e005: 125M rerun of softmax_xs / asentmax_zi_xs / wstieltjes_q3_d2_xs x lr {6e-4, 1.2e-3} on
# fineweb25B (1 epoch) with the RNG-restore fix (ckpt.pt stores the batch-sampler RNG, so resumes do not replay data).
# One chain of 6 segment arrays (4:15, WALL_LIMIT_S=13800, afterany); segments per task softmax 4, ASEntmax 5, Stieltjes 6.
# Results: $LM_RESULTS/125m_fw25_rng/<row>/lr<lr> (new tag: the replayed e004 runs in 125m_fw25 stay intact).
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
TAG=125m_fw25_rng
mkdir -p $LM_RESULTS/slurm $LM_RESULTS/$TAG; cd $LM_RESULTS/slurm
M=$LM_RESULTS/$TAG/manifest_e005.txt; : > $M
for row in softmax_xs asentmax_zi_xs wstieltjes_q3_d2_xs; do for lr in 6e-4 1.2e-3; do
  echo "CONFIG=lm_stieltjes_125m.py RESULTS_TAG=$TAG $row $lr" >> $M; done; done
NSEG=(4 4 5 5 6 6)
EXP=ALL,REPO=$REPO,MANIFEST=$M,DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B,MIN_TRAIN_TOKENS=2490000000,WALL_LIMIT_S=13800
dep=""; ids=""
for s in $(seq 1 6); do
  arr=$(for i in ${!NSEG[@]}; do [ ${NSEG[$i]} -ge $s ] && echo $i; done | paste -sd,)
  j=$(sbatch --parsable --job-name=lm_e005 --time=4:15:00 --exclude=a0339 --array=$arr --export=$EXP $dep $REPO/osc/lm_array.sbatch)
  dep="--dependency=afterany:$j"; ids="$ids $j"
done
echo "e005 arrays:$ids  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)" | tee $LM_RESULTS/$TAG/jobs.txt
