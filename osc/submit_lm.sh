#!/usr/bin/env bash
# submit_lm.sh  -> probe job + 12-task array (rows x LRs) held until the probe finishes (afterany:
# the array falls back to num_iter=30 if the probe failed to write NUM_ITER).
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh
cd $REPO; [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
mkdir -p $LM_RESULTS/slurm; cd $LM_RESULTS/slurm
M=$LM_RESULTS/manifest.txt; : > $M
for row in softmax stieltjes_q4 wstieltjes_q4_d2 aswstieltjes_q2_d4_zi; do for lr in 7.5e-4 1.5e-3 3e-3; do echo "$row $lr" >> $M; done; done
P=$(sbatch --parsable $REPO/osc/probe.sbatch)
A=$(sbatch --parsable --dependency=afterany:$P --array=0-$(( $(wc -l < $M) - 1 )) --export=ALL,MANIFEST=$M $REPO/osc/lm_array.sbatch)
echo "probe $P  array $A  manifest $M  commit $(git -C $REPO rev-parse --short HEAD)"
