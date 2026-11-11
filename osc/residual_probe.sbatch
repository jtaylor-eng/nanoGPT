#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm_resid
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.log
# Solver residual |S-1| at num_iter 15 vs 30 (residual_probe.py) on existing Stieltjes checkpoints, to check that
# $LM_RESULTS/NUM_ITER=15 holds for a row before training it. Arguments: one <run_dir>:<lengths>:<n_seqs> per
# checkpoint. The run dirs are not written to: ckpt.pt is symlinked into $TMPDIR and residual.json goes to
# $OUT/<row>__<lr dir>.json. Submit with --export=ALL,REPO=<checkout>,OUT=<dir>.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source ${REPO:-$SCRATCH/nanoGPT}/osc/env.sh; source $VENV/bin/activate
mkdir -p $OUT; cd $TMPDIR
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
rc=0
for spec in "$@"; do
  IFS=: read -r run lens nseq <<< "$spec"; name=$(basename $(dirname $run))__$(basename $run)
  mkdir -p $TMPDIR/$name; ln -sfn $run/ckpt.pt $TMPDIR/$name/ckpt.pt
  echo "== $run lengths=$lens n_seqs=$nseq $(date '+%H:%M:%S')"
  python $REPO/residual_probe.py --out_dir=$TMPDIR/$name --data_dir=$SCRATCH/nanogpt_data/fineweb25B --lengths=$lens --iters=15,30 --n_seqs=$nseq \
    && cp $TMPDIR/$name/residual.json $OUT/$name.json || rc=1
done
exit $rc
