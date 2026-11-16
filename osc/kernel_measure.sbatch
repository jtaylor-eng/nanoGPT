#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=ks_meas
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/fs/scratch/PAS2836/jacktaylor/nanogpt_results/lm1012/kernel_speed/slurm/%x_%j.log
# Measurements that choose the next kernel candidate (no kernel edit). Direction stieltjes-kernel-speed.
# Submit with --export=ALL,TAG=e00N.
#   1. bench_kernel.py tiles: live (query block, key tile) fraction per Newton sweep on trained q/k (FineWeb 2048, ArXiv 16384)
#   2. bench_kernel.py sweep: tile sizes / num_warps / num_stages of the four kernels at the training shape and at 16k
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$SCRATCH/wt/nanogpt/stieltjes-kernel-speed VENV=$SCRATCH/venvs/nanogpt DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B
source $REPO/osc/env.sh; source $VENV/bin/activate
TAG=${TAG:?set TAG=e00N}; OUT=$LM_RESULTS/kernel_speed/$TAG; mkdir -p $OUT
[ -z "$(git -C $REPO status --porcelain)" ] || { echo "dirty tree in $REPO"; exit 2; }
cd $TMPDIR
echo "commit $(git -C $REPO rev-parse HEAD) job $SLURM_JOB_ID host $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $(date '+%F %T')"
CK=$LM_RESULTS/125m_fw25_rng
fail=0
echo "== tiles $(date '+%T')"
python $REPO/bench_kernel.py tiles --ckpts=$CK/wstieltjes_q3_d2_xs/lr1.2e-3/ckpt.pt,$CK/aswstieltjes_q3_d2_zi_xs/lr1.2e-3/ckpt.pt \
  --fineweb_val=$DATA_SRC/val.bin --arxiv=$REPO/data/arxiv_long/arxiv_long.bin --out=$OUT/tiles.json 2>&1 | tee $OUT/tiles.log
rc=${PIPESTATUS[0]}; echo "== tiles rc=$rc $(date '+%T')"; [ $rc -eq 0 ] || fail=1
echo "== sweep $(date '+%T')"
python $REPO/bench_kernel.py sweep --out=$OUT/sweep.json 2>&1 | tee $OUT/sweep.log
rc=${PIPESTATUS[0]}; echo "== sweep rc=$rc $(date '+%T')"; [ $rc -eq 0 ] || fail=1
echo DONE $(date '+%T')
exit $fail
