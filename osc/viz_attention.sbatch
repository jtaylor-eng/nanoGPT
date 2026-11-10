#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=attn_viz
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%A_%a.log
# attn-viz: per-position attention stats + NoPE rows on 8 ArXiv papers at 16k for one 125M checkpoint per array task.
# Array index = word of $ROWS (default: the 7 finished 125M rows at lr1.2e-3, headline three first). Writes $LM_RESULTS/125m/<row>/<lr>/attn_stats/.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
REPO=$SCRATCH/wt/nanogpt/attn-viz
VENV=$SCRATCH/venvs/nanogpt
LM_RESULTS=$SCRATCH/nanogpt_results/lm1012
export HF_HUB_OFFLINE=1 WANDB_MODE=offline TRITON_CACHE_DIR=$SCRATCH/.triton-cache TORCHINDUCTOR_CACHE_DIR=$SCRATCH/.inductor-cache
source $VENV/bin/activate
ROWS=${ROWS:-"softmax_xs/lr1.2e-3 asentmax_zi_xs/lr1.2e-3 wstieltjes_q3_d2_xs/lr1.2e-3 wstieltjes_q3_d3_xs/lr1.2e-3 wstieltjes_q3_d4_xs/lr1.2e-3 wstieltjes_q4_d2_xs/lr1.2e-3 wstieltjes_q4_d3_xs/lr1.2e-3"}
R=$(echo $ROWS | tr ' ' '\n' | sed -n "${SLURM_ARRAY_TASK_ID:-1}p")
D=$LM_RESULTS/${TAG-125m}/$R   # TAG=t2048 for 51M T=2048, TAG= (empty) for 51M T=512 (rows at the top level)
cd ${TMPDIR:-/tmp}
echo "commit $(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain | wc -l) $(nvidia-smi --query-gpu=name --format=csv,noheader) row=$R $(date '+%F %T')"
OUT=$D/${STATS_DIR:-attn_stats}
[ -f $OUT/summary.json ] && { echo "skip $OUT"; exit 0; }
python $REPO/viz_attention.py --out_dir=$D --save_dir=$OUT --data_file=$REPO/data/arxiv_long/arxiv_long.bin \
  --doc_len=16385 --n_docs=8 --L=16384 "$@"
echo "exit $? $(date '+%F %T')"
