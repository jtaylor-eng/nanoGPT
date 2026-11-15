#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=de_lmeval
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --array=0-9
#SBATCH --output=/fs/scratch/PAS2836/jacktaylor/nanogpt_results/logs/%x_%A_%a.log
# direction downstream-eval: lm-eval-harness (LAMBADA-openai, HellaSwag, 0-shot) on one 125M checkpoint per
# array task -> $LM_RESULTS/$TAG/<run>/$OUT_NAME. TAG= results tag (default 125m); RUNS_FILE= file of <row>/<lr dir>
# lines (relative to $DE_REPO; default: the 10 e001 runs). EXTRA= passes extra eval_downstream.py args (smoke: --limit=50).
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
DE_REPO=$SCRATCH/wt/nanogpt/downstream-eval
source $DE_REPO/osc/env.sh
REPO=$DE_REPO VENV=$SCRATCH/venvs/nanogpt-eval
export HF_DATASETS_OFFLINE=1 TIKTOKEN_CACHE_DIR=$SCRATCH/.tiktoken-cache
source $VENV/bin/activate
TAG=${TAG:-125m}
if [ -n "${RUNS_FILE:-}" ]; then
  mapfile -t RUNS < <(grep -vE '^\s*(#|$)' $DE_REPO/$RUNS_FILE)
else
  RUNS=(softmax_xs/lr1.2e-3 asentmax_zi_xs/lr1.2e-3 wstieltjes_q3_d2_xs/lr1.2e-3
        softmax_xs/lr6e-4 asentmax_zi_xs/lr6e-4 wstieltjes_q3_d2_xs/lr6e-4
        wstieltjes_q3_d3_xs/lr1.2e-3 wstieltjes_q3_d4_xs/lr1.2e-3 wstieltjes_q4_d2_xs/lr1.2e-3 wstieltjes_q4_d3_xs/lr1.2e-3)
fi
[ -n "${RUNS[$SLURM_ARRAY_TASK_ID]:-}" ] || { echo "no run for task $SLURM_ARRAY_TASK_ID (${#RUNS[@]} runs)"; exit 1; }
D=$LM_RESULTS/$TAG/${RUNS[$SLURM_ARRAY_TASK_ID]}
OUT_NAME=${OUT_NAME:-lmeval.json}
[ -z "$(git -C $REPO status --porcelain)" ] || { echo "dirty tree $REPO"; exit 1; }
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader) run $D"
[ -f $D/TRAIN_DONE ] || { echo "no TRAIN_DONE in $D"; exit 1; }
[ -f $D/$OUT_NAME ] && { echo "exists $D/$OUT_NAME"; exit 0; }
cd $TMPDIR
python $REPO/eval_downstream.py --out_dir=$D --out_name=$OUT_NAME ${EXTRA:-} 2>&1 | grep -vE "it/s\]|s/it\]" || exit 1
echo DONE $(date '+%H:%M:%S')
