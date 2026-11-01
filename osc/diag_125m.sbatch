#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm125m_diag
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH --output=%x_%j.log
# Free diagnostics on the trained 125M windowed q3/d2 checkpoint (lr 1.2e-3): inference-time d and q sweep on ArXiv
# (64 papers), and dL/dq, dL/dd per head at L=2048 (8 papers). Same protocol as stage 0 at 51M.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh; source $VENV/bin/activate
cd $TMPDIR
DATA=$REPO/data/arxiv_long/arxiv_long.bin
R=$LM_RESULTS/125m/wstieltjes_q3_d2_xs/lr1.2e-3
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
ev() { local tag=$1; shift
  [ -f $R/$tag.json ] && { echo "skip $tag"; return; }
  echo "== $tag $(date '+%H:%M:%S')"
  python $REPO/eval_lengths.py --out_dir=$R --data_file=$DATA --doc_len=16385 --max_docs=64 \
    --lengths=512,2048,4096,8192,16384 --batch_tokens=16384 --stats_lengths= --out_name=$tag.json "$@" 2>&1 | grep -E "^L=|wrote|Error"
}
ev arxiv_base64
for d in 1.5 2.5 3 4; do ev arxiv_d$d --window_override=$d; done
for q in 2 4 6; do ev arxiv_q$q --q_override=$q; done
[ -f $R/qd_grad_L2048.json ] || python $REPO/qd_grad_probe.py --out_dir=$R --data_file=$DATA --doc_len=16385 --L=2048 --n_docs=8 2>&1 | grep -E "^L=|^  |Error"
echo DONE $(date '+%H:%M:%S')
