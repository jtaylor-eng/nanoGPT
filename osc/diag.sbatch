#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=lm1012_diag
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.log
# Stage 0 diagnostics on existing exp-slope checkpoints (no training). All on the ArXiv long-doc set.
#  (a) inference-time q/d sweep on the trained windowed q4/d2 model (selected LR 1.5e-3, and 3e-3)
#  (b) dL/dq, dL/dd per head at L=2048 and 8192 on the three Stieltjes rows
#  (c) NoPE heads capped to the last 2048 keys, softmax / entmax / ASEntmax / windowed q4/d2, selected LRs
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/nanoGPT/osc/env.sh; source $VENV/bin/activate
cd $TMPDIR
DATA=$REPO/data/arxiv_long/arxiv_long.bin
D=$LM_RESULTS/diag; mkdir -p $D
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
ev() { # ev <run_dir> <tag> extra-args...   (64 docs, lengths 512..16k, no stats)
  local run=$1 tag=$2; shift 2
  [ -f $run/$tag.json ] && { echo "skip $run/$tag"; return; }
  echo "== $run $tag $(date '+%H:%M:%S')"
  python $REPO/eval_lengths.py --out_dir=$run --data_file=$DATA --doc_len=16385 --max_docs=64 \
    --lengths=512,2048,4096,8192,16384 --batch_tokens=16384 --stats_lengths= --out_name=$tag.json "$@" 2>&1 | grep -E "^L=|wrote|Error"
}
# (a) q/d sweep at inference, windowed q4/d2
for lr in 1.5e-3 3e-3; do
  R=$LM_RESULTS/wstieltjes_q4_d2_xs/lr$lr
  for d in 1.25 1.5 3 4 0; do ev $R arxiv_d$d --window_override=$d; done
  for q in 3 6 8; do ev $R arxiv_q$q --q_override=$q; done
done
# (b) q/d gradients (eager per-head fp32; 8 docs)
for row in wstieltjes_q4_d2_xs aswstieltjes_q2_d4_zi_xs stieltjes_q4_xs; do
  for lr in 1.5e-3 3e-3; do
    R=$LM_RESULTS/$row/lr$lr
    for L in 2048 4096; do   # 8192 OOMs (fp32 per-head maps + autograd on 40 GB)
      [ -f $R/qd_grad_L$L.json ] && continue
      echo "== grad $row lr$lr L=$L $(date '+%H:%M:%S')"
      python $REPO/qd_grad_probe.py --out_dir=$R --data_file=$DATA --doc_len=16385 --L=$L --n_docs=8 2>&1 | grep -E "^L=|^  |Error"
    done
  done
done
# (c) NoPE cap 2048 (eager per-head path: slow at 16k, so 32 docs and no 16k for the dense maps)
for rl in softmax_xs:3e-3 entmax_xs:1.5e-3 asentmax_zi_xs:1.5e-3 wstieltjes_q4_d2_xs:1.5e-3; do
  R=$LM_RESULTS/${rl%%:*}/lr${rl##*:}
  [ -f $R/arxiv_cap2048.json ] && continue
  echo "== cap $R $(date '+%H:%M:%S')"
  python $REPO/eval_lengths.py --out_dir=$R --data_file=$DATA --doc_len=16385 --max_docs=32 \
    --lengths=2048,4096,8192 --batch_tokens=8192 --stats_lengths= --out_name=arxiv_cap2048.json --nope_cap=2048 2>&1 | grep -E "^L=|wrote|Error"
done
echo DONE $(date '+%H:%M:%S')
