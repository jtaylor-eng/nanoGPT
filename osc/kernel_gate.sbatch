#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=ks_gate
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/fs/scratch/PAS2836/jacktaylor/nanogpt_results/lm1012/kernel_speed/slurm/%x_%j.log
# Correctness gate for the candidate kernel (stieltjes/triton_stieltjes.py) against eager and the frozen baseline
# (stieltjes/triton_stieltjes_ref.py). Direction stieltjes-kernel-speed. Submit with --export=ALL,TAG=e00N.
#   1. tests/test_attn_paths.py, SLOPES=linear and exp
#   2. tests/test_kernel_equiv.py --part=real  (trained wstieltjes_q3_d2_xs and aswstieltjes_q3_d2_zi_xs, 12 layers)
#   3. eval_lengths.py with the baseline and the candidate: FineWeb 2048 and ArXiv 16384, |d loss| < 1e-3
#   4. bench_kernel.py skipcheck  (e004: tile flags trusted / never trusted / trust failing, bit for bit; e005: gradients too;
#      e006: the cycle exit at other sweep counts) and, first of all, bench_kernel.py aliascheck (Triton loop-carried aliases)
#   5. tests/test_kernel_equiv.py --part=random
# Checkpoints are read through symlinks in $TMPDIR (the run dirs are read-only for this direction).
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$SCRATCH/wt/nanogpt/stieltjes-kernel-speed VENV=$SCRATCH/venvs/nanogpt DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B
source $REPO/osc/env.sh; source $VENV/bin/activate
TAG=${TAG:?set TAG=e00N}; OUT=$LM_RESULTS/kernel_speed/$TAG; mkdir -p $OUT
[ -z "$(git -C $REPO status --porcelain)" ] || { echo "dirty tree in $REPO"; exit 2; }
# Triton/inductor caches: env.sh defaults on scratch (persistent, keyed by source hash: the frozen kernel compiles once)
cd $TMPDIR
echo "commit $(git -C $REPO rev-parse HEAD) job $SLURM_JOB_ID host $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $(date '+%F %T')"
CK=$LM_RESULTS/125m_fw25_rng
fail=0
step() { local name=$1; shift; echo "== $name $(date '+%T')"; "$@"; local rc=$?; echo "== $name rc=$rc $(date '+%T')"; [ $rc -eq 0 ] || fail=1; }

step aliascheck python $REPO/bench_kernel.py aliascheck
step attn_paths_linear env SLOPES=linear python $REPO/tests/test_attn_paths.py
step attn_paths_exp env SLOPES=exp python $REPO/tests/test_attn_paths.py
step equiv_real python $REPO/tests/test_kernel_equiv.py --part=real \
  --ckpts=$CK/wstieltjes_q3_d2_xs/lr1.2e-3/ckpt.pt,$CK/aswstieltjes_q3_d2_zi_xs/lr1.2e-3/ckpt.pt \
  --fineweb_val=$DATA_SRC/val.bin --arxiv=$REPO/data/arxiv_long/arxiv_long.bin --out=$OUT/equiv_real.json

for kern in ref cand; do
  E=$TMPDIR/e2e_$kern; mkdir -p $E; ln -sfn $CK/wstieltjes_q3_d2_xs/lr1.2e-3/ckpt.pt $E/ckpt.pt
  step e2e_fw2048_$kern python $REPO/bench_kernel.py wrap --kernel=$kern -- $REPO/eval_lengths.py --out_dir=$E \
    --data_dir=$DATA_SRC --lengths=2048 --batch_tokens=16384 --stats_lengths= --out_name=fw2048.json
  step e2e_arxiv16k_$kern python $REPO/bench_kernel.py wrap --kernel=$kern -- $REPO/eval_lengths.py --out_dir=$E \
    --data_file=$REPO/data/arxiv_long/arxiv_long.bin --doc_len=16385 --lengths=16384 --batch_tokens=16384 \
    --stats_lengths= --out_name=arxiv16k.json
  cp $E/fw2048.json $OUT/e2e_fw2048_$kern.json; cp $E/arxiv16k.json $OUT/e2e_arxiv16k_$kern.json
done
step e2e_compare python - $OUT <<'PY'
import json, sys
d = sys.argv[1]; ok = True; res = {}
for name, L in (('fw2048', '2048'), ('arxiv16k', '16384')):
    r = json.load(open(f'{d}/e2e_{name}_ref.json'))['lengths'][L]; c = json.load(open(f'{d}/e2e_{name}_cand.json'))['lengths'][L]
    tok = lambda x: x['n_seqs'] * int(L) / x['eval_s']
    res[name] = dict(loss_ref=r['val_loss'], loss_cand=c['val_loss'], diff=c['val_loss'] - r['val_loss'], tok_s_ref=tok(r), tok_s_cand=tok(c))
    good = abs(res[name]['diff']) < 1e-3; ok &= good
    print(('OK ' if good else 'BAD'), name, json.dumps(res[name]))
json.dump(res, open(f'{d}/e2e.json', 'w'), indent=1)
sys.exit(0 if ok else 1)
PY
step skipcheck python $REPO/bench_kernel.py skipcheck
# last: the most compile-heavy part (cold cache), so a walltime kill cannot take the real-input and e2e results with it
step equiv_random python $REPO/tests/test_kernel_equiv.py --part=random --out=$OUT/equiv_random.json
[ $fail -eq 0 ] && echo "GATE PASS $TAG" || echo "GATE FAIL $TAG"
echo DONE $(date '+%T')
exit $fail
