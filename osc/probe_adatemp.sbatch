#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=adatemp_probe
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=%x_%j.log
# e010 probe (lm350m-qdf extension 2026-11-15): which implementation of adaptive-temperature softmax trains at 125M.
# 1. tests on the OSC torch/triton (ssmax + adatemp regression, fused-vs-eager paths).
# 2. train.py throughput / peak memory, 40 steps at accum 4, compile on (as probe_350m): softmax_xs and ssmax_xs at
#    micro-batch 16, adatemp_xs (streamed entropy) at 16 and 8.
# 3. the alternatives without compile: streamed vs per-head materialised training steps at small micro-batches, and the
#    16k-context eval paths (plain and collect_stats) on a random-init 125M model.
# Writes $LM_RESULTS/probe_adatemp/{speed.tsv,alt.tsv}. Submit with --export=ALL,REPO=<direction checkout>.
set -uo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
source ${REPO:-$SCRATCH/nanoGPT}/osc/env.sh; source $REPO/osc/rows.sh; source $REPO/osc/stage.sh
source $VENV/bin/activate
export DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B
P=$LM_RESULTS/probe_adatemp; mkdir -p $P
RUNDIR=$TMPDIR/run; mkdir -p $RUNDIR; stage_data; cd $RUNDIR; ln -sfn $REPO/config config
echo "commit $(git -C $REPO rev-parse --short HEAD) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) DATA_SRC=$DATA_SRC"
echo "== tests $(date '+%H:%M:%S')"
bash $REPO/tests/test_rows.sh | tail -3
python $REPO/tests/test_ssmax_adatemp.py 2>&1 | grep -E "^(OK|BAD|skip|ALL OK|FAILURES)|Error"; echo "test_ssmax_adatemp rc ${PIPESTATUS[0]}"
SLOPES=exp python $REPO/tests/test_attn_paths.py 2>&1 | grep -E "ssmax|adatemp|BAD|ALL OK|FAILURES|Error"; echo "test_attn_paths rc ${PIPESTATUS[0]}"
T=$P/speed.tsv; printf "row\tmicro_batch\trc\tms_per_step_accum4\ttok_per_s\tpeak_mem_gb\thours_4770_steps\n" > $T
speed() {  # row micro-batch
  local row=$1 bs=$2 n=${1}_bs$2
  echo "== $n $(date '+%H:%M:%S')"
  python $REPO/train.py $REPO/config/lm_stieltjes_125m.py --data_root=$DATA_DIR $(row_args $row) \
     --batch_size=$bs --gradient_accumulation_steps=4 --learning_rate=1.2e-3 --min_lr=1.2e-4 --out_dir=$RUNDIR/speed_$n \
     --max_iters=40 --lr_decay_iters=4770 --eval_interval=100000 --final_eval_iters=0 --log_interval=5 \
     --always_save_checkpoint=False > $P/speed_$n.log 2>&1
  local rc=$?
  cp -f $RUNDIR/speed_$n/metrics.jsonl $P/speed_$n.jsonl 2>/dev/null; rm -rf $RUNDIR/speed_$n
  python - "$row" "$bs" "$rc" "$P/speed_$n.jsonl" <<'PY' | tee -a $T
import json, sys, os, statistics as s
row, bs, rc, f = sys.argv[1:]
r = []
if os.path.exists(f):
    r = [json.loads(l) for l in open(f) if '"train"' in l]; r = [x for x in r if x['iter'] >= 15]
if rc != '0' or len(r) < 3:
    print(f"{row}\t{bs}\t{rc}\tnan\tnan\tnan\tnan"); sys.exit()
ms = s.median(x['dt_ms'] for x in r); tps = s.median(x['tok_per_s'] for x in r); mem = max(x.get('mem_gb', 0) for x in r)
print(f"{row}\t{bs}\t{rc}\t{ms:.0f}\t{tps:.0f}\t{mem:.1f}\t{4770 * 2**19 / tps / 3600:.2f}")
PY
  grep -iE "out of memory|Error" $P/speed_$n.log | head -2
}
speed softmax_xs 16
speed ssmax_xs 16
speed adatemp_xs 16
speed adatemp_xs 8
echo "== alternatives $(date '+%H:%M:%S')"
python - $REPO <<'PY' | tee $P/alt.tsv
import sys, time, torch
sys.path.insert(0, sys.argv[1])
from model import GPT, GPTConfig
m = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=2048, bias=False, vocab_size=50304, pos_emb='nape',
                  nape_slope_type='exp', attn_scale='adatemp')).cuda()
print("cell\tmicro_batch\tT\tms\ttok_per_s\tpeak_mem_gb\thours_4770_steps")
def cell(name, bs, T, eager=False, grad=True, stats=False):
    for b in m.transformer.h:
        b.attn.force_per_head = eager; b.attn.collect_stats = stats
    m.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 50304, (bs, T), device='cuda')
    try:
        ts = []
        for _ in range(3):
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if grad:
                    m(x, x)[1].backward(); m.zero_grad(set_to_none=True)
                else:
                    with torch.no_grad(): m(x, x)
            torch.cuda.synchronize(); ts.append(time.time() - t)
        dt = min(ts[1:]); tps = bs * T / dt
        h = f"{4770 * 2**19 / tps / 3600:.1f}" if grad else "-"
        print(f"{name}\t{bs}\t{T}\t{dt * 1000:.0f}\t{tps:.0f}\t{torch.cuda.max_memory_allocated() / 2**30:.1f}\t{h}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"{name}\t{bs}\t{T}\tOOM\tnan\tnan\tnan", flush=True)
cell('train_streamed_nocompile', 16, 2048)
cell('train_streamed_nocompile', 8, 2048)
for bs in (1, 2, 4):
    cell('train_materialised_per_head', bs, 2048, eager=True)
cell('eval_streamed', 1, 16384, grad=False)
cell('eval_collect_stats', 1, 16384, eager=True, grad=False, stats=True)
PY
echo "== done $(date '+%H:%M:%S')"; cat $T
