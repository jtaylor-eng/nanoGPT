"""
Length-extrapolation eval for a trained checkpoint (fork addition).

For each context length L: val loss over the same deterministic, non-overlapping L-token
windows from the start of val.bin (identical tokens for every method), plus loss by
position bucket (so "loss at positions 2048-4095 given a 4096 context" is separable from
"loss at short positions"). Then attention statistics per layer/head (entropy, exact
support size, max prob; all rows and last-quarter rows) from the eager path on a few
sequences at stats_lengths.

    python eval_lengths.py --out_dir=<run> [--lengths=512,1024,2048,4096] [--tokens_per_length=2097152]
writes <run>/lengths.json
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

out_dir = 'out'
data_dir = 'data/fineweb10B'
lengths = '512,1024,2048,4096'
tokens_per_length = 2 * 1024 * 1024
batch_tokens = 8192           # tokens per forward batch (logits are 50k-wide: 8k tok = 1.6 GB fp32)
stats_lengths = '512,4096'
stats_seqs = 4
out_name = 'lengths.json'        # output file name (override to keep an existing lengths.json)
data_file = ''                   # '' = <data_dir>/val.bin, consecutive non-overlapping windows
doc_len = 0                      # >0: data_file holds documents of exactly doc_len tokens; each window = first L+1 tokens of a doc
max_docs = 0                     # doc mode: use only the first max_docs documents (0 = all)
q_override = 0.0                 # >0: evaluate a Stieltjes checkpoint with this q instead of the trained one
window_override = -1.0           # >=0: evaluate with this window d (0 = dense) instead of the trained one
nope_cap = 0                     # >0: NoPE heads see only the last nope_cap keys (eager per-head path, slow at 16k)
num_iter_override = 0         # >0: evaluate with a different Stieltjes Newton-sweep count
device = 'cuda'
for arg in sys.argv[1:]:
    k, v = arg.lstrip('-').split('=', 1)
    g = globals()[k]
    globals()[k] = type(g)(v) if not isinstance(g, str) else v

ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
args = dict(ckpt['model_args'])
if num_iter_override > 0:
    args['stieltjes_num_iter'] = num_iter_override
model = GPT(GPTConfig(**args))
sd = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
model.load_state_dict(sd)
model.to(device).eval()
attns = [b.attn for b in model.transformer.h]
for a in attns:
    if q_override > 0:
        a.stieltjes_q = float(q_override)
    if window_override >= 0:
        a.stieltjes_window = None if window_override == 0 else float(window_override)
    if nope_cap > 0:
        a.nope_cap = int(nope_cap); a.force_per_head = True
args['eval_overrides'] = {'q': q_override, 'window': window_override, 'nope_cap': nope_cap}
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
val = np.memmap(data_file if data_file else os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def windows(L, n):
    if doc_len > 0:
        assert L + 1 <= doc_len, f'L={L} exceeds doc_len={doc_len}'
        n = min(n, len(val) // doc_len)
        idx = np.arange(n) * doc_len
    else:
        idx = np.arange(n) * (L + 1)
    assert idx[-1] + L + 1 <= len(val), 'val.bin too small'
    w = np.stack([val[i:i + L + 1].astype(np.int64) for i in idx])
    return torch.from_numpy(w)

res = {'iter': ckpt['iter_num'], 'model_args': args, 'lengths': {}}
for L in [int(x) for x in lengths.split(',')]:
    if args['pos_emb'] == 'learned' and L > args['block_size']:
        continue
    n = max(1, tokens_per_length // L)
    if doc_len > 0:
        n = len(val) // doc_len          # every document at every length
        if max_docs > 0:
            n = min(n, max_docs)
    W = windows(L, n)
    bs = max(1, batch_tokens // L)
    nb = 16                                     # position buckets
    bucket_sum = torch.zeros(nb, dtype=torch.float64); bucket_cnt = torch.zeros(nb, dtype=torch.float64)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, bs):
            w = W[i:i + bs].to(device)
            x, y = w[:, :-1], w[:, 1:]
            with ctx:
                logits, _ = model(x, return_all_logits=True)
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction='none').view(y.shape)
            b = (torch.arange(L, device=device) * nb // L)
            bucket_sum += torch.zeros(nb, device=device, dtype=torch.float64).index_add_(0, b, nll.double().sum(0)).cpu()
            bucket_cnt += torch.bincount(b, minlength=nb).double().cpu() * y.shape[0]
    torch.cuda.synchronize()
    loss = (bucket_sum.sum() / bucket_cnt.sum()).item()
    res['lengths'][L] = {'val_loss': loss, 'n_seqs': n, 'eval_s': time.time() - t0,
                         'bucket_edges': [L * j // nb for j in range(nb + 1)],
                         'bucket_loss': (bucket_sum / bucket_cnt).tolist()}
    print(f"L={L}: val loss {loss:.4f} over {n} seqs ({time.time() - t0:.0f}s); "
          f"first/last bucket {res['lengths'][L]['bucket_loss'][0]:.3f}/{res['lengths'][L]['bucket_loss'][-1]:.3f}", flush=True)

res['stats'] = {}
for L in [int(x) for x in stats_lengths.split(',') if x]:
    if args['pos_emb'] == 'learned' and L > args['block_size']:
        continue
    W = windows(L, stats_seqs)
    for a in attns:
        a.collect_stats = True
    acc = None
    with torch.no_grad():
        for i in range(stats_seqs):
            with ctx:
                model(W[i:i + 1, :-1].to(device), return_all_logits=True)
            s = [{k: v for k, v in a.last_stats.items()} for a in attns]
            acc = s if acc is None else [{k: acc[l][k] + s[l][k] for k in s[l]} for l in range(len(s))]
    for a in attns:
        a.collect_stats = False
    # per layer, per head lists
    res['stats'][L] = [{k: (v / stats_seqs).tolist() for k, v in layer.items()} for layer in acc]
    sup = np.mean([np.mean(l['support_tail']) for l in res['stats'][L]])
    ent = np.mean([np.mean(l['entropy_tail']) for l in res['stats'][L]])
    print(f"stats L={L}: last-quarter rows mean support {sup:.1f} keys, entropy {ent:.3f} nats", flush=True)

with open(os.path.join(out_dir, out_name if num_iter_override <= 0 else f'lengths_it{num_iter_override}.json'), 'w') as f:
    json.dump(res, f, indent=1)
print('wrote', out_dir)
