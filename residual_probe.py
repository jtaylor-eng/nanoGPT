"""
Stieltjes solver-residual probe (fork addition): does num_iter=15 converge as well as 30?

Loads <out_dir>/ckpt.pt, runs real val sequences, captures every layer's (scaled) q, k and
calls the fused kernel's stieltjes_solver_residual (per-row |sum_j w_ij - 1| BEFORE the final
normalisation, so a wrong lambda cannot hide) at each num_iter. Also reports the val-loss
change from swapping num_iter at evaluation time.

    python residual_probe.py --out_dir=<run> [--lengths=512,4096] [--iters=15,30]
writes <out_dir>/residual.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT
from stieltjes.triton_stieltjes import stieltjes_solver_residual

out_dir = 'out'
data_dir = 'data/fineweb10B'
lengths = '512,4096'
iters = '15,30'
n_seqs = 4
for arg in sys.argv[1:]:
    k, v = arg.lstrip('-').split('=', 1)
    g = globals()[k]
    globals()[k] = type(g)(v) if not isinstance(g, str) else v

ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location='cuda')
args = ckpt['model_args']
assert args['attn_type'] == 'stieltjes'
val = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
res = {'iter': ckpt['iter_num'], 'model_args': args, 'residual': {}, 'val_loss': {}}

def build(num_iter):
    a = dict(args); a['stieltjes_num_iter'] = num_iter
    m = GPT(GPTConfig(**a))
    m.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
    return m.cuda().eval()

for L in [int(x) for x in lengths.split(',')]:
    W = torch.from_numpy(np.stack([val[i * (L + 1): i * (L + 1) + L + 1].astype(np.int64) for i in range(n_seqs)])).cuda()
    m = build(30)
    for b in m.transformer.h:
        b.attn.capture_qk = True
    with torch.no_grad(), ctx:
        m(W[:, :-1], return_all_logits=True)
    qks = [b.attn.last_qk for b in m.transformer.h]
    slopes = m.transformer.h[0].attn.nape_slopes if args['pos_emb'] == 'nape' else None
    window = None if args['stieltjes_window'] <= 0 else args['stieltjes_window']
    hd = args['n_embd'] // args['n_head']
    res['residual'][L] = {}
    for it in [int(x) for x in iters.split(',')]:
        per_layer = []
        for q, k in qks:
            r = stieltjes_solver_residual(q.contiguous(), k.contiguous(), k.contiguous(), causal=True,
                                          sm_scale=1.0 / hd ** 0.5, stieltjes_q=args['stieltjes_q'],
                                          num_iter=it, window=window, alibi_slopes=slopes)
            per_layer.append({'max': r.max().item(), 'p999': torch.quantile(r.flatten()[:2**24].float(), 0.999).item(),
                              'mean': r.mean().item()})
        res['residual'][L][it] = per_layer
        print(f"L={L} num_iter={it}: max|S-1| over layers {max(p['max'] for p in per_layer):.2e}, "
              f"worst p99.9 {max(p['p999'] for p in per_layer):.2e}", flush=True)
    # val loss on the same windows at each num_iter (bf16 autocast, fused kernel)
    res['val_loss'][L] = {}
    for it in [int(x) for x in iters.split(',')]:
        m = build(it)
        with torch.no_grad(), ctx:
            logits, _ = m(W[:, :-1], return_all_logits=True)
        res['val_loss'][L][it] = F.cross_entropy(logits.float().flatten(0, 1), W[:, 1:].flatten()).item()
    print(f"L={L} val loss by num_iter: {res['val_loss'][L]}", flush=True)

with open(os.path.join(out_dir, 'residual.json'), 'w') as f:
    json.dump(res, f, indent=1)
