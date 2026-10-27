"""Stage-0 diagnostic: gradient of the LM loss w.r.t. per-head (q, d) of a trained Stieltjes checkpoint, on long
documents. Tells which way learnable q / d would move before we build the training backward.

Each layer's attention gets leaf tensors head_q (H,) and head_window (H,) initialised at the trained values; the
forward uses the differentiable fp32 map (stieltjes_normalize_qd). We accumulate dL/dq and dL/dd over n_docs
documents at context L, plus a finite-difference check of the total-loss change for a global +-delta in d and q.

    python qd_grad_probe.py --out_dir=<run> --data_file=data/arxiv_long/arxiv_long.bin --doc_len=16385 --L=4096 --n_docs=8
writes <out_dir>/qd_grad_L<L>.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

out_dir = 'out'
data_file = 'data/arxiv_long/arxiv_long.bin'
doc_len = 16385
L = 4096
n_docs = 8
device = 'cuda'
for arg in sys.argv[1:]:
    k, v = arg.lstrip('-').split('=', 1); g = globals()[k]; globals()[k] = type(g)(v) if not isinstance(g, str) else v

ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
args = dict(ckpt['model_args']); assert args['attn_type'] == 'stieltjes'
model = GPT(GPTConfig(**args))
model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
model.to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
H = args['n_head']
attns = [b.attn for b in model.transformer.h]
q0, d0 = float(args['stieltjes_q']), float(args['stieltjes_window'])
for a in attns:
    a.force_per_head = True
    a.head_q = torch.full((H,), q0, device=device, requires_grad=True)
    a.head_window = torch.full((H,), d0, device=device, requires_grad=True)

data = np.memmap(data_file, dtype=np.uint16, mode='r')
n_docs = min(n_docs, len(data) // doc_len)
W = torch.from_numpy(np.stack([data[i * doc_len: i * doc_len + L + 1].astype(np.int64) for i in range(n_docs)]))

def run_loss(grad):
    tot = 0.0
    for i in range(n_docs):
        w = W[i:i + 1].to(device); x, y = w[:, :-1], w[:, 1:]
        with torch.set_grad_enabled(grad), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x, return_all_logits=True)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        if grad:
            (loss / n_docs).backward()
        tot += loss.item() / n_docs
        del logits, loss
    return tot

base = run_loss(True)
gq = torch.stack([a.head_q.grad for a in attns]).cpu()        # (layers, H)
gd = torch.stack([a.head_window.grad for a in attns]).cpu()
nope = [h for h in range(H) if float(attns[0].nape_slopes[h]) == 0.0]
alibi = [h for h in range(H) if h not in nope]
res = {'iter': ckpt['iter_num'], 'q0': q0, 'd0': d0, 'L': L, 'n_docs': n_docs, 'loss': base,
       'dL_dq': gq.tolist(), 'dL_dd': gd.tolist(),
       'sum_dL_dq': gq.sum().item(), 'sum_dL_dd': gd.sum().item(),
       'sum_dL_dq_nope': gq[:, nope].sum().item(), 'sum_dL_dd_nope': gd[:, nope].sum().item(),
       'sum_dL_dq_alibi': gq[:, alibi].sum().item(), 'sum_dL_dd_alibi': gd[:, alibi].sum().item()}
print(f"L={L} loss {base:.4f}  sum dL/dq {res['sum_dL_dq']:+.4f} (nope {res['sum_dL_dq_nope']:+.4f}, alibi {res['sum_dL_dq_alibi']:+.4f})"
      f"  sum dL/dd {res['sum_dL_dd']:+.4f} (nope {res['sum_dL_dd_nope']:+.4f}, alibi {res['sum_dL_dd_alibi']:+.4f})", flush=True)
# finite-difference check on global shifts (also directly useful: loss at d0 +- 0.25, q0 +- 0.5)
fd = {}
for name, dq, dd in (('d+0.25', 0, 0.25), ('d-0.25', 0, -0.25), ('q+0.5', 0.5, 0), ('q-0.5', -0.5, 0)):
    for a in attns:
        a.head_q = torch.full((H,), q0 + dq, device=device); a.head_window = torch.full((H,), d0 + dd, device=device)
    fd[name] = run_loss(False)
    pred = base + res['sum_dL_dq'] * dq + res['sum_dL_dd'] * dd
    print(f"  {name}: loss {fd[name]:.4f} (linear prediction {pred:.4f})", flush=True)
res['finite_diff'] = fd
with open(os.path.join(out_dir, f'qd_grad_L{L}.json'), 'w') as f:
    json.dump(res, f, indent=1)
