"""
Per-query-position attention statistics and raw attention rows on long ArXiv papers (direction attn-viz, e001/e002).

Wraps each CausalSelfAttention.forward (model.py is untouched) with a chunked eager path: query chunks of `chunk`
rows against all causal keys, fp32, one head at a time, same math as CausalSelfAttention._probs_head (NAPE bias,
AS scale, masking, map). For every layer, head and query position it records, averaged over documents:
  support   exact count of nonzero weights (softmax: ~ i+1, fp32 exp only underflows ~100 nats below the top)
  k90       number of keys holding 90 % of the mass (top-weight first), comparable across maps
  entropy   nats;  pmax;  first64 = mass on keys 0..63 (sink);  local2k = mass on keys < 2048 back;  dist = E[i - j]
and saves the full attention row (weights and pre-map scores) of every NoPE head at query positions `rows_pos`
for the first `row_docs` documents.

    python viz_attention.py --out_dir=<run with ckpt.pt> --data_file=data/arxiv_long/arxiv_long.bin --doc_len=16385
writes <out_dir>/attn_stats/{stats.npz, rows.npz, summary.json}  (or --save_dir)
"""
import os, sys, json, math, time, types
import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT
from entmax import entmax_bisect
from stieltjes.stieltjes_eager import stieltjes_normalize

out_dir = ''
save_dir = ''                     # '' = <out_dir>/attn_stats
data_file = ''
doc_len = 16385
n_docs = 8
L = 16384
chunk = 512
rows_pos = '1024,4096,16384'      # 1-based query positions (query index pos-1)
row_docs = 2
check_L = 2048                    # >0: loss of doc 0 at check_L through the wrapped vs the stock forward
table_pos = '512,2048,8192,16384'
device = 'cuda'
for arg in sys.argv[1:]:
    k, v = arg.lstrip('-').split('=', 1)
    g = globals()[k]
    globals()[k] = type(g)(v) if not isinstance(g, str) else v
save_dir = save_dir or os.path.join(out_dir, 'attn_stats')
os.makedirs(save_dir, exist_ok=True)
STATS = ('support', 'k90', 'entropy', 'pmax', 'first64', 'local2k', 'dist', 'scale')   # scale: AS factor on q (1 = none)
rows_pos = [int(x) for x in rows_pos.split(',') if int(x) <= L]
table_pos = [int(x) for x in table_pos.split(',') if int(x) <= L]

ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
args = dict(ckpt['model_args'])
# checkpoints from other direction branches may carry config keys this branch lacks; only inert ones may be dropped
extra = {k: args.pop(k) for k in list(args) if k not in GPTConfig.__dataclass_fields__}
assert all(k == 'stieltjes_learn_qd' and not v for k, v in extra.items()), f'unsupported model_args {extra}'
model = GPT(GPTConfig(**args))
model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
model.to(device).eval()
del ckpt
attns = [b.attn for b in model.transformer.h]
NL, H = len(attns), attns[0].n_head
slopes = attns[0].nape_slopes.float().cpu() if args['pos_emb'] == 'nape' else torch.zeros(H)
nope = [h for h in range(H) if float(slopes[h]) == 0.0]
print(f"{out_dir}: attn {args['attn_type']} scale {args.get('attn_scale')} q {args.get('stieltjes_q')} "
      f"d {args.get('stieltjes_window')} iters {args.get('stieltjes_num_iter')}; NoPE heads {nope}", flush=True)
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda'))
data = np.memmap(data_file, dtype=np.uint16, mode='r')
n_docs = min(n_docs, len(data) // doc_len)
assert L + 1 <= doc_len


def prob_map(self, s):
    if self.attn_type == 'softmax':
        return F.softmax(s, dim=-1)
    if self.attn_type == 'entmax':
        return entmax_bisect(s, alpha=self.entmax_alpha, dim=-1, n_iter=50)
    return stieltjes_normalize(s, q=self.stieltjes_q, num_iter=self.stieltjes_num_iter, window=self.stieltjes_window)


def chunked_forward(self, x):
    B, T, C = x.size()
    assert B == 1
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
    st = self.acc                                                    # (n_stats, H, T) for this layer, this doc
    st[7] = 1.0
    if self.attn_scale == 'as':
        q0 = q
        q = self._apply_scale(q, x)
        st[7] = q[0].float().norm(dim=-1) / q0[0].float().norm(dim=-1).clamp_min(1e-12)   # scalar per (h, i)
    qf, kf = q[0].float(), k[0].float()                              # (H, T, D)
    y = torch.empty_like(v[0])
    fill = float('-inf') if self.attn_type in ('softmax', 'entmax') else -1e30
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        qi = torch.arange(c0, c1, device=x.device)
        kj = torch.arange(c1, device=x.device)
        back = (qi[:, None] - kj[None, :])                           # i - j
        causal = back >= 0
        dist = back.clamp(min=0).float()
        local = (back < 2048) & causal
        bias = -dist                                                 # nape_bias: slope * min(j - i, 0)
        for h in range(self.n_head):
            s = (qf[h, c0:c1] @ kf[h, :c1].T) * (1.0 / math.sqrt(self.head_dim))
            if self.pos_emb == 'nape' and float(self.nape_slopes[h]) != 0.0:
                s = s + float(self.nape_slopes[h]) * bias
            s = s.masked_fill(~causal, fill)
            p = prob_map(self, s)
            y[h, c0:c1] = p.to(v.dtype) @ v[0, h, :c1]
            ps = torch.sort(p, dim=-1, descending=True)[0].cumsum(-1)
            st[0, h, c0:c1] = (p > 0).sum(-1).float()
            st[1, h, c0:c1] = ((ps < 0.9).sum(-1) + 1).clamp(max=c1).float()
            st[2, h, c0:c1] = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
            st[3, h, c0:c1] = p.amax(-1)
            st[4, h, c0:c1] = p[:, :64].sum(-1)
            st[5, h, c0:c1] = (p * local).sum(-1)
            st[6, h, c0:c1] = (p * dist).sum(-1)
            if self.keep_rows and h in self.row_heads:
                for r, pos in enumerate(rows_pos):
                    i = pos - 1
                    if c0 <= i < c1:
                        hi = self.row_heads.index(h)
                        self.rows_p[hi, r, :i + 1] = p[i - c0, :i + 1].cpu()
                        self.rows_s[hi, r, :i + 1] = s[i - c0, :i + 1].cpu()
            del s, p, ps
    y = y.unsqueeze(0).transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y))


stock = [a.forward for a in attns]
def wrap(on):
    for a, f in zip(attns, stock):
        a.forward = types.MethodType(chunked_forward, a) if on else f

def doc(i, n):
    w = torch.from_numpy(data[i * doc_len: i * doc_len + n + 1].astype(np.int64)).to(device)
    return w[None, :-1], w[None, 1:]

res = {'out_dir': out_dir, 'model_args': args, 'stats': STATS, 'nope_heads': nope, 'slopes': slopes.tolist(),
       'L': L, 'n_docs': n_docs, 'chunk': chunk}
with torch.no_grad():
    if check_L > 0:
        x, y = doc(0, check_L)
        for a in attns:
            a.acc = torch.zeros(len(STATS), H, check_L, device=device); a.keep_rows = False
        with ctx:
            l_stock = model(x, targets=y)[1].item()
            wrap(True); l_wrap = model(x, targets=y)[1].item(); wrap(False)
        res['check'] = {'L': check_L, 'loss_stock': l_stock, 'loss_wrapped': l_wrap}
        print(f"check L={check_L} doc 0: stock {l_stock:.4f} wrapped {l_wrap:.4f} diff {l_wrap - l_stock:+.4f}", flush=True)
        assert abs(l_wrap - l_stock) < 0.02, 'wrapped forward disagrees with the model'

    wrap(True)
    total = torch.zeros(len(STATS), NL, H, L, device=device)
    nb = L // 64
    binned = np.zeros((n_docs, len(STATS), NL, H, nb), dtype=np.float32)
    rows_p = np.zeros((min(row_docs, n_docs), NL, len(nope), len(rows_pos), L), dtype=np.float32)
    rows_s = np.zeros_like(rows_p)
    losses = []
    for d in range(n_docs):
        t0 = time.time()
        for l, a in enumerate(attns):
            a.acc = torch.zeros(len(STATS), H, L, device=device)
            a.keep_rows = d < row_docs; a.row_heads = nope
            a.rows_p = torch.zeros(len(nope), len(rows_pos), L); a.rows_s = torch.full((len(nope), len(rows_pos), L), float('nan'))
        x, y = doc(d, L)
        with ctx:
            loss = model(x, targets=y)[1].item()
        losses.append(loss)
        for l, a in enumerate(attns):
            total[:, l] += a.acc
            binned[d, :, l] = a.acc.view(len(STATS), H, nb, 64).mean(-1).cpu().numpy()
            if d < row_docs:
                rows_p[d, l] = a.rows_p.numpy(); rows_s[d, l] = a.rows_s.numpy()
        torch.cuda.synchronize() if device == 'cuda' else None
        print(f"doc {d}: loss@{L} {loss:.4f}  {time.time() - t0:.0f}s", flush=True)
    mean = (total / n_docs).cpu().numpy()

res['loss_per_doc'] = losses
# table: mean over the 64 query positions ending at each table position; per head (mean over docs) and per group
# (mean over layers and heads of the group; SE over documents from the 64-position bins)
groups = {'alibi': [h for h in range(H) if h not in nope], 'nope': nope}
res['table'] = {}
for pos in table_pos:
    b = pos // 64 - 1
    ent = {'per_head': {s: binned[:, j, :, :, b].mean(0).round(5).tolist() for j, s in enumerate(STATS)}}
    for g, hs in groups.items():
        if not hs:
            continue
        per_doc = binned[:, :, :, hs, b].mean(axis=(2, 3))            # (docs, stats)
        ent[g] = {s: {'mean': float(per_doc[:, j].mean()),
                      'se': float(per_doc[:, j].std(ddof=1) / math.sqrt(n_docs)) if n_docs > 1 else None}
                  for j, s in enumerate(STATS)}
    res['table'][pos] = ent
np.savez_compressed(os.path.join(save_dir, 'stats.npz'), mean=mean, binned=binned, stats=np.array(STATS),
                    slopes=slopes.numpy(), nope=np.array(nope))
np.savez_compressed(os.path.join(save_dir, 'rows.npz'), p=rows_p, s=rows_s, rows_pos=np.array(rows_pos),
                    nope=np.array(nope))
with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
    json.dump(res, f, indent=1)
for pos in table_pos:
    t = res['table'][pos]
    print(f"pos {pos:>5}: " + '  '.join(f"{g} support {t[g]['support']['mean']:.1f} k90 {t[g]['k90']['mean']:.1f} "
                                        f"H {t[g]['entropy']['mean']:.3f}" for g in groups if groups[g]), flush=True)
print('wrote', save_dir, flush=True)
