"""Correctness gate for edits to stieltjes/triton_stieltjes.py (direction stieltjes-kernel-speed). CUDA required.

Candidate (stieltjes/triton_stieltjes.py) vs eager fp32 (stieltjes_eager.stieltjes_normalize, num_iter 30) AND vs the
frozen baseline (stieltjes/triton_stieltjes_ref.py), on the output and on dq, dk, dv, plus the solver residual |S - 1|.

    python tests/test_kernel_equiv.py --part=random [--quick] [--out=equiv_random.json]
    python tests/test_kernel_equiv.py --part=real --ckpts=<run>/ckpt.pt,<run2>/ckpt.pt --fineweb_val=<val.bin> \
        --arxiv=<arxiv_long.bin> [--heads16k=0,2,5,6,8,11] [--out=equiv_real.json]

A case passes when, for x in (out, dq, dk, dv):
  (a) rel err candidate-vs-eager < tol   (tests/test_attn_paths.py: out 2e-3 fp32 / 4e-2 bf16, grads 5x that), and
  (b) rel err candidate-vs-eager <= 1.2 x rel err baseline-vs-eager on the same input, and
  (c) real inputs only: max |S - 1| of the candidate <= the baseline's.
Where the frozen baseline itself misses (a), the case is tagged REFTOL and judged on (b), (c) only (the baseline is
the definition of "unchanged"); such cases are listed in the summary.
Exit code 0 iff every case passes.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import stieltjes.triton_stieltjes as cand
import stieltjes.triton_stieltjes_ref as ref
import stieltjes.stieltjes_eager as eager
from model import GPTConfig, GPT, nape_slopes

eager._solve_compiled = eager._solve_lambda   # reference solve without torch.compile: same ops, no per-shape recompiles

p = argparse.ArgumentParser()
p.add_argument('--part', default='random', choices=['random', 'real', 'all'])
p.add_argument('--quick', action='store_true', help='random part at N in {64, 300} only (desktop smoke)')
p.add_argument('--ckpts', default='')
p.add_argument('--fineweb_val', default='')
p.add_argument('--arxiv', default='')
p.add_argument('--heads16k', default='0,2,5,6,8,11', help="heads compared against eager at N=16384 ('all' = every head)")
p.add_argument('--num_iter', type=int, default=15)
p.add_argument('--out', default='')
args = p.parse_args()
dev = 'cuda'
RATIO = 1.2


def rel(u, v):
    return ((u.float() - v.float()).norm() / v.float().norm().clamp_min(1e-12)).item()


def fused(mod, q, k, v, do, sq, window, num_iter, slopes):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o = mod.stieltjes_attention(q, k, v, causal=True, sm_scale=q.shape[-1] ** -0.5, stieltjes_q=sq,
                                num_iter=num_iter, window=window, alibi_slopes=slopes)
    o.backward(do)
    return [o.detach().float(), q.grad.float(), k.grad.float(), v.grad.float()]


def eager_ref(q, k, v, do, sq, window, slopes, heads, chunk):
    """fp32 materialised reference on the same input values, `chunk` heads at a time. Returns [o, dq, dk, dv] for `heads`."""
    N, D = q.shape[2], q.shape[3]
    i = torch.arange(N, device=dev)
    diff = (i[None, :] - i[:, None]).clamp(max=0).float()
    causal = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
    outs = [[], [], [], []]
    for a in range(0, len(heads), chunk):
        hs = heads[a:a + chunk]
        qf, kf, vf = (t[:, hs].float().detach().requires_grad_(True) for t in (q, k, v))
        s = (qf @ kf.transpose(-2, -1)) * D ** -0.5
        if slopes is not None:
            s = s + slopes[hs].float().view(1, -1, 1, 1) * diff
        s = s.masked_fill(~causal, -1e30)
        o = eager.stieltjes_normalize(s, q=sq, num_iter=30, window=window) @ vf
        o.backward(do[:, hs].float())
        for lst, t in zip(outs, (o.detach(), qf.grad, kf.grad, vf.grad)):
            lst.append(t)
        del s, o, qf, kf, vf
    return [torch.cat(x, dim=1) for x in outs]


results = []

def check(tag, q, k, v, sq, window, num_iter, slopes, heads=None, chunk=None, gate_resid=False):
    B, H, N, D = q.shape
    heads = list(range(H)) if heads is None else heads
    do = torch.randn_like(q)
    c = fused(cand, q, k, v, do, sq, window, num_iter, slopes)
    r = fused(ref, q, k, v, do, sq, window, num_iter, slopes)
    e = eager_ref(q, k, v, do, sq, window, slopes, heads, chunk or len(heads))
    ec = [rel(c[i][:, heads], e[i]) for i in range(4)]
    er = [rel(r[i][:, heads], e[i]) for i in range(4)]
    ecr = [rel(c[i], r[i]) for i in range(4)]
    kw = dict(causal=True, sm_scale=D ** -0.5, stieltjes_q=sq, num_iter=num_iter, window=window, alibi_slopes=slopes)
    rc = cand.stieltjes_solver_residual(q, k, v, **kw).max().item()
    rr = ref.stieltjes_solver_residual(q, k, v, **kw).max().item()
    tol = 2e-3 if q.dtype == torch.float32 else 4e-2
    tols = [tol, 5 * tol, 5 * tol, 5 * tol]
    ref_tol_ok = all(x < t for x, t in zip(er, tols))
    ok_tol = all(x < t for x, t in zip(ec, tols))
    ok_ratio = all(x <= RATIO * y for x, y in zip(ec, er))
    ok_res = (rc <= rr) if gate_resid else True
    ok = (ok_tol or not ref_tol_ok) and ok_ratio and ok_res
    rec = dict(tag=tag, shape=[B, H, N, D], dtype=str(q.dtype)[6:], q=sq, window=window, num_iter=num_iter,
               slopes=slopes is not None, heads_vs_eager=len(heads), err_cand=ec, err_ref=er, err_cand_vs_ref=ecr,
               resid_cand=rc, resid_ref=rr, ref_within_tol=ref_tol_ok, ok=ok)
    results.append(rec)
    flag = 'OK ' if ok else 'BAD'
    if not ref_tol_ok:
        flag += ' REFTOL'
    print(f"{flag} {tag:44s} cand-vs-eager out {ec[0]:.1e} dq {ec[1]:.1e} dk {ec[2]:.1e} dv {ec[3]:.1e} | "
          f"ref-vs-eager {er[0]:.1e} {er[1]:.1e} {er[2]:.1e} {er[3]:.1e} | cand-vs-ref max {max(ecr):.1e} | "
          f"resid cand {rc:.1e} ref {rr:.1e}", flush=True)
    del c, r, e
    torch.cuda.empty_cache()


def random_part():
    torch.manual_seed(0)
    D, H = 64, 4
    maps = [('w_q3_d2', 3.0, 2.0), ('w_q4_d2', 4.0, 2.0), ('w_q3_d1.5', 3.0, 1.5), ('dense_q4', 4.0, None), ('w_q2.5_d2', 2.5, 2.0)]
    Ns = (64, 300) if args.quick else (64, 300, 2048, 4096)
    for dtype in (torch.float32, torch.bfloat16):
        for name, sq, window in maps:
            # with NAPE slopes for every map; without slopes for the main windowed row and the dense row
            for use_slopes in ((True, False) if name in ('w_q3_d2', 'dense_q4') else (True,)):
                for N in Ns:
                    B = 2 if N <= 300 else 1
                    q = (torch.randn(B, H, N, D, device=dev) * 3).to(dtype)     # logit std ~3: the window truncates
                    k = torch.randn(B, H, N, D, device=dev).to(dtype)
                    v = torch.randn(B, H, N, D, device=dev).to(dtype)
                    slopes = nape_slopes(H, 'exp').to(dev) if use_slopes else None
                    # dense rows need the full 30 sweeps from the left bracket; windowed rows run at the LM setting
                    # (plus 30 sweeps once, on the main row; every (q, NUM_ITER, dtype, slopes) is a Triton compile)
                    iters = (30,) if window is None else ((args.num_iter, 30) if (N == 300 and name == 'w_q3_d2' and use_slopes) else (args.num_iter,))
                    for ni in iters:
                        check(f"rand {name} {'nape' if use_slopes else 'nopos'} {str(dtype)[6:]} N={N} it={ni}",
                              q, k, v, sq, window, ni, slopes, chunk=H if N <= 2048 else 1)


def capture(ckpt_path, tokens):
    """Per layer (q, k, v) of a trained model on `tokens` (B, N): q after the AS scale (capture_qk), v from c_attn."""
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    a = ck['model_args']
    m = GPT(GPTConfig(**a))
    m.load_state_dict({kk.removeprefix('_orig_mod.'): vv for kk, vv in ck['model'].items()})
    m.to(dev).eval()
    vs = {}
    for li, b in enumerate(m.transformer.h):
        b.attn.capture_qk = True
        b.attn.c_attn.register_forward_hook(lambda mod, inp, out, li=li: vs.__setitem__(li, out.detach()))
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        m(tokens, return_all_logits=True)
    B, N = tokens.shape
    H, C = a['n_head'], a['n_embd']
    out = []
    for li, b in enumerate(m.transformer.h):
        q, k = b.attn.last_qk
        v = vs[li].split(C, dim=2)[2].view(B, N, H, C // H).transpose(1, 2)
        out.append((q.contiguous(), k.contiguous(), v.contiguous()))
    slopes = m.transformer.h[0].attn.nape_slopes.to(dev) if a['pos_emb'] == 'nape' else None
    window = None if a['stieltjes_window'] <= 0 else float(a['stieltjes_window'])
    del m, ck
    torch.cuda.empty_cache()
    return out, float(a['stieltjes_q']), window, slopes


def real_part():
    torch.manual_seed(0)
    fw = np.memmap(args.fineweb_val, dtype=np.uint16, mode='r')
    ax = np.memmap(args.arxiv, dtype=np.uint16, mode='r')
    tok_fw = torch.from_numpy(np.stack([fw[i * 2049: i * 2049 + 2048].astype(np.int64) for i in range(2)])).to(dev)
    tok_ax = torch.from_numpy(ax[:16384].astype(np.int64))[None].to(dev)
    for path in [x for x in args.ckpts.split(',') if x]:
        row = os.path.basename(os.path.dirname(os.path.dirname(path)))
        qkv, sq, window, slopes = capture(path, tok_fw)
        for li, (q, k, v) in enumerate(qkv):
            check(f"real {row} fineweb N=2048 L{li} bf16", q, k, v, sq, window, args.num_iter, slopes, chunk=6, gate_resid=True)
            check(f"real {row} fineweb N=2048 L{li} fp32", q.float(), k.float(), v.float(), sq, window, args.num_iter, slopes,
                  chunk=6, gate_resid=True)
        del qkv
        qkv, sq, window, slopes = capture(path, tok_ax)
        H = qkv[0][0].shape[1]
        heads = list(range(H)) if args.heads16k == 'all' else [int(x) for x in args.heads16k.split(',')]
        for li, (q, k, v) in enumerate(qkv):
            check(f"real {row} arxiv N=16384 L{li} bf16", q, k, v, sq, window, args.num_iter, slopes, heads=heads, chunk=1,
                  gate_resid=True)
        del qkv


if args.part in ('random', 'all'):
    random_part()
if args.part in ('real', 'all'):
    real_part()

bad = [r for r in results if not r['ok']]
reftol = [r for r in results if not r['ref_within_tol']]
summary = dict(cases=len(results), failed=len(bad), ref_outside_tol=len(reftol),
               max_err_cand=[max(r['err_cand'][i] for r in results) for i in range(4)],
               max_err_ref=[max(r['err_ref'][i] for r in results) for i in range(4)],
               max_ratio=max(c / max(e, 1e-30) for r in results for c, e in zip(r['err_cand'], r['err_ref'])),
               max_cand_vs_ref=max(max(r['err_cand_vs_ref']) for r in results),
               max_resid_cand=max(r['resid_cand'] for r in results), max_resid_ref=max(r['resid_ref'] for r in results),
               device=torch.cuda.get_device_name(), tiles_bf16=list(cand._pick_blocks(64, 2, torch.device('cuda'))),
               tiles_fp32=list(cand._pick_blocks(64, 4, torch.device('cuda'))))
print('SUMMARY', json.dumps(summary))
for r in reftol:
    print('REFTOL (frozen baseline outside the absolute tolerance):', r['tag'], ['%.1e' % x for x in r['err_ref']])
if args.out:
    with open(args.out, 'w') as f:
        json.dump(dict(summary=summary, cases=results), f, indent=1)
print('GATE PASS' if not bad else f'GATE FAIL ({len(bad)} cases)')
sys.exit(0 if not bad else 1)
