"""Fused-vs-eager equivalence for every attention row (fork addition). CUDA required.
    python tests/test_attn_paths.py
Checks CausalSelfAttention output and q/k/v-projection grads: the training path
(SDPA+NAPE bias for softmax, fused Triton kernel for Stieltjes) vs the materialised
eager path used by eval_lengths.py stats (collect_stats=True)."""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GPTConfig, CausalSelfAttention

ROWS = {
    'softmax':              dict(attn_type='softmax'),
    'softmax_as':           dict(attn_type='softmax', attn_scale='as', attn_scale_zero_init=0.05),
    'entmax':               dict(attn_type='entmax', entmax_alpha=1.5),
    'asentmax_zi':          dict(attn_type='entmax', entmax_alpha=1.5, attn_scale='as', attn_scale_zero_init=0.05),
    'stieltjes_q4':         dict(attn_type='stieltjes', stieltjes_q=4.0),
    'wstieltjes_q4_d2':     dict(attn_type='stieltjes', stieltjes_q=4.0, stieltjes_window=2.0),
    'aswstieltjes_q2_d4_zi': dict(attn_type='stieltjes', stieltjes_q=2.0, stieltjes_window=4.0,
                                  attn_scale='as', attn_scale_zero_init=0.05),
}
torch.manual_seed(0)
ok = True
for dtype in (torch.float32, torch.bfloat16):
    for name, kw in ROWS.items():
        for T in (64, 300):
            cfg = GPTConfig(n_layer=1, n_head=8, n_embd=512, block_size=256, bias=False, pos_emb='nape', nape_slope_type=os.environ.get('SLOPES', 'linear'), **kw)
            a = CausalSelfAttention(cfg).cuda()
            a.reset_scale_params()
            if cfg.attn_scale == 'as':   # perturb so the scale is non-trivial
                with torch.no_grad():
                    a.scale_beta.weight.normal_(0, 0.02); a.scale_gamma.weight.normal_(0, 0.02)
            b = copy.deepcopy(a); b.collect_stats = True
            x = (torch.randn(2, T, 512, device='cuda') * 3).requires_grad_(True)
            x2 = x.detach().clone().requires_grad_(True)
            with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                ya = a(x); yb = b(x2)
            g = torch.randn_like(ya)
            ya.backward(g); yb.backward(g)
            tol = 2e-3 if dtype == torch.float32 else 4e-2
            def rel(u, v): return ((u.float() - v.float()).norm() / v.float().norm().clamp_min(1e-12)).item()
            eo = rel(ya, yb); eg = rel(a.c_attn.weight.grad, b.c_attn.weight.grad); ex = rel(x.grad, x2.grad)
            good = eo < tol and eg < tol * 5 and ex < tol * 5
            ok &= good
            print(f"{'OK ' if good else 'BAD'} {str(dtype)[6:]:8s} {name:24s} T={T:3d}  out {eo:.1e}  dW {eg:.1e}  dx {ex:.1e}")
print('ALL OK' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
