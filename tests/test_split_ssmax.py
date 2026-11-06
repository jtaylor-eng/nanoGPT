"""Regression for the ssmax row and the _a<N> ALiBi/NoPE split (direction pe-ssmax-headsplit).
    python tests/test_split_ssmax.py [base_rev]      (base_rev default c712396 = before the change)
1. Existing rows: a model built by this model.py and by model.py at base_rev from the same seed has identical
   parameters and bit-identical outputs (CPU eager; CUDA training path too if available).
2. 'exp' == 'exp_a6' at n_head 12 (the default split is 6/6); 'exp_a2' has 2 nonzero slopes.
3. ssmax: NoPE-head scale equals s_h * log(pos+2); ALiBi heads untouched; s is not weight-decayed."""
import os, sys, math, subprocess, types
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch
import model as new

base = sys.argv[1] if len(sys.argv) > 1 else 'c712396'
src = subprocess.check_output(['git', '-C', REPO, 'show', f'{base}:model.py'], text=True)
old = types.ModuleType('model_old'); old.__file__ = os.path.join(REPO, 'model_old.py'); exec(compile(src, 'model_old', 'exec'), old.__dict__)

ROWS = {
    'softmax_xs':          dict(attn_type='softmax'),
    'asentmax_zi_xs':      dict(attn_type='entmax', entmax_alpha=1.5, attn_scale='as', attn_scale_zero_init=0.05),
    'wstieltjes_q3_d2_xs': dict(attn_type='stieltjes', stieltjes_q=3.0, stieltjes_window=2.0, stieltjes_num_iter=15),
    'aswstieltjes_q2_d4_zi': dict(attn_type='stieltjes', stieltjes_q=2.0, stieltjes_window=4.0, attn_scale='as', attn_scale_zero_init=0.05),
}
ok = True
def report(good, msg):
    global ok; ok &= bool(good); print(('OK  ' if good else 'BAD ') + msg)

devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
for name, kw in ROWS.items():
    for dev in devs:
        if kw['attn_type'] == 'entmax' and dev == 'cpu' and new._entmax_bisect is None:
            print(f'skip {name} cpu (no pip entmax)'); continue
        impl = 'eager' if dev == 'cpu' else 'triton'
        mk = lambda m: m.GPT(m.GPTConfig(n_layer=2, n_head=12, n_embd=192, block_size=128, vocab_size=256, bias=False,
                                         pos_emb='nape', nape_slope_type='exp', stieltjes_impl=impl, **kw))
        torch.manual_seed(0); a = mk(old).to(dev)
        torch.manual_seed(0); b = mk(new).to(dev)
        sa, sb = a.state_dict(), b.state_dict()
        same_p = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        idx = torch.randint(0, 256, (2, 96), device=dev)
        with torch.no_grad():
            la, _ = a(idx, return_all_logits=True); lb, _ = b(idx, return_all_logits=True)
        report(same_p and torch.equal(la, lb), f'{name:24s} {dev}: params equal {same_p}, logits max|diff| {(la - lb).abs().max().item():.1e}')

report(torch.equal(new.nape_slopes(12, 'exp'), new.nape_slopes(12, 'exp_a6')), "exp == exp_a6 at 12 heads")
report(torch.equal(new.nape_slopes(12, 'exp'), old.nape_slopes(12, 'exp')), "exp slopes unchanged vs base")
s2 = new.nape_slopes(12, 'exp_a2'); report((s2 > 0).sum() == 2 and torch.allclose(s2[:2], torch.tensor([2 ** -4, 2 ** -8])), f"exp_a2 slopes {s2[:3].tolist()}")
report((new.nape_slopes(12, 'exp_a0') == 0).all(), "exp_a0 all NoPE")

cfg = new.GPTConfig(n_layer=1, n_head=12, n_embd=192, block_size=64, vocab_size=256, pos_emb='nape', nape_slope_type='exp_a4', attn_scale='ssmax')
att = new.CausalSelfAttention(cfg)
report(att.n_scaled == 8 and att.ssmax_s.shape == (8,), f"ssmax a4: n_scaled {att.n_scaled}")
q = torch.randn(1, 12, 10, 16)
qs = att._apply_scale(q, None)
ref = q[:, 4:] * att.ssmax_s.view(1, 8, 1, 1) * torch.log(torch.arange(2, 12.)).view(1, 1, 10, 1)
report(torch.equal(qs[:, :4], q[:, :4]) and torch.allclose(qs[:, 4:], ref), "ssmax scale = s_h log(pos+2) on NoPE heads only")
g = new.GPT(new.GPTConfig(n_layer=1, n_head=12, n_embd=192, block_size=64, vocab_size=256, pos_emb='nape', nape_slope_type='exp', attn_scale='ssmax'))
opt = g.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
nod = [p for grp in opt.param_groups if grp['weight_decay'] == 0 for p in grp['params']]
report(any(p is g.transformer.h[0].attn.ssmax_s for p in nod), "ssmax_s in the no-decay group")
print('ALL OK' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
