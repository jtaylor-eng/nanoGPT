"""ssmax and adatemp rows (lm350m-qdf extension 2026-11-15; ssmax ported from direction/pe-ssmax-headsplit 2bf7b37,
tests/test_split_ssmax.py without the _a<N> head split).
    python tests/test_ssmax_adatemp.py [base_rev]      (base_rev default 2240e48 = before the change)
1. Existing rows: this model.py and model.py at base_rev give identical parameters and bit-identical logits.
2. ssmax: NoPE-head scale equals s_h * log(pos+2); ALiBi heads untouched; s is not weight-decayed.
3. adatemp: (a) beta(H) against the paper's numbers; (b) the materialised function equals an independent port of the
   paper's JAX code; (c) streamed row entropy and its analytic backward equal autograd on the materialised formula,
   over several chunks; (d) ALiBi heads are plain softmax, NoPE heads are sharpened; (e) whole model: training path
   (streamed entropy -> scaled queries -> SDPA) equals the per-head materialised path, logits and parameter grads."""
import os, sys, math, subprocess, types
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch
import torch.nn.functional as F
import model as new

base = sys.argv[1] if len(sys.argv) > 1 else '2240e48'
src = subprocess.check_output(['git', '-C', REPO, 'show', f'{base}:model.py'], text=True)
old = types.ModuleType('model_old'); old.__file__ = os.path.join(REPO, 'model_old.py'); exec(compile(src, 'model_old', 'exec'), old.__dict__)

ok = True
def report(good, msg):
    global ok; ok &= bool(good); print(('OK  ' if good else 'BAD ') + msg)

# 1. existing rows unchanged
ROWS = {
    'softmax_xs':          dict(attn_type='softmax'),
    'asentmax_zi_xs':      dict(attn_type='entmax', entmax_alpha=1.5, attn_scale='as', attn_scale_zero_init=0.05),
    'wstieltjes_q3_d2_xs': dict(attn_type='stieltjes', stieltjes_q=3.0, stieltjes_window=2.0, stieltjes_num_iter=15),
    'aswstieltjes_q3_d2_zi_xs': dict(attn_type='stieltjes', stieltjes_q=3.0, stieltjes_window=2.0, attn_scale='as', attn_scale_zero_init=0.05),
}
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

# 2. ssmax
cfg = new.GPTConfig(n_layer=1, n_head=12, n_embd=192, block_size=64, vocab_size=256, pos_emb='nape', nape_slope_type='exp', attn_scale='ssmax')
att = new.CausalSelfAttention(cfg)
report(att.n_scaled == 6 and att.ssmax_s.shape == (6,), f"ssmax: n_scaled {att.n_scaled}")
q = torch.randn(1, 12, 10, 16)
qs = att._apply_scale(q, None)
ref = q[:, 6:] * att.ssmax_s.view(1, 6, 1, 1) * torch.log(torch.arange(2, 12.)).view(1, 1, 10, 1)
report(torch.equal(qs[:, :6], q[:, :6]) and torch.allclose(qs[:, 6:], ref), "ssmax scale = s_h log(pos+2) on NoPE heads only")
g = new.GPT(cfg)
opt = g.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
nod = [p for grp in opt.param_groups if grp['weight_decay'] == 0 for p in grp['params']]
report(any(p is g.transformer.h[0].attn.ssmax_s for p in nod), "ssmax_s in the no-decay group")

# 3a. beta(H): hand-evaluated polynomial of Fig. 7 / the theta formula in the Fig. 6 caption
H = torch.tensor([0.3, 0.5, 0.8, 1.0, 2.0, 4.0, 5.0, 6.0, 7.6])
want = torch.tensor([1.0, 1.0, 1.0, 1.27, 2.099, 2.389, 2.294, 1.0, 1.0])
report(torch.allclose(new.adatemp_beta(H), want, atol=2e-3), f"adatemp beta(H) = {[round(v, 3) for v in new.adatemp_beta(H).tolist()]}")

# 3b. independent port of the paper's JAX function (finfo.min mask, as CLRS_Eval/clrs_eval/attn.py)
def paper_fn(logits):
    probs = torch.softmax(logits, dim=-1)
    poly = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791])
    entropy = torch.sum(-probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)
    pv = sum(c * entropy ** (4 - i) for i, c in enumerate(poly))
    beta = torch.where(entropy > 0.5, torch.maximum(pv, torch.ones(())), torch.ones(()))
    return torch.softmax(torch.clamp(logits * beta, min=torch.finfo(logits.dtype).min), dim=-1), entropy, beta
torch.manual_seed(1)
T = 200
causal = torch.ones(T, T, dtype=torch.bool).tril()
s = torch.randn(2, 3, T, T) * torch.tensor([0.3, 2.0, 8.0]).view(1, 3, 1, 1)
p_ref, ent_ref, beta_ref = paper_fn(s.masked_fill(~causal, torch.finfo(torch.float32).min))
p_new = new.adaptive_temperature_softmax(s, causal)
frac = (beta_ref > 1).float().mean().item()
report(torch.allclose(p_new, p_ref, atol=1e-6) and 0.2 < frac < 0.95, f"adatemp materialised == paper port, max|diff| {(p_new - p_ref).abs().max().item():.1e}, beta>1 on {frac:.0%} of rows")

# 3c. streamed entropy + analytic backward, several chunks
B, Hh, T, D = 2, 3, 150, 16
qq = (torch.randn(B, Hh, T, D) * 1.5).requires_grad_(True); kk = (torch.randn(B, Hh, T, D) * 1.5).requires_grad_(True)
sc = 1 / math.sqrt(D)
keep = new.ADATEMP_CHUNK
new.ADATEMP_CHUNK = B * Hh * T * 37          # 37 query rows per chunk -> 5 chunks
ent = new._row_entropy(qq, kk, sc)
w = torch.randn_like(ent)
gq, gk = torch.autograd.grad((ent * w).sum(), (qq, kk))
new.ADATEMP_CHUNK = keep
causal = torch.ones(T, T, dtype=torch.bool).tril()
pm = F.softmax(((qq @ kk.transpose(-2, -1)) * sc).masked_fill(~causal, float('-inf')), dim=-1)
ent_m = -(pm * torch.log(pm + 1e-9)).sum(-1)
gq_m, gk_m = torch.autograd.grad((ent_m * w).sum(), (qq, kk))
rel = lambda u, v: ((u - v).norm() / v.norm()).item()
report(torch.allclose(ent, ent_m, atol=1e-5) and rel(gq, gq_m) < 1e-4 and rel(gk, gk_m) < 1e-4,
       f"streamed entropy == materialised (max|diff| {(ent - ent_m).abs().max().item():.1e}); grad rel err q {rel(gq, gq_m):.1e} k {rel(gk, gk_m):.1e}")

# 3d / 3e. whole model, training path vs per-head materialised path
for dev in devs:
    torch.manual_seed(2)
    cfg = new.GPTConfig(n_layer=2, n_head=12, n_embd=192, block_size=128, vocab_size=256, bias=False, pos_emb='nape',
                        nape_slope_type='exp', attn_scale='adatemp')
    m = new.GPT(cfg).to(dev)
    with torch.no_grad():   # default init gives near-uniform rows (H > 5.95 at long T is beta = 1): sharpen the scores
        for blk in m.transformer.h:
            blk.attn.c_attn.weight.mul_(12.0)
    idx = torch.randint(0, 256, (2, 120), device=dev); tgt = torch.randint(0, 256, (2, 120), device=dev)
    def run(eager):
        for blk in m.transformer.h:
            blk.attn.force_per_head = eager
        m.zero_grad()
        logits, loss = m(idx, tgt); loss.backward()
        return logits.detach(), torch.cat([p.grad.flatten() for p in m.parameters()])
    lt, gt = run(False); le, ge = run(True)
    att = m.transformer.h[0].attn
    x = m.transformer.wte(idx); x = m.transformer.h[0].ln_1(x)
    q, k, v = att.c_attn(x).split(192, dim=2)
    q = q.view(2, 120, 12, 16).transpose(1, 2); k = k.view(2, 120, 12, 16).transpose(1, 2)
    with torch.no_grad():
        beta = new.adatemp_beta(new._row_entropy(q[:, 6:], k[:, 6:], 0.25))
        qs = att._apply_scale(q, None, k)
        p_alibi = att._probs_head(q[:, 0:1], k[:, 0:1], 0)
        s0 = (q[:, 0:1] @ k[:, 0:1].transpose(-2, -1)) * 0.25 + new.nape_bias(att.nape_slopes[0:1], 120, q.device, torch.float32)
        p_plain = F.softmax(s0.masked_fill(~torch.ones(120, 120, dtype=torch.bool, device=dev).tril(), float('-inf')), dim=-1)
    frac = (beta > 1).float().mean().item()
    report(torch.equal(qs[:, :6], q[:, :6]) and torch.allclose(p_alibi, p_plain, atol=1e-6) and frac > 0.1,
           f"adatemp {dev}: ALiBi heads plain softmax, NoPE beta>1 on {frac:.0%} of rows (max {beta.max().item():.2f})")
    report((lt - le).abs().max().item() < 2e-3 and rel(gt, ge) < 2e-3,
           f"adatemp {dev}: training path == materialised path, logits max|diff| {(lt - le).abs().max().item():.1e}, grad rel err {rel(gt, ge):.1e}")
try:
    new.CausalSelfAttention(new.GPTConfig(attn_type='entmax', attn_scale='adatemp')); report(False, "adatemp + entmax must raise")
except ValueError:
    report(True, "adatemp with a non-softmax attn_type raises")
print('ALL OK' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
