# Vendored verbatim from jtaylor-eng/asentmax@c7093f0 synthetic/src/attention/stieltjes_eager.py -- edit there, re-copy here.
"""
Eager (dense-materialised) Stieltjes attention normalisation, dense or windowed.

    w_j = [ (λ - s_j)^{-q} - c ]_+ ,   c = d^{-q}   (window d; c = 0 is the dense map)
    λ solved per row so that Σ_j w_j = 1;  p = w / Σ w  (the division only removes
    the solver residual).

Windowing (context/stieltjes_proposal.tex §2): tokens more than ~d below the top
get exactly zero mass, so the map is sparse with an n-independent threshold;
d -> inf recovers the dense map bit-for-bit (max(w - 0, 0) == w).

Solver: after centring (row max = 0) the root is bracketed by
    lo = (1 + c)^{-1/q}   (top token alone already gives w_top = 1 => f(lo) >= 0)
    hi = min(d, K^{1/q})  (every term clipped at λ = d; Σ (K^{1/q} - s)^{-q} <= 1)
with K = number of finite entries. 12 bisections then safeguarded Newton; f is
convex and decreasing in λ so this is monotone for any q > 0, any N, any d.

Gradient (implicit-function theorem through Σ_A w = 1 over the support A):
    r_j   = (λ - s_j)^{-q-1} [j in A]
    δ     = Σ_j dP_j r_j / Σ_j r_j
    dL/ds_j = (q / S) r_j (dP_j - δ)
identical to the dense formula with r zeroed off the support. Masked logits
(<= -1e30 or non-finite) get zero weight and zero gradient.

Reference semantics for ``kernels/adasplash/triton_stieltjes.py``. O(N^2) fp32:
fine for training at short context and decode; not for prefill beyond ~4k.
"""
import math
import torch


def _window_c(q: float, window) -> float:
    return 0.0 if (window is None or window <= 0 or math.isinf(window)) else float(window) ** (-q)


def _solve_lambda(x, q: float, c: float, num_iter: int = 30, eps: float = 1e-6):
    """λ per row for centred logits x (row max == 0). Shape x.shape[:-1] + (1,)."""
    valid = torch.isfinite(x) & (x > -1e30)
    K = valid.sum(dim=-1, keepdim=True).clamp(min=1).to(x.dtype)
    lo = torch.full_like(K, (1.0 + c) ** (-1.0 / q))
    hi = K.pow(1.0 / q) + eps
    if c > 0:
        hi = hi.clamp(max=c ** (-1.0 / q) + eps)
    lam = 0.5 * (lo + hi)
    n_bisect = min(12, num_iter // 2)
    for it in range(num_iter):
        inv = (lam - x).clamp(min=eps).reciprocal()
        w = inv.pow(q) - c
        on = valid & (w > 0)
        f = torch.where(on, w, torch.zeros_like(w)).sum(dim=-1, keepdim=True) - 1.0
        lo = torch.where(f > 0, lam, lo)
        hi = torch.where(f <= 0, lam, hi)
        if it < n_bisect:
            lam = 0.5 * (lo + hi)
            continue
        fp = -q * torch.where(on, (w + c) * inv, torch.zeros_like(w)).sum(dim=-1, keepdim=True)
        newton = lam - f / fp.clamp(max=-1e-30)
        inside = (newton > lo) & (newton < hi)
        lam = torch.where(inside, newton, 0.5 * (lo + hi))
    return lam


# torch.compile fuses the per-iteration elementwise kernels (~7x at training shapes);
# falls back to eager if compile is unavailable.
_solve_compiled = None

def _solve(x, q, c, num_iter, eps):
    global _solve_compiled
    if x.is_cuda:
        if _solve_compiled is None:
            try:
                _solve_compiled = torch.compile(_solve_lambda, dynamic=True)
            except Exception:
                _solve_compiled = _solve_lambda
        try:
            return _solve_compiled(x, q, c, num_iter, eps)
        except Exception:
            _solve_compiled = _solve_lambda
    return _solve_lambda(x, q, c, num_iter, eps)


class _StieltjesNormalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, q: float, c: float, num_iter: int, eps: float):
        s = scores if scores.dtype == torch.float64 else scores.to(torch.float32)
        x = s - s.max(dim=-1, keepdim=True).values
        with torch.no_grad():
            lam = _solve(x, q, c, num_iter, eps)
        inv = (lam - x).clamp(min=eps).reciprocal()
        w = inv.pow(q) - c
        on = torch.isfinite(s) & (s > -1e30) & (w > 0)
        w = torch.where(on, w, torch.zeros_like(w))
        S = w.sum(dim=-1, keepdim=True).clamp(min=eps)
        r = torch.where(on, (w + c) * inv, torch.zeros_like(w))
        ctx.save_for_backward(r, S)
        ctx.q = q
        return w / S

    @staticmethod
    def backward(ctx, dP):
        r, S = ctx.saved_tensors
        dP = dP.to(r.dtype)
        D = r.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        delta = (dP * r).sum(dim=-1, keepdim=True) / D
        return (ctx.q / S) * r * (dP - delta), None, None, None, None


def stieltjes_normalize(scores, q: float = 4.0, num_iter: int = 30, window=None, eps: float = 1e-6):
    """Row-wise Stieltjes mapping over the last dim (drop-in for softmax).
    window=d gives the sparse windowed map (c = d^-q); None/0/inf is dense."""
    return _StieltjesNormalize.apply(scores, float(q), _window_c(float(q), window), int(num_iter), float(eps))


def stieltjes_normalize_qd(scores, q, d, num_iter: int = 30, eps: float = 1e-6):
    """Fork addition (diagnostics): the windowed map with q and d as 0-dim TENSORS, differentiable in both.
    λ is solved without grad, then one Newton step is taken inside autograd, so d(out)/dq and d(out)/dd are the
    exact implicit-function derivatives at the root (same trick as stieltjes_reference). d <= 0 means dense.
    fp32 (fp64 if given fp64), materialised; eval-only."""
    s = scores if scores.dtype == torch.float64 else scores.to(torch.float32)
    x = s - s.max(dim=-1, keepdim=True).values.detach()
    valid = torch.isfinite(s) & (s > -1e30)
    dense = float(d) <= 0
    c = torch.zeros((), dtype=s.dtype, device=s.device) if dense else d.to(s.dtype).pow(-q.to(s.dtype))
    with torch.no_grad():
        lam = _solve(x, float(q), float(c), num_iter, eps)

    def w_of(lam_, xx):
        w = (lam_ - xx).clamp(min=eps).pow(-q) - c
        return torch.where(valid & (w > 0), w, torch.zeros_like(w))

    w = w_of(lam, x)
    f = w.sum(-1, keepdim=True) - 1
    fp = -q * torch.where(w > 0, (w + c) / (lam - x).clamp(min=eps), torch.zeros_like(w)).sum(-1, keepdim=True)
    lam = lam - f / fp
    w = w_of(lam, x)
    return (w / w.sum(-1, keepdim=True).clamp(min=eps)).to(scores.dtype)


def stieltjes_reference(scores, q: float = 4.0, window=None, n_bisect: int = 200):
    """Slow fp64 reference for tests: 200 bisections for λ, then one differentiable
    Newton step so autograd gives the exact implicit-function gradient at the root."""
    q = float(q); c = _window_c(q, window)
    s = scores.to(torch.float64)
    x = s - s.max(dim=-1, keepdim=True).values.detach()
    valid = torch.isfinite(s) & (s > -1e30)
    K = valid.sum(-1, keepdim=True).to(torch.float64)
    lo = torch.full_like(K, (1.0 + c) ** (-1.0 / q))
    hi = K.pow(1.0 / q) + 1e-6
    if c > 0:
        hi = hi.clamp(max=c ** (-1.0 / q) + 1e-6)

    def w_of(lam, xx):
        w = (lam - xx).clamp(min=1e-12).pow(-q) - c
        return torch.where(valid & (w > 0), w, torch.zeros_like(w))

    with torch.no_grad():
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f = w_of(mid, x).sum(-1, keepdim=True) - 1
            lo = torch.where(f > 0, mid, lo); hi = torch.where(f <= 0, mid, hi)
        lam = 0.5 * (lo + hi)
    w = w_of(lam, x)
    f = w.sum(-1, keepdim=True) - 1
    fp = -q * torch.where(w > 0, (w + c) / (lam - x).clamp(min=1e-12), torch.zeros_like(w)).sum(-1, keepdim=True)
    lam = lam - f / fp
    w = w_of(lam, x)
    return (w / w.sum(-1, keepdim=True)).to(scores.dtype)
