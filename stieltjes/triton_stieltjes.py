# Vendored verbatim from jtaylor-eng/asentmax@c7093f0 synthetic/src/kernels/adasplash/triton_stieltjes.py -- edit there, re-copy here.
"""
Fused (flash-style) Stieltjes attention, dense or windowed. Triton.

    O = P V,   P_ij = w_ij / S_i,   w_ij = [ (λ_i - s_ij)^{-q} - c ]_+ ,   c = d^{-q}
    s_ij = scale * q_i.k_j + slope_h * min(j - i, 0)   (NAPE/ALiBi bias, optional)
    λ_i solved per query row so that Σ_j w_ij = 1  (S_i = 1 up to solver residual)

c = 0 is the dense map (window d = inf). Same semantics as
``attention/stieltjes_eager.stieltjes_normalize`` (the O(N^2) reference).

Forward = 3 kinds of sweeps over K (never materialises N x N):
    1. row max            (1 sweep)
    2. Newton for λ       (NUM_ITER sweeps)  f(λ) = Σ w - 1 is convex, decreasing;
                                             started left of the root (λ = (1+c)^{-1/q}
                                             after centring) Newton is monotone.
    3. P V, S             (1 sweep)
Backward (implicit-function gradient through Σ_A w = 1 over the support A):
    r_ij = (λ_i - s_ij)^{-q-1} [ij in A],  δ_i = Σ_j dP_ij r_ij / Σ_j r_ij
    dS_ij = (q / S_i) r_ij (dP_ij - δ_i);   dQ = dS K scale,  dK = dS^T Q scale,  dV = P^T dO
Three backward kernels (δ; dK,dV; dQ), all recomputing scores on the fly.
Saved per row: λ (absolute) and S.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------

@triton.jit
def _pow_pair(diff, sq: tl.constexpr):
    """(diff^-q, diff^-q-1). Integer q via multiply chains (transcendentals dominate
    the sweep cost); other q via exp/log. Masked entries (diff ~ 1e30) underflow to 0."""
    r = 1.0 / diff
    if sq == 1.0:
        inv_q = r
    elif sq == 2.0:
        inv_q = r * r
    elif sq == 3.0:
        inv_q = r * r * r
    elif sq == 4.0:
        r2 = r * r
        inv_q = r2 * r2
    elif sq == 8.0:
        r2 = r * r
        r4 = r2 * r2
        inv_q = r4 * r4
    elif sq == 16.0:
        r2 = r * r
        r4 = r2 * r2
        r8 = r4 * r4
        inv_q = r8 * r8
    else:
        inv_q = tl.exp(tl.log(diff) * (-sq))
    return inv_q, inv_q * r


@triton.jit
def _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX,
            CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr):
    """Scaled QK^T tile + ALiBi bias, with padded rows/columns and the causal
    upper triangle set to -1e30 (== sparse_gemma.make_bias_window + causal mask).
    A padded query row therefore gets w = r = 0 everywhere."""
    qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee") * sm_scale
    if USE_ALIBI:
        qk = qk + alibi_slope * tl.minimum((offs_n[None, :] - offs_m[:, None]).to(tl.float32), 0.0)
    keep = (offs_n[None, :] < N_CTX) & (offs_m[:, None] < N_CTX)
    if CAUSAL:
        keep = keep & (offs_m[:, None] >= offs_n[None, :])
    return tl.where(keep, qk, -1e30)


@triton.jit
def _weights(qk, lam_row, c, sq: tl.constexpr, EPS: tl.constexpr):
    """(w, r) for a tile: w = [(λ-s)^-q - c]_+, r = (λ-s)^-q-1 on the support, 0 off it."""
    diff = tl.maximum(lam_row[:, None] - qk, EPS)
    inv_q, inv_q1 = _pow_pair(diff, sq)
    w = inv_q - c
    on = w > 0
    return tl.where(on, w, 0.0), tl.where(on, inv_q1, 0.0)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel(
    Q, K, V, O, Lam, S, Slopes,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale, c, lam0, N_CTX, H,
    sq: tl.constexpr, NUM_ITER: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H          # H passed explicitly: stride-derived H is wrong for
    off_h = off_hz % H           # fused-qkv transpose views at B > 1
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(Slopes + off_h)
    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)

    # 1) row max
    row_max = tl.full([BLOCK_M], value=-1e30, dtype=tl.float32)
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_block = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
        qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
        row_max = tl.maximum(row_max, tl.max(qk, axis=1))

    # 2) Newton for λ (absolute), from the left of the root
    lam = row_max + lam0
    for _it in tl.static_range(NUM_ITER):
        f = tl.zeros([BLOCK_M], dtype=tl.float32)
        fp = tl.zeros([BLOCK_M], dtype=tl.float32)
        for start_n in tl.range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_block = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
            qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
            w, r = _weights(qk, lam, c, sq, EPS)
            f += tl.sum(w, axis=1)
            fp += tl.sum(r, axis=1)
        lam = lam + (f - 1.0) / (sq * tl.maximum(fp, 1e-30))   # f' = -q Σ r

    # 3) O = Σ w v / S
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    s_row = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_block = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
        v_block = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, mask=offs_n[:, None] < N_CTX, other=0.0)
        qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
        w, _r = _weights(qk, lam, c, sq, EPS)
        s_row += tl.sum(w, axis=1)
        acc += tl.dot(w.to(v_block.dtype), v_block, input_precision="ieee")
    acc = acc / tl.maximum(s_row, EPS)[:, None]

    o_ptrs = O + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(q_block.dtype), mask=offs_m[:, None] < N_CTX)
    tl.store(Lam + off_hz * N_CTX + offs_m, lam, mask=offs_m < N_CTX)
    tl.store(S + off_hz * N_CTX + offs_m, s_row, mask=offs_m < N_CTX)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
# Padded query rows (offs_m >= N_CTX) are masked to -1e30 in _scores, so every
# weight is 0 there and they contribute nothing to dK/dV.

@triton.jit
def _bwd_delta_kernel(
    Q, K, V, DO, Lam, Delta, Slopes,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    sm_scale, c, N_CTX, H,
    sq: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr,
):
    """δ_i = Σ_j dP_ij r_ij / Σ_j r_ij,  dP = dO V^T."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_block = tl.load(Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX, other=0.0)
    do_block = tl.load(DO + off_z * stride_doz + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok, mask=offs_m[:, None] < N_CTX, other=0.0)
    lam = tl.load(Lam + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(Slopes + off_h)
    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)

    num = tl.zeros([BLOCK_M], dtype=tl.float32)
    den = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_block = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
        v_block = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, mask=offs_n[:, None] < N_CTX, other=0.0)
        qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
        _w, r = _weights(qk, lam, c, sq, EPS)
        dP = tl.dot(do_block, tl.trans(v_block), input_precision="ieee")
        num += tl.sum(dP * r, axis=1)
        den += tl.sum(r, axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, num / tl.maximum(den, 1e-30), mask=offs_m < N_CTX)


@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, DO, Lam, Coef, Delta, Slopes, DK, DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    sm_scale, c, N_CTX, H,
    sq: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr,
):
    """Fixed K/V block, sweep Q blocks:  dK = Σ dS^T Q scale,  dV = Σ P^T dO."""
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    k_block = tl.load(K + off_z * stride_kz + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
    v_block = tl.load(V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, mask=offs_n[:, None] < N_CTX, other=0.0)
    q_base = Q + off_z * stride_qz + off_h * stride_qh
    do_base = DO + off_z * stride_doz + off_h * stride_doh
    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(Slopes + off_h)
    lo = 0
    if CAUSAL:
        lo = start_n * BLOCK_N

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    for start_m in tl.range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_block = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX, other=0.0)
        do_block = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok, mask=offs_m[:, None] < N_CTX, other=0.0)
        lam = tl.load(Lam + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        coef = tl.load(Coef + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)   # q / S
        delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
        w, r = _weights(qk, lam, c, sq, EPS)
        dP = tl.dot(do_block, tl.trans(v_block), input_precision="ieee")
        dS = (coef[:, None] * r * (dP - delta[:, None])).to(q_block.dtype)
        dk += tl.dot(tl.trans(dS), q_block, input_precision="ieee") * sm_scale
        p = w * (coef * (1.0 / sq))[:, None]
        dv += tl.dot(tl.trans(p.to(do_block.dtype)), do_block, input_precision="ieee")

    tl.store(DK + off_z * stride_dkz + off_h * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk, dk.to(k_block.dtype), mask=offs_n[:, None] < N_CTX)
    tl.store(DV + off_z * stride_dvz + off_h * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk, dv.to(v_block.dtype), mask=offs_n[:, None] < N_CTX)


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, DO, Lam, Coef, Delta, Slopes, DQ,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    sm_scale, c, N_CTX, H,
    sq: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr,
):
    """Fixed Q block, sweep K/V blocks:  dQ = Σ dS K scale."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_block = tl.load(Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX, other=0.0)
    do_block = tl.load(DO + off_z * stride_doz + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok, mask=offs_m[:, None] < N_CTX, other=0.0)
    lam = tl.load(Lam + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    coef = tl.load(Coef + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)[:, None]   # q / S
    delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(Slopes + off_h)
    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_block = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
        v_block = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, mask=offs_n[:, None] < N_CTX, other=0.0)
        qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
        _w, r = _weights(qk, lam, c, sq, EPS)
        dP = tl.dot(do_block, tl.trans(v_block), input_precision="ieee")
        dS = (coef * r * (dP - delta[:, None])).to(q_block.dtype)
        dq += tl.dot(dS, k_block, input_precision="ieee") * sm_scale

    tl.store(DQ + off_z * stride_dqz + off_h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk, dq.to(q_block.dtype), mask=offs_m[:, None] < N_CTX)


# ---------------------------------------------------------------------------
# autograd wrapper
# ---------------------------------------------------------------------------

def _pick_blocks(D, elem_size, device):
    """Tile sizes shared by forward and backward (they must match). The backward
    (K/V + Q/dO tiles resident) binds shared memory: fp32 D=64 at 128x64 needs
    ~200 KB (H100 ok, A100's 164 KB not); consumer GPUs (~100 KB) need 32x32.
    bf16/fp16: 64x32 measured fastest on A100 at every N in 64..2048 (job 7462343,
    2026-09-21: 128x64 was 2-4x slower than 64x32 for this kernel)."""
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        smem = triton.runtime.driver.active.utils.get_device_properties(idx)["max_shared_mem"]
    except Exception:
        smem = 1 << 20
    if smem < 140 * 1024:
        return 32, 32
    if elem_size >= 4:
        if D >= 128:
            return 32, 32
        if D >= 64 and smem < 200 * 1024:
            return 64, 64
        return 128, 64
    return 64, 32


def _strides(t):
    return t.stride(0), t.stride(1), t.stride(2), t.stride(3)


def window_c(q: float, window) -> float:
    """c = d^-q; None / 0 / inf -> 0 (dense)."""
    return 0.0 if (window is None or window <= 0 or window == float("inf")) else float(window) ** (-q)


def _forward(q, k, v, causal, sm_scale, sq, num_iter, c, alibi_slopes):
    """Launch the forward kernel. Returns (o, λ, S, slopes, BLOCK_M, BLOCK_N)."""
    B, H, N, D = q.shape
    assert k.shape == v.shape == (B, H, N, D)
    assert D in {16, 32, 64, 128, 256}
    if B * H > 65535:
        raise ValueError(f"B*H = {B * H} exceeds the CUDA grid limit; split the batch")
    o = torch.empty_like(q)
    lam = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    S = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    use_alibi = alibi_slopes is not None
    if use_alibi:
        alibi_slopes = alibi_slopes.reshape(-1).to(device=q.device, dtype=torch.float32).contiguous()
        assert alibi_slopes.shape == (H,), "alibi_slopes must have shape (H,)"
    else:
        alibi_slopes = lam   # never read (constexpr-eliminated); any valid pointer
    BLOCK_M, BLOCK_N = _pick_blocks(D, q.element_size(), q.device)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _fwd_kernel[grid](
        q, k, v, o, lam, S, alibi_slopes,
        *_strides(q), *_strides(k), *_strides(v), *_strides(o),
        sm_scale, c, (1.0 + c) ** (-1.0 / sq), N, H,
        sq=sq, NUM_ITER=num_iter, EPS=1e-6, HEAD_DIM=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal, USE_ALIBI=use_alibi,
    )
    return o, lam, S, alibi_slopes, BLOCK_M, BLOCK_N


class StieltjesAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, sq, num_iter, c, alibi_slopes):
        o, lam, S, slopes, BLOCK_M, BLOCK_N = _forward(q, k, v, causal, sm_scale, sq, num_iter, c, alibi_slopes)
        ctx.save_for_backward(q, k, v, lam, S, slopes)
        ctx.meta = (causal, sm_scale, sq, c, alibi_slopes is not None, BLOCK_M, BLOCK_N)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, lam, S, alibi_slopes = ctx.saved_tensors
        causal, sm_scale, sq, c, use_alibi, BLOCK_M, BLOCK_N = ctx.meta
        B, H, N, D = q.shape
        do = do.contiguous()
        common = dict(sq=sq, EPS=1e-6, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                      CAUSAL=causal, USE_ALIBI=use_alibi)
        grid_m = (triton.cdiv(N, BLOCK_M), B * H)
        grid_n = (triton.cdiv(N, BLOCK_N), B * H)

        delta = torch.empty_like(lam)
        _bwd_delta_kernel[grid_m](
            q, k, v, do, lam, delta, alibi_slopes,
            *_strides(q), *_strides(k), *_strides(v), *_strides(do),
            sm_scale, c, N, H, **common)
        coef = (sq / S.clamp(min=1e-6)).contiguous()
        dk, dv, dq = torch.empty_like(k), torch.empty_like(v), torch.empty_like(q)
        _bwd_dkdv_kernel[grid_n](
            q, k, v, do, lam, coef, delta, alibi_slopes, dk, dv,
            *_strides(q), *_strides(k), *_strides(v), *_strides(do), *_strides(dk), *_strides(dv),
            sm_scale, c, N, H, **common)
        _bwd_dq_kernel[grid_m](
            q, k, v, do, lam, coef, delta, alibi_slopes, dq,
            *_strides(q), *_strides(k), *_strides(v), *_strides(do), *_strides(dq),
            sm_scale, c, N, H, **common)
        return dq, dk, dv, None, None, None, None, None, None


def stieltjes_attention(q, k, v, causal=False, sm_scale=None, stieltjes_q=4.0, num_iter=30,
                        window=None, alibi_slopes=None):
    """Fused Stieltjes attention on (B, H, N, D) tensors.

    stieltjes_q: exponent q.  window: d (sparse map, c = d^-q) or None for dense.
    num_iter: Newton sweeps for λ (each is one pass over K; 30 converges to fp32
    precision for q <= 16 and N <= 64k from the left-bracket init).
    alibi_slopes: optional (H,) NAPE/ALiBi slopes; adds slope_h * min(j - i, 0) to
    every score (forward and backward recompute), as sparse_gemma.make_bias_window.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    sq = float(stieltjes_q)
    return StieltjesAttention.apply(q, k, v, causal, sm_scale, sq, int(num_iter),
                                    window_c(sq, window), alibi_slopes)


def stieltjes_solver_residual(q, k, v, causal=False, sm_scale=None, stieltjes_q=4.0, num_iter=30,
                              window=None, alibi_slopes=None):
    """Diagnostic: per-row |S - 1| (the unnormalised weight sum is 1 at the root),
    so a wrong λ cannot hide behind the final normalisation. Shape (B*H, N)."""
    sq = float(stieltjes_q)
    with torch.no_grad():
        _, _, S, _, _, _ = _forward(q, k, v, causal, sm_scale or 1.0 / (q.shape[-1] ** 0.5), sq,
                                    int(num_iter), window_c(sq, window), alibi_slopes)
    return (S - 1.0).abs()
