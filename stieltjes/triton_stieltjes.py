# Direction stieltjes-kernel-speed CANDIDATE. Base: jtaylor-eng/asentmax@c7093f0 synthetic/src/kernels/adasplash/triton_stieltjes.py (frozen copy: triton_stieltjes_ref.py). Diverges: e002 Newton loop is a runtime loop (tl.range), not unrolled (tl.static_range); e004 windowed forward skips key tiles outside every row's support (SKIP); e005 the three backward kernels skip the tiles the forward flagged dead at the final λ (SKIP in the backward kernels, FLAGS_OUT in the forward); e006 a block leaves the Newton loop once every row's λ repeats with period 1 or 2 (CYC), the remaining sweeps being determined; e007 no copies around the kernels: q/k/v/dO are read through their strides as the caller's views and O, dQ, dK, dV are written in the caller's memory order (ZERO_COPY).
"""
Fused (flash-style) Stieltjes attention, dense or windowed. Triton.

    O = P V,   P_ij = w_ij / S_i,   w_ij = [ (λ_i - s_ij)^{-q} - c ]_+ ,   c = d^{-q}
    s_ij = scale * q_i.k_j + slope_h * min(j - i, 0)   (NAPE/ALiBi bias, optional)
    λ_i solved per query row so that Σ_j w_ij = 1  (S_i = 1 up to solver residual)

c = 0 is the dense map (window d = inf). Same semantics as
``attention/stieltjes_eager.stieltjes_normalize`` (the O(N^2) reference).

Forward = 3 kinds of sweeps over K (never materialises N x N):
    1. row max            (1 sweep)
    2. Newton for λ       (<= NUM_ITER sweeps, one runtime loop)  f(λ) = Σ w - 1 is convex, decreasing;
                                             started left of the root (λ = (1+c)^{-1/q}
                                             after centring) Newton is monotone.
    3. P V, S             (1 sweep)
Windowed map only (SKIP): a (query block, key tile) whose every row is outside the window by a margin is flagged dead
in Live and skipped by later sweeps; a skipped tile would have added exact zeros to f, f', S and O. The flag is
trusted in a sweep only while no row's λ is more than μ below its running maximum (λ rises monotonically except for
rounding), otherwise the block sweeps every tile, so the result does not depend on the flags.
CYC: the Newton update is a deterministic function of a row's λ (a skipped tile adds exact zeros), so once every row
of a block has λ_t == λ_{t-1} or λ_t == λ_{t-2} the remaining sweeps only alternate between λ_t and λ_{t-1}: the block
stops sweeping and takes the one the full NUM_ITER sweeps would have ended on. Bit-identical by construction; a NaN λ
never compares equal, so it runs every sweep as before.
With FLAGS_OUT the output sweep leaves in Live the flags at the final λ (a tile it processed is re-flagged from the
final λ with the 2μ margin; a tile it skipped was trusted, so it is outside the window at the final λ by >= μ). The
backward kernels use the saved λ, so a tile flagged dead has r = w = 0 on every row and adds exact zeros to δ, dQ, dK,
dV: they skip it. (As in the forward, a skipped tile no longer turns a non-finite dO / V into NaN on rows that give it
zero weight; every row still has a live tile, so a non-finite input still reaches every row's gradient.)
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
    Live, NB, NT, dthr2, mu,
    sq: tl.constexpr, NUM_ITER: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr, SKIP: tl.constexpr, FLAGS_OUT: tl.constexpr, CYC: tl.constexpr,
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
    # SKIP: Live[off_hz, start_m, tile] (uint8, 1 at launch) is set to 0 once every valid row of the tile has
    # λ - max_j s_ij >= dthr2 = d (1 + 2e-3), i.e. is outside the window (w > 0 <=> λ - s < d) by 2μ. A flag set at
    # sweep t0 holds at sweep t if λ_t >= λ_t0 - μ on every row, which `trust` checks against the running maximum.
    # The remaining μ = 1e-3 d covers rounding and FMA differences in the w test by a factor > 1e3.
    live_base = Live + (off_hz * NB + start_m) * NT
    row_ok = offs_m < N_CTX
    lam_hw = lam
    # runtime loop: ONE copy of the sweep body. tl.static_range unrolled NUM_ITER copies into the kernel and the
    # cost per sweep grew with NUM_ITER (A100, job 8603935: 0.5 / 1.3 / 2.0 ms per sweep at NUM_ITER 5 / 15 / 30).
    # CYC: λ_{t+1} = G(λ_t) row by row, bit for bit (skipped tiles add exact zeros), so if after sweep t every valid row
    # has λ_t == λ_{t-1} (fixed point) or λ_t == λ_{t-2} (2-cycle), the sweeps still to run would only alternate between
    # λ_t and λ_{t-1}: stop, and keep λ_t if an even number of sweeps remains, λ_{t-1} if odd (the same value on a
    # fixed-point row). The flags stay valid for the output sweep: `trust` there is tested against lam_hw as before.
    # `+ 0.0`: a value of its own. A name that only aliases `lam` here and is re-bound by plain assignment in the loop is
    # not carried by triton 3.6 (it keeps its initial value: bench_kernel.py aliascheck); `lam_hw = lam` above is updated
    # through a new op and is carried on both versions (same check).
    lam_p1 = lam + 0.0           # λ two sweeps back at the test (λ_0 before the first sweep: the test is then λ_1 == λ_0)
    active = N_CTX > 0           # runtime scalar, true
    for _it in tl.range(0, NUM_ITER):
        if active:
            f = tl.zeros([BLOCK_M], dtype=tl.float32)
            fp = tl.zeros([BLOCK_M], dtype=tl.float32)
            if SKIP:
                # NaN / inf λ compare false: no trust, every tile is swept and propagates as in the unskipped kernel
                trust = tl.min((((lam_hw - lam) <= mu) | (offs_m >= N_CTX)).to(tl.int32), axis=0)
                lam_hw = tl.maximum(lam_hw, lam)
            for start_n in tl.range(0, hi, BLOCK_N):
                if SKIP:
                    go = (tl.load(live_base + start_n // BLOCK_N) != 0) | (trust == 0)
                else:
                    go = True
                if go:
                    offs_n = start_n + tl.arange(0, BLOCK_N)
                    k_block = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
                    qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
                    w, r = _weights(qk, lam, c, sq, EPS)
                    f += tl.sum(w, axis=1)
                    fp += tl.sum(r, axis=1)
                    if SKIP:
                        alive = tl.max((((lam - tl.max(qk, axis=1)) < dthr2) & row_ok).to(tl.int32), axis=0)
                        tl.store(live_base + start_n // BLOCK_N, alive.to(tl.uint8))
            lam_new = lam + (f - 1.0) / (sq * tl.maximum(fp, 1e-30))   # f' = -q Σ r
            if CYC:
                cyc = tl.min((((lam_new == lam) | (lam_new == lam_p1)) | (offs_m >= N_CTX)).to(tl.int32), axis=0)
                odd_left = ((NUM_ITER - 1 - _it) % 2) == 1
                lam_p1 = lam
                lam = tl.where((cyc != 0) & odd_left, lam, lam_new)
                active = cyc == 0
            else:
                lam = lam_new

    # 3) O = Σ w v / S
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    s_row = tl.zeros([BLOCK_M], dtype=tl.float32)
    if SKIP:
        trust = tl.min((((lam_hw - lam) <= mu) | (offs_m >= N_CTX)).to(tl.int32), axis=0)
    for start_n in tl.range(0, hi, BLOCK_N):
        if SKIP:
            go = (tl.load(live_base + start_n // BLOCK_N) != 0) | (trust == 0)
        else:
            go = True
        if go:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_block = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < N_CTX, other=0.0)
            v_block = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, mask=offs_n[:, None] < N_CTX, other=0.0)
            qk = _scores(q_block, k_block, offs_m, offs_n, sm_scale, alibi_slope, N_CTX, CAUSAL, USE_ALIBI)
            w, _r = _weights(qk, lam, c, sq, EPS)
            s_row += tl.sum(w, axis=1)
            acc += tl.dot(w.to(v_block.dtype), v_block, input_precision="ieee")
            if FLAGS_OUT:
                # for the backward: the flag at the FINAL λ (same 2μ margin; a skipped tile keeps a flag that holds by μ)
                alive = tl.max((((lam - tl.max(qk, axis=1)) < dthr2) & row_ok).to(tl.int32), axis=0)
                tl.store(live_base + start_n // BLOCK_N, alive.to(tl.uint8))
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
    Live, NB, NT,
    sq: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr, SKIP: tl.constexpr,
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
    live_base = Live + (off_hz * NB + start_m) * NT      # SKIP: the forward's tile flags at the final λ
    for start_n in tl.range(0, hi, BLOCK_N):
        if SKIP:
            go = tl.load(live_base + start_n // BLOCK_N) != 0
        else:
            go = True
        if go:
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
    Live, NB, NT,
    sq: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr, SKIP: tl.constexpr,
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
    live_base = Live + off_hz * NB * NT + start_n        # SKIP: the forward's tile flags at the final λ, this key tile
    for start_m in tl.range(lo, N_CTX, BLOCK_M):
        if SKIP:
            # the Q blocks here start at lo = start_n * BLOCK_N, which need not be a multiple of BLOCK_M: rows
            # start_m .. start_m + BLOCK_M - 1 lie in the forward's blocks b0 and b1 (b1 past the end = padding rows)
            b0 = start_m // BLOCK_M
            b1 = tl.minimum((start_m + BLOCK_M - 1) // BLOCK_M, NB - 1)
            go = (tl.load(live_base + b0 * NT) != 0) | (tl.load(live_base + b1 * NT) != 0)
        else:
            go = True
        if go:
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
    Live, NB, NT,
    sq: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr, SKIP: tl.constexpr,
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
    live_base = Live + (off_hz * NB + start_m) * NT      # SKIP: the forward's tile flags at the final λ
    for start_n in tl.range(0, hi, BLOCK_N):
        if SKIP:
            go = tl.load(live_base + start_n // BLOCK_N) != 0
        else:
            go = True
        if go:
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


SKIP_TILES = True       # windowed forward: skip key tiles outside every row's support (False = the e002 kernel)
SKIP_MU_REL = 1e-3      # μ / d. A negative value never trusts a flag (every tile swept): the test of the fallback path
SKIP_BWD = True         # backward kernels skip the tiles flagged dead at the final λ (False = the e004 kernel)
EXIT_CYCLE = True       # a block leaves the Newton loop once every row's λ has period 1 or 2 (False = the e005 kernel)
ZERO_COPY = True        # outputs in the caller's memory order, dO read through its strides (False = the e006 wrapper)


def _like(t):
    """Uninitialised kernel output shaped like t (B, H, N, D). ZERO_COPY: if t is a heads-inner view (the model's q/k/v:
    (B, N, H, D) memory seen through transpose(1, 2)) the output gets the same memory order, so the caller's
    transpose(1, 2).contiguous() on O and autograd's reshape of dQ, dK, dV are no copies. The kernels address every
    tensor through its strides, so the values do not depend on the layout."""
    if ZERO_COPY and t.stride(1) < t.stride(2):
        B, H, N, D = t.shape
        return torch.empty((B, N, H, D), device=t.device, dtype=t.dtype).transpose(1, 2)
    return torch.empty_like(t)


def _forward(q, k, v, causal, sm_scale, sq, num_iter, c, alibi_slopes, return_live=False):
    """Launch the forward kernel. Returns (o, λ, S, slopes, BLOCK_M, BLOCK_N) [+ the tile flags if return_live]."""
    B, H, N, D = q.shape
    assert k.shape == v.shape == (B, H, N, D)
    assert D in {16, 32, 64, 128, 256}
    if B * H > 65535:
        raise ValueError(f"B*H = {B * H} exceeds the CUDA grid limit; split the batch")
    o = _like(v)
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
    NB, NT = triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N)
    skip = bool(SKIP_TILES and c > 0.0 and num_iter > 0)     # dense: every key is in the support, nothing to skip
    if skip:
        live = torch.ones((B * H, NB, NT), device=q.device, dtype=torch.uint8)
        d = c ** (-1.0 / sq)
        mu, dthr2 = SKIP_MU_REL * d, d * (1.0 + 2.0 * abs(SKIP_MU_REL))
    else:
        live, mu, dthr2 = lam, 0.0, 0.0                      # never read (constexpr-eliminated)
    _fwd_kernel[grid](
        q, k, v, o, lam, S, alibi_slopes,
        *_strides(q), *_strides(k), *_strides(v), *_strides(o),
        sm_scale, c, (1.0 + c) ** (-1.0 / sq), N, H,
        live, NB, NT, dthr2, mu,
        sq=sq, NUM_ITER=num_iter, EPS=1e-6, HEAD_DIM=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal, USE_ALIBI=use_alibi, SKIP=skip,
        FLAGS_OUT=bool(skip and SKIP_BWD), CYC=bool(EXIT_CYCLE and num_iter > 1),
    )
    if return_live:
        return o, lam, S, alibi_slopes, BLOCK_M, BLOCK_N, (live if skip else None)
    return o, lam, S, alibi_slopes, BLOCK_M, BLOCK_N


class StieltjesAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, sq, num_iter, c, alibi_slopes):
        o, lam, S, slopes, BLOCK_M, BLOCK_N, live = _forward(q, k, v, causal, sm_scale, sq, num_iter, c, alibi_slopes,
                                                             return_live=True)
        ctx.save_for_backward(q, k, v, lam, S, slopes)
        ctx.live = live if SKIP_BWD else None     # tile flags at the final λ (None: dense map, num_iter 0, or switched off)
        ctx.meta = (causal, sm_scale, sq, c, alibi_slopes is not None, BLOCK_M, BLOCK_N)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, lam, S, alibi_slopes = ctx.saved_tensors
        causal, sm_scale, sq, c, use_alibi, BLOCK_M, BLOCK_N = ctx.meta
        B, H, N, D = q.shape
        if not ZERO_COPY:
            do = do.contiguous()                    # e006 wrapper; the kernels read dO through its strides
        live = ctx.live
        bskip = live is not None
        if not bskip:
            live = lam                              # never read (constexpr-eliminated)
        NB, NT = triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N)
        common = dict(sq=sq, EPS=1e-6, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                      CAUSAL=causal, USE_ALIBI=use_alibi, SKIP=bskip)
        grid_m = (triton.cdiv(N, BLOCK_M), B * H)
        grid_n = (triton.cdiv(N, BLOCK_N), B * H)

        delta = torch.empty_like(lam)
        _bwd_delta_kernel[grid_m](
            q, k, v, do, lam, delta, alibi_slopes,
            *_strides(q), *_strides(k), *_strides(v), *_strides(do),
            sm_scale, c, N, H, live, NB, NT, **common)
        coef = (sq / S.clamp(min=1e-6)).contiguous()
        dk, dv, dq = _like(k), _like(v), _like(q)
        _bwd_dkdv_kernel[grid_n](
            q, k, v, do, lam, coef, delta, alibi_slopes, dk, dv,
            *_strides(q), *_strides(k), *_strides(v), *_strides(do), *_strides(dk), *_strides(dv),
            sm_scale, c, N, H, live, NB, NT, **common)
        _bwd_dq_kernel[grid_m](
            q, k, v, do, lam, coef, delta, alibi_slopes, dq,
            *_strides(q), *_strides(k), *_strides(v), *_strides(do), *_strides(dq),
            sm_scale, c, N, H, live, NB, NT, **common)
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
