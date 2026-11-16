"""Timing harness for the fused Stieltjes kernel (direction stieltjes-kernel-speed). CUDA required; A100 for numbers.

  python bench_kernel.py micro [--out=micro.json] [--small]
      Kernel microbench, candidate (stieltjes/triton_stieltjes.py) and frozen baseline (triton_stieltjes_ref.py) in one
      process: fwd kernel at NUM_ITER in {0, 5, 15, 30}, each backward kernel, bwd, fwd+bwd at the 125M training shape
      (8, 12, 2048, 64) bf16 causal + exp NAPE slopes, q3/d2; the model call site (non-contiguous q/k/v views);
      AdaSplash entmax and SDPA softmax at the same shape; forward at (1, 12, 16384, 64).
  python bench_kernel.py wrap [--kernel=cand|cand_nocache|cand_noexit|cand_nobskip|cand_noskip|ref] [--timers=0|1] [--out=x.json] -- <script.py> [script args...]
      Run an UNMODIFIED repo script (train.py, eval_lengths.py) with the chosen kernel and, with --timers=1,
      CUDA-synchronised timers around the attention call site and the autograd Function forward/backward.
      cand_noskip = the candidate with SKIP_TILES off (the e002 kernel); cand_nobskip = SKIP_BWD off (the e004 kernel);
      cand_noexit = EXIT_CYCLE off (the e005 kernel); cand_nocache = CACHE_SMEM off (the e006 wrapper).
  python bench_kernel.py summary <dir>
      Table from <dir>/micro.json, step_*.log, step_*.json.
  python bench_kernel.py sweep [--out=sweep.json] [--small]
      Launch parameters of the candidate's four kernels at the training shape: tile sizes, num_warps, num_stages, each
      variant timed and compared bit for bit with the default launch (64x32 tiles, 4 warps, 3 stages). No kernel edit:
      the kernels are launched directly with other constexpr tiles / Triton launch options. Written for the kernel
      signature at 0b3c77a (e003); the forward launch does not pass the e004 tile-flag arguments.
  python bench_kernel.py tiles --ckpts=<run>/ckpt.pt[,..] --fineweb_val=<val.bin> --arxiv=<arxiv_long.bin> [--out=tiles.json]
      [--layers=0,5,11] [--n_long=16384] [--heads_long=0,2,5,6,8,11] [--selftest]
      On q/k captured from trained checkpoints: the fraction of (query block, key tile) tiles that hold any key of any
      row's support at each Newton sweep and at the final lambda (what tile skipping could save), sweeps until lambda
      stops changing, and whether a dead tile ever comes back. fp32 emulation of the kernel's Newton loop, no kernel run.
  python bench_kernel.py real --ckpts=<run>/ckpt.pt[,..] --fineweb_val=<val.bin> --arxiv=<arxiv_long.bin> [--out=real.json]
      [--layers=0,11] [--batch=8] [--n_long=16384]
      Forward kernel on q/k captured from trained checkpoints (random logits have no dead tiles to speak of): frozen
      baseline, candidate without tile skipping, candidate; per layer ms, bit identity, live-flag fraction at exit.
      e005: also forward + backward (autograd, random v and dO) with the backward skipping off (the e004 kernel) and
      on, and dq, dk, dv bit for bit against the e004 kernel and the baseline.
      e006: also the candidate with EXIT_CYCLE off (the e005 kernel), forward and forward + backward, and the sweep at
      which each 64-row block's lambda starts to repeat (lambda_t from kernels launched at NUM_ITER = 1..num_iter).
  python bench_kernel.py skipcheck [--quick]
      Tile skipping is invisible: candidate with flags trusted, never trusted (fallback path) and at logits large
      enough that the trust test fails by rounding, each bit for bit against the unskipped candidate and the baseline
      (o, lambda, S; e005: and dq, dk, dv with the backward skipping on against off and against the baseline).
      e006: plus the cycle exit at other sweep counts (both parities of the sweeps left) against the baseline.
  python bench_kernel.py aliascheck
      The Triton behaviour the kernel's loop-carried lambda history relies on (a name aliasing another before a loop).
"""
import os, sys, json, time, glob, re, statistics, runpy
from collections import defaultdict
import torch


def _flags(argv, **defaults):
    out = dict(defaults)
    rest = []
    for a in argv:
        if a.startswith('--') and '=' in a and a[2:].split('=', 1)[0] in out:
            k, v = a[2:].split('=', 1)
            out[k] = type(defaults[k])(v)
        elif a.startswith('--') and a[2:] in out and isinstance(defaults[a[2:]], bool):
            out[a[2:]] = True
        else:
            rest.append(a)
    return out, rest


def bench(fn, repeats=5, target_s=0.6):
    """Median over `repeats` of the mean ms per call (CUDA-synchronised), after warmup."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    one = time.perf_counter() - t
    iters = int(min(50, max(3, target_s / max(one, 1e-6))))
    reps = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(); reps.append((time.perf_counter() - t) / iters * 1e3)
    return statistics.median(reps)


def micro(argv):
    import triton
    import model
    import stieltjes.triton_stieltjes as cand
    import stieltjes.triton_stieltjes_ref as ref
    from stieltjes.adasplash_entmax import sparse_attn
    import torch.nn.functional as F
    fl, _ = _flags(argv, out='', small=False)
    dev = 'cuda'
    torch.manual_seed(0)
    smem = triton.runtime.driver.active.utils.get_device_properties(0)["max_shared_mem"]
    res = dict(device=torch.cuda.get_device_name(), max_shared_mem=smem, torch=torch.__version__, triton=triton.__version__,
               tiles_bf16=list(cand._pick_blocks(64, 2, torch.device('cuda'))), rows={})
    print(res, flush=True)
    R = res['rows']

    def put(name, ms):
        R[name] = ms
        print(f"{name:52s} {ms:9.2f} ms", flush=True)

    B, H, N, D = (2, 12, 512, 64) if fl['small'] else (8, 12, 2048, 64)
    NL = 2048 if fl['small'] else 16384
    sq, window, NI = 3.0, 2.0, 15
    sm = D ** -0.5
    slopes = model.nape_slopes(H, 'exp').to(dev)

    def mk(B_, N_):
        q = (torch.randn(B_, H, N_, D, device=dev) * 3).to(torch.bfloat16).requires_grad_(True)
        k = torch.randn(B_, H, N_, D, device=dev).to(torch.bfloat16).requires_grad_(True)
        v = torch.randn(B_, H, N_, D, device=dev).to(torch.bfloat16).requires_grad_(True)
        return q, k, v

    seen = set()
    res['compiled'] = {}
    def kinfo(tag, jitfn):
        # what the compiler produced for the variant just run: shared memory per block bounds how many blocks an SM
        # can hold at once, PTX size tracks the unrolling. Optional (Triton internals), never fatal.
        try:
            for dc in jitfn.cache.values():
                for kern in dc.values():
                    if id(kern) in seen:
                        continue
                    seen.add(id(kern))
                    info = dict(shared_bytes=int(kern.metadata.shared), num_warps=int(kern.metadata.num_warps),
                                ptx_chars=len(kern.asm.get('ptx', '')), cubin_bytes=len(kern.asm.get('cubin', b'')))
                    res['compiled'][tag] = info
                    print(f"{tag:52s} {info}", flush=True)
        except Exception as e:
            print(f"{tag}: compiled-kernel info skipped ({e!r})", flush=True)

    q, k, v = mk(B, N)
    do = torch.randn_like(q)
    shape = f"B{B} H{H} N{N}"
    for name, mod in (('ref', ref), ('cand', cand)):
        c = mod.window_c(sq, window)
        for ni in (0, 5, 15, 30):
            with torch.no_grad():
                put(f"{name} fwd kernel it={ni} {shape}", bench(lambda: mod._forward(q, k, v, True, sm, sq, ni, c, slopes)))
            kinfo(f"{name} fwd kernel it={ni} compiled", mod._fwd_kernel)
        try:   # per-kernel backward: valid while the module keeps the three-kernel backward of the baseline
            with torch.no_grad():
                bsk = hasattr(mod, 'SKIP_BWD')      # e005 signature: tile flags + SKIP in the backward kernels
                fo = mod._forward(q, k, v, True, sm, sq, NI, c, slopes, **(dict(return_live=True) if bsk else {}))
                o, lam, S, sl, BM, BN = fo[:6]
                common = dict(sq=sq, EPS=1e-6, HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, CAUSAL=True, USE_ALIBI=True)
                gm, gn = (triton.cdiv(N, BM), B * H), (triton.cdiv(N, BN), B * H)
                ex = ()
                if bsk:
                    common['SKIP'] = fo[6] is not None and mod.SKIP_BWD
                    ex = (fo[6] if common['SKIP'] else lam, triton.cdiv(N, BM), triton.cdiv(N, BN))
                st = mod._strides
                delta = torch.empty_like(lam); coef = (sq / S.clamp(min=1e-6)).contiguous()
                dk, dv, dq = torch.empty_like(k), torch.empty_like(v), torch.empty_like(q)
                put(f"{name} bwd delta kernel {shape}", bench(lambda: mod._bwd_delta_kernel[gm](
                    q, k, v, do, lam, delta, sl, *st(q), *st(k), *st(v), *st(do), sm, c, N, H, *ex, **common)))
                put(f"{name} bwd dkdv kernel {shape}", bench(lambda: mod._bwd_dkdv_kernel[gn](
                    q, k, v, do, lam, coef, delta, sl, dk, dv, *st(q), *st(k), *st(v), *st(do), *st(dk), *st(dv), sm, c, N, H, *ex, **common)))
                put(f"{name} bwd dq kernel {shape}", bench(lambda: mod._bwd_dq_kernel[gm](
                    q, k, v, do, lam, coef, delta, sl, dq, *st(q), *st(k), *st(v), *st(do), *st(dq), sm, c, N, H, *ex, **common)))
        except AttributeError as e:
            print(f"{name}: per-kernel backward skipped ({e})", flush=True)
        kw = dict(causal=True, sm_scale=sm, stieltjes_q=sq, num_iter=NI, window=window, alibi_slopes=slopes)
        # `* 1.0`: the kernel must not be the first backward node. Its backward launches Triton through the CUDA driver
        # API from the autograd worker thread, which has no current context until a torch CUDA op has run there
        # ("Triton Error [CUDA]: invalid device context", smoke job 8603546). In training c_proj's backward runs first.
        o = mod.stieltjes_attention(q, k, v, **kw) * 1.0
        put(f"{name} bwd (autograd, 3 kernels) {shape}", bench(lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True)))
        del o
        put(f"{name} fwd+bwd it={NI} {shape}", bench(lambda: torch.autograd.grad(mod.stieltjes_attention(q, k, v, **kw) * 1.0, (q, k, v), do)))

    if hasattr(cand, 'SKIP_TILES'):      # the e002 kernel on the same (random) inputs
        cand.SKIP_TILES = False
        with torch.no_grad():
            put(f"cand noskip fwd kernel it={NI} {shape}", bench(lambda: cand._forward(q, k, v, True, sm, sq, NI, cand.window_c(sq, window), slopes)))
        cand.SKIP_TILES = True
    if hasattr(cand, 'SKIP_BWD'):        # the e004 backward on the same (random) inputs
        cand.SKIP_BWD = False
        kw = dict(causal=True, sm_scale=sm, stieltjes_q=sq, num_iter=NI, window=window, alibi_slopes=slopes)
        o = cand.stieltjes_attention(q, k, v, **kw) * 1.0
        put(f"cand nobskip bwd (autograd, 3 kernels) {shape}", bench(lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True)))
        del o
        put(f"cand nobskip fwd+bwd it={NI} {shape}", bench(lambda: torch.autograd.grad(cand.stieltjes_attention(q, k, v, **kw) * 1.0, (q, k, v), do)))
        cand.SKIP_BWD = True
    if hasattr(cand, 'EXIT_CYCLE'):      # the e005 kernel on the same (random) inputs
        cand.EXIT_CYCLE = False
        kw = dict(causal=True, sm_scale=sm, stieltjes_q=sq, num_iter=NI, window=window, alibi_slopes=slopes)
        with torch.no_grad():
            put(f"cand noexit fwd kernel it={NI} {shape}", bench(lambda: cand._forward(q, k, v, True, sm, sq, NI, cand.window_c(sq, window), slopes)))
        put(f"cand noexit fwd+bwd it={NI} {shape}", bench(lambda: torch.autograd.grad(cand.stieltjes_attention(q, k, v, **kw) * 1.0, (q, k, v), do)))
        cand.EXIT_CYCLE = True
    # call site as in CausalSelfAttention.forward: q/k/v are transposed views of one c_attn output
    x = torch.randn(B, N, 3 * H * D, device=dev).to(torch.bfloat16).requires_grad_(True)
    def views():
        a, b, c_ = x.split(H * D, dim=2)
        return [t.view(B, N, H, D).transpose(1, 2) for t in (a, b, c_)]
    if hasattr(cand, 'ZERO_COPY'):      # e007 (reverted: no gain on the step); rows only for a kernel file with the switch
        # e007: the whole call site of CausalSelfAttention.forward, from the c_attn output x to y (B, N, H*D) and back to
        # dx. "copy" = the e006 site (q/k/v .contiguous(), O and gradients in (B, H, N, D) memory, dO copied), "zero-copy" =
        # the candidate (views in, outputs in the views' memory order). "kernel on views" = the forward alone on strided q/k/v.
        def site(copy):
            qq, kk, vv = views()
            cand.ZERO_COPY = not copy
            try:
                if copy:
                    qq, kk, vv = qq.contiguous(), kk.contiguous(), vv.contiguous()
                y = cand.stieltjes_attention(qq, kk, vv, causal=True, sm_scale=sm, stieltjes_q=sq, num_iter=NI, window=window, alibi_slopes=slopes)
            finally:
                cand.ZERO_COPY = True
            return y.transpose(1, 2).contiguous().view(B, N, H * D)
        dy = torch.randn(B, N, H * D, device=dev).to(torch.bfloat16)
        def site_fb(copy):
            cand.ZERO_COPY = not copy       # the backward reads the switch too
            try:
                return torch.autograd.grad(site(copy), x, dy)
            finally:
                cand.ZERO_COPY = True
        with torch.no_grad():
            qv, kv, vv_ = views()
            put(f"cand fwd kernel on views it={NI} {shape}", bench(lambda: cand._forward(qv, kv, vv_, True, sm, sq, NI, cand.window_c(sq, window), slopes)))
            put(f"cand call site fwd copy {shape}", bench(lambda: site(True)))
            put(f"cand call site fwd zero-copy {shape}", bench(lambda: site(False)))
            same = bool(torch.equal(site(True), site(False)))
        put(f"cand call site fwd+bwd copy {shape}", bench(lambda: site_fb(True)))
        put(f"cand call site fwd+bwd zero-copy {shape}", bench(lambda: site_fb(False)))
        gsame = bool(torch.equal(site_fb(True)[0], site_fb(False)[0]))
        res['site_identical'] = [same, gsame]
        print(f"call site copy vs zero-copy identical (y, dx): {same} {gsame}", flush=True)
        del qv, kv, vv_, dy
    if hasattr(cand, 'CACHE_SMEM'):      # e008: the e006 wrapper (driver query on every forward) on the same inputs
        cand.CACHE_SMEM = False
        with torch.no_grad():
            put(f"cand nocache fwd kernel it={NI} {shape}", bench(lambda: cand._forward(q, k, v, True, sm, sq, NI, cand.window_c(sq, window), slopes)))
        put(f"cand nocache fwd+bwd it={NI} {shape}", bench(lambda: torch.autograd.grad(cand.stieltjes_attention(q, k, v, causal=True, sm_scale=sm, stieltjes_q=sq, num_iter=NI, window=window, alibi_slopes=slopes) * 1.0, (q, k, v), do)))
        cand.CACHE_SMEM = True

    # references: AdaSplash entmax (niter 10) and SDPA softmax + NAPE bias, same shape; softmax also at micro-batch 16
    varlen = torch.full((B,), N, dtype=torch.long, device=dev)
    sl_e = slopes.to(q.dtype)
    ent = lambda: sparse_attn(q, k, v, alpha=1.5, is_causal=True, varlen=varlen, niter=10, alibi_slopes=sl_e)
    with torch.no_grad():
        put(f"entmax fwd niter=10 {shape}", bench(ent))
    put(f"entmax fwd+bwd niter=10 {shape}", bench(lambda: torch.autograd.grad(ent() * 1.0, (q, k, v), do)))
    mask = model.nape_bias(slopes, N, dev, q.dtype, causal_fill=float('-inf'))
    sd = lambda a, b, c_: F.scaled_dot_product_attention(a, b, c_, attn_mask=mask)
    with torch.no_grad():
        put(f"softmax SDPA+bias fwd {shape}", bench(lambda: sd(q, k, v)))
    put(f"softmax SDPA+bias fwd+bwd {shape}", bench(lambda: torch.autograd.grad(sd(q, k, v), (q, k, v), do)))
    if not fl['small']:
        q2, k2, v2 = mk(2 * B, N); do2 = torch.randn_like(q2)
        put(f"softmax SDPA+bias fwd+bwd B{2 * B} H{H} N{N}", bench(lambda: torch.autograd.grad(sd(q2, k2, v2), (q2, k2, v2), do2)))
        del q2, k2, v2, do2
    del q, k, v, do, x, mask
    torch.cuda.empty_cache()

    # long-context eval forward
    q, k, v = mk(1, NL)
    shape = f"B1 H{H} N{NL}"
    with torch.no_grad():
        for name, mod in (('ref', ref), ('cand', cand)):
            put(f"{name} fwd kernel it={NI} {shape}", bench(lambda: mod._forward(q, k, v, True, sm, sq, NI, mod.window_c(sq, window), slopes), repeats=3))
        varlen = torch.full((1,), NL, dtype=torch.long, device=dev)
        put(f"entmax fwd niter=10 {shape}", bench(lambda: sparse_attn(q, k, v, alpha=1.5, is_causal=True, varlen=varlen, niter=10, alibi_slopes=sl_e), repeats=3))
    # fixed per-call cost of the Python wrapper (shows up 12 layers x 32 micro-steps = 384 times per 125M step)
    try:
        qs, ks, vs = mk(1, 64)
        with torch.no_grad():
            put("cand fwd wrapper it=0 B1 H12 N64 (launch overhead)", bench(lambda: cand._forward(qs, ks, vs, True, sm, sq, 0, cand.window_c(sq, window), slopes)))
            put("ref fwd wrapper it=0 B1 H12 N64 (launch overhead)", bench(lambda: ref._forward(qs, ks, vs, True, sm, sq, 0, ref.window_c(sq, window), slopes)))
            put("cand _pick_blocks", bench(lambda: cand._pick_blocks(D, 2, qs.device)))
            if hasattr(cand, 'CACHE_SMEM'):
                cand.CACHE_SMEM = False
                put("cand nocache fwd wrapper it=0 B1 H12 N64 (launch overhead)", bench(lambda: cand._forward(qs, ks, vs, True, sm, sq, 0, cand.window_c(sq, window), slopes)))
                put("cand nocache _pick_blocks (device query)", bench(lambda: cand._pick_blocks(D, 2, qs.device)))
                cand.CACHE_SMEM = True
    except Exception as e:      # optional rows: never lose the table above
        print(f"overhead rows skipped: {e!r}", flush=True)
    res['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30
    if fl['out']:
        with open(fl['out'], 'w') as f:
            json.dump(res, f, indent=1)


def wrap(argv):
    i = argv.index('--')
    fl, _ = _flags(argv[:i], kernel='cand', timers=0, out='')
    script, sargs = argv[i + 1], argv[i + 2:]
    import model
    import stieltjes.triton_stieltjes as kmod
    if fl['kernel'] == 'ref':
        import stieltjes.triton_stieltjes_ref as kmod
        model._triton_stieltjes = kmod.stieltjes_attention     # _stieltjes_fused looks this global up at call time
    elif fl['kernel'] == 'cand_noskip':
        kmod.SKIP_TILES = False                                # read by _forward at call time
    elif fl['kernel'] == 'cand_nobskip':
        kmod.SKIP_BWD = False                                  # read by _forward / the Function at call time
    elif fl['kernel'] == 'cand_noexit':
        kmod.EXIT_CYCLE = False                                # read by _forward at call time (the e005 kernel)
    elif fl['kernel'] == 'cand_nocache':
        kmod.CACHE_SMEM = False                                # read by _pick_blocks at call time (the e006 wrapper)
    else:
        assert fl['kernel'] == 'cand', fl['kernel']
    T = defaultdict(list)       # name -> [(t_end, seconds)]
    state = dict(train=True)
    if fl['timers']:
        import stieltjes.adasplash_entmax as ae

        def timed(name, fn, outer=False):
            def w(*a, **k):
                if outer:
                    state['train'] = torch.is_grad_enabled()     # eval forwards are kept apart (no backward there)
                torch.cuda.synchronize(); t = time.perf_counter()
                out = fn(*a, **k)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                T[name + ('' if state['train'] else '_eval')].append((t1, t1 - t))
                return out
            return w
        for cls in (kmod.StieltjesAttention, ae._sparse_attention):
            cls.forward = staticmethod(timed('kernel_fwd', cls.forward))
            cls.backward = staticmethod(timed('kernel_bwd', cls.backward))
        model._stieltjes_fused = torch.compiler.disable(timed('call_fwd', model._stieltjes_fused, outer=True))
        model._entmax_fused = torch.compiler.disable(timed('call_fwd', model._entmax_fused, outer=True))
    print(f"[bench_kernel wrap] kernel={fl['kernel']} ({kmod.__name__}) timers={fl['timers']} script={script}", flush=True)
    sys.argv = [script] + sargs
    t0 = time.perf_counter()
    try:
        runpy.run_path(script, run_name='__main__')
    finally:
        wall = time.perf_counter() - t0
        res = dict(kernel=fl['kernel'], timers=fl['timers'], script=os.path.basename(script), args=sargs, wall_s=wall,
                   peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0, timers_ms={})
        if T.get('kernel_fwd'):
            # steady state = second half of the training calls; share = time inside the timers / wall of that window
            ts = T['kernel_fwd']
            t_lo, t_hi = ts[len(ts) // 2][0], ts[-1][0]
            for name, lst in T.items():
                sel = [d for (te, d) in lst if t_lo < te <= t_hi] if not name.endswith('_eval') else [d for _, d in lst]
                if sel:
                    res['timers_ms'][name] = dict(calls=len(lst), steady_calls=len(sel), median=statistics.median(sel) * 1e3,
                                                  mean=statistics.fmean(sel) * 1e3, steady_sum_s=sum(sel))
            res['steady_window_s'] = t_hi - t_lo
        print('[bench_kernel wrap] ' + json.dumps(res), flush=True)
        if fl['out']:
            with open(fl['out'], 'w') as f:
                json.dump(res, f, indent=1)


def summary(argv):
    d = argv[0]
    out = {}
    for log in sorted(glob.glob(os.path.join(d, 'step_*.log'))):
        name = os.path.basename(log)[5:-4]
        ms = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^iter (\d+): loss [\d.naninf]+, time ([\d.]+)ms", open(log).read(), re.M)]
        ms = [x for _, x in ms][3:]      # first 3 logged steps (compile, warm-up) dropped by position: resumed runs start at iter ~4770
        row = dict(steps=len(ms), s_per_step=statistics.median(ms) / 1e3 if ms else None)
        js = os.path.join(d, f'step_{name}.json')
        if os.path.exists(js):
            j = json.load(open(js))
            row.update(peak_alloc_gb=j['peak_alloc_gb'], peak_reserved_gb=j['peak_reserved_gb'])
            tm = j.get('timers_ms', {})
            if 'kernel_fwd' in tm and j.get('steady_window_s'):
                w = j['steady_window_s']
                row.update(kernel_fwd_ms=tm['kernel_fwd']['median'], kernel_bwd_ms=tm['kernel_bwd']['median'],
                           call_fwd_ms=tm['call_fwd']['median'],
                           share_call_fwd=tm['call_fwd']['steady_sum_s'] / w, share_kernel_fwd=tm['kernel_fwd']['steady_sum_s'] / w,
                           share_kernel_bwd=tm['kernel_bwd']['steady_sum_s'] / w)
                row['share_attention'] = row['share_call_fwd'] + row['share_kernel_bwd']
        if 'out of memory' in open(log).read():
            row['oom'] = True
        out[name] = row
        print(name, json.dumps(row))
    with open(os.path.join(d, 'step_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)


def sweep(argv):
    import triton
    import model
    import stieltjes.triton_stieltjes as cand
    fl, _ = _flags(argv, out='', small=False)
    dev = 'cuda'
    torch.manual_seed(0)
    small = fl['small']
    B, H, N, D = (2, 12, 512, 64) if small else (8, 12, 2048, 64)
    NL = 2048 if small else 16384
    sq, window, NI = 3.0, 2.0, 15
    sm, c = D ** -0.5, cand.window_c(sq, window)
    lam0 = (1.0 + c) ** (-1.0 / sq)
    slopes = model.nape_slopes(H, 'exp').reshape(-1).to(device=dev, dtype=torch.float32).contiguous()
    st = cand._strides
    DEF = (64, 32, 4, 3)
    res = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
               shape=[B, H, N, D], long_shape=[1, H, NL, D], default=list(DEF), rows=[])
    print({k: v for k, v in res.items() if k != 'rows'}, flush=True)

    def mk(B_, N_):
        q = (torch.randn(B_, H, N_, D, device=dev) * 3).to(torch.bfloat16)
        return q, torch.randn(B_, H, N_, D, device=dev).to(torch.bfloat16), torch.randn(B_, H, N_, D, device=dev).to(torch.bfloat16)

    def launch(kern, q, k, v, do, lam, coef, delta, outs, BM, BN, W, S):
        Bq, _, Nq, _ = q.shape
        common = dict(sq=sq, EPS=1e-6, HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, CAUSAL=True, USE_ALIBI=True, num_warps=W, num_stages=S)
        gm, gn = (triton.cdiv(Nq, BM), Bq * H), (triton.cdiv(Nq, BN), Bq * H)
        if kern == 'fwd':
            o, lam_o, S_o = outs
            return lambda: cand._fwd_kernel[gm](q, k, v, o, lam_o, S_o, slopes, *st(q), *st(k), *st(v), *st(o),
                                                sm, c, lam0, Nq, H, NUM_ITER=NI, **common)
        if kern == 'delta':
            return lambda: cand._bwd_delta_kernel[gm](q, k, v, do, lam, outs[0], slopes, *st(q), *st(k), *st(v), *st(do),
                                                      sm, c, Nq, H, **common)
        if kern == 'dkdv':
            return lambda: cand._bwd_dkdv_kernel[gn](q, k, v, do, lam, coef, delta, slopes, outs[0], outs[1], *st(q), *st(k), *st(v),
                                                     *st(do), *st(outs[0]), *st(outs[1]), sm, c, Nq, H, **common)
        return lambda: cand._bwd_dq_kernel[gm](q, k, v, do, lam, coef, delta, slopes, outs[0], *st(q), *st(k), *st(v), *st(do),
                                               *st(outs[0]), sm, c, Nq, H, **common)

    def new_outs(kern, q):
        Bq, _, Nq, _ = q.shape
        if kern == 'fwd':
            return [torch.empty_like(q), torch.empty((Bq * H, Nq), device=dev), torch.empty((Bq * H, Nq), device=dev)]
        if kern == 'delta':
            return [torch.empty((Bq * H, Nq), device=dev)]
        return [torch.empty_like(q), torch.empty_like(q)] if kern == 'dkdv' else [torch.empty_like(q)]

    q, k, v = mk(B, N)
    do = torch.randn_like(q)
    base = {}                   # kern -> outputs of the default launch (the bit-identity reference)
    for kern in ('fwd', 'delta', 'dkdv', 'dq'):
        outs = new_outs(kern, q)
        if kern == 'fwd':
            launch(kern, q, k, v, do, None, None, None, outs, *DEF)()
            lam, coef = outs[1], (sq / outs[2].clamp(min=1e-6)).contiguous()
        else:
            launch(kern, q, k, v, do, lam, coef, base['delta'][0] if 'delta' in base else None, outs, *DEF)()
        torch.cuda.synchronize()
        base[kern] = outs
    delta = base['delta'][0]

    def run(kern, cfg, tag, repeats=3, target_s=0.4):
        BM, BN, W, S = cfg
        row = dict(kern=kern, tag=tag, BM=BM, BN=BN, num_warps=W, num_stages=S)
        outs = new_outs(kern, q)
        try:
            t = time.perf_counter()
            fn = launch(kern, q, k, v, do, lam, coef, delta, outs, BM, BN, W, S)
            fn(); torch.cuda.synchronize()
            row['first_call_s'] = time.perf_counter() - t           # compile (or cache load) + one run
            row['identical'] = [bool(torch.equal(a, b)) for a, b in zip(outs, base[kern])]   # fwd: o, lam, S; dkdv: dk, dv
            row['bit_identical'] = all(row['identical'])
            row['max_abs_diff'] = max((a.float() - b.float()).abs().max().item() for a, b in zip(outs, base[kern]))
            row['ms'] = bench(fn, repeats=repeats, target_s=target_s)
        except Exception as e:       # tiles too large for shared memory, unsupported option: recorded, never fatal
            row['error'] = repr(e)[:300]
        res['rows'].append(row)
        print(json.dumps(row), flush=True)
        if fl['out']:
            with open(fl['out'], 'w') as f:
                json.dump(res, f, indent=1)
        return row

    ok = lambda rows: sorted([r for r in rows if 'ms' in r], key=lambda r: r['ms'])
    cfg_of = lambda r: (r['BM'], r['BN'], r['num_warps'], r['num_stages'])
    # forward (num_iter 15): tiles at the default options, then warps x stages on the fastest tiles
    f_tiles = [(64, 32), (32, 32)] if small else [(bm, bn) for bm in (32, 64, 128) for bn in (16, 32, 64, 128)]
    rows = [run('fwd', (bm, bn, 4, 3), 'tiles') for bm, bn in f_tiles]
    top = ok(rows)[:1 if small else 3]
    for r in top:
        for W in ((4, 8) if small else (2, 4, 8)):
            for S in ((2, 3) if small else (1, 2, 3, 4)):
                if (W, S) != (4, 3):
                    rows.append(run('fwd', (r['BM'], r['BN'], W, S), 'opts'))
    best_fwd = [cfg_of(r) for r in ok(rows)[:4]]
    # backward kernels: tiles x warps (each kernel has its own grid: nothing ties its tiles to the forward's)
    b_tiles = [(64, 32), (32, 32)] if small else [(bm, bn) for bm in (32, 64, 128) for bn in (32, 64)]
    best_bwd = {}
    for kern in ('delta', 'dkdv', 'dq'):
        rows = [run(kern, (bm, bn, W, 3), 'tiles') for bm, bn in b_tiles for W in ((4,) if small else (4, 8))]
        r0 = ok(rows)[0]
        rows += [run(kern, (r0['BM'], r0['BN'], r0['num_warps'], S), 'opts') for S in ((2,) if small else (1, 2, 4))]
        best_bwd[kern] = [cfg_of(r) for r in ok(rows)[:2]]
    # confirmation: default and the best variants again, 5 repeats, back to back
    for cfg in [DEF] + [x for x in best_fwd if x != DEF]:
        run('fwd', cfg, 'confirm', repeats=5, target_s=0.6)
    for kern in ('delta', 'dkdv', 'dq'):
        for cfg in [DEF] + [x for x in best_bwd[kern] if x != DEF]:
            run(kern, cfg, 'confirm', repeats=5, target_s=0.6)
    # long-context forward for the same forward variants
    del q, k, v, do
    torch.cuda.empty_cache()
    q, k, v = mk(1, NL)
    do = lam = coef = delta = None
    outs = new_outs('fwd', q)
    launch('fwd', q, k, v, None, None, None, None, outs, *DEF)(); torch.cuda.synchronize()
    base['fwd'] = outs
    for cfg in [DEF] + [x for x in best_fwd if x != DEF]:
        run('fwd', cfg, f'long N={NL}', repeats=3, target_s=0.6)
    res['best_fwd'], res['best_bwd'] = best_fwd, best_bwd
    if fl['out']:
        with open(fl['out'], 'w') as f:
            json.dump(res, f, indent=1)


def _capture(ckpt_path, tokens, dev='cuda'):
    """Per layer (q, k) of a trained model on `tokens` (B, N), q after the AS scale (as tests/test_kernel_equiv.py)."""
    from model import GPTConfig, GPT
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    a = ck['model_args']
    m = GPT(GPTConfig(**a))
    m.load_state_dict({kk.removeprefix('_orig_mod.'): vv for kk, vv in ck['model'].items()})
    m.to(dev).eval()
    for b in m.transformer.h:
        b.attn.capture_qk = True
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        m(tokens, return_all_logits=True)
    out = [tuple(t.contiguous() for t in b.attn.last_qk) for b in m.transformer.h]
    slopes = m.transformer.h[0].attn.nape_slopes.to(dev) if a['pos_emb'] == 'nape' else None
    window = None if a['stieltjes_window'] <= 0 else float(a['stieltjes_window'])
    del m, ck
    torch.cuda.empty_cache()
    return out, float(a['stieltjes_q']), window, slopes


TILE_SHAPES = ((64, 32), (32, 32), (32, 16), (64, 64))


def _tile_stats(q, k, slopes, sq, window, num_iter, heads):
    """Sums over (batch, heads) for one layer. lam[t] is the lambda sweep t uses (t < num_iter); lam[num_iter] is the
    final one (output sweep, backward). A tile is live at t when any causal (row, key) in it has w > 0, i.e.
    s > lam[t] - d. `cons0`: live by the test a row-max sweep could do for free (tile max > min lam[0] of the block - d)."""
    B, H, N, D = q.shape
    dev = q.device
    c = window ** -sq
    lam0 = (1.0 + c) ** (-1.0 / sq)
    i = torch.arange(N, device=dev)
    dist = (i[None, :] - i[:, None]).clamp(max=0).float()
    upper = torch.ones(N, N, dtype=torch.bool, device=dev).triu(1)
    T = num_iter + 1
    A = dict(pairs=0, live={sh: [0] * T for sh in TILE_SHAPES}, total={sh: 0 for sh in TILE_SHAPES}, cons0={sh: 0 for sh in TILE_SHAPES},
             resurrected=0, support0=0.0, support_final=0.0, rows=0, lam_decrease_rows=0, max_lam_decrease=0.0,
             fix_hist=[0] * (T + 1), resid_max=0.0)
    for b in range(B):
        for h in heads:
            s = (q[b, h].float() @ k[b, h].float().T) * D ** -0.5
            if slopes is not None:
                s = s + slopes[h].float() * dist
            s.masked_fill_(upper, -1e30)
            lam = s.amax(1) + lam0
            lams = [lam]
            for _ in range(num_iter):
                r = 1.0 / (lam[:, None] - s).clamp_min(1e-6)
                iq = r * r * r if sq == 3.0 else ((r * r) * (r * r) if sq == 4.0 else r ** sq)
                w = iq - c
                on = w > 0
                f = torch.where(on, w, 0.0).sum(1)
                fp = torch.where(on, iq * r, 0.0).sum(1)
                lam = lam + (f - 1.0) / (sq * fp.clamp_min(1e-30))
                lams.append(lam)
            del r, iq, w, on
            L = torch.stack(lams)                                   # (T, N)
            dl = L[1:] - L[:-1]
            A['lam_decrease_rows'] += int((dl < 0).any(0).sum()); A['max_lam_decrease'] = max(A['max_lam_decrease'], float((-dl).max()))
            # sweeps until lambda stops changing, per 64-row block (the unit that could exit early): first t with dl[t:] == 0
            moving = (dl != 0).view(num_iter, N // 64, 64).any(2)   # (num_iter, blocks)
            last = torch.where(moving.any(0), num_iter - moving.flip(0).float().argmax(0), torch.zeros_like(moving[0], dtype=torch.float))
            for x in last.long().tolist():
                A['fix_hist'][x] += 1                               # x = number of sweeps that changed lambda (num_iter = never settled)
            prev = None
            for t in range(T):
                live = s > (L[t][:, None] - window)
                if t == 0:
                    A['support0'] += float(live.sum())
                if t == T - 1:
                    A['support_final'] += float(live.sum())
                    wf = (1.0 / (L[t][:, None] - s).clamp_min(1e-6)) ** sq - c
                    A['resid_max'] = max(A['resid_max'], float((torch.where(wf > 0, wf, 0.0).sum(1) - 1).abs().max()))
                    del wf
                for (BM, BN) in TILE_SHAPES:
                    lt = live.view(N // BM, BM, N // BN, BN).any(3).any(1)
                    A['live'][(BM, BN)][t] += int(lt.sum())
                    if (BM, BN) == (64, 32):
                        if prev is not None:
                            A['resurrected'] += int((lt & ~prev).sum())
                        prev = lt
                del live
            for (BM, BN) in TILE_SHAPES:
                nb = N // BM
                A['total'][(BM, BN)] += sum(-(-min((m + 1) * BM, N) // BN) for m in range(nb))
                tmax = s.view(nb, BM, N // BN, BN).amax(3).amax(1)
                A['cons0'][(BM, BN)] += int((tmax > (L[0].view(nb, BM).amin(1)[:, None] - window)).sum())
            A['rows'] += N; A['pairs'] += 1
            del s, L, dl
    return A


def _tile_row(A, num_iter):
    out = dict(rows=A['rows'], support0=A['support0'] / A['rows'], support_final=A['support_final'] / A['rows'],
               resurrected=A['resurrected'], lam_decrease_rows=A['lam_decrease_rows'], max_lam_decrease=A['max_lam_decrease'],
               fix_hist=A['fix_hist'], resid_max=A['resid_max'])
    nblk = max(sum(A['fix_hist']), 1)
    out['mean_sweeps_until_fixed'] = sum(i * n for i, n in enumerate(A['fix_hist'])) / nblk
    for sh in TILE_SHAPES:
        f = [x / A['total'][sh] for x in A['live'][sh]]
        key = f"{sh[0]}x{sh[1]}"
        # forward with a mask carried from the previous sweep: row max (1) + first Newton sweep (1) + Newton sweeps
        # 1..num_iter-1 at f[t-1] + output sweep at f[num_iter-1]; the backward sweeps run at f[num_iter]
        out[key] = dict(live=f, cons0=A['cons0'][sh] / A['total'][sh], fwd_sweep_equiv=2 + sum(f[:num_iter]), fwd_sweeps=num_iter + 2,
                        bwd_frac=f[num_iter])
    return out


def tiles(argv):
    import numpy as np
    import model
    fl, _ = _flags(argv, out='', ckpts='', fineweb_val='', arxiv='', layers='', n_long=16384, heads_long='0,2,5,6,8,11',
                   num_iter=15, selftest=False)
    dev = 'cuda'
    torch.manual_seed(0)
    NI = fl['num_iter']
    res = dict(num_iter=NI, tile_shapes=[list(x) for x in TILE_SHAPES], cases=[])

    def emit(tag, layer, A):
        row = dict(tag=tag, layer=layer, **_tile_row(A, NI))
        res['cases'].append(row)
        t = row['64x32']
        print(f"{tag:46s} L{layer:<2d} support {row['support0']:7.1f} -> {row['support_final']:6.1f} | 64x32 live t0 {t['live'][0]:.3f} "
              f"t1 {t['live'][1]:.3f} t4 {t['live'][4]:.3f} final {t['live'][NI]:.3f} cons0 {t['cons0']:.3f} | fwd sweeps "
              f"{t['fwd_sweep_equiv']:.2f}/{t['fwd_sweeps']} | 32x16 final {row['32x16']['live'][NI]:.3f} | settled after "
              f"{row['mean_sweeps_until_fixed']:.1f} sweeps | resurrected {row['resurrected']} lam-dec rows {row['lam_decrease_rows']} "
              f"resid {row['resid_max']:.1e}", flush=True)
        if fl['out']:
            with open(fl['out'], 'w') as f:
                json.dump(res, f, indent=1)

    if fl['selftest']:           # random logits, no checkpoint: exercises everything but the capture
        H, N, D = 4, 512, 64
        q = (torch.randn(1, H, N, D, device=dev) * 3).to(torch.bfloat16); k = torch.randn(1, H, N, D, device=dev).to(torch.bfloat16)
        emit('selftest random w_q3_d2 N=512', 0, _tile_stats(q, k, model.nape_slopes(H, 'exp').to(dev), 3.0, 2.0, NI, list(range(H))))
        return
    fw = np.memmap(fl['fineweb_val'], dtype=np.uint16, mode='r')
    ax = np.memmap(fl['arxiv'], dtype=np.uint16, mode='r')
    NLg = fl['n_long']
    tok_fw = torch.from_numpy(np.stack([fw[i * 2049: i * 2049 + 2048].astype(np.int64) for i in range(2)])).to(dev)
    tok_ax = torch.from_numpy(ax[:NLg].astype(np.int64))[None].to(dev)
    for path in [x for x in fl['ckpts'].split(',') if x]:
        row = os.path.basename(os.path.dirname(os.path.dirname(path)))
        for name, tok, heads in (('fineweb N=2048', tok_fw, None), (f'arxiv N={NLg}', tok_ax, fl['heads_long'])):
            qk, sq, window, slopes = _capture(path, tok)
            H = qk[0][0].shape[1]
            hs = list(range(H)) if not heads or heads == 'all' else [int(x) for x in heads.split(',')]
            layers = [int(x) for x in fl['layers'].split(',')] if fl['layers'] else list(range(len(qk)))
            for li in layers:
                with torch.no_grad():
                    emit(f"{row} {name}", li, _tile_stats(qk[li][0], qk[li][1], slopes, sq, window, NI, hs))
            del qk
            torch.cuda.empty_cache()
    # layer means per (row, dataset): each layer costs the same in the step
    agg = {}
    for cse in res['cases']:
        agg.setdefault(cse['tag'], []).append(cse)
    res['mean'] = {}
    for tag, lst in agg.items():
        m = dict(layers=len(lst), support_final=statistics.fmean(x['support_final'] for x in lst),
                 mean_sweeps_until_fixed=statistics.fmean(x['mean_sweeps_until_fixed'] for x in lst),
                 resurrected=sum(x['resurrected'] for x in lst), lam_decrease_rows=sum(x['lam_decrease_rows'] for x in lst))
        for sh in TILE_SHAPES:
            key = f"{sh[0]}x{sh[1]}"
            m[key] = dict(live_t0=statistics.fmean(x[key]['live'][0] for x in lst), live_final=statistics.fmean(x[key]['live'][NI] for x in lst),
                          cons0=statistics.fmean(x[key]['cons0'] for x in lst),
                          fwd_sweep_equiv=statistics.fmean(x[key]['fwd_sweep_equiv'] for x in lst), fwd_sweeps=NI + 2)
        res['mean'][tag] = m
        print('MEAN', tag, json.dumps(m), flush=True)
    if fl['out']:
        with open(fl['out'], 'w') as f:
            json.dump(res, f, indent=1)


def _live_frac(lv, BM, BN, N):
    """Fraction of the causal tiles the forward sweeps whose flag is still 1."""
    NB = lv.shape[1]
    nt = [-(-min((m + 1) * BM, N) // BN) for m in range(NB)]
    return float(sum(lv[:, m, :nt[m]].float().sum() for m in range(NB))) / (sum(nt) * lv.shape[0])


def real(argv):
    import numpy as np
    import triton
    import stieltjes.triton_stieltjes as cand
    import stieltjes.triton_stieltjes_ref as ref
    fl, _ = _flags(argv, out='', ckpts='', fineweb_val='', arxiv='', layers='', n_long=16384, batch=8, num_iter=15)
    dev = 'cuda'
    torch.manual_seed(0)
    NI = fl['num_iter']
    res = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__, num_iter=NI, batch=fl['batch'], cases=[], mean={})
    print({k: v for k, v in res.items() if k not in ('cases', 'mean')}, flush=True)
    fw = np.memmap(fl['fineweb_val'], dtype=np.uint16, mode='r')
    ax = np.memmap(fl['arxiv'], dtype=np.uint16, mode='r')
    NLg = fl['n_long']
    tok_fw = torch.from_numpy(np.stack([fw[i * 2049: i * 2049 + 2048].astype(np.int64) for i in range(fl['batch'])])).to(dev)
    tok_ax = torch.from_numpy(ax[:NLg].astype(np.int64))[None].to(dev)

    def fwd(mod, skip, q, k, v, sq, c, slopes, ex=True, ni=None, **kw):
        cand.SKIP_TILES = skip; cand.EXIT_CYCLE = ex
        try:
            return mod._forward(q, k, v, True, q.shape[-1] ** -0.5, sq, NI if ni is None else ni, c, slopes, **kw)
        finally:
            cand.EXIT_CYCLE = True

    def cycle_stats(a, BM):
        # lambda_t for t = 1..NI from kernels launched at NUM_ITER = t (exit off: the plain iterates). A row repeats at
        # sweep t if lambda_t == lambda_{t-1} or lambda_{t-2} (the kernel's test; P4: or t-3, t-4, what a longer history
        # would add); a block leaves the loop at the first sweep where all its rows do.
        L = torch.stack([fwd(cand, True, *a, ex=False, ni=t)[1] for t in range(1, NI + 1)])
        L = torch.cat([fwd(cand, True, *a, ex=False, ni=0)[1][None], L])            # L[t] = lambda_t, t = 0..NI
        N = L.shape[-1]; out = {}
        for name, P in (('p2', 2), ('p4', 4)):
            blk_exit = torch.full((L.shape[1], -(-N // BM)), NI, device=L.device)
            for t in range(NI, 0, -1):
                ok = torch.zeros_like(L[0], dtype=torch.bool)
                for p in range(1, P + 1):
                    if t - p >= 0:
                        ok |= L[t] == L[t - p]
                ok = torch.nn.functional.pad(ok, (0, blk_exit.shape[1] * BM - N), value=True).view(L.shape[1], -1, BM).all(-1)
                blk_exit = torch.where(ok, torch.full_like(blk_exit, t), blk_exit)
            # earliest t with the test true (the test stays true once true: the row is then on its cycle)
            out[f'sweeps_run_{name}'] = float(blk_exit.float().mean())
            out[f'blocks_exit_early_{name}'] = float((blk_exit < NI).float().mean())
        return out

    for path in [x for x in fl['ckpts'].split(',') if x]:
        row = os.path.basename(os.path.dirname(os.path.dirname(path)))
        for name, tok in (('fineweb N=2048', tok_fw), (f'arxiv N={NLg}', tok_ax)):
            qk, sq, window, slopes = _capture(path, tok)
            c = cand.window_c(sq, window)
            layers = [int(x) for x in fl['layers'].split(',')] if fl['layers'] else list(range(len(qk)))
            rows = []
            for li in layers:
                q, k = qk[li]
                v = torch.randn_like(q)
                a = (q, k, v, sq, c, slopes)
                with torch.no_grad():
                    r = fwd(ref, True, *a); n = fwd(cand, False, *a); x = fwd(cand, True, *a, return_live=True)
                    e = fwd(cand, True, *a, ex=False)
                    cse = dict(tag=f"{row} {name}", layer=li, shape=list(q.shape), dtype=str(q.dtype),
                               identical_noskip=[bool(torch.equal(x[i], n[i])) for i in range(3)],
                               identical_ref=[bool(torch.equal(x[i], r[i])) for i in range(3)],
                               identical_noexit=[bool(torch.equal(x[i], e[i])) for i in range(3)],
                               resid_max=float((x[2] - 1).abs().max()), resid_max_ref=float((r[2] - 1).abs().max()),
                               live_at_exit=_live_frac(x[6], x[4], x[5], q.shape[2]) if x[6] is not None else None)
                    cse.update(cycle_stats(a, x[4]))
                    del r, n, x, e
                    cse['ms_ref'] = bench(lambda: fwd(ref, True, *a), repeats=3, target_s=0.3)
                    cse['ms_noskip'] = bench(lambda: fwd(cand, False, *a), repeats=3, target_s=0.3)
                    cse['ms_noexit'] = bench(lambda: fwd(cand, True, *a, ex=False), repeats=3, target_s=0.3)
                    cse['ms_skip'] = bench(lambda: fwd(cand, True, *a), repeats=3, target_s=0.3)
                # forward + backward through autograd (random dO): baseline, backward skipping off (e004), on
                qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
                do = torch.randn_like(q)
                kw = dict(causal=True, sm_scale=q.shape[-1] ** -0.5, stieltjes_q=sq, num_iter=NI, window=window, alibi_slopes=slopes)

                def fb(mod, bskip, ex=True):
                    cand.SKIP_TILES = True; cand.SKIP_BWD = bskip; cand.EXIT_CYCLE = ex
                    try:
                        return torch.autograd.grad(mod.stieltjes_attention(qg, kg, vg, **kw) * 1.0, (qg, kg, vg), do)
                    finally:
                        cand.SKIP_BWD = True; cand.EXIT_CYCLE = True
                gr = fb(ref, True); gn = fb(cand, False); gx = fb(cand, True); ge = fb(cand, True, ex=False)
                cse['grad_identical_nobskip'] = [bool(torch.equal(gx[i], gn[i])) for i in range(3)]
                cse['grad_identical_ref'] = [bool(torch.equal(gx[i], gr[i])) for i in range(3)]
                cse['grad_identical_noexit'] = [bool(torch.equal(gx[i], ge[i])) for i in range(3)]
                del gr, gn, gx, ge
                cse['ms_fb_ref'] = bench(lambda: fb(ref, True), repeats=3, target_s=0.3)
                cse['ms_fb_nobskip'] = bench(lambda: fb(cand, False), repeats=3, target_s=0.3)
                cse['ms_fb_noexit'] = bench(lambda: fb(cand, True, ex=False), repeats=3, target_s=0.3)
                cse['ms_fb_skip'] = bench(lambda: fb(cand, True), repeats=3, target_s=0.3)
                del qg, kg, vg, do
                rows.append(cse); res['cases'].append(cse)
                print(json.dumps(cse), flush=True)
                if fl['out']:
                    with open(fl['out'], 'w') as f:
                        json.dump(res, f, indent=1)
            m = dict(layers=len(rows), ms_ref=statistics.fmean(x['ms_ref'] for x in rows), ms_noskip=statistics.fmean(x['ms_noskip'] for x in rows),
                     ms_skip=statistics.fmean(x['ms_skip'] for x in rows), live_at_exit=statistics.fmean(x['live_at_exit'] for x in rows),
                     identical_noskip=all(all(x['identical_noskip']) for x in rows), identical_ref=all(all(x['identical_ref']) for x in rows),
                     ms_fb_ref=statistics.fmean(x['ms_fb_ref'] for x in rows), ms_fb_nobskip=statistics.fmean(x['ms_fb_nobskip'] for x in rows),
                     ms_fb_skip=statistics.fmean(x['ms_fb_skip'] for x in rows),
                     grad_identical_nobskip=all(all(x['grad_identical_nobskip']) for x in rows),
                     grad_identical_ref=all(all(x['grad_identical_ref']) for x in rows),
                     ms_noexit=statistics.fmean(x['ms_noexit'] for x in rows), ms_fb_noexit=statistics.fmean(x['ms_fb_noexit'] for x in rows),
                     identical_noexit=all(all(x['identical_noexit']) for x in rows),
                     grad_identical_noexit=all(all(x['grad_identical_noexit']) for x in rows),
                     **{key: statistics.fmean(x[key] for x in rows) for key in ('sweeps_run_p2', 'blocks_exit_early_p2', 'sweeps_run_p4', 'blocks_exit_early_p4')})
            res['mean'][f"{row} {name}"] = m
            print('MEAN', f"{row} {name}", json.dumps(m), flush=True)
            del qk
            torch.cuda.empty_cache()
    cand.SKIP_TILES = True; cand.SKIP_BWD = True
    if fl['out']:
        with open(fl['out'], 'w') as f:
            json.dump(res, f, indent=1)
    sys.exit(0 if all(m['identical_noskip'] and m['identical_ref'] and m['grad_identical_nobskip'] and m['grad_identical_ref']
                      and m['identical_noexit'] and m['grad_identical_noexit']
                      for m in res['mean'].values()) else 1)


def skipcheck(argv):
    import model
    import stieltjes.triton_stieltjes as cand
    import stieltjes.triton_stieltjes_ref as ref
    fl, _ = _flags(argv, quick=False)
    dev = 'cuda'; H, D = 4, 64
    torch.manual_seed(0)
    bad = n = 0
    for dt in (torch.bfloat16, torch.float32):
        for N in ((64, 300) if fl['quick'] else (64, 300, 1000, 2048)):
            for (sq, win) in ((3.0, 2.0), (4.0, 2.0), (3.0, 1.5), (2.5, 2.0), (4.0, None)):
                for sl in (None, model.nape_slopes(H, 'exp').to(dev)):
                    for qs in (3.0, 12.0, 2e4):      # 2e4: |lambda| > 16384, one ulp of lambda exceeds mu and the trust test fails
                        q = (torch.randn(2, H, N, D, device=dev) * qs).to(dt); k = torch.randn(2, H, N, D, device=dev).to(dt)
                        v = torch.randn(2, H, N, D, device=dev).to(dt)
                        a = (q, k, v, True, D ** -0.5, sq, 15, cand.window_c(sq, win), sl)
                        qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v)); do = torch.randn_like(q)
                        kw = dict(causal=True, sm_scale=D ** -0.5, stieltjes_q=sq, num_iter=15, window=win, alibi_slopes=sl)
                        grad = lambda mod: torch.autograd.grad(mod.stieltjes_attention(qg, kg, vg, **kw) * 1.0, (qg, kg, vg), do)
                        with torch.no_grad():
                            r = ref._forward(*a)
                            cand.SKIP_TILES = False; base = cand._forward(*a); cand.SKIP_TILES = True
                        gr = grad(ref)
                        cand.SKIP_BWD = False; gb = grad(cand); cand.SKIP_BWD = True      # the e004 backward (no tile skipped)
                        for name, mu in (('trust', 1e-3), ('notrust', -1.0)):
                            cand.SKIP_MU_REL = mu
                            with torch.no_grad():
                                x = cand._forward(*a, return_live=True)
                            gx = grad(cand)
                            cand.SKIP_MU_REL = 1e-3
                            same = [bool(torch.equal(x[i], base[i])) and bool(torch.equal(x[i], r[i])) for i in range(3)]
                            gsame = [bool(torch.equal(gx[i], gb[i])) and bool(torch.equal(gx[i], gr[i])) for i in range(3)]
                            fr = _live_frac(x[6], x[4], x[5], N) if x[6] is not None else None
                            n += 1; bad += not (all(same) and all(gsame))
                            print(f"{'OK ' if all(same) and all(gsame) else 'BAD'} {str(dt)[6:]:9s} N{N:5d} q{sq} d{win} slopes{int(sl is not None)} qs{qs:7.0f} {name:8s} "
                                  f"identical(o,lam,S)={same} (dq,dk,dv)={gsame} live_at_exit={fr if fr is None else round(fr, 3)}", flush=True)
    # e006: the cycle exit at other sweep counts (odd and even numbers of sweeps left at the exit; 1 and 2 = no history yet)
    for dt in (torch.bfloat16, torch.float32):
        for N in (64, 300):
            for (sq, win) in ((3.0, 2.0), (4.0, None)):
                for qs in (3.0, 12.0):
                    sl = model.nape_slopes(H, 'exp').to(dev)
                    q = (torch.randn(2, H, N, D, device=dev) * qs).to(dt); k = torch.randn(2, H, N, D, device=dev).to(dt)
                    v = torch.randn(2, H, N, D, device=dev).to(dt)
                    for ni in (1, 2, 3, 4, 7, 8, 14):
                        a = (q, k, v, True, D ** -0.5, sq, ni, cand.window_c(sq, win), sl)
                        with torch.no_grad():
                            r = ref._forward(*a); x = cand._forward(*a)
                            cand.EXIT_CYCLE = False; e = cand._forward(*a); cand.EXIT_CYCLE = True
                        same = [bool(torch.equal(x[i], r[i])) and bool(torch.equal(x[i], e[i])) for i in range(3)]
                        n += 1; bad += not all(same)
                        print(f"{'OK ' if all(same) else 'BAD'} {str(dt)[6:]:9s} N{N:5d} q{sq} d{win} qs{qs:7.0f} num_iter{ni:3d} exit     identical(o,lam,S)={same}", flush=True)
    print(f"skipcheck: {n - bad} of {n} OK")
    sys.exit(1 if bad else 0)


def aliascheck(argv):
    """Toolchain assumption of the kernel: a name that aliases another before a runtime loop (`lam_hw = lam`, e004) must
    stay a loop-carried variable of its own. On triton 3.6 it does when the loop updates it through a new op (the
    `lam_hw` pattern) and does not when the loop re-binds it by plain assignment (it keeps its initial value), which is
    why e006's `lam_p1` starts as `lam + 0.0`. Exit 1 unless the two patterns the kernel uses behave."""
    global triton, tl     # triton 3.1.0 resolves a jitted function's names in the module globals, not in this closure
    import triton
    import triton.language as tl

    @triton.jit
    def k_alias(X, Y, n, BM: tl.constexpr, COPY: tl.constexpr, INIF: tl.constexpr):
        offs = tl.arange(0, BM)
        a = tl.load(X + offs)
        if COPY:
            b = a + 0.0
        else:
            b = a
        hw = a                       # as lam_hw = lam
        go = n > 0
        for i in tl.range(0, n):
            if INIF:
                if go:
                    hw = tl.maximum(hw, a)
                    t = tl.where(i < 3, a + 1.0, a - 1.0)
                    b = a
                    a = t
            else:
                hw = tl.maximum(hw, a)
                t = tl.where(i < 3, a + 1.0, a - 1.0)
                b = a
                a = t
        tl.store(Y + offs, a)
        tl.store(Y + BM + offs, b)
        tl.store(Y + 2 * BM + offs, hw)

    print('triton', triton.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(), flush=True)
    bad = 0
    for inif in (False, True):
        for copy in (False, True):
            x = torch.zeros(4, device='cuda'); y = torch.zeros(12, device='cuda')
            k_alias[(1,)](x, y, 5, BM=4, COPY=copy, INIF=inif)
            a_, b_, hw_ = y[0].item(), y[4].item(), y[8].item()      # a: 0 1 2 3 2 -> 1; b = the a before it = 2; hw = 3
            used = (a_ == 1.0 and hw_ == 3.0) and (b_ == 2.0 or not copy)   # the kernel uses: hw aliased, b not aliased
            bad += not used
            print(f"{'OK ' if used else 'BAD'} if-wrapped={inif} b {'= a + 0.0' if copy else 'aliases a'}: a={a_} (1) b={b_} (2) hw={hw_} (3)"
                  f"{'' if b_ == 2.0 else '   <- aliased plain-assigned name NOT carried (not used by the kernel)'}", flush=True)
    print('aliascheck:', 'OK' if not bad else f'{bad} BAD')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    modes = {'micro': micro, 'wrap': wrap, 'summary': summary, 'sweep': sweep, 'tiles': tiles, 'real': real, 'skipcheck': skipcheck, 'aliascheck': aliascheck}
    if mode not in modes:
        print(__doc__); sys.exit(2)
    modes[mode](sys.argv[2:])
