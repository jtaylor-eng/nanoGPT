"""Timing harness for the fused Stieltjes kernel (direction stieltjes-kernel-speed). CUDA required; A100 for numbers.

  python bench_kernel.py micro [--out=micro.json] [--small]
      Kernel microbench, candidate (stieltjes/triton_stieltjes.py) and frozen baseline (triton_stieltjes_ref.py) in one
      process: fwd kernel at NUM_ITER in {0, 5, 15, 30}, each backward kernel, bwd, fwd+bwd at the 125M training shape
      (8, 12, 2048, 64) bf16 causal + exp NAPE slopes, q3/d2; the model call site (non-contiguous q/k/v views);
      AdaSplash entmax and SDPA softmax at the same shape; forward at (1, 12, 16384, 64).
  python bench_kernel.py wrap [--kernel=cand|ref] [--timers=0|1] [--out=x.json] -- <script.py> [script args...]
      Run an UNMODIFIED repo script (train.py, eval_lengths.py) with the chosen kernel and, with --timers=1,
      CUDA-synchronised timers around the attention call site and the autograd Function forward/backward.
  python bench_kernel.py summary <dir>
      Table from <dir>/micro.json, step_*.log, step_*.json.
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
                o, lam, S, sl, BM, BN = mod._forward(q, k, v, True, sm, sq, NI, c, slopes)
                common = dict(sq=sq, EPS=1e-6, HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, CAUSAL=True, USE_ALIBI=True)
                gm, gn = (triton.cdiv(N, BM), B * H), (triton.cdiv(N, BN), B * H)
                st = mod._strides
                delta = torch.empty_like(lam); coef = (sq / S.clamp(min=1e-6)).contiguous()
                dk, dv, dq = torch.empty_like(k), torch.empty_like(v), torch.empty_like(q)
                put(f"{name} bwd delta kernel {shape}", bench(lambda: mod._bwd_delta_kernel[gm](
                    q, k, v, do, lam, delta, sl, *st(q), *st(k), *st(v), *st(do), sm, c, N, H, **common)))
                put(f"{name} bwd dkdv kernel {shape}", bench(lambda: mod._bwd_dkdv_kernel[gn](
                    q, k, v, do, lam, coef, delta, sl, dk, dv, *st(q), *st(k), *st(v), *st(do), *st(dk), *st(dv), sm, c, N, H, **common)))
                put(f"{name} bwd dq kernel {shape}", bench(lambda: mod._bwd_dq_kernel[gm](
                    q, k, v, do, lam, coef, delta, sl, dq, *st(q), *st(k), *st(v), *st(do), *st(dq), sm, c, N, H, **common)))
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

    # call site as in CausalSelfAttention.forward: q/k/v are transposed views of one c_attn output
    x = torch.randn(B, N, 3 * H * D, device=dev).to(torch.bfloat16).requires_grad_(True)
    def views():
        a, b, c_ = x.split(H * D, dim=2)
        return [t.view(B, N, H, D).transpose(1, 2) for t in (a, b, c_)]
    def site():
        qq, kk, vv = views()
        return model._stieltjes_fused(qq, kk, vv, sm, sq, NI, window, slopes)
    with torch.no_grad():
        put(f"cand call site fwd (views + contiguous) {shape}", bench(site))
    put(f"cand call site fwd+bwd (views) {shape}", bench(lambda: torch.autograd.grad(site() * 1.0, x, do)))

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
            put("cand _pick_blocks (device query)", bench(lambda: cand._pick_blocks(D, 2, qs.device)))
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
        ms = [x for i, x in ms if i >= 3]
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


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode not in ('micro', 'wrap', 'summary'):
        print(__doc__); sys.exit(2)
    {'micro': micro, 'wrap': wrap, 'summary': summary}[mode](sys.argv[2:])
