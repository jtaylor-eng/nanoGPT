"""The fixed-(q, d) Triton path must stay bit-identical after the PER_HEAD addition (direction learnable-qd).
    git show <old sha>:stieltjes/triton_stieltjes.py > /some/dir/triton_stieltjes_old.py
    python tests/test_fixed_path_identical.py /some/dir/triton_stieltjes_old.py
Compares O, dQ, dK, dV with torch.equal for windowed and dense rows, with and without NAPE slopes. CUDA required."""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from stieltjes import triton_stieltjes as new
from model import nape_slopes

spec = importlib.util.spec_from_file_location('triton_stieltjes_old', sys.argv[1])
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

torch.manual_seed(0)
ok = True
for dtype in (torch.float32, torch.bfloat16):
    for sq, window in ((3.0, 2.0), (4.0, 2.0), (4.0, None), (2.5, 3.0)):
        for use_slopes in (False, True):
            B, H, N, D = 2, 8, 300, 64
            q, k, v = (torch.randn(B, H, N, D, device='cuda', dtype=dtype) * 1.5 for _ in range(3))
            g = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
            sl = nape_slopes(H, 'exp').cuda() if use_slopes else None
            outs = []
            for mod in (old, new):
                qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
                o = mod.stieltjes_attention(qq, kk, vv, causal=True, stieltjes_q=sq, num_iter=15, window=window, alibi_slopes=sl)
                o.backward(g)
                outs.append((o.detach(), qq.grad, kk.grad, vv.grad))
            same = all(torch.equal(a, b) for a, b in zip(*outs))
            ok &= same
            print(f"{'OK ' if same else 'BAD'} {str(dtype)[6:]:8s} q={sq} d={window} slopes={use_slopes}")
print('ALL IDENTICAL' if ok else 'DIFFERENCES')
sys.exit(0 if ok else 1)
