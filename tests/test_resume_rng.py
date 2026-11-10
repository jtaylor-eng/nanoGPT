"""Resume equivalence (fork addition): a run chained over resumes draws the same training batches as an
uninterrupted run. Covers the final-save path (max_iters) and the wall-limit path (exit 4).
    python tests/test_resume_rng.py [attn_type]      # CUDA if available, else CPU
Run A: iters 0..400 in one process. Run B: 0..150 (final save) -> resume, wall stop after 1 s -> resume to 400.
Before the RNG fix every resume reseeded and replayed segment 1's batch indices."""
import os, sys, json, subprocess, tempfile
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
attn = sys.argv[1] if len(sys.argv) > 1 else 'softmax'
tmp = tempfile.mkdtemp(prefix='resume_rng_')
data = os.path.join(tmp, 'data'); os.makedirs(data)
rng = np.random.default_rng(0)
rng.integers(0, 50000, 200_000, dtype=np.uint16).tofile(os.path.join(data, 'train.bin'))
rng.integers(0, 50000, 20_000, dtype=np.uint16).tofile(os.path.join(data, 'val.bin'))
cuda = torch.cuda.is_available()
BASE = ['--data_root=' + data, '--n_layer=2', '--n_head=2', '--n_embd=64', '--block_size=64', '--batch_size=4',
        '--gradient_accumulation_steps=2', '--lr_decay_iters=400', '--warmup_iters=5', '--eval_interval=50',
        '--eval_iters=2', '--log_interval=10', '--final_eval_iters=2', '--compile=False', '--attn_type=' + attn,
        '--device=' + ('cuda' if cuda else 'cpu'), '--dtype=' + ('bfloat16' if cuda else 'float32')]

def run(out, log, *extra):
    r = subprocess.run([sys.executable, os.path.join(REPO, 'train.py'), *BASE, '--out_dir=' + out,
                        '--batch_log=' + log, *extra], cwd=tmp, capture_output=True, text=True)
    assert r.returncode in (0, 4), r.stdout[-2000:] + r.stderr[-3000:]
    return r.returncode, r.stdout

def lines(p):
    return open(p).read().splitlines()

A, B = os.path.join(tmp, 'A'), os.path.join(tmp, 'B')
run(A, A + '.log', '--max_iters=400')
rc1, _ = run(B, B + '1.log', '--max_iters=150')
rc2, o2 = run(B, B + '2.log', '--max_iters=400', '--init_from=resume', '--wall_limit_s=1')
rc3, o3 = run(B, B + '3.log', '--max_iters=400', '--init_from=resume')
assert (rc1, rc2, rc3) == (0, 4, 0), (rc1, rc2, rc3)
assert 'restored RNG state' in o2 and 'restored RNG state' in o3
# each non-final segment prefetches one batch that the next segment redraws: drop it
b = lines(B + '1.log')[:-1] + lines(B + '2.log')[:-1] + lines(B + '3.log')
a = lines(A + '.log')
same = sum(x == y for x, y in zip(a, b))
fa, fb = json.load(open(os.path.join(A, 'final.json'))), json.load(open(os.path.join(B, 'final.json')))
print(f"attn={attn} device={'cuda' if cuda else 'cpu'} batches A={len(a)} B={len(b)} identical={same}")
print(f"final val A={fa['val_loss']:.5f} B={fb['val_loss']:.5f} train A={fa['train_loss']:.5f} B={fb['train_loss']:.5f}")
ok = len(a) == len(b) == same and abs(fa['val_loss'] - fb['val_loss']) < 0.02
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
