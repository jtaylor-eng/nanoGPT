"""
Single-needle retrieval at length (RULER S-NIAH-1 style passkey) on a nanoGPT checkpoint (fork addition).

Each sample is exactly L tokens: PG-19 filler (doc i % n_docs of the eval set) with the needle
"\nThe special magic number is NNNNN.\n" inserted at depth ~U(bin) (snapped forward to the next sentence end),
then the question "\n\nWhat is the special magic number mentioned in the text above? The special magic number is"
and the answer " NNNNN". One teacher-forced forward per sample gives:
  correct = argmax matches every answer token AND the argmax after the answer is not a digit-led token,
which is exactly "greedy decoding emits ' NNNNN' and then stops the number" (exact match of the immediate
continuation), plus the answer log-prob as a continuous score. Needles, numbers, depths and filler are seeded
by (seed, L) so every checkpoint sees identical samples. `gen_check` > 0 runs true greedy decoding (8 tokens)
on that many samples at the shortest length and reports agreement with the teacher-forced score.

    python eval_niah.py --out_dir=<run> --data_file=<pg19_long.bin> [--lengths=2048,4096,8192,16384] [--n_per_length=300]
writes <run>/<out_name> (default niah.json)
"""
import os, sys, json, time, re
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from eval_downstream import load_model, hidden

out_dir = 'out'
data_file = ''
doc_len = 16385
lengths = '2048,4096,8192,16384'
n_per_length = 300
n_bins = 10
seed = 1234
batch_tokens = 16384
gen_check = 20
out_name = 'niah.json'
device = 'cuda'
for arg in sys.argv[1:]:
    k, v = arg.lstrip('-').split('=', 1)
    g = globals()[k]
    globals()[k] = type(g)(v) if not isinstance(g, str) else v

enc = tiktoken.get_encoding('gpt2')
model, args, it = load_model(out_dir, device)
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
docs = np.memmap(data_file, dtype=np.uint16, mode='r')
n_docs = len(docs) // doc_len
QUESTION = enc.encode_ordinary('\n\nWhat is the special magic number mentioned in the text above? The special magic number is')
digit_led = torch.tensor([bool(re.match(r'\s*\d', enc.decode([t]))) for t in range(enc.n_vocab)]
                         + [False] * (args['vocab_size'] - enc.n_vocab), device=device)
sent_end = {t for t in range(enc.n_vocab) if re.search(r'[.!?]["\')]*\s*$', enc.decode([t]))}


def make_sample(L, i, rng):
    num = int(rng.integers(10000, 100000))
    b = i % n_bins
    depth = (b + rng.random()) / n_bins
    needle = enc.encode_ordinary(f'\nThe special magic number is {num}.\n')
    answer = enc.encode_ordinary(f' {num}')
    n_fill = L - len(needle) - len(QUESTION) - len(answer)
    fill = docs[(i % n_docs) * doc_len:(i % n_docs) * doc_len + n_fill].astype(np.int64).tolist()
    pos = int(depth * n_fill)
    if pos > 0:
        pos = next((p for p in range(pos, min(pos + 200, n_fill)) if fill[p - 1] in sent_end), pos)  # sentence end
    toks = fill[:pos] + needle + fill[pos:] + QUESTION + answer
    assert len(toks) == L
    return dict(num=num, bin=b, depth=depth, depth_tok=pos / n_fill, toks=toks, na=len(answer))


def score(batch):
    x = torch.tensor([s['toks'] for s in batch], device=device)
    with torch.no_grad(), ctx:
        h = hidden(model, x)
    out = []
    for j, s in enumerate(batch):
        L, na = x.shape[1], s['na']
        with torch.no_grad(), ctx:
            lp = F.log_softmax(model.lm_head(h[j, L - 1 - na:L]).float(), dim=-1)   # predicts answer[0..na-1], then next
        tgt = x[j, L - na:]
        am = lp.argmax(-1)
        ok = bool((am[:na] == tgt).all()) and not bool(digit_led[am[na]])
        out.append(dict(correct=ok, logp=lp[:na].gather(-1, tgt[:, None]).sum().item(),
                        first_tok_ok=bool(am[0] == tgt[0])))
    return out


def greedy(toks, n_new=8):
    seq = list(toks)
    for _ in range(n_new):
        T = len(seq)
        Lp = (T + 63) // 64 * 64                     # right-pad: causal, so padding cannot affect position T-1
        x = torch.full((1, Lp), enc.eot_token, dtype=torch.long, device=device)
        x[0, :T] = torch.tensor(seq, device=device)
        with torch.no_grad(), ctx:
            nxt = int(model.lm_head(hidden(model, x)[0, T - 1]).float().argmax())
        seq.append(nxt)
    return enc.decode(seq[len(toks):])


res = {'iter': it, 'model_args': args, 'cfg': dict(lengths=lengths, n_per_length=n_per_length, n_bins=n_bins,
       seed=seed, data_file=data_file), 'lengths': {}}
for L in [int(v) for v in lengths.split(',')]:
    rng = np.random.default_rng([seed, L])
    samples = [make_sample(L, i, rng) for i in range(n_per_length)]
    bs = max(1, batch_tokens // L)
    t0 = time.time()
    scored = []
    for k in range(0, len(samples), bs):
        scored += score(samples[k:k + bs])
    acc = np.mean([r['correct'] for r in scored])
    by_bin = [float(np.mean([r['correct'] for r, s in zip(scored, samples) if s['bin'] == b])) for b in range(n_bins)]
    res['lengths'][L] = dict(acc=float(acc), mean_logp=float(np.mean([r['logp'] for r in scored])),
                             first_tok_acc=float(np.mean([r['first_tok_ok'] for r in scored])),
                             acc_by_bin=by_bin, n=len(scored), eval_s=time.time() - t0,
                             samples=[dict(num=s['num'], bin=s['bin'], depth=round(s['depth_tok'], 4), **r)
                                      for s, r in zip(samples, scored)])
    print(f'L={L}: acc {acc:.3f}  logp {res["lengths"][L]["mean_logp"]:.2f}  by bin '
          + ' '.join(f'{a:.2f}' for a in by_bin) + f'  ({time.time() - t0:.0f}s)', flush=True)

if gen_check > 0:
    L = min(int(v) for v in lengths.split(','))
    rng = np.random.default_rng([seed, L])
    samples = [make_sample(L, i, rng) for i in range(gen_check)]
    agree, gens = 0, []
    for s, r in zip(samples, res['lengths'][L]['samples'][:gen_check]):
        g = greedy(s['toks'][:-s['na']])
        m = re.match(r'\s*(\d+)', g)
        hit = bool(m) and int(m.group(1)) == s['num']
        agree += hit == r['correct']
        gens.append(dict(num=s['num'], gen=g, gen_correct=hit, tf_correct=r['correct']))
    res['gen_check'] = dict(L=L, n=gen_check, agree=agree, samples=gens)
    print(f'gen_check L={L}: teacher-forced vs greedy agree {agree}/{gen_check}; e.g. {gens[0]}', flush=True)

with open(os.path.join(out_dir, out_name), 'w') as f:
    json.dump(res, f, indent=1)
print('wrote', os.path.join(out_dir, out_name))
