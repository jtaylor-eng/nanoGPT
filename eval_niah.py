"""
Single-needle retrieval at length (RULER S-NIAH-1 style passkey) on a nanoGPT checkpoint (fork addition).

Each sample is exactly L tokens: PG-19 filler (doc i % n_docs of the eval set) with the needle
"\nThe special magic number is NNNNN.\n" inserted at depth ~U(bin) (snapped forward to the next sentence end),
then the question "\n\nWhat is the special magic number mentioned in the text above? The special magic number is"
and the answer " NNNNN". correct = greedy decoding (<= 8 new tokens, batched with early exit, see score()) emits
optional whitespace, then exactly NNNNN, then a non-digit. The 125M models answer "\nNNNNN" with the needle's digit
pieces, which the teacher-forced chains of e002/e003 mis-scored. log p(' NNNNN') is kept as a continuous score. Needles, numbers, depths and filler are seeded
by (seed, L) so every checkpoint sees identical samples. `gen_check` > 0 runs true greedy decoding (8 tokens)
on that many samples at the shortest length, one at a time, and reports agreement with the batched score.

    python eval_niah.py --out_dir=<run> --data_file=<pg19_long.bin> [--lengths=2048,4096,8192,16384] [--n_per_length=200]
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
n_per_length = 200
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


def tf_check(pairs):
    """pairs: [(prefix, answer)]. One right-padded forward; per pair: (argmax chain == answer and the argmax after
    it is not digit-led, log p(answer | prefix), argmax at the end of the prefix)."""
    T = max(len(p) + len(a) for p, a in pairs)
    x = torch.full((len(pairs), (T + 63) // 64 * 64), enc.eot_token, dtype=torch.long, device=device)
    for j, (p, a) in enumerate(pairs):
        x[j, :len(p) + len(a)] = torch.tensor(p + a, device=device)
    with torch.no_grad(), ctx:
        h = hidden(model, x)
    out = []
    for j, (p, a) in enumerate(pairs):
        P, m = len(p), len(a)
        with torch.no_grad(), ctx:
            lp = F.log_softmax(model.lm_head(h[j, P - 1:P + m]).float(), dim=-1)   # predicts a[0..m-1], then next
        tgt = torch.tensor(a, device=device)
        am = lp.argmax(-1)
        out.append((bool((am[:m] == tgt).all()) and not bool(digit_led[am[m]]),
                    lp[:m].gather(-1, tgt[:, None]).sum().item(), int(am[0])))
    return out


def verdict(g, num):
    """Greedy text so far -> True/False once decided (digits ended, or first non-space char is not a digit), else None."""
    m = re.match(r'\s*(\d+)(\D)?', g)
    if m is None:
        return False if g.strip() else None
    return int(m.group(1)) == num if m.group(2) else None


def score(batch):
    """True greedy exact match (same rule as greedy() + gen_check, up to 8 new tokens), batched with early exit.
    Forward 1 is teacher-forced on prompt + ' NNNNN': its argmax at the prompt end is greedy token 1, and it gives
    the continuous score log p(' NNNNN' | prompt). Then every undecided sample takes one more greedy step per forward
    (all prompts have the same length, so live samples share one length). The 125M models answer '\nNNNNN' with the
    needle's own digit pieces (non-canonical BPE), which a teacher-forced chain cannot score (e003)."""
    s1 = tf_check([(s['toks'][:-s['na']], s['toks'][-s['na']:]) for s in batch])
    gen = [[a0] for _, _, a0 in s1]
    res = [verdict(enc.decode(g), s['num']) for g, s in zip(gen, batch)]
    for _ in range(7):
        live = [j for j, r in enumerate(res) if r is None]
        if not live:
            break
        T = len(batch[0]['toks']) - batch[0]['na'] + len(gen[live[0]])
        x = torch.full((len(live), (T + 63) // 64 * 64), enc.eot_token, dtype=torch.long, device=device)
        for i, j in enumerate(live):
            x[i, :T] = torch.tensor(batch[j]['toks'][:-batch[j]['na']] + gen[j], device=device)
        with torch.no_grad(), ctx:
            nxt = model.lm_head(hidden(model, x)[:, T - 1]).float().argmax(-1).tolist()
        for i, j in enumerate(live):
            gen[j].append(min(nxt[i], enc.n_vocab - 1))
            res[j] = verdict(enc.decode(gen[j]), batch[j]['num'])
    out = []
    for s, (ok1, lp1, a0), g, r in zip(batch, s1, gen, res):
        text = enc.decode(g)
        if r is None:                                   # 8 tokens and still undecided: same regex as gen_check
            m = re.match(r'\s*(\d+)', text)
            r = bool(m) and int(m.group(1)) == s['num']
        out.append(dict(correct=bool(r), correct_space=ok1, logp=lp1, gen=text, n_gen=len(g),
                        first_tok=enc.decode([a0]), first_tok_ok=a0 == s['toks'][-s['na']]))
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
                             acc_space=float(np.mean([r['correct_space'] for r in scored])),
                             mean_n_gen=float(np.mean([r['n_gen'] for r in scored])),
                             first_ws_rate=float(np.mean([r['first_tok'].strip() == '' for r in scored])),
                             acc_by_bin=by_bin, n=len(scored), eval_s=time.time() - t0,
                             samples=[dict(num=s['num'], bin=s['bin'], depth=round(s['depth_tok'], 4), **r)
                                      for s, r in zip(samples, scored)])
    print(f'L={L}: acc {acc:.3f}  logp {res["lengths"][L]["mean_logp"]:.2f}  tok/sample {res["lengths"][L]["mean_n_gen"]:.2f}  by bin '
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
        gens.append(dict(num=s['num'], gen=g, gen_correct=hit, batch_correct=r['correct']))
    res['gen_check'] = dict(L=L, n=gen_check, agree=agree, samples=gens)
    print(f'gen_check L={L}: batched vs single greedy agree {agree}/{gen_check}; e.g. {gens[0]}', flush=True)

with open(os.path.join(out_dir, out_name), 'w') as f:
    json.dump(res, f, indent=1)
print('wrote', os.path.join(out_dir, out_name))
