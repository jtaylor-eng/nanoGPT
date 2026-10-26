"""Build a long-document eval set from the Pile ArXiv test split (ArmelR/the-pile-splitted, LaTeX sources).

Keeps documents with >= min_tokens GPT-2 tokens (the paper's protocol uses >=16K-token documents) and writes
<out>/arxiv_long.bin as uint16 tokens laid out as n_docs consecutive blocks of exactly (min_tokens) tokens
(the first min_tokens tokens of each document; no <|endoftext|> inside a block). eval_lengths.py with
--data_file=arxiv_long.bin --doc_len=16385 then takes, for each eval length L, the FIRST L+1 tokens of every
document, so every length scores the same documents and a longer context only ever adds earlier text of the
same paper.

    python data/arxiv_long/prepare.py --n_docs=128 --min_tokens=16385 --shards=0-11
"""
import os, sys, io, urllib.request
import numpy as np
import pyarrow.ipc as ipc
import tiktoken

n_docs = 128
min_tokens = 16385        # 16384 context + 1 target
shards = '0-11'
out = os.path.dirname(os.path.abspath(__file__))
for arg in sys.argv[1:]:
    k, v = arg.lstrip('-').split('=', 1); globals()[k] = type(globals()[k])(v)

enc = tiktoken.get_encoding('gpt2')
a, b = (int(x) for x in shards.split('-'))
kept, seen = [], 0
for i in range(a, b + 1):
    url = f'https://huggingface.co/datasets/ArmelR/the-pile-splitted/resolve/main/data/ArXiv/test/data-{i:05d}-of-00096.arrow'
    buf = urllib.request.urlopen(url, timeout=120).read()
    t = ipc.open_stream(io.BytesIO(buf)).read_all()
    for doc in t.column('text').to_pylist():
        seen += 1
        if len(doc) < 3 * min_tokens:        # cheap char filter (LaTeX is ~3-4 chars/token)
            continue
        ids = enc.encode_ordinary(doc)
        if len(ids) >= min_tokens:
            kept.append(np.array(ids[:min_tokens], dtype=np.uint16))
            if len(kept) >= n_docs:
                break
    print(f'shard {i}: seen {seen} docs, kept {len(kept)}', flush=True)
    if len(kept) >= n_docs:
        break
assert len(kept) == n_docs, f'only {len(kept)} docs >= {min_tokens} tokens; widen --shards'
arr = np.concatenate(kept)
arr.tofile(os.path.join(out, 'arxiv_long.bin'))
with open(os.path.join(out, 'README.txt'), 'w') as f:
    f.write(f'{n_docs} docs x {min_tokens} GPT-2 tokens from Pile ArXiv test shards {shards} (ArmelR/the-pile-splitted); '
            f'first {min_tokens} tokens of each doc with >= {min_tokens} tokens; {seen} docs scanned.\n')
print('wrote', arr.shape, arr.nbytes / 1e6, 'MB')
