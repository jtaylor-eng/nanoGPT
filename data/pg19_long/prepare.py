"""Build a long-document eval set from the PG-19 test split (deepmind/pg19 via emozilla/pg19 parquet; Project
Gutenberg books published before 1919, plain English prose). Same layout as data/arxiv_long: n_docs blocks of
exactly min_tokens GPT-2 tokens, the first min_tokens tokens of each book. Books are long (median ~100k tokens),
so to avoid 128 near-identical front matter samples we skip the first `skip_tokens` tokens of each book.

    python data/pg19_long/prepare.py --n_docs=128 --min_tokens=16385 --skip_tokens=2048
"""
import os, sys, io, urllib.request
import numpy as np
import pyarrow.parquet as pq
import tiktoken

n_docs = 128
min_tokens = 16385
skip_tokens = 2048
out = os.path.dirname(os.path.abspath(__file__))
for arg in sys.argv[1:]:
    k, v = arg.lstrip('-').split('=', 1); globals()[k] = type(globals()[k])(v)

url = 'https://huggingface.co/datasets/emozilla/pg19/resolve/main/data/test-00000-of-00001-29a571947c0b5ccc.parquet'
buf = urllib.request.urlopen(url, timeout=300).read()
t = pq.read_table(io.BytesIO(buf))
print(t.schema.names, t.num_rows)
enc = tiktoken.get_encoding('gpt2')
# the test split has 100 books (91 long enough): take up to two non-overlapping segments per book
# (first pass: tokens [skip, skip+min); second pass: [skip+min+skip, skip+2*min+skip)) until n_docs
toks, seen = [], 0
for doc in t.column('text').to_pylist():
    seen += 1
    if len(doc) < 3 * (min_tokens + skip_tokens):
        continue
    toks.append(enc.encode_ordinary(doc))
kept, nseg = [], [0, 0]
for pass_ in range(2):
    start = skip_tokens + pass_ * (min_tokens + skip_tokens)
    for ids in toks:
        if len(kept) >= n_docs:
            break
        if len(ids) >= start + min_tokens:
            kept.append(np.array(ids[start:start + min_tokens], dtype=np.uint16)); nseg[pass_] += 1
print(f'seen {seen} books, {len(toks)} tokenised, segments per pass {nseg}, kept {len(kept)}')
assert len(kept) == n_docs, f'only {len(kept)} segments'
arr = np.concatenate(kept)
arr.tofile(os.path.join(out, 'pg19_long.bin'))
with open(os.path.join(out, 'README.txt'), 'w') as f:
    f.write(f'{n_docs} segments x {min_tokens} GPT-2 tokens from the 100-book PG-19 test split (emozilla/pg19 parquet): '
            f'{nseg[0]} books give tokens [{skip_tokens}, {skip_tokens + min_tokens}), then {nseg[1]} of them give a second '
            f'non-overlapping segment starting at {2 * skip_tokens + min_tokens}. File order.\n')
print('wrote', arr.shape, arr.nbytes / 1e6, 'MB')
