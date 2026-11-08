"""Table of final val, FineWeb 16k, ArXiv 2k..16k, PG-19 16k for the 125M rows in results/lm1012/125m."""
import json, os, sys

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'lm1012', '125m')


def L(path, n):
    try:
        return json.load(open(path))['lengths'][str(n)]['val_loss']
    except (FileNotFoundError, KeyError):
        return float('nan')


rows = sys.argv[1:] or sorted(os.listdir(ROOT))
print('| row | lr | val@2048 | FW 16k | ArXiv 2k | ArXiv 4k | ArXiv 8k | ArXiv 16k | PG-19 16k |')
print('|---|---|---|---|---|---|---|---|---|')
for row in rows:
    rd = os.path.join(ROOT, row)
    if not os.path.isdir(rd):
        continue
    for lr in sorted(os.listdir(rd)):
        d = os.path.join(rd, lr)
        try:
            v = json.load(open(os.path.join(d, 'final.json')))['val_loss']
        except FileNotFoundError:
            continue
        vals = [v, L(f'{d}/lengths_16k.json', 16384)] + [L(f'{d}/arxiv.json', n) for n in (2048, 4096, 8192, 16384)] \
            + [L(f'{d}/pg19.json', 16384)]
        print(f'| `{row}` | {lr[2:]} | ' + ' | '.join(f'{x:.3f}' for x in vals) + ' |')
