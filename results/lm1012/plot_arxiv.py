"""ArXiv long-document ladder (128 Pile-ArXiv papers, first 16385 GPT-2 tokens each), exp-slope rows at their
validation-selected LR. Left: mean loss over the first L tokens vs L. Right: loss by position within the
16k context (1024-token buckets); the prefix is identical at every L, so this is the per-position curve.
    python plot_arxiv.py -> arxiv_lengths.png / .pdf"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [
    ('softmax_xs',               'softmax',                            '#000000', '-'),
    ('entmax_xs',                '1.5-entmax',                         '#ff7f0e', '-'),
    ('asentmax_zi_xs',           'ASEntmax (1.5-entmax + AS)',         '#ff7f0e', '--'),
    ('stieltjes_q4_xs',          'Stieltjes q=4 (dense)',              '#d62728', '-'),
    ('wstieltjes_q4_d2_xs',      'Stieltjes q=4, d=2 (windowed)',      '#1f77b4', '-'),
    ('aswstieltjes_q2_d4_zi_xs', 'Stieltjes q=2, d=4 + AS (windowed)', '#2ca02c', '-'),
]
LRS = ['7.5e-4', '1.5e-3', '3e-3']
TRAIN_LEN = 512

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
for d, label, color, ls in ROWS:
    final = {lr: json.load(open(os.path.join(HERE, d, f'lr{lr}', 'final.json')))['val_loss'] for lr in LRS}
    lr = min(final, key=lambda k: final[k])
    L = json.load(open(os.path.join(HERE, d, f'lr{lr}', 'arxiv.json')))['lengths']
    xs = sorted(int(k) for k in L)
    ys = [L[str(x)]['val_loss'] for x in xs]
    ax.plot(xs, ys, ls, color=color, marker='o', ms=4, lw=1.8, label=f'{label}  (lr {lr})')
    b = L['16384']['bucket_loss']; e = L['16384']['bucket_edges']
    mids = [(e[i] + e[i + 1]) / 2 for i in range(len(b))]
    ax2.plot(mids, b, ls, color=color, marker='.', ms=4, lw=1.8, label=label)
    print(f'{d:26s} lr={lr:7s} ' + '  '.join(f'{x}:{y:.3f}' for x, y in zip(xs, ys)))
ax.axvline(TRAIN_LEN, color='0.6', lw=0.8, ls=':')
ax.set_xscale('log', base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
ax.set_xlabel('context length L (first L tokens of each paper)')
ax.set_ylabel('mean loss over the first L tokens (nats/token)')
ax.set_title('mean loss vs context length', fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc='upper left')
ax.text(TRAIN_LEN * 1.04, ax.get_ylim()[0] + 0.01, 'train length', va='bottom', ha='left', fontsize=8, color='0.4')
ax2.axvline(TRAIN_LEN, color='0.6', lw=0.8, ls=':')
ax2.set_xlabel('token position within the paper (16k context)')
ax2.set_ylabel('loss at that position (nats/token)')
ax2.set_title('loss by position (1024-token buckets)', fontsize=9)
for a in (ax, ax2):
    a.grid(alpha=0.3)
fig.suptitle('Pile ArXiv papers, 51M GPT trained on FineWeb at T=512 (NAPE, exponential slopes); best LR per model by FineWeb val @512; 1 seed', fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'arxiv_lengths.{ext}'), dpi=200)
print('wrote', os.path.join(HERE, 'arxiv_lengths.png'))
