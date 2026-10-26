"""Val loss vs eval context length 512..16384, exp-slope rows at their validation-selected LR (final val @512).
Left: absolute loss. Right: loss minus each row's own loss at 512 (extrapolation cost).
    python plot_lengths_16k.py  ->  lengths_16k.png / .pdf"""
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
    L = json.load(open(os.path.join(HERE, d, f'lr{lr}', 'lengths_16k.json')))['lengths']
    xs = sorted(int(k) for k in L)
    ys = [L[str(x)]['val_loss'] for x in xs]
    ax.plot(xs, ys, ls, color=color, marker='o', ms=4, lw=1.8, label=f'{label}  (lr {lr})')
    ax2.plot(xs, [y - ys[0] for y in ys], ls, color=color, marker='o', ms=4, lw=1.8, label=label)
    print(f'{d:26s} lr={lr:7s} ' + '  '.join(f'{x}:{y:.3f}' for x, y in zip(xs, ys)))
for a in (ax, ax2):
    a.axvline(TRAIN_LEN, color='0.6', lw=0.8, ls=':')
    a.set_xscale('log', base=2)
    a.set_xticks(xs); a.set_xticklabels([str(x) for x in xs])
    a.set_xlabel('evaluation context length (tokens)')
    a.grid(alpha=0.3)
ax2.axhline(0, color='0.6', lw=0.8)
ax.set_ylabel('validation loss (nats/token)')
ax2.set_ylabel('loss minus own loss at 512')
ax.set_title('absolute', fontsize=9); ax2.set_title('extrapolation cost relative to 512', fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc='upper left')
ax.text(TRAIN_LEN * 1.04, ax.get_ylim()[0] + 0.005, 'train length', va='bottom', ha='left', fontsize=8, color='0.4')
fig.suptitle('Length extrapolation to 32x, 51M GPT on FineWeb (T=512, NAPE with exponential ALiBi slopes); best LR per model by val @512; 1 seed', fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'lengths_16k.{ext}'), dpi=200)
print('wrote', os.path.join(HERE, 'lengths_16k.png'))
