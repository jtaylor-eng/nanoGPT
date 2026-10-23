"""Val loss vs eval context length, each row at its validation-selected LR (final val loss @512).
Two panels: left = synthetic-task NAPE slopes (1/h), right = the paper's exponential ALiBi slopes (2^-8h/4).
    python plot_lengths.py   ->  lengths_best_lr.png / .pdf next to this file"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = {  # base row -> (label, color, linestyle)
    'softmax':               ('softmax',                            '#000000', '-'),
    'softmax_as_zi':         ('softmax + AS (control)',             '#7f7f7f', '--'),
    'entmax':                ('1.5-entmax',                         '#ff7f0e', '-'),
    'asentmax_zi':           ('ASEntmax (1.5-entmax + AS)',         '#ff7f0e', '--'),
    'stieltjes_q4':          ('Stieltjes q=4 (dense)',              '#d62728', '-'),
    'wstieltjes_q4_d2':      ('Stieltjes q=4, d=2 (windowed)',      '#1f77b4', '-'),
    'aswstieltjes_q2_d4_zi': ('Stieltjes q=2, d=4 + AS (windowed)', '#2ca02c', '-'),
}
PANELS = [
    ('NAPE slopes 1/h (asentmax synthetic setting)', ''),
    ('NAPE exponential ALiBi slopes (paper LM setting)', '_xs'),
]
LRS = ['7.5e-4', '1.5e-3', '3e-3']
TRAIN_LEN = 512

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (title, suffix) in zip(axes, PANELS):
    xs = None
    for base, (label, color, ls) in STYLE.items():
        d = base + suffix
        if not os.path.isdir(os.path.join(HERE, d)):
            continue
        final = {lr: json.load(open(os.path.join(HERE, d, f'lr{lr}', 'final.json')))['val_loss'] for lr in LRS}
        lr = min(final, key=lambda k: final[k])
        L = json.load(open(os.path.join(HERE, d, f'lr{lr}', 'lengths.json')))['lengths']
        xs = sorted(int(k) for k in L)
        ys = [L[str(x)]['val_loss'] for x in xs]
        ax.plot(xs, ys, ls, color=color, marker='o', ms=4, lw=1.8, label=f'{label}  (lr {lr})')
        print(f'{d:26s} lr={lr:7s} ' + '  '.join(f'{x}:{y:.4f}' for x, y in zip(xs, ys)))
    ax.axvline(TRAIN_LEN, color='0.6', lw=0.8, ls=':')
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel('evaluation context length (tokens)')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
axes[0].set_ylabel('validation loss (nats/token)')
axes[0].text(TRAIN_LEN * 1.04, axes[0].get_ylim()[0] + 0.005, 'train length', va='bottom', ha='left', fontsize=8, color='0.4')
fig.suptitle('Length extrapolation, 51M GPT on FineWeb (trained at T=512); best LR per model by val loss @512; 1 seed', fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'lengths_best_lr.{ext}'), dpi=200)
print('wrote', os.path.join(HERE, 'lengths_best_lr.png'))
