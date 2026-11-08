"""figures/arxiv_vs_length.png: ArXiv loss vs eval length, 125M, best LR per row (by final val)."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..', 'results', 'lm1012', '125m')
ROWS = ['softmax_xs', 'asentmax_zi_xs', 'wstieltjes_q3_d2_xs', 'wstieltjes_q3_d3_xs',
        'aswstieltjes_q3_d2_zi_xs', 'lwstieltjes_q3_d2_xs']
STYLE = {'aswstieltjes_q3_d2_zi_xs': dict(lw=2.5), 'lwstieltjes_q3_d2_xs': dict(lw=2.5)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for row in ROWS:
    rd = os.path.join(ROOT, row)
    if not os.path.isdir(rd):
        continue
    best = None
    for lr in os.listdir(rd):
        f = os.path.join(rd, lr, 'final.json')
        if os.path.exists(f) and os.path.exists(os.path.join(rd, lr, 'arxiv.json')):
            v = json.load(open(f))['val_loss']
            if best is None or v < best[0]:
                best = (v, lr)
    if best is None:
        continue
    for ax, fname in zip(axes, ['arxiv.json', 'pg19.json']):
        L = json.load(open(os.path.join(rd, best[1], fname)))['lengths']
        ns = sorted(int(n) for n in L)
        ax.plot(ns, [L[str(n)]['val_loss'] for n in ns], marker='o', label=f'{row} {best[1][2:]}',
                **STYLE.get(row, dict(lw=1.2)))
for ax, t in zip(axes, ['ArXiv', 'PG-19']):
    ax.set_xscale('log', base=2)
    ax.axvline(2048, color='grey', ls=':', lw=1)
    ax.set_xlabel('eval length (train T=2048)')
    ax.set_ylabel('loss')
    ax.set_title(f'125M, 2.5B tok, seed 1337, best LR by val: {t}')
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.tight_layout()
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'arxiv_vs_length.png')
fig.savefig(out, dpi=130)
print(out)
