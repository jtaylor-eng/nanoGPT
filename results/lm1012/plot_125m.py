"""Stage 4 figure (125M, T=2048, 2.5B tokens, 1 seed, selected LR): ArXiv and PG-19 ladders with the 51M T=2048
3-seed means dashed; third panel = loss by position within the 16k ArXiv context.
    python plot_125m.py -> stage4_125m.png / .pdf"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p)))
Ls = [512, 1024, 2048, 4096, 8192, 16384]
ROWS = [('softmax_xs', '3e-3', 'softmax', '#000000'), ('asentmax_zi_xs', '1.5e-3', 'ASEntmax', '#ff7f0e'),
        ('wstieltjes_q3_d2_xs', '1.5e-3', 'windowed Stieltjes q3 d2', '#1f77b4')]
def ladder(d, f): L = J(f'{d}/{f}')['lengths']; return [L[str(x)]['val_loss'] for x in Ls]
def mean_ladder(base, lr, f):
    ls = [ladder(f't2048/{base}/lr{lr}{s}', f) for s in ['', '_s1', '_s2']]
    return [sum(v) / len(v) for v in zip(*ls)]
def sel(base): return min(['6e-4', '1.2e-3'], key=lambda l: J(f'125m/{base}/lr{l}/final.json')['val_loss'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (f, title) in zip(axes[:2], [('arxiv.json', 'ArXiv papers'), ('pg19.json', 'PG-19 books')]):
    for base, lr51, label, color in ROWS:
        ax.plot(Ls, ladder(f'125m/{base}/lr{sel(base)}', f), '-', color=color, marker='o', ms=4, lw=1.8, label=f'{label}, 125M (lr {sel(base)})')
        ax.plot(Ls, mean_ladder(base, lr51, f), '--', color=color, marker='.', ms=4, lw=1.2, alpha=0.6, label=f'{label}, 51M (3-seed mean)')
    ax.axvline(2048, color='0.6', lw=0.8, ls=':')
    ax.set_xscale('log', base=2); ax.set_xticks(Ls); ax.set_xticklabels([str(x) for x in Ls])
    ax.set_xlabel('context length (first L tokens of each document)'); ax.set_ylabel('mean loss (nats/token)')
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
ax = axes[2]
for base, _, label, color in ROWS:
    L = J(f'125m/{base}/lr{sel(base)}/arxiv.json')['lengths']['16384']
    e = L['bucket_edges']; b = L['bucket_loss']
    ax.plot([(e[i] + e[i + 1]) / 2 for i in range(len(b))], b, '-', color=color, marker='.', ms=4, lw=1.8, label=label)
ax.axvline(2048, color='0.6', lw=0.8, ls=':')
ax.set_xlabel('token position within the paper (16k context)'); ax.set_ylabel('loss at that position (nats/token)')
ax.set_title('125M: ArXiv loss by position at 16k', fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
fig.suptitle('Stage 4: 125M (12L x 12H x 768), T=2048, 2.5B FineWeb tokens, 1 seed; dashed = 51M at T=2048', fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'stage4_125m.{ext}'), dpi=200)
print('wrote', os.path.join(HERE, 'stage4_125m.png'))
