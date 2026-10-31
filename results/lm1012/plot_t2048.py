"""T=2048 (stage 3) figure: ArXiv and PG-19 ladders for softmax / ASEntmax / windowed q3 d2 trained at T=2048
(selected LR), with the T=512 3-seed means of the same rows dashed for reference.
    python plot_t2048.py -> t2048_1030.png / .pdf"""
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
    ls = [ladder(f'{base}/lr{lr}{s}', f) for s in ['', '_s1', '_s2'] if os.path.exists(os.path.join(HERE, f'{base}/lr{lr}{s}', f))]
    return [sum(v) / len(v) for v in zip(*ls)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (f, title) in zip(axes, [('arxiv.json', 'ArXiv papers'), ('pg19.json', 'PG-19 books')]):
    for base, lr, label, color in ROWS:
        t2 = {'1.5e-3': '1.5e-3', '3e-3': '3e-3'}
        best = min(['1.5e-3', '3e-3'], key=lambda l: J(f't2048/{base}/lr{l}/final.json')['val_loss'])
        ax.plot(Ls, ladder(f't2048/{base}/lr{best}', f), '-', color=color, marker='o', ms=4, lw=1.8, label=f'{label}, T=2048 (lr {best})')
        ax.plot(Ls, mean_ladder(base, lr, f), '--', color=color, marker='.', ms=4, lw=1.2, alpha=0.7, label=f'{label}, T=512 (3-seed mean)')
    ax.axvline(2048, color='0.6', lw=0.8, ls=':'); ax.axvline(512, color='0.8', lw=0.8, ls=':')
    ax.set_xscale('log', base=2); ax.set_xticks(Ls); ax.set_xticklabels([str(x) for x in Ls])
    ax.set_xlabel('context length (first L tokens of each document)'); ax.set_ylabel('mean loss (nats/token)')
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
fig.suptitle('Training length 2048 vs 512, 51M GPT, 1B FineWeb tokens, NAPE exp slopes', fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f't2048_1030.{ext}'), dpi=200)
print('wrote', os.path.join(HERE, 't2048_1030.png'))
