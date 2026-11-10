"""ArXiv loss vs eval length for every finished 125m_fw25 row at lr 1.2e-3.
Colour = map, linestyle = ALiBi/NoPE split. Usage: python plot_arxiv.py (from anywhere)."""
import json, os, re, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../../../..'))
RES = os.path.join(ROOT, 'results/lm1012/125m_fw25')
COLOR = {'softmax': '#4C72B0', 'ssmax': '#DD8452', 'asentmax': '#55A868', 'wstieltjes': '#C44E52'}
STYLE = {6: '-', 4: '--', 2: ':'}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for path in sorted(glob.glob(os.path.join(RES, '*/lr1.2e-3/arxiv.json'))):
    row = path.split('/')[-3]
    m = re.search(r'_a(\d+)$', row)
    n_alibi = int(m.group(1)) if m else 6
    fam = next(k for k in ['wstieltjes', 'asentmax', 'ssmax', 'softmax'] if row.startswith(k))
    for ax, name in zip(axes, ['arxiv.json', 'lengths_16k.json']):
        p = os.path.join(os.path.dirname(path), name)
        if not os.path.exists(p):
            continue
        L = json.load(open(p))['lengths']
        xs = sorted(int(k) for k in L)
        ax.plot(xs, [L[str(x)]['val_loss'] for x in xs], STYLE.get(n_alibi, '-.'), marker='o', ms=3,
                color=COLOR[fam], label=f'{row} ({n_alibi}/{12 - n_alibi})')
for ax, t in zip(axes, ['ArXiv', 'FineWeb val']):
    ax.set_xscale('log', base=2)
    ax.axvline(2048, color='0.7', lw=0.8)
    ax.set_xlabel('eval length (tokens)')
    ax.set_ylabel('loss')
    ax.set_title(f'{t}, 125M fineweb25B, lr 1.2e-3, seed 1337')
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, title='row (ALiBi/NoPE)', title_fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'arxiv_vs_length.png'), dpi=150)
print('wrote', os.path.join(HERE, 'arxiv_vs_length.png'))
