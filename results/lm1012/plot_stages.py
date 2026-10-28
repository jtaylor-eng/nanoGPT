"""Stage 1-2 figure: (left) ArXiv ladder per seed for softmax / ASEntmax / windowed q4/d2, (middle) the AS
disentangle at lr 1.5e-3, (right) q x d grid heatmaps of ArXiv 16k loss and FineWeb val@512.
    python plot_stages.py -> stages_1028.png / .pdf"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p)))
Ls = [512, 1024, 2048, 4096, 8192, 16384]
def ladder(d): L = J(d + '/arxiv.json')['lengths']; return [L[str(x)]['val_loss'] for x in Ls]

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
# left: seeds
for base, lr, label, color in [('softmax_xs', '3e-3', 'softmax', '#000000'), ('asentmax_zi_xs', '1.5e-3', 'ASEntmax', '#ff7f0e'),
                               ('wstieltjes_q4_d2_xs', '1.5e-3', 'windowed Stieltjes q4 d2', '#1f77b4')]:
    for i, s in enumerate(['', '_s1', '_s2']):
        a1.plot(Ls, ladder(f'{base}/lr{lr}{s}'), '-', color=color, marker='o', ms=3, lw=1.2, alpha=0.8, label=label if i == 0 else None)
a1.set_title('ArXiv papers, 3 seeds per row (selected LR)', fontsize=9)
# middle: AS disentangle at 1.5e-3
for base, label, color, ls in [('wstieltjes_q4_d2_xs', 'q4 d2', '#1f77b4', '-'), ('aswstieltjes_q4_d2_zi_xs', 'q4 d2 + AS', '#1f77b4', '--'),
                               ('wstieltjes_q2_d4_xs', 'q2 d4', '#2ca02c', '-'), ('aswstieltjes_q2_d4_zi_xs', 'q2 d4 + AS', '#2ca02c', '--')]:
    a2.plot(Ls, ladder(f'{base}/lr1.5e-3'), ls, color=color, marker='o', ms=3, lw=1.6, label=label)
a2.set_title('AS scale vs (q, d), lr 1.5e-3, 1 seed', fontsize=9)
for a in (a1, a2):
    a.set_xscale('log', base=2); a.set_xticks(Ls); a.set_xticklabels([str(x) for x in Ls])
    a.set_xlabel('context length (first L tokens of each paper)'); a.set_ylabel('mean loss (nats/token)')
    a.axvline(512, color='0.6', lw=0.8, ls=':'); a.grid(alpha=0.3); a.legend(fontsize=7.5, frameon=False)
# right: grid
qs, ds = [3, 4, 6], ['1.5', '2', '3']
M16 = np.full((3, 3), np.nan); M512 = np.full((3, 3), np.nan)
for i, q in enumerate(qs):
    for j, d in enumerate(ds):
        p = f'wstieltjes_q{q}_d{d}_xs/lr1.5e-3'
        if os.path.exists(os.path.join(HERE, p, 'arxiv.json')):
            M16[i, j] = ladder(p)[-1]; M512[i, j] = J(p + '/final.json')['val_loss']
im = a3.imshow(M16, cmap='viridis_r', origin='lower')
for i in range(3):
    for j in range(3):
        a3.text(j, i, f'{M16[i, j]:.2f}\n({M512[i, j]:.3f})', ha='center', va='center', fontsize=8, color='w' if M16[i, j] > 4.6 else 'k')
a3.set_xticks(range(3)); a3.set_xticklabels([f'd={d}' for d in ds]); a3.set_yticks(range(3)); a3.set_yticklabels([f'q={q}' for q in qs])
a3.set_title('windowed Stieltjes q x d, lr 1.5e-3:\nArXiv loss @16k (FineWeb val @512)', fontsize=9)
fig.colorbar(im, ax=a3, fraction=0.046, label='ArXiv loss @16k')
fig.suptitle('51M GPT, FineWeb 1B tokens, T=512, NAPE exp slopes', fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'stages_1028.{ext}'), dpi=200)
print('wrote', os.path.join(HERE, 'stages_1028.png'))
