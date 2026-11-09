"""figures/qd_vs_iter.png: learned per-head q_h, d_h (median over heads, per layer) over training, per LR."""
import json, os, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..', 'results', 'lm1012', '125m', 'lwstieltjes_q3_d2_xs')
LRS = ['lr3e-4', 'lr6e-4', 'lr1.2e-3']

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey='row')
cmap = plt.get_cmap('viridis')
for j, lr in enumerate(LRS):
    recs = {}
    for l in open(os.path.join(ROOT, lr, 'metrics.jsonl')):
        r = json.loads(l)
        if r.get('kind') == 'qd':
            recs[r['iter']] = r
    its = sorted(recs)
    nl = len(recs[its[0]]['q'])
    for i, key in enumerate(['q', 'd']):
        ax = axes[i, j]
        for layer in range(nl):
            ax.plot(its, [st.median(recs[t][key][layer]) for t in its], color=cmap(layer / (nl - 1)),
                    lw=1.2, label=f'L{layer}')
        ax.axhline(3 if key == 'q' else 2, color='grey', ls=':', lw=1)
        ax.set_title(f'{key}_h, {lr[2:]}')
        ax.grid(alpha=0.3)
        if i == 1:
            ax.set_xlabel('iter')
        if j == 0:
            ax.set_ylabel(f'median {key}_h over 12 heads')
axes[0, 2].legend(fontsize=6, ncol=2)
fig.suptitle('lwstieltjes_q3_d2_xs, 125M, init q3/d2 (dotted), seed 1337')
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'qd_vs_iter.png')
fig.savefig(out, dpi=130)
print(out)
