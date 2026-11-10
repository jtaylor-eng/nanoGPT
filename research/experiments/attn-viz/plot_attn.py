"""attn-viz figures from viz_attention.py outputs (<stats dir> = .../125m/<row>/<lr>/attn_stats with
stats.npz, rows.npz, summary.json). Runs with the login node's /usr/bin/python3 (numpy + matplotlib).

    python3 plot_attn.py <figures dir> <stats dir> [<stats dir> ...]
(a) support_vs_position.png, k90_vs_position.png, entropy_vs_position.png, first64_vs_position.png: per row, mean
    over layers and heads of each group (ALiBi solid, NoPE dashed) vs query position, log-spaced bins, band = +-1 SE
    over papers.
(b) rows_<row>_L<l>H<h>.png for the 3 NoPE heads with the largest k90 growth 2k -> 16k (picked per row).
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

figs = sys.argv[1]
DIRS = {os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p)))): p for p in sys.argv[2:]}
rows = list(DIRS)
os.makedirs(figs, exist_ok=True)
COL = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#17becf', '#7f7f7f']
LABEL = {'support': 'support (nonzero keys)', 'k90': 'keys holding 90 % of mass', 'entropy': 'entropy (nats)',
         'first64': 'mass on first 64 keys', 'local2k': 'mass within 2048 back', 'dist': 'E[i - j]',
         'scale': 'AS scale s on q (window = d / s)'}


def load(r):
    s = np.load(os.path.join(DIRS[r], 'stats.npz'))
    return s, json.load(open(os.path.join(DIRS[r], 'summary.json')))


def logbin(binned):
    # binned: (docs, NL, H, nb) on 64-position linear bins -> geometric bins of bin index
    nb = binned.shape[-1]
    edges = np.unique(np.round(np.geomspace(1, nb, 40)).astype(int))
    starts = np.r_[0, edges[:-1]]
    vals = np.stack([binned[..., a:b].mean(-1) for a, b in zip(starts, edges)], -1)
    centres = 64 * np.sqrt(np.maximum(starts, 0.5) * edges)
    return centres, vals


for stat in ('support', 'k90', 'entropy', 'first64', 'local2k', 'dist', 'scale'):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c, r in zip(COL, rows):
        s, summ = load(r)
        if stat not in list(s['stats']):
            continue                                         # stats.npz written before the stat existed
        j = list(s['stats']).index(stat)
        nope = list(s['nope']); H = s['binned'].shape[3]
        for g, hs, ls in (('ALiBi', [h for h in range(H) if h not in nope], '-'), ('NoPE', nope, '--')):
            x, v = logbin(s['binned'][:, j][:, :, hs])          # (docs, NL, len(hs), nbins)
            per_doc = v.mean(axis=(1, 2))
            m = per_doc.mean(0)
            se = per_doc.std(0, ddof=1) / np.sqrt(len(per_doc)) if len(per_doc) > 1 else 0 * m
            ax.plot(x, m, ls, color=c, label=f'{r} {g}')
            ax.fill_between(x, m - se, m + se, color=c, alpha=0.15, lw=0)
    ax.set_xscale('log')
    if stat in ('support', 'k90', 'dist'):
        ax.set_yscale('log')
    ax.set_xlabel('query position'); ax.set_ylabel(LABEL[stat])
    ax.set_title(f'125M, lr 1.2e-3, seed 1337, ArXiv, mean over 12 layers x 6 heads per group')
    ax.grid(alpha=0.3, which='both', lw=0.5); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figs, f'{stat}_vs_position.png'), dpi=110); plt.close(fig)

for r in rows:
    p_rows = os.path.join(DIRS[r], 'rows.npz')
    if not os.path.exists(p_rows):
        continue
    s, summ = load(r)
    R = np.load(p_rows)
    nope = list(R['nope']); pos = list(R['rows_pos'])
    k90 = np.array(summ['table']['16384']['per_head']['k90']) / np.array(summ['table']['2048']['per_head']['k90'])
    growth = k90[:, nope]                                        # (NL, n_nope)
    top = np.dstack(np.unravel_index(np.argsort(-growth, axis=None)[:3], growth.shape))[0]
    ma = summ['model_args']
    d = ma.get('stieltjes_window') if ma['attn_type'] == 'stieltjes' else None
    for l, hi in top:
        h = nope[hi]
        fig, axs = plt.subplots(2, len(pos), figsize=(4 * len(pos), 6.5))
        for c, (P, ix) in enumerate(zip(pos, range(len(pos)))):
            p = R['p'][0, l, hi, ix, :P]; sc = R['s'][0, l, hi, ix, :P]
            ax = axs[0, c]
            nz = p > 0
            ax.scatter(np.arange(P)[nz], p[nz], s=2, color=COL[0])
            ax.set_yscale('log'); ax.set_xlabel('key position'); ax.set_ylabel('weight')
            ax.set_title(f'query {P}: {nz.sum()} nonzero, first64 {p[:64].sum():.2f}', fontsize=9)
            ax = axs[1, c]
            gap = sc.max() - sc
            o = np.argsort(gap)
            ax.scatter(gap[o][nz[o]], p[o][nz[o]], s=2, color=COL[1])
            if d:
                ax.axvline(gap[nz].max(), color='k', lw=0.8, ls='--', label=f'support edge (d={d:g})')
                ax.legend(fontsize=7)
            ax.set_yscale('log'); ax.set_xscale('symlog', linthresh=0.1)
            ax.set_xlabel('score gap from top key'); ax.set_ylabel('weight')
        fig.suptitle(f'{r} layer {l} head {h} (NoPE), ArXiv paper 0; k90 growth 2k->16k x{growth[l, hi]:.1f}', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(figs, f'rows_{r}_L{l}H{h}.png'), dpi=100); plt.close(fig)
print('wrote', figs)
