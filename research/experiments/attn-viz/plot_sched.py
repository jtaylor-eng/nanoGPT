"""e004 figure: NoPE support vs position and per-position loss delta vs a = 0, for the inference-time NoPE window
schedule on wstieltjes_q3_d2_xs. Login node: /usr/bin/python3 plot_sched.py <run dir> <out png>"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run, out = sys.argv[1], sys.argv[2]
A = ["-0.2", "-0.1", "-0.05", "0", "0.05", "0.1", "0.2"]
S = {a: np.load(os.path.join(run, f'attn_stats_sched_a{a}', 'stats.npz')) for a in A}
nope = list(S['0']['nope']); j = list(S['0']['stats']).index('support')
nb = S['0']['loss_binned'].shape[1]
edges = np.unique(np.round(np.geomspace(1, nb, 40)).astype(int)); starts = np.r_[0, edges[:-1]]
x = 64 * np.sqrt(np.maximum(starts, 0.5) * edges)
lb = lambda v: np.stack([v[..., a:b].mean(-1) for a, b in zip(starts, edges)], -1)
base = S['0']['loss_binned']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c, a in zip(["#7b3294", "#c2a5cf", "#a6dba0", "#000000", "#2ca02c", "#ff7f0e", "#d62728"], A):
    sup = S[a]['binned'][:, j][:, :, nope].mean(axis=(1, 2))           # (docs, nbins)
    ax1.plot(x, lb(sup).mean(0), color=c, label=f'a = {a}')
    d = lb(S[a]['loss_binned'] - base)                                  # paired per paper
    m, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
    ax2.plot(x, m, color=c, label=f'a = {a}'); ax2.fill_between(x, m - se, m + se, color=c, alpha=0.15, lw=0)
for ax in (ax1, ax2):
    ax.set_xscale('log'); ax.axvline(2048, color='k', lw=0.6, ls=':'); ax.grid(alpha=0.3, which='both', lw=0.5)
    ax.set_xlabel('query position'); ax.legend(fontsize=8)
ax1.set_yscale('log'); ax1.set_ylabel('NoPE support (nonzero keys)')
ax2.set_ylabel('loss - loss(a = 0), nats (+-1 SE over 8 papers)')
fig.suptitle('wstieltjes_q3_d2_xs 125M lr 1.2e-3 seed 1337, ArXiv: NoPE window d / (1 + a ln(max(i, 2048) / 2048))', fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=110)
for a in A:
    t = S[a]['loss_binned'][:, 32:].mean(1) - base[:, 32:].mean(1)
    h = S[a]['loss_binned'][:, :32].mean(1) - base[:, :32].mean(1)
    sup = S[a]['binned'][:, j][:, :, nope].mean(axis=(1, 2))
    print(f'a={a}: tail loss {S[a]["loss_binned"][:, 32:].mean():.4f} delta {t.mean():+.4f} +- {t.std(ddof=1)/np.sqrt(len(t)):.4f} '
          f'(papers worse {int((t > 0).sum())}/8); head delta max |{np.abs(h).max():.1e}|; NoPE support 2k {sup[:, 31].mean():.1f} 16k {sup[:, -1].mean():.1f}')
