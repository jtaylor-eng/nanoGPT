"""Stage 4b figure (125M, T=2048, lr 1.2e-3, 1 seed): left = train-length loss vs ArXiv 16k loss per attention row
(the trade-off); right = ArXiv ladder per row; middle = inference-time d/q sweep on the trained q3/d2 model.
    python plot_125m_qd.py -> stage4b_qd.png / .pdf"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p)))
Ls = [512, 1024, 2048, 4096, 8192, 16384]
ROWS = [('softmax_xs', 'softmax', '#000000', 'o'), ('asentmax_zi_xs', 'ASEntmax', '#ff7f0e', 'o'),
        ('wstieltjes_q3_d2_xs', 'Stieltjes q3 d2', '#1f77b4', 's'), ('wstieltjes_q4_d2_xs', 'Stieltjes q4 d2', '#17becf', 's'),
        ('wstieltjes_q3_d3_xs', 'Stieltjes q3 d3', '#2ca02c', '^'), ('wstieltjes_q4_d3_xs', 'Stieltjes q4 d3', '#98df8a', '^')]
if os.path.exists(os.path.join(HERE, '125m/wstieltjes_q3_d4_xs/lr1.2e-3/arxiv.json')):
    ROWS.append(('wstieltjes_q3_d4_xs', 'Stieltjes q3 d4', '#9467bd', 'v'))
def ladder(d, f): L = J(f'{d}/{f}')['lengths']; return [L[str(x)]['val_loss'] if str(x) in L else float('nan') for x in Ls]

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
for base, label, color, mk in ROWS:
    d = f'125m/{base}/lr1.2e-3'
    v = J(d + '/final.json')['val_loss']; a = ladder(d, 'arxiv.json')
    a1.scatter([v], [a[-1]], color=color, marker=mk, s=60, label=label, zorder=3)
    a1.annotate(label, (v, a[-1]), fontsize=7, xytext=(4, 3), textcoords='offset points')
    a3.plot(Ls, a, '-', color=color, marker=mk, ms=4, lw=1.6, label=label)
a1.set_xlabel('FineWeb val loss at the training length (2048)'); a1.set_ylabel('ArXiv loss at 16k')
a1.set_title('train-length loss vs long-document loss (lower-left is better)', fontsize=9); a1.grid(alpha=0.3)
# middle: inference overrides
R = '125m/wstieltjes_q3_d2_xs/lr1.2e-3'
base = ladder(R, 'arxiv_base64.json')
for tag, label, color, ls in [('d1.5', 'd=1.5', '#d62728', '-'), ('d2.5', 'd=2.5', '#2ca02c', '-'), ('d3', 'd=3', '#9467bd', '-'),
                              ('q2', 'q=2', '#ff7f0e', '--'), ('q4', 'q=4', '#17becf', '--'), ('q6', 'q=6', '#8c564b', '--')]:
    y = ladder(R, f'arxiv_{tag}.json')
    a2.plot(Ls, [b - c for b, c in zip(y, base)], ls, color=color, marker='.', ms=4, lw=1.4, label=label)
a2.axhline(0, color='k', lw=1, label='trained q3 d2')
a2.set_xscale('log', base=2); a2.set_xticks(Ls); a2.set_xticklabels([str(x) for x in Ls])
a2.set_xlabel('context length'); a2.set_ylabel('ArXiv loss minus trained-setting loss')
a2.set_title('inference-time q / d override on the trained q3 d2 model (64 papers)', fontsize=9); a2.grid(alpha=0.3); a2.legend(fontsize=7, frameon=False, ncol=2)
a2.set_ylim(-0.05, 0.5)
a3.set_xscale('log', base=2); a3.set_xticks(Ls); a3.set_xticklabels([str(x) for x in Ls]); a3.axvline(2048, color='0.6', lw=0.8, ls=':')
a3.set_xlabel('context length (first L tokens of each paper)'); a3.set_ylabel('ArXiv mean loss'); a3.set_title('ArXiv ladder per row', fontsize=9)
a3.grid(alpha=0.3); a3.legend(fontsize=7, frameon=False)
fig.suptitle('Stage 4b: windowed Stieltjes q x d at 125M, T=2048, 2.5B tokens, lr 1.2e-3, 1 seed', fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'stage4b_qd.{ext}'), dpi=200)
print('wrote', os.path.join(HERE, 'stage4b_qd.png'))
